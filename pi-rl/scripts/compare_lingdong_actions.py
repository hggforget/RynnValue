"""Run offline Lingdong action comparisons and export aligned line plots."""

import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.parquet as pq


def log(event, **values):
    print(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': event, **values}), flush=True)


def select_episodes(root, requested):
    episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line]
    indexed = {episode['episode_index']: episode for episode in episodes}
    if requested is not None:
        return [indexed[index] for index in requested]
    result = []
    for task in ('pick up the object', 'put down the object'):
        group = [episode for episode in episodes if episode['tasks'] == [task]]
        result.extend(group[index] for index in dict.fromkeys((0, len(group) // 2, len(group) - 1)))
    return result


def read_video(path, frames, timestamps, length):
    selected = set(frames.tolist())
    images = {}
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if index in selected:
                if frame.time is None or abs(frame.time - timestamps[index]) > 1 / 60:
                    raise ValueError(f'Video/Parquet timestamp mismatch at {path}, frame {index}')
                images[index] = frame.to_ndarray(format='rgb24')
    if count != length or set(images) != selected:
        raise ValueError(f'Video frame count mismatch: {path}, expected {length}, got {count}')
    return images


def load_episode(root, info, episode, frame_step, video_keys, action_indices):
    index = episode['episode_index']
    fields = {'episode_index': index, 'episode_chunk': index // info['chunks_size']}
    table = pq.read_table(root / info['data_path'].format(**fields))
    if len(table) != episode['length']:
        raise ValueError(f'Parquet length mismatch in episode {index}')
    timestamps = np.asarray(table['timestamp'].to_pylist(), dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(f'Non-increasing timestamps in episode {index}')
    frames = np.unique(np.r_[np.arange(0, len(table), frame_step), len(table) - 1]).astype(np.int64)
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    gt = np.asarray(table['action'].to_pylist(), dtype=np.float32)[:, action_indices]
    if not np.isfinite(state).all() or not np.isfinite(gt).all():
        raise ValueError(f'Nonfinite recorded state/action in episode {index}')
    task_indices = np.asarray(table['task_index'].to_pylist(), dtype=np.int64)
    tasks = {record['task_index']: record['task'] for record in
             (json.loads(line) for line in (root / 'meta/tasks.jsonl').read_text().splitlines() if line)}
    images = {key: read_video(root / info['video_path'].format(**fields, video_key=key),
                              frames, timestamps, len(table)) for key in video_keys}
    log('episode_decoded', episode=index, frames=len(table), requests=len(frames), task=episode['tasks'][0])
    return {'frames': frames, 'timestamps': timestamps, 'state': state, 'gt': gt,
            'task_indices': task_indices, 'tasks': tasks, 'images': images}


def build_inference(args):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies import lingdong_policy
    from openpi.shared import array_typing
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from openpi.training import sharding

    config = config_lib.get_config(lingdong_policy.CONFIG_NAME)
    data_factory = dataclasses.replace(
        config.data, dataset_root=str(args.dataset.parent),
        assets=dataclasses.replace(config.data.assets, assets_dir=str(args.checkpoint / 'assets')),
    )
    with jax.default_device(jax.devices('cpu')[0]):
        data = data_factory.create(config.assets_dirs, config.model)
    norm = normalize.load(args.checkpoint / 'assets' / lingdong_policy.ASSET_ID)
    input_transform = transforms.compose([
        *data.repack_transforms.inputs, *data.data_transforms.inputs,
        transforms.Normalize(norm, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs,
    ])
    output_transform = transforms.compose([
        *data.model_transforms.outputs,
        transforms.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
        *data.data_transforms.outputs, *data.repack_transforms.outputs,
    ])
    devices = jax.devices('gpu')
    log('restore_started', checkpoint=str(args.checkpoint), gpu_devices=len(devices))
    started = time.monotonic()
    # Restore on the host before placing parameters on the selected device mesh.
    params = model_lib.restore_params(args.checkpoint / 'params', restore_type=np.ndarray, dtype=jnp.bfloat16)
    log('host_params_restored', seconds=round(time.monotonic() - started, 2),
        bytes=int(sum(value.nbytes for value in jax.tree.leaves(params))))
    mesh = sharding.make_mesh(len(devices))
    param_sharding = sharding.fsdp_sharding(params, mesh, min_size_mbytes=1)
    params = jax.device_put(params, param_sharding)
    jax.block_until_ready(params)
    with jax.default_device(jax.devices('cpu')[0]):
        model = config.model.load(params)
    graph, state = nnx.split(model)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def sample(model_state, observation, key, noise):
        module = nnx.merge(graph, model_state)
        return module.sample_actions(key, observation, num_steps=args.denoising_steps, noise=noise)

    sampler = jax.jit(sample, out_shardings=replicated)
    fixed_noise = np.random.default_rng(args.seed).standard_normal(
        (1, config.model.action_horizon, config.model.action_dim)).astype(np.float32)
    np.save(args.output / 'fixed_noise.npy', fixed_noise)
    noise = jax.device_put(fixed_noise, replicated)
    key = jax.device_put(np.array([0, args.seed], dtype=np.uint32), replicated)
    compiled = None

    def infer(raw):
        nonlocal compiled
        transformed = input_transform(raw)
        batched = jax.tree.map(lambda value: np.asarray(value)[None, ...], transformed)
        observation = model_lib.Observation.from_dict(batched)
        observation = jax.device_put(observation, replicated)
        if compiled is None:
            log('compilation_started')
            begin = time.monotonic()
            # JAX lowering reconstructs Observation with ArgInfo placeholders.
            with array_typing.disable_typechecking():
                compiled = sampler.lower(state, observation, key, noise).compile()
            memory = compiled.memory_analysis()
            log('compilation_finished', seconds=round(time.monotonic() - begin, 2),
                argument_bytes=memory.argument_size_in_bytes, temporary_bytes=memory.temp_size_in_bytes,
                output_bytes=memory.output_size_in_bytes)
        begin = time.monotonic()
        with array_typing.disable_typechecking():
            actions = np.asarray(compiled(state, observation, key, noise)[0], dtype=np.float32)
        result = output_transform({'state': transformed['state'], 'actions': actions})['actions']
        if result.shape != (30, 8) or not np.isfinite(result).all():
            raise ValueError(f'Invalid predicted action shape/values: {result.shape}')
        return result, (time.monotonic() - begin) * 1000

    return infer


def collect(args):
    from openpi.policies import lingdong_policy

    args.output.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    state_indices, action_indices = lingdong_policy.dataset_indices((str(args.dataset),))
    episodes = select_episodes(args.dataset, args.episodes)
    metadata = {
        'checkpoint': str(args.checkpoint), 'dataset': str(args.dataset), 'fps': info['fps'],
        'frame_step': args.frame_step, 'seed': args.seed, 'denoising_steps': args.denoising_steps,
        'channels': list(lingdong_policy.RIGHT_NAMES),
        'episode_selection': 'First, middle and last episode of each task, chosen before inference.'
                             if args.episodes is None else 'Explicit episode IDs.',
        'comparison': 'Prediction[t, 0, :] versus recorded action[t, :]; all observations are recorded dataset observations.',
        'noise': 'One fixed seeded Gaussian noise tensor is reused across observations to control sampling variability.',
        'scope': 'Offline fitting on samples from the training dataset; no closed-loop robot execution.',
        'norm_stats_sha256': hashlib.sha256((args.checkpoint / 'assets/lingdong_right_arm/norm_stats.json').read_bytes()).hexdigest(),
        'episode_ids': [episode['episode_index'] for episode in episodes],
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    log('evaluation_plan', **metadata)
    infer = build_inference(args)
    for episode in episodes:
        index = episode['episode_index']
        loaded = load_episode(args.dataset, info, episode, args.frame_step, lingdong_policy.VIDEO_KEYS, action_indices)
        chunks, timings = [], []
        for count, frame in enumerate(loaded['frames']):
            raw = {'observation.state': loaded['state'][frame],
                   'prompt': loaded['tasks'][int(loaded['task_indices'][frame])],
                   **{name: images[int(frame)] for name, images in loaded['images'].items()}}
            prediction, timing = infer(raw)
            chunks.append(prediction)
            timings.append(timing)
            if count == 0 or (count + 1) % 20 == 0 or count + 1 == len(loaded['frames']):
                log('inference_progress', episode=index, completed=count + 1,
                    requests=len(loaded['frames']), last_inference_ms=round(timing, 2))
        np.savez_compressed(
            args.output / f'episode_{index:06d}.npz', frames=loaded['frames'],
            timestamps=loaded['timestamps'], gt=loaded['gt'], predictions=np.stack(chunks),
            right_state=loaded['state'][:, state_indices],
            inference_ms=np.asarray(timings), task=np.asarray(episode['tasks'][0]),
        )
        log('episode_saved', episode=index)


def metrics_for(records):
    errors = np.concatenate([record['predictions'][:, 0, :] - record['gt'][record['frames']] for record in records])
    sums = np.zeros((30, 8), dtype=np.float64)
    counts = np.zeros(30, dtype=np.int64)
    for record in records:
        for offset in range(30):
            targets = record['frames'] + offset
            valid = targets < len(record['gt'])
            sums[offset] += np.abs(record['predictions'][valid, offset] - record['gt'][targets[valid]]).sum(axis=0)
            counts[offset] += np.count_nonzero(valid)
    result = {'requests': len(errors), 'mae': np.abs(errors).mean(axis=0).tolist(),
            'rmse': np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
            'horizon_mae': (sums / counts[:, None]).tolist(), 'horizon_valid_requests': counts.tolist()}
    if all('right_state' in record for record in records):
        baseline = np.concatenate([record['right_state'][record['frames']] - record['gt'][record['frames']]
                                   for record in records])
        future = []
        for record in records:
            frames = record['frames'][record['frames'] + 29 < len(record['gt'])]
            future.append(record['right_state'][frames] - record['gt'][frames + 29])
        result['state_copy_baseline'] = {
            'description': 'Predict the current recorded right-arm state at every horizon.',
            'first_step_mae': np.abs(baseline).mean(axis=0).tolist(),
            'horizon29_mae': np.abs(np.concatenate(future)).mean(axis=0).tolist(),
        }
    return result


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    metadata = json.loads((args.output / 'metadata.json').read_text())
    records = []
    for index in metadata['episode_ids']:
        with np.load(args.output / f'episode_{index:06d}.npz', allow_pickle=False) as archive:
            record = dict(archive)
        record['episode'] = index
        record['task'] = str(record['task'])
        records.append(record)
    channels = metadata['channels']
    short_names = [name.replace('_Joint', '') for name in channels]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#cbd5e1', 'text.color': '#0f172a'})

    def panel(axis, record, channel):
        frames = record['frames']
        gt = record['gt'][:, channel]
        prediction = record['predictions'][:, 0, channel]
        mae = float(np.abs(prediction - gt[frames]).mean())
        axis.plot(record['timestamps'], gt, color='#334155', linewidth=1.7, label='GT action')
        axis.plot(record['timestamps'][frames], prediction, color='#ea580c', linewidth=1.2,
                  marker='.', markersize=2.5, alpha=0.9, label='Pi0.5 prediction (first step)')
        axis.set_title(f'{short_names[channel]}   |   MAE {mae:.4g}', loc='left', fontsize=11)
        axis.set_ylabel('Recorded gripper target' if channel == 7 else 'Recorded joint target')
        axis.set_xlabel('Episode time (s)')
        axis.grid(alpha=0.18)
        axis.set_xlim(record['timestamps'][0], record['timestamps'][-1])

    pdf_path = args.output / 'action_comparison.pdf'
    with PdfPages(pdf_path) as pdf:
        task_groups = [[record for record in records if record['task'] == task]
                       for task in ('pick up the object', 'put down the object')]
        representatives = [group[len(group) // 2] for group in task_groups if group]
        fig, axes = plt.subplots(8, len(representatives), figsize=(9 * len(representatives), 19), squeeze=False)
        for column, record in enumerate(representatives):
            for channel in range(8):
                panel(axes[channel, column], record, channel)
            axes[0, column].set_title(f"Episode {record['episode']} | {record['task']}\n" +
                                      axes[0, column].get_title(loc='left'), loc='left', fontsize=12)
        fig.suptitle('Lingdong Pi0.5 checkpoint 10000: predicted actions vs GT', x=0.06, ha='left', fontsize=20)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.985, 0.978), ncol=2)
        fig.text(0.06, 0.954, f"Recorded observations; action chunk step 0 aligned to the same-frame GT; one request every {metadata['frame_step']} frames.", fontsize=10)
        fig.tight_layout(rect=(0.01, 0.015, 0.99, 0.94), h_pad=1.4, w_pad=2)
        fig.savefig(args.output / 'overview.png', dpi=150, facecolor='white')
        pdf.savefig(fig)
        plt.close(fig)
        # A second view compares one complete forecast with its future GT,
        # without replanning from the intervening recorded observations.
        fig, axes = plt.subplots(8, len(representatives), figsize=(9 * len(representatives), 19), squeeze=False)
        chunk_examples = []
        for column, record in enumerate(representatives):
            eligible = np.flatnonzero(record['frames'] + 29 < len(record['gt']))
            sample = eligible[np.argmin(np.abs(record['frames'][eligible] - len(record['gt']) / 2))]
            anchor = int(record['frames'][sample])
            prediction = record['predictions'][sample]
            reference = record['gt'][anchor:anchor + 30]
            offsets = np.arange(30) / metadata['fps']
            errors = np.abs(prediction - reference).mean(axis=0)
            chunk_examples.append({'episode': record['episode'], 'frame': anchor,
                                   'timestamp': float(record['timestamps'][anchor]), 'mae': errors.tolist()})
            for channel in range(8):
                axis = axes[channel, column]
                axis.plot(offsets, reference[:, channel], color='#334155', linewidth=1.7, label='GT future actions')
                axis.plot(offsets, prediction[:, channel], color='#ea580c', linewidth=1.2,
                          marker='.', markersize=3, label='Pi0.5 prediction (30-step chunk)')
                title = f'{short_names[channel]}   |   chunk MAE {errors[channel]:.4g}'
                if channel == 0:
                    title = f"Episode {record['episode']} | frame {anchor} | {record['task']}\n" + title
                axis.set_title(title, loc='left', fontsize=11)
                axis.set_ylabel('Recorded gripper target' if channel == 7 else 'Recorded joint target')
                axis.set_xlabel('Prediction horizon (s)')
                axis.set_xlim(offsets[0], offsets[-1])
                axis.grid(alpha=0.18)
            for channel, axis in enumerate(axes[:, column]):
                if np.ptp(np.r_[prediction[:, channel], reference[:, channel]]) < 0.02:
                    center = float(np.mean(reference[:, channel]))
                    axis.set_ylim(center - 0.02, center + 0.02)
        fig.suptitle('Checkpoint 10000: a complete 30-step prediction vs future GT',
                     x=0.06, ha='left', fontsize=20)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.985, 0.978), ncol=2)
        fig.text(0.06, 0.954, 'One observation near the middle of each episode; all 30 future actions are predicted from that observation.', fontsize=10)
        fig.tight_layout(rect=(0.01, 0.015, 0.99, 0.94), h_pad=1.4, w_pad=2)
        fig.savefig(args.output / 'action_chunk_comparison.png', dpi=150, facecolor='white')
        pdf.savefig(fig)
        plt.close(fig)
        for record in records:
            fig, axes = plt.subplots(4, 2, figsize=(16, 12))
            for channel, axis in enumerate(axes.flat):
                panel(axis, record, channel)
            fig.suptitle(f"Checkpoint 10000 | episode {record['episode']} | {record['task']}",
                         x=0.065, ha='left', fontsize=18)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.98, 0.946), ncol=2)
            fig.text(0.065, 0.945, 'Offline recorded observations; raw action units; fixed sampling noise.', fontsize=10)
            fig.tight_layout(rect=(0.015, 0.015, 0.99, 0.915), h_pad=1.4)
            fig.savefig(args.output / f"episode_{record['episode']:06d}_actions.png", dpi=150, facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)
    summary = {'metadata': metadata, 'overall': metrics_for(records), 'chunk_examples': chunk_examples,
               'tasks': {task: metrics_for(group) for task, group in zip(
                   ('pick up the object', 'put down the object'), task_groups) if group},
               'episodes': {str(record['episode']): {'task': record['task'], **metrics_for([record])}
                            for record in records}}
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    csv_path = args.output / 'first_step_comparison.csv'
    with csv_path.open('w') as output:
        header = ['episode', 'frame', 'time_seconds', *[f'gt_{name}' for name in channels],
                  *[f'prediction_{name}' for name in channels]]
        output.write(','.join(header) + '\n')
        for record in records:
            frames = record['frames']
            values = np.column_stack([np.full(len(frames), record['episode']), frames,
                                      record['timestamps'][frames], record['gt'][frames],
                                      record['predictions'][:, 0]])
            np.savetxt(output, values, delimiter=',', fmt='%.9g')
    log('plots_saved', directory=str(args.output), requests=summary['overall']['requests'],
        mae=summary['overall']['mae'], rmse=summary['overall']['rmse'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/oss/embodied-ai-data/lingdong/lerobot_2026-10-26_12-09-32_final'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/oss/embodied-ai-data/lingdong/pi05_lingdong_right_arm/lingdong_right_arm_1027_163358/10000'))
    parser.add_argument('--output', type=Path, default=Path('../output/evaluation/lingdong_step10000_action_compare'))
    parser.add_argument('--episodes', type=int, nargs='+')
    parser.add_argument('--frame-step', type=int, default=5)
    parser.add_argument('--denoising-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.frame_step < 1 or args.denoising_steps < 1:
        parser.error('Frame step and denoising steps must be positive')
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not args.plot_only:
        collect(args)
    plot(args)


if __name__ == '__main__':
    main()
