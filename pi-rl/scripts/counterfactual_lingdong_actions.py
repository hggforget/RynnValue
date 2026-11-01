"""Paired task swaps on recorded observations with fixed checkpoint and noise."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from compare_lingdong_actions import build_inference, load_episode, log, select_episodes


TASK_SWAP = {
    'pick up the object': 'put down the object',
    'put down the object': 'pick up the object',
}


def collect(args):
    from openpi.policies import lingdong_policy

    args.output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((args.baseline / 'metadata.json').read_text())
    norm_sha = hashlib.sha256((args.checkpoint / 'assets/lingdong_right_arm/norm_stats.json').read_bytes()).hexdigest()
    assert norm_sha == prior['norm_stats_sha256'], 'Normalization differs from the baseline'
    assert args.checkpoint.name == prior['checkpoint_step'], 'Checkpoint step differs from the baseline'
    assert args.checkpoint.parent.name == Path(prior['checkpoint']).parent.name, 'Training run differs from the baseline'
    assert args.seed == prior['seed'] and args.denoising_steps == prior['denoising_steps']
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    state_indices, action_indices = lingdong_policy.dataset_indices((str(args.dataset),))
    episodes = select_episodes(args.dataset, prior['episode_ids'])
    metadata = {
        'checkpoint': str(args.checkpoint), 'baseline': str(args.baseline), 'dataset': str(args.dataset),
        'checkpoint_step': args.checkpoint.name, 'episode_ids': prior['episode_ids'],
        'episode_selection': prior['episode_selection'], 'fps': info['fps'],
        'frame_step': prior['frame_step'], 'seed': args.seed, 'denoising_steps': args.denoising_steps,
        'channels': list(lingdong_policy.RIGHT_NAMES), 'norm_stats_sha256': norm_sha,
        'task_swap': TASK_SWAP,
        'controls': 'Identical recorded images, state, checkpoint, Gaussian noise, RNG key and denoising steps. Only prompt changes.',
        'baseline_validation': 'Recompute original task at first/middle/last sampled observations of every episode. Recompute all originals if max difference exceeds 1e-5.',
        'repeat_control': 'Repeat the identical original-task request on the first observation of each episode.',
        'metric_alignment': 'Use only within-episode future GT for every horizon. Joint metrics exclude the binary gripper.',
        'scope': 'Offline task swaps on sampled training trajectories. GT belongs to the recorded task; there is no counterfactual-task GT or robot rollout.',
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    log('counterfactual_plan', **metadata)
    infer = build_inference(args)
    assert np.array_equal(np.load(args.output / 'fixed_noise.npy'), np.load(args.baseline / 'fixed_noise.npy'))
    controls = []
    for episode in episodes:
        index = episode['episode_index']
        loaded = load_episode(args.dataset, info, episode, prior['frame_step'], lingdong_policy.VIDEO_KEYS, action_indices)
        with np.load(args.baseline / f'episode_{index:06d}.npz', allow_pickle=False) as archive:
            original = archive['predictions'].copy()
            for name in ('frames', 'timestamps', 'gt'):
                assert np.array_equal(loaded[name], archive[name]), f'{name} changed in episode {index}'
            assert np.array_equal(loaded['state'][:, state_indices], archive['right_state'])
        tasks = [loaded['tasks'][int(loaded['task_indices'][frame])] for frame in loaded['frames']]
        assert len(set(tasks)) == 1 and tasks[0] in TASK_SWAP
        task, swapped_task = tasks[0], TASK_SWAP[tasks[0]]

        def observation(sample, prompt):
            frame = int(loaded['frames'][sample])
            return {'observation.state': loaded['state'][frame].copy(), 'prompt': prompt,
                    **{name: images[frame] for name, images in loaded['images'].items()}}

        anchors = sorted(set((0, len(loaded['frames']) // 2, len(loaded['frames']) - 1)))
        checks = {}
        for anchor in anchors:
            checks[anchor], _ = infer(observation(anchor, task))
        repeated, _ = infer(observation(0, task))
        repeat_error = float(np.max(np.abs(repeated - checks[0])))
        assert repeat_error < 1e-5, f'Unexpected same-prompt sampling variation: {repeat_error}'
        cached_error = max(float(np.max(np.abs(value - original[anchor]))) for anchor, value in checks.items())
        recompute = cached_error > 1e-5
        control = {'episode': index, 'checked_frames': loaded['frames'][anchors].tolist(),
                   'cached_baseline_max_abs_difference': cached_error,
                   'same_prompt_repeat_max_abs_difference': repeat_error, 'recomputed_all_originals': recompute}
        controls.append(control)
        log('baseline_validated', **control)
        swapped, timings = [], []
        for sample in range(len(loaded['frames'])):
            if recompute:
                original[sample] = checks[sample] if sample in checks else infer(observation(sample, task))[0]
            prediction, timing = infer(observation(sample, swapped_task))
            swapped.append(prediction)
            timings.append(timing)
            if sample == 0 or (sample + 1) % 20 == 0 or sample + 1 == len(loaded['frames']):
                log('counterfactual_progress', episode=index, completed=sample + 1,
                    requests=len(loaded['frames']), last_inference_ms=round(timing, 2))
        np.savez_compressed(
            args.output / f'episode_{index:06d}.npz', frames=loaded['frames'], timestamps=loaded['timestamps'],
            gt=loaded['gt'], right_state=loaded['state'][:, state_indices], original=original,
            swapped=np.stack(swapped), task=np.asarray(task), swapped_task=np.asarray(swapped_task),
            inference_ms=np.asarray(timings),
        )
        (args.output / 'controls.json').write_text(json.dumps(controls, indent=2) + '\n')
        log('counterfactual_episode_saved', episode=index)


def metrics_for(records):
    sums = {key: np.zeros((30, 8), dtype=np.float64) for key in ('original_error', 'swapped_error', 'paired_delta')}
    counts = np.zeros(30, dtype=np.int64)
    gripper_flips = np.zeros(30, dtype=np.int64)
    original_correct = np.zeros(30, dtype=np.int64)
    swapped_correct = np.zeros(30, dtype=np.int64)
    first_deltas = []
    for record in records:
        first_deltas.append(np.abs(record['swapped'][:, 0, :7] - record['original'][:, 0, :7]).mean(axis=1))
        for horizon in range(30):
            targets = record['frames'] + horizon
            valid = targets < len(record['gt'])
            original = record['original'][valid, horizon]
            swapped = record['swapped'][valid, horizon]
            gt = record['gt'][targets[valid]]
            sums['original_error'][horizon] += np.abs(original - gt).sum(axis=0)
            sums['swapped_error'][horizon] += np.abs(swapped - gt).sum(axis=0)
            sums['paired_delta'][horizon] += np.abs(swapped - original).sum(axis=0)
            counts[horizon] += len(original)
            gripper_flips[horizon] += np.sum((original[:, 7] >= 0.5) != (swapped[:, 7] >= 0.5))
            original_correct[horizon] += np.sum((original[:, 7] >= 0.5) == (gt[:, 7] >= 0.5))
            swapped_correct[horizon] += np.sum((swapped[:, 7] >= 0.5) == (gt[:, 7] >= 0.5))
    averages = {key: value / counts[:, None] for key, value in sums.items()}
    flattened = np.concatenate(first_deltas)
    result = {
        'observations': int(counts[0]), 'valid_predictions_per_horizon': counts.tolist(),
        **{f'{key}_by_horizon': value.tolist() for key, value in averages.items()},
        'gripper_flip_rate_by_horizon': (gripper_flips / counts).tolist(),
        'original_gripper_accuracy_by_horizon': (original_correct / counts).tolist(),
        'swapped_gripper_accuracy_by_horizon': (swapped_correct / counts).tolist(),
        'first_step_joint_delta_median': float(np.median(flattened)),
        'first_step_joint_delta_p95': float(np.quantile(flattened, 0.95)),
    }
    for horizon, name in ((0, 'first_step'), (29, 'last_step')):
        original_error = float(averages['original_error'][horizon, :7].mean())
        swapped_error = float(averages['swapped_error'][horizon, :7].mean())
        result[name] = {
            'joint_paired_mean_absolute_difference': float(averages['paired_delta'][horizon, :7].mean()),
            'joint_original_gt_mae': original_error, 'joint_swapped_gt_mae': swapped_error,
            'swapped_to_original_gt_mae_ratio': swapped_error / original_error if original_error else None,
            'gripper_mean_absolute_difference': float(averages['paired_delta'][horizon, 7]),
            'gripper_flip_rate': float(gripper_flips[horizon] / counts[horizon]),
            'gripper_flip_count': int(gripper_flips[horizon]),
            'original_gripper_accuracy': float(original_correct[horizon] / counts[horizon]),
            'swapped_gripper_accuracy': float(swapped_correct[horizon] / counts[horizon]),
        }
    result['all_valid_horizons'] = {
        'joint_paired_mean_absolute_difference': float(sums['paired_delta'][:, :7].sum() / (counts.sum() * 7)),
        'joint_original_gt_mae': float(sums['original_error'][:, :7].sum() / (counts.sum() * 7)),
        'joint_swapped_gt_mae': float(sums['swapped_error'][:, :7].sum() / (counts.sum() * 7)),
        'gripper_flip_rate': float(gripper_flips.sum() / counts.sum()),
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
        record.update(episode=index, task=str(record['task']), swapped_task=str(record['swapped_task']))
        assert record['original'].shape == record['swapped'].shape == (len(record['frames']), 30, 8)
        assert np.isfinite(record['original']).all() and np.isfinite(record['swapped']).all()
        records.append(record)
    groups = {task: [record for record in records if record['task'] == task] for task in TASK_SWAP}
    summary = {'metadata': metadata, 'overall': metrics_for(records),
               'tasks': {task: metrics_for(group) for task, group in groups.items()},
               'episodes': {str(record['episode']): {'task': record['task'], **metrics_for([record])} for record in records},
               'controls': json.loads((args.output / 'controls.json').read_text())}
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    names = [name.replace('_Joint', '') for name in metadata['channels']]
    representatives = [group[len(group) // 2] for group in groups.values()]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'text.color': '#0f172a'})
    colors = {'gt': '#334155', 'original': '#ea580c', 'swapped': '#7c3aed'}

    def panel(axis, record, channel, chunk_sample=None):
        if chunk_sample is None:
            x = record['timestamps'][record['frames']]
            gt_x, gt = record['timestamps'], record['gt'][:, channel]
            original, swapped = record['original'][:, 0, channel], record['swapped'][:, 0, channel]
            axis.set_xlabel('Episode time (s)')
        else:
            frame = int(record['frames'][chunk_sample])
            x = gt_x = np.arange(30) / metadata['fps']
            gt = record['gt'][frame:frame + 30, channel]
            original, swapped = record['original'][chunk_sample, :, channel], record['swapped'][chunk_sample, :, channel]
            axis.set_xlabel('Prediction horizon (s)')
        axis.plot(gt_x, gt, color=colors['gt'], linewidth=2, alpha=0.8, label='GT (recorded task)')
        axis.plot(x, original, color=colors['original'], linewidth=1.4, label='Original task')
        axis.plot(x, swapped, color=colors['swapped'], linewidth=1.2, linestyle='--', label='Swapped task')
        delta = float(np.abs(original - swapped).mean())
        axis.set_title(f'{names[channel]} | mean task-swap difference {delta:.4g}', loc='left', fontsize=11)
        axis.set_ylabel('Gripper command' if channel == 7 else 'Joint target (dataset units)')
        axis.set_xlim(x[0], x[-1])
        axis.grid(alpha=0.18)
        if channel == 7:
            axis.set_ylim(-0.08, 1.08)
        elif chunk_sample is not None and np.ptp(np.r_[gt, original, swapped]) < 0.04:
            center = float(np.mean(gt))
            axis.set_ylim(center - 0.025, center + 0.025)

    with PdfPages(args.output / 'counterfactual_comparison.pdf') as pdf:
        for mode in ('trajectory', 'chunk'):
            fig, axes = plt.subplots(8, 2, figsize=(18, 19), squeeze=False)
            for column, record in enumerate(representatives):
                sample = None
                if mode == 'chunk':
                    eligible = np.flatnonzero(record['frames'] + 29 < len(record['gt']))
                    sample = int(eligible[np.argmin(np.abs(record['frames'][eligible] - len(record['gt']) / 2))])
                for channel in range(8):
                    panel(axes[channel, column], record, channel, sample)
                caption = f"Episode {record['episode']} | {record['task']} -> {record['swapped_task']}"
                if sample is not None:
                    caption += f" | frame {record['frames'][sample]}"
                axes[0, column].set_title(caption + '\n' + axes[0, column].get_title(loc='left'), loc='left', fontsize=11)
            title = 'first-step actions across the trajectory' if mode == 'trajectory' else 'a complete 30-step action chunk'
            fig.suptitle(f"Checkpoint {metadata['checkpoint_step']}: task swap | {title}", x=0.06, ha='left', fontsize=19)
            fig.text(0.06, 0.954, 'Same recorded images + state + Gaussian noise; only task text changes. GT belongs to the original task.', fontsize=10)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.99, 0.982), ncol=3)
            fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.94), h_pad=1.5, w_pad=2)
            filename = 'counterfactual_overview.png' if mode == 'trajectory' else 'counterfactual_chunks.png'
            fig.savefig(args.output / filename, dpi=150, facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)

        overall = summary['overall']
        horizons = np.arange(30) / metadata['fps']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for task, values in summary['tasks'].items():
            delta = np.asarray(values['paired_delta_by_horizon'])
            axes[0, 0].plot(horizons, delta[:, :7].mean(axis=1), label=f'{task} -> {TASK_SWAP[task]}')
        axes[0, 0].set_title('How much do predicted joints change when the task is swapped?', loc='left', fontsize=11)
        axes[0, 0].set_ylabel('Mean absolute difference, 7 joints')
        axes[0, 0].legend(fontsize=8)
        for key, label, color in (('original_error_by_horizon', 'Original task', colors['original']),
                                  ('swapped_error_by_horizon', 'Swapped task', colors['swapped'])):
            axes[0, 1].plot(horizons, np.asarray(overall[key])[:, :7].mean(axis=1), label=label, color=color)
        axes[0, 1].set_title('Error against recorded original-task GT', loc='left')
        axes[0, 1].set_ylabel('Mean of 7 joint MAEs (dataset units)')
        axes[0, 1].legend()
        positions = np.arange(7)
        deltas = np.asarray(overall['paired_delta_by_horizon'])
        axes[1, 0].bar(positions - 0.18, deltas[0, :7], 0.36, label='First step', color='#60a5fa')
        axes[1, 0].bar(positions + 0.18, deltas[-1, :7], 0.36, label='Step 30', color=colors['swapped'])
        axes[1, 0].set_xticks(positions, names[:7], rotation=25, ha='right')
        axes[1, 0].set_title('Task-swap difference by joint', loc='left')
        axes[1, 0].set_ylabel('Mean absolute paired difference')
        axes[1, 0].legend()
        axes[1, 1].plot(horizons, np.asarray(overall['gripper_flip_rate_by_horizon']) * 100, color=colors['swapped'])
        axes[1, 1].set_title('Does the task swap change the binary gripper decision?', loc='left', fontsize=11)
        axes[1, 1].set_ylabel('Opposite decisions (%) at threshold 0.5')
        for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
            axis.set_xlabel('Prediction horizon (s)')
        for axis in axes.flat:
            axis.grid(alpha=0.18)
        fig.suptitle(f"Checkpoint {metadata['checkpoint_step']} | paired task swaps | {overall['observations']} fixed observations", fontsize=17)
        fig.text(0.02, 0.01, 'Only prompt changes. Training-trajectory samples; differences do not measure counterfactual task success.', fontsize=10)
        fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=2)
        fig.savefig(args.output / 'task_swap_effect.png', dpi=160, facecolor='white')
        pdf.savefig(fig)
        plt.close(fig)

        for record in records:
            fig, axes = plt.subplots(4, 2, figsize=(16, 12))
            for channel, axis in enumerate(axes.flat):
                panel(axis, record, channel)
            fig.suptitle(f"Episode {record['episode']} | {record['task']} -> {record['swapped_task']}", x=0.06, ha='left', fontsize=17)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.99, 0.96), ncol=3)
            fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.93), h_pad=1.4)
            fig.savefig(args.output / f"episode_{record['episode']:06d}_counterfactual.png", dpi=150, facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)

    with (args.output / 'first_step_paired_actions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['episode', 'frame', 'time_seconds', 'original_task', 'swapped_task',
                         *[f'{prefix}_{name}' for prefix in ('gt', 'original', 'swapped', 'delta') for name in metadata['channels']]])
        for record in records:
            for sample, frame in enumerate(record['frames']):
                original, swapped = record['original'][sample, 0], record['swapped'][sample, 0]
                writer.writerow([record['episode'], int(frame), float(record['timestamps'][frame]), record['task'], record['swapped_task'],
                                 *record['gt'][frame].tolist(), *original.tolist(), *swapped.tolist(), *(swapped - original).tolist()])
    log('counterfactual_plots_saved', directory=str(args.output), first_step=overall['first_step'], last_step=overall['last_step'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--denoising-steps', type=int, default=10)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    for name in ('dataset', 'checkpoint', 'baseline', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not args.plot_only:
        collect(args)
    plot(args)


if __name__ == '__main__':
    main()
