"""Collect replay predictions on the robot host, then plot them on the GPU host."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

MODES = ("absolute", "relative")
ACTION_KEYS = {"absolute": "action", "relative": "action.relative"}
DIMENSIONS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def collect(args):
    from openpi_client.franka_collect.client import create_client
    from openpi_client.franka_collect.client import iter_dataset_observations
    from openpi_client.franka_collect.client import prepare_observation
    import pyarrow.parquet as pq

    dataset = args.dataset.expanduser().resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    if not 0 <= args.episode < info["total_episodes"] or args.frame_step < 1:
        raise ValueError("Use a valid episode index and a positive frame step")
    indices = {"episode_index": args.episode, "episode_chunk": args.episode // info["chunks_size"]}
    table = pq.read_table(dataset / info["data_path"].format(**indices), columns=["timestamp", *ACTION_KEYS.values()])
    frames = np.arange(0, table.num_rows, args.frame_step)
    arrays = {"frames": frames, "time_seconds": np.asarray(table["timestamp"].to_pylist())[frames]}
    metadata = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset": str(dataset),
        "episode": args.episode,
        "episode_frames": table.num_rows,
        "fps": info["fps"],
        "frame_step": args.frame_step,
        "comparison": "prediction[t, 0, :] against dataset action[t, :]; recorded observations at every request",
        "servers": {},
    }
    clients = {}
    for mode in MODES:
        key = ACTION_KEYS[mode]
        reference = np.asarray(table[key].to_pylist(), dtype=np.float32)[frames]
        if reference.shape != (len(frames), 7) or not np.isfinite(reference).all():
            raise ValueError(f"Invalid dataset action field: {key}")
        arrays[f"{mode}_reference"] = reference
        arrays[f"{mode}_prediction"] = np.empty((len(frames), 30, 7), dtype=np.float32)
        arrays[f"{mode}_round_trip_ms"] = np.empty(len(frames))
        arrays[f"{mode}_server_infer_ms"] = np.empty(len(frames))
        port = args.absolute_port if mode == "absolute" else args.relative_port
        clients[mode] = create_client(host=args.host, port=port)
        server = clients[mode].get_server_metadata()
        expected = "pi05_pku_franka_collect_objects" + ("_relative" if mode == "relative" else "")
        if server.get("config") != expected:
            raise ValueError(f"Expected {expected}, got {server}")
        metadata["servers"][mode] = server
    start = time.perf_counter()
    observations = iter_dataset_observations(dataset, args.episode, 0, len(frames), args.frame_step)
    try:
        for index, raw in enumerate(observations):
            observation = prepare_observation(raw)
            for mode, client in clients.items():
                before = time.perf_counter()
                result = client.infer(observation)
                elapsed = (time.perf_counter() - before) * 1000
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.shape != (30, 7) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid {mode} prediction at frame {frames[index]}")
                arrays[f"{mode}_prediction"][index] = actions
                arrays[f"{mode}_round_trip_ms"][index] = elapsed
                arrays[f"{mode}_server_infer_ms"][index] = result["server_timing"]["infer_ms"]
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(frames):
                print(f"{index + 1}/{len(frames)} frames, both models, {time.perf_counter() - start:.1f}s", flush=True)
    finally:
        observations.close()
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {args.output}", flush=True)


def plot(args):
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    with np.load(args.input, allow_pickle=False) as archive:
        arrays = dict(archive)
    metadata = json.loads(str(arrays["metadata"]))
    frames = arrays["frames"]
    times = arrays["time_seconds"]
    if len(frames) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Need at least two ordered frames to draw a trajectory")
    args.output.mkdir(parents=True, exist_ok=True)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next(
        (name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK JP") if name in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": [chinese_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
        }
    )
    fig, axes = plt.subplots(7, 2, figsize=(17, 18), sharex=True)
    colors = {"absolute": "#2563eb", "relative": "#ea580c"}
    metrics = {"metadata": metadata, "samples": len(frames), "modes": {}}
    csv_columns = [frames, times]
    csv_headers = ["frame", "time_seconds"]
    for column, mode in enumerate(MODES):
        reference = arrays[f"{mode}_reference"].astype(np.float64)
        chunks = arrays[f"{mode}_prediction"]
        if chunks.shape != (len(frames), 30, 7) or reference.shape != (len(frames), 7):
            raise ValueError(f"Unexpected {mode} prediction/reference shape")
        prediction = chunks[:, 0, :].astype(np.float64)
        if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite {mode} prediction/reference")
        error = prediction - reference
        error[:, 3:6] = (error[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
        mae = np.mean(np.abs(error), axis=0)
        rmse = np.sqrt(np.mean(error**2, axis=0))
        metrics["modes"][mode] = {
            "action_key": ACTION_KEYS[mode],
            "mae": dict(zip(DIMENSIONS, mae.tolist(), strict=True)),
            "rmse": dict(zip(DIMENSIONS, rmse.tolist(), strict=True)),
            "units": ["m", "m", "m", "rad", "rad", "rad", "normalized"],
            "gripper_binary_accuracy": float(np.mean((prediction[:, 6] >= 0.5) == (reference[:, 6] >= 0.5))),
            "round_trip_ms_median": float(np.median(arrays[f"{mode}_round_trip_ms"])),
            "angular_error": "Euler XYZ component errors wrapped to [-pi, pi)",
        }
        # Plot equivalent Euler angles on the reference branch to avoid artificial 2*pi jumps.
        displayed_reference = reference.copy()
        displayed_reference[:, 3:6] = np.unwrap(reference[:, 3:6], axis=0)
        displayed_prediction = prediction.copy()
        displayed_prediction[:, 3:6] = displayed_reference[:, 3:6] + error[:, 3:6]
        for dimension, name in enumerate(DIMENSIONS):
            axis = axes[dimension, column]
            scale = 1000 if mode == "relative" and dimension < 3 else 1
            unit = ("mm" if scale == 1000 else "m") if dimension < 3 else ("rad" if dimension < 6 else "0-1")
            label = ("Δ" if mode == "relative" and dimension < 6 else "") + name
            axis.plot(
                times,
                displayed_reference[:, dimension] * scale,
                color="#334155",
                linewidth=1.5,
                label="数据集动作",
                zorder=2,
            )
            axis.plot(
                times,
                displayed_prediction[:, dimension] * scale,
                color=colors[mode],
                linewidth=1.15,
                alpha=0.85,
                label="Pi0.5 预测 (第 1 步)",
                zorder=3,
            )
            axis.set_ylabel(f"{label} ({unit})")
            axis.grid(alpha=0.15)
            axis.set_xlim(times[0], times[-1])
            if (
                3 <= dimension < 6
                and np.ptp(np.concatenate([displayed_reference[:, dimension], displayed_prediction[:, dimension]]))
                < 1e-6
            ):
                # Keep floating-point noise in constant rotation dimensions from filling the panel.
                center = float(np.mean(displayed_reference[:, dimension]))
                axis.set_ylim(center - 1e-4, center + 1e-4)
                axis.text(0.02, 0.94, "近零变化", transform=axis.transAxes, va="top", fontsize=9, color="#64748b")
            axis.text(
                0.985,
                0.94,
                f"MAE {mae[dimension] * scale:.4g}" + (f" {unit}" if dimension < 6 else ""),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
            )
            if dimension == 0:
                title = "绝对动作 · action" if mode == "absolute" else "相对动作 · action.relative"
                axis.set_title(title, loc="left", fontsize=15, pad=13, weight="bold")
                axis.legend(loc="lower left", fontsize=9, framealpha=0.9)
            if dimension == 6:
                axis.set_xlabel("数据集时间 (s)")
        csv_columns.extend(reference.T)
        csv_columns.extend(prediction.T)
        csv_headers.extend(f"{mode}_reference_{name}" for name in DIMENSIONS)
        csv_headers.extend(f"{mode}_prediction_{name}" for name in DIMENSIONS)
    fig.suptitle(
        f"Pi0.5 预测曲线与数据集动作对比 | 轨迹 {metadata['episode']}",
        x=0.055,
        y=0.984,
        ha="left",
        fontsize=22,
        weight="bold",
    )
    step = metadata["frame_step"]
    fig.text(
        0.055,
        0.96,
        f"录制观测逐帧请求 → 对齐动作块第 1 步与同帧标签  ·  {len(frames)} 个观测  ·  原始 {metadata['fps']} Hz / 采样步长 {step} 帧",
        fontsize=11,
        color="#475569",
    )
    fig.text(
        0.055,
        0.015,
        "每个请求使用录制的双相机与实测状态; 角度按 2π 等价对齐; 夹爪 0=闭合、1=张开。曲线反映离线动作拟合。",
        fontsize=10,
        color="#64748b",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.055, hspace=0.28, wspace=0.22)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"replay_comparison.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        args.output / "first_step_comparison.csv",
        np.column_stack(csv_columns),
        delimiter=",",
        header=",".join(csv_headers),
        comments="",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved plots and data to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("collect", help="Record both models' complete action chunks")
    record.add_argument("--dataset", type=Path, required=True)
    record.add_argument("--episode", type=int, default=1)
    record.add_argument("--frame-step", type=int, default=1)
    record.add_argument("--host", default="219.223.195.77")
    record.add_argument("--absolute-port", type=int, default=8000)
    record.add_argument("--relative-port", type=int, default=8001)
    record.add_argument("--output", type=Path, required=True)
    record.set_defaults(run=collect)
    draw = commands.add_parser("plot", help="Draw aligned first-step curves from a collected NPZ")
    draw.add_argument("--input", type=Path, required=True)
    draw.add_argument("--output", type=Path, required=True)
    draw.set_defaults(run=plot)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
