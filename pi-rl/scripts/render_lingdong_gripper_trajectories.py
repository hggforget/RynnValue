"""Render recorded cameras with whole-episode gripper events and previous-rule comparisons."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import textwrap

import av
import cv2
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from openpi.policies import lingdong_policy as policy


def plot_trace(axes, sample):
    time = np.arange(len(sample["raw"])) / sample["fps"]
    axes[0].plot(time, sample["raw"], color="#64748b", label="Recorded opening", linewidth=1.5)
    axes[0].step(
        time, sample["old_binary"], where="post", color="#dc2626", label="Previous rule", linewidth=0.8, alpha=0.5
    )
    axes[0].step(time, sample["binary"], where="post", color="#2563eb", label="Event labels", linewidth=1.8)
    axes[0].set_ylim(-0.08, 1.12)
    axes[0].set_ylabel("Gripper")
    axes[0].set_title(f"Episode {sample['episode_index']:06d} | {sample['task']}", loc="left", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8, ncol=3)
    axes[1].plot(time, sample["delta"], color="#7c3aed", linewidth=1)
    axes[1].axhline(0, color="#94a3b8", linewidth=0.7)
    axes[1].set_ylabel("Next - current")
    axes[1].set_title("Confirm motion from cumulative change; hold commands through plateaus", loc="left", fontsize=10)
    for event in sample["events"]:
        axes[1].axvline(event["frame"] / sample["fps"], color="#16a34a" if event["command"] else "#dc2626", alpha=0.5)
    for axis in axes:
        axis.set_xlim(0, time[-1])
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)


def graph_strip(sample):
    fig, ax = plt.subplots(figsize=(12.8, 1.9), dpi=100)
    time = np.arange(len(sample["raw"])) / sample["fps"]
    ax.plot(time, sample["raw"], color="#64748b", linewidth=1.5, label="Recorded opening")
    if sample.get("old_binary") is not None:
        ax.step(
            time, sample["old_binary"], where="post", color="#dc2626", linewidth=0.8, alpha=0.4, label="Previous rule"
        )
    ax.step(time, sample["binary"], where="post", color="#2563eb", linewidth=1.8, label="Event labels")
    ax.set(xlim=(0, time[-1]), ylim=(-0.08, 1.12), xlabel="Time (s)")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    ax.grid(alpha=0.15)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.95, bottom=0.25)
    fig.canvas.draw()
    strip = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    left, bottom, width, height = ax.get_position().bounds
    bounds = (int(left * 1280), int((1 - bottom - height) * 190), int(width * 1280), int(height * 190))
    plt.close(fig)
    return cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), bounds


def render_video(sample, info, dataset, output):
    cameras = []
    episode = sample["episode_index"]
    header_height = 144 if sample.get("instruction") else 80
    video_height = header_height + 480 + 190
    try:
        for key in policy.VIDEO_KEYS:
            path = dataset / info["video_path"].format(
                episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=key
            )
            container = av.open(str(path))
            container.streams.video[0].thread_count = 1
            cameras.append((container, iter(container.decode(video=0))))
        strip, bounds = graph_strip(sample)
        writer = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"1280x{video_height}",
                "-r",
                str(sample["fps"] / 2),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            stdin=subprocess.PIPE,
        )
        try:
            for index in range(len(sample["raw"])):
                try:
                    frames = [next(decoder) for _, decoder in cameras]
                except StopIteration as exc:
                    raise ValueError(f"Video ends before numeric trajectory: episode {episode}, frame {index}") from exc
                if index % 2:
                    continue
                image = np.full((video_height, 1280, 3), 255, dtype=np.uint8)
                for column, frame in enumerate(frames):
                    rgb = cv2.resize(frame.to_ndarray(format="bgr24"), (640, 480))
                    image[header_height : header_height + 480, column * 640 : (column + 1) * 640] = rgb
                binary = int(sample["binary"][index])
                title = f"Episode {episode:06d} | {sample['task']} | frame {index} | {index / sample['fps']:.2f}s"
                if sample.get("instruction"):
                    title += f" | {sample['scene_status']}"
                details = f"opening={sample['raw'][index]:.3f}   next-current={sample['delta'][index]:+.3f}   event_label={binary}"
                cv2.putText(image, title, (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2, cv2.LINE_AA)
                cv2.putText(image, details, (20, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (175, 75, 20), 2, cv2.LINE_AA)
                if sample.get("instruction"):
                    for line, text in enumerate(textwrap.wrap("Instruction: " + sample["instruction"], width=100)):
                        cv2.putText(
                            image,
                            text,
                            (20, 91 + line * 27),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (35, 35, 35),
                            2,
                            cv2.LINE_AA,
                        )
                cv2.putText(
                    image,
                    "HEAD",
                    (15, header_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "RIGHT WRIST",
                    (655, header_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                graph_top = header_height + 480
                image[graph_top:] = strip
                x0, y0, width, height = bounds
                cursor = x0 + round(width * index / max(len(sample["raw"]) - 1, 1))
                cv2.line(image, (cursor, graph_top + y0), (cursor, graph_top + y0 + height), (80, 170, 30), 2)
                writer.stdin.write(image.tobytes())
        finally:
            writer.stdin.close()
            code = writer.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited with {code}: {output}")
    finally:
        for container, _ in cameras:
            container.close()


def main(dataset: Path, output: Path, episodes: list[int]):
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset / "meta/info.json").read_text())
    metadata = {
        e["episode_index"]: e for e in map(json.loads, (dataset / "meta/episodes.jsonl").read_text().splitlines())
    }
    action_ids = policy.right_indices(info["features"], "action")
    samples = []
    manifest = []
    for episode in episodes:
        meta = metadata[episode]
        path = dataset / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        actions = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)[
            :, action_ids
        ]
        raw = actions[:, -1]
        binary, events = policy.detect_gripper_events(raw)
        delta = np.diff(raw, append=raw[-1])
        sample = {
            "episode_index": episode,
            "task": meta["tasks"][0],
            "fps": info["fps"],
            "raw": raw,
            "delta": delta,
            "binary": binary,
            "old_binary": np.where(delta == 0, raw >= 0.6, delta > 0),
            "events": events,
        }
        samples.append(sample)
        with (output / f"episode_{episode:06d}.csv").open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "time_s", "recorded_gripper", "next_minus_current", "binary_gripper"])
            writer.writerows(
                (i, i / info["fps"], float(raw[i]), float(delta[i]), int(binary[i])) for i in range(len(raw))
            )
        fig, axes = plt.subplots(1, 2, figsize=(16, 3.6), constrained_layout=True)
        plot_trace(axes, sample)
        fig.savefig(output / f"episode_{episode:06d}_gripper.png", dpi=160)
        plt.close(fig)
        video = output / f"episode_{episode:06d}.mp4"
        render_video(sample, info, dataset, video)
        manifest.append(
            {
                "episode_index": episode,
                "task": sample["task"],
                "frames": len(raw),
                "duration_s": len(raw) / info["fps"],
                "opening_frames": int(np.count_nonzero(delta > 0)),
                "closing_frames": int(np.count_nonzero(delta < 0)),
                "stationary_or_terminal_frames": int(np.count_nonzero(delta == 0)),
                "video": video.name,
                "events": events,
            }
        )
        print(f"Rendered episode {episode:06d}: {sample['task']}, {len(raw)} frames", flush=True)
    fig, axes = plt.subplots(len(samples), 2, figsize=(17, 3.5 * len(samples)), constrained_layout=True, squeeze=False)
    for row, sample in zip(axes, samples, strict=True):
        plot_trace(row, sample)
    fig.savefig(output / "gripper_overview.png", dpi=160)
    plt.close(fig)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "rule": policy.gripper_binarization_spec(),
                "episodes": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "gripper_overview.png", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(policy.DEFAULT_DATASET_ROOT) / policy.DATASET_NAMES[0])
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_gripper_event_review"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[1, 1558, 0, 1202, 128, 880])
    args = parser.parse_args()
    main(args.dataset, args.output, args.episodes)
