"""Prepare timestamp-checked dual-camera evidence for offline episode annotation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pyarrow.parquet as pq

from openpi.policies import lingdong_language as language
from openpi.policies import lingdong_policy as policy

SAMPLER = "dual_camera_event_anchors_v1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def choose_frames(length, events, task):
    command = 0 if task == "pick_up" else 1
    main = next((e for e in events if e["command"] == command), None)
    anchors = [0, length - 1]
    if main:
        anchors += [max(0, main["frame"] - 6), main["confirmed_frame"], min(length - 1, main["confirmed_frame"] + 9)]
    for frame in np.linspace(0, length - 1, 11, dtype=int):
        if len(set(anchors)) >= min(8, length):
            break
        anchors.append(int(frame))
    return sorted(set(anchors))


def camera_sheet(path, wanted, fps, camera):
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_count = 1
        images = {}
        for frame in video.decode(video=0):
            index = round(float(frame.time) * fps)
            if index in wanted:
                if abs(float(frame.time) * fps - index) > 0.02:
                    raise ValueError(f"Nonaligned video timestamp: {path}, {frame.time}")
                images[index] = frame.to_image().resize((320, 240))
            if index >= wanted[-1]:
                break
    if set(images) != set(wanted):
        raise ValueError(f"Missing requested video frames: {path}, {set(wanted) - set(images)}")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    canvas = Image.new("RGB", (1280, 2 * 264), "white")
    draw = ImageDraw.Draw(canvas)
    for cell, index in enumerate(wanted):
        x, y = (cell % 4) * 320, (cell // 4) * 264
        draw.text((x + 5, y + 3), f"{camera} | frame {index} | {index / fps:.2f}s", font=font, fill="black")
        canvas.paste(images[index], (x, y + 24))
    return canvas


def prepare_one(root, output, info, episode, source_signature):
    index = episode["episode_index"]
    destination = output / "prepared" / f"episode_{index:06d}.json"
    if destination.is_file():
        existing = json.loads(destination.read_text())
        if (
            existing.get("dataset_signature") == source_signature
            and existing.get("sampler") == SAMPLER
            and all((output / name).is_file() for name in existing["images"].values())
        ):
            return index
    fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
    data_path = root / info["data_path"].format(**fields)
    values = np.asarray(
        pq.read_table(data_path, columns=["action"], use_threads=False)["action"].to_pylist(), dtype=np.float32
    )
    if len(values) != episode["length"]:
        raise ValueError(f"Wrong episode length: {index}")
    gripper_index = info["features"]["action"]["names"].index("right_gripper")
    _, events = policy.detect_gripper_events(values[:, gripper_index])
    task = next(task for task, prompt in language.CANONICAL_TASKS.items() if prompt in episode["tasks"])
    frames = choose_frames(len(values), events, task)
    images = {}
    for camera, key in zip(language.CAMERAS, policy.VIDEO_KEYS, strict=True):
        sheet = camera_sheet(root / info["video_path"].format(video_key=key, **fields), frames, info["fps"], camera)
        relative = f"images/episode_{index:06d}_{camera}.jpg"
        (output / "images").mkdir(parents=True, exist_ok=True)
        sheet.save(output / relative, quality=92)
        images[camera] = relative
    write_json(
        destination,
        {
            "episode_index": index,
            "length": len(values),
            "task_type": task,
            "frames": frames,
            "gripper_events": events,
            "images": images,
            "dataset_signature": source_signature,
            "sampler": SAMPLER,
        },
    )
    return index


def main(dataset, output, episodes, workers):
    info = json.loads((dataset / "meta/info.json").read_text())
    rows = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line]
    signature = language.dataset_signature(dataset)
    manifest = {"dataset": str(dataset.resolve()), "dataset_signature": signature, "sampler": SAMPLER, "episodes": rows}
    write_json(output / "manifest.json", manifest)
    chosen = [row for row in rows if episodes is None or row["episode_index"] in episodes]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_one, dataset, output, info, row, signature) for row in chosen]
        for done, future in enumerate(as_completed(futures), 1):
            index = future.result()
            if len(chosen) <= 16 or done % 100 == 0 or done == len(chosen):
                print(f"Prepared {done}/{len(chosen)} (episode {index:06d})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(policy.DEFAULT_DATASET_ROOT) / policy.DATASET_NAMES[0])
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_scene_annotations"))
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    main(args.dataset, args.output, args.episodes, args.workers)
