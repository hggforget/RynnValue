"""Render stratified full-episode videos with their actual training instructions."""

import argparse
import csv
import json
from pathlib import Path
import random
import textwrap

import av
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pyarrow.parquet as pq
import render_lingdong_gripper_trajectories as renderer

from openpi.policies import lingdong_language as language
from openpi.policies import lingdong_policy as policy


def representative_instruction(record):
    variants = record["scene_variants"]
    if not variants:
        return record["canonical_prompt"]
    if record["task_type"] == "put_down":
        return variants[42] if len(variants) == 48 else variants[18]
    return variants[6] if len(variants) == 48 else variants[0]


def main(dataset, asset_path, output, per_group):
    output.mkdir(parents=True, exist_ok=True)
    records = language.load_scene_prompts(asset_path, (str(dataset),))[0]
    info = json.loads((dataset / "meta/info.json").read_text())
    groups = {}
    for record in records.values():
        key = (record["task_type"], record["container_support"] if record["accepted"] else "generic_fallback")
        groups.setdefault(key, []).append(record)
    rng = random.Random(20261029)
    chosen = []
    for key in sorted(groups):
        chosen += rng.sample(
            sorted(groups[key], key=lambda row: row["episode_index"]), min(per_group, len(groups[key]))
        )
    selected_ids = {row["episode_index"] for row in chosen}
    with (output / "all_2810_video_instruction_pairs.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode_index",
                "task",
                "augmentation",
                "support",
                "instruction_example",
                "head_video",
                "right_wrist_video",
                "annotated_sample",
            ]
        )
        for index, record in sorted(records.items()):
            fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            cameras = [str(dataset / info["video_path"].format(video_key=key, **fields)) for key in policy.VIDEO_KEYS]
            writer.writerow(
                [
                    index,
                    record["task_type"],
                    "scene" if record["accepted"] else "generic_fallback",
                    record["container_support"],
                    representative_instruction(record),
                    *cameras,
                    f"episode_{index:06d}.mp4" if index in selected_ids else "",
                ]
            )
    gripper_index = info["features"]["action"]["names"].index("right_gripper")
    entries = []
    for record in chosen:
        index = record["episode_index"]
        fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        path = dataset / info["data_path"].format(**fields)
        actions = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)
        raw = actions[:, gripper_index]
        labels, _ = policy.detect_gripper_events(raw)
        instruction = representative_instruction(record)
        assert instruction in (*policy.PROMPT_VARIANTS[record["canonical_prompt"]], *record["scene_variants"])
        status = record["container_support"] if record["accepted"] else "GENERIC FALLBACK"
        sample = {
            "episode_index": index,
            "task": record["canonical_prompt"],
            "fps": info["fps"],
            "raw": raw,
            "binary": labels,
            "delta": np.diff(raw, append=raw[-1]),
            "instruction": instruction,
            "scene_status": status,
        }
        video = output / f"episode_{index:06d}.mp4"
        renderer.render_video(sample, info, dataset, video)
        with av.open(str(video)) as generated:
            generated.seek(int(generated.duration / 2))
            frame = next(generated.decode(video=0)).to_image()
            frame.save(output / f"episode_{index:06d}_preview.jpg", quality=94)
        entry = {
            "episode_index": index,
            "video": video.name,
            "instruction": instruction,
            "task": record["task_type"],
            "support": record["container_support"],
            "scene_augmented": record["accepted"],
            "reason_codes": record["reason_codes"],
            "evidence": record["evidence"],
            "model_prediction": record["model_prediction"],
        }
        entries.append(entry)
        (output / f"episode_{index:06d}_instruction.json").write_text(json.dumps(entry, indent=2) + "\n")
        print(f"Rendered episode {index:06d} | {record['task_type']} | {status}: {instruction}", flush=True)
    (output / "video_instruction_pairs.json").write_text(
        json.dumps(
            {
                "selection_seed": 20261029,
                "source_asset": str(asset_path.resolve()),
                "dataset_signature": language.dataset_signature(dataset),
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    canvas = Image.new("RGB", (1440, ((len(entries) + 2) // 3) * 280), "white")
    draw = ImageDraw.Draw(canvas)
    for cell, entry in enumerate(entries):
        x, y = (cell % 3) * 480, (cell // 3) * 280
        with Image.open(output / f"episode_{entry['episode_index']:06d}_preview.jpg") as image:
            canvas.paste(image.crop((0, 144, 1280, 624)).resize((480, 180)), (x, y))
        label = f"Ep {entry['episode_index']:06d} | {entry['task']} | {entry['support']}"
        if not entry["scene_augmented"]:
            label += " | FALLBACK"
        draw.text((x + 5, y + 184), label, font=font, fill="black")
        for line, text in enumerate(textwrap.wrap(entry["instruction"], width=60)):
            draw.text((x + 5, y + 205 + line * 18), text, font=font, fill="#1d4ed8")
    canvas.save(output / "video_instruction_overview.jpg", quality=94)
    print(
        f"Rendered {len(entries)} videos; full mapping: {output / 'all_2810_video_instruction_pairs.csv'}", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(policy.DEFAULT_DATASET_ROOT) / policy.DATASET_NAMES[0])
    parser.add_argument(
        "--asset", type=Path, default=Path("assets") / policy.CONFIG_NAME / policy.ASSET_ID / language.ASSET_NAME
    )
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_video_instruction_review"))
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()
    main(args.dataset, args.asset, args.output, args.per_group)
