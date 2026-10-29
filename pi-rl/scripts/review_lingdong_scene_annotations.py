"""Create stratified scene/instruction contact sheets from real evidence frames."""

import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def main(output, asset):
    records = json.loads(asset.read_text())["episodes"]
    groups = {}
    for record in records.values():
        key = (record["task_type"], record["container_support"] if record["accepted"] else "fallback")
        groups.setdefault(key, []).append(record)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["episode_index"])
        count = min(6, len(group))
        selected += [group[round(i * (len(group) - 1) / max(count - 1, 1))] for i in range(count)]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    for start in range(0, len(selected), 12):
        page = selected[start : start + 12]
        canvas = Image.new("RGB", (1280, 40 + 6 * 328), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            "Scene / instruction audit: actual first and last head-camera frames",
            font=title_font,
            fill="black",
        )
        for cell, record in enumerate(page):
            index = record["episode_index"]
            prepared = json.loads((output / "prepared" / f"episode_{index:06d}.json").read_text())
            x, y = (cell % 2) * 640, 40 + (cell // 2) * 328
            with Image.open(output / prepared["images"]["head"]) as image:
                canvas.paste(image.crop((0, 0, 320, 264)), (x, y))
                canvas.paste(image.crop((960, 264, 1280, 528)), (x + 320, y))
            color = "#1d4ed8" if record["accepted"] else "#b45309"
            caption = f"Ep {index:06d} | {record['task_type']} | {record['container_support']} | accepted={record['accepted']}"
            draw.text((x + 5, y + 266), caption, font=font, fill=color)
            variants = record["scene_variants"]
            instruction = (
                variants[6] if len(variants) == 48 else variants[0] if variants else record["canonical_prompt"]
            )
            for line, text in enumerate(textwrap.wrap(instruction, width=75)[:2]):
                draw.text((x + 5, y + 286 + line * 18), text, font=font, fill="black")
        target = output / f"scene_instruction_audit_{start // 12 + 1}.jpg"
        canvas.save(target, quality=94)
        print(target, flush=True)
    (output / "review_sample_ids.json").write_text(
        json.dumps([row["episode_index"] for row in selected], indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_scene_annotations"))
    parser.add_argument("--asset", type=Path)
    args = parser.parse_args()
    main(args.output, args.asset or args.output / "compiled_preview.json")
