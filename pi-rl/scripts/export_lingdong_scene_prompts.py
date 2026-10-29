"""Validate local VLM results, render review artifacts, and publish complete training assets."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import textwrap

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from openpi.policies import lingdong_language as language
from openpi.policies import lingdong_policy as policy


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def review_image(output, prepared, annotation):
    canvas = Image.new("RGB", (1280, 1280), "white")
    for row, camera in enumerate(language.CAMERAS):
        with Image.open(output / prepared["images"][camera]) as image:
            canvas.paste(image, (0, row * 528))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    draw = ImageDraw.Draw(canvas)
    index = prepared["episode_index"]
    lines = [
        f"Episode {index:06d} | task={annotation['task_type']} | accepted={annotation['accepted']} | support={annotation['container_support']}"
    ]
    variants = annotation["scene_variants"]
    if variants:
        examples = [variants[0], variants[6] if len(variants) == 48 else variants[-1]]
        if annotation["task_type"] == "put_down":
            examples.append(variants[-1])
        for example in dict.fromkeys(examples):
            lines.extend(textwrap.wrap("Instruction: " + example, width=109))
    else:
        lines.append("Instruction: " + annotation["canonical_prompt"])
        lines.extend(textwrap.wrap("Fallback: " + ", ".join(annotation["reason_codes"]), width=109))
    notes = annotation["model_prediction"].get("notes", "")
    lines.extend(textwrap.wrap("Model observation: " + notes, width=109))
    for row, line in enumerate(lines[:9]):
        draw.text((12, 1066 + row * 23), line, font=font, fill="black")
    path = output / "review" / f"episode_{index:06d}.jpg"
    path.parent.mkdir(exist_ok=True)
    canvas.save(path, quality=94)


def main(output, asset_path, preview):
    manifest = json.loads((output / "manifest.json").read_text())
    dataset = Path(manifest["dataset"])
    if language.dataset_signature(dataset) != manifest["dataset_signature"]:
        raise ValueError("Dataset changed since evidence preparation")
    records, errors, signatures, models = {}, {}, set(), {}
    predictions = {
        int(path.stem.removeprefix("episode_")): path for path in (output / "predictions").glob("episode_*.json")
    }
    missing = [row["episode_index"] for row in manifest["episodes"] if row["episode_index"] not in predictions]

    def load(row):
        index = row["episode_index"]
        saved = json.loads(predictions[index].read_text())
        prepared = json.loads((output / "prepared" / f"episode_{index:06d}.json").read_text())
        if saved.get("dataset_signature") != manifest["dataset_signature"]:
            raise ValueError(f"Stale prediction source fingerprint: {index}")
        try:
            text = saved["raw_response"]
            raw = json.loads(text[text.find("{") : text.rfind("}") + 1])
            annotation = language.validate_prediction(raw, prepared)
            return index, annotation, None, saved
        except (ValueError, KeyError, TypeError) as error:
            return index, None, str(error), saved

    selected = [row for row in manifest["episodes"] if row["episode_index"] in predictions]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, annotation, error, saved in pool.map(load, selected):
            if error:
                errors[str(index)] = error
            else:
                records[str(index)] = annotation
                signatures.add(saved["inference_signature"])
                models[saved["inference_signature"]] = saved["model"]
    if len(signatures) > 1:
        raise ValueError("Mixed model/prompt configurations; rerun stale predictions")
    categories = Counter(
        row["container_support"] if row["accepted"] else "generic_fallback" for row in records.values()
    )
    summary = {
        "expected_episodes": len(manifest["episodes"]),
        "valid_predictions": len(records),
        "missing": missing,
        "invalid": errors,
        "categories": dict(categories),
        "models": models,
        "fallback_reasons": dict(Counter(reason for row in records.values() for reason in row["reason_codes"])),
    }
    write_json(output / "summary.json", summary)
    review_ids = {0, 1, 33, 128, 880, 1202, 1558, 2339, 2431, 2437, 2808, 2809}
    review_groups = {}
    for record in records.values():
        group = (record["task_type"], record["container_support"] if record["accepted"] else "generic_fallback")
        review_groups.setdefault(group, []).append(record["episode_index"])
    for indices in review_groups.values():
        indices.sort()
        count = min(6, len(indices))
        review_ids.update(indices[round(i * (len(indices) - 1) / max(count - 1, 1))] for i in range(count))
    for record in records.values():
        index = record["episode_index"]
        if len(records) <= 24 or index in review_ids:
            prepared = json.loads((output / "prepared" / f"episode_{index:06d}.json").read_text())
            review_image(output, prepared, record)
    asset = {
        "version": language.VERSION,
        "dataset_signature": manifest["dataset_signature"],
        "complete": not missing and not errors,
        "models": models,
        "inference_signatures": sorted(signatures),
        "episodes": records,
    }
    if preview:
        write_json(output / "compiled_preview.json", asset)
    else:
        if missing or errors:
            raise ValueError(f"Cannot publish incomplete scene language: {len(missing)} missing, {len(errors)} invalid")
        write_json(asset_path, asset)
        language.load_scene_prompts(asset_path, (str(dataset),))
        print(f"Published {len(records)} episode annotations: {asset_path}", flush=True)
    print(
        json.dumps({key: value for key, value in summary.items() if key not in {"missing", "invalid"}}, indent=2),
        flush=True,
    )
    print(f"Missing: {len(missing)}, invalid: {errors}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../output/lingdong_scene_annotations"))
    parser.add_argument(
        "--asset", type=Path, default=Path("assets") / policy.CONFIG_NAME / policy.ASSET_ID / language.ASSET_NAME
    )
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    main(args.output, args.asset, args.preview)
