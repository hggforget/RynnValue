"""Compile episode-specific language only from validated visual evidence."""

import hashlib
import json
from pathlib import Path

CANONICAL_TASKS = {"pick_up": "pick up the object", "put_down": "put down the object"}
CAMERAS = ("head", "right_wrist")
FACTS = {
    "blue_container",
    "container_on_table",
    "container_on_metal_shelf",
    "object_in_container",
    "object_held",
    "object_released",
    "pick_up_attempt",
    "put_down_attempt",
}
ASSET_NAME = "lingdong_scene_prompts.json"
VERSION = "grounded_episode_language_v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dataset_signature(root: Path) -> str:
    files = {}
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "file_checksums.json", "episode_mapping.jsonl"):
        path = root / "meta" / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not all(name in files for name in ("info.json", "episodes.jsonl", "tasks.jsonl")):
        raise ValueError(f"Missing Lingdong source metadata: {root}")
    return digest(files)


def grounded_variants(task: str, support: str = "unknown") -> tuple[str, ...]:
    """Describe the verified container; support qualifiers require their own evidence."""
    locations = (
        "blue bin",
        "blue tray",
        "blue container",
        "blue plastic bin",
        "blue plastic tray",
        "blue plastic container",
    )
    if support == "table":
        locations += tuple(f"{name} on the table" for name in locations)
    elif support == "metal_shelf":
        locations += tuple(f"{name} on the metal shelf" for name in locations)
    elif support != "unknown":
        raise ValueError(f"Unsupported container support: {support}")
    templates = {
        "pick_up": (
            "pick up the object from the {location}",
            "take the object out of the {location}",
            "lift the object out of the {location}",
            "grasp the object in the {location} and lift it",
        ),
        "put_down": (
            "put the object into the {location}",
            "place the object in the {location}",
            "move the held object to the {location} and set it down",
            "transfer the object from the gripper to the {location}",
        ),
    }
    if task not in templates:
        raise ValueError(f"Unsupported Lingdong task: {task}")
    return tuple(template.format(location=location) for template in templates[task] for location in locations)


def _confidence(raw: dict, key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise ValueError(f"Invalid model confidence: {key}")
    return float(value)


def validate_prediction(raw: dict, prepared: dict) -> dict:
    """Reject unsupported references and compile only independently supported fields."""
    if not isinstance(raw, dict):
        raise ValueError("Model prediction must be a JSON object")
    if raw.get("task_type") not in {"pick_up", "put_down", "unknown"}:
        raise ValueError("Invalid visually observed task")
    if raw.get("container_support") not in {"table", "metal_shelf", "unknown"}:
        raise ValueError("Invalid container support")
    if raw.get("container_color") not in {"blue", "unknown"}:
        raise ValueError("Invalid container color")
    if raw.get("source") not in {"container", "gripper", "unknown"} or raw.get("destination") not in {
        "container",
        "gripper",
        "unknown",
    }:
        raise ValueError("Invalid source/destination")
    if raw.get("visual_support") not in {"object_transfer", "visible_attempt", "insufficient"}:
        raise ValueError("Invalid visual support")
    container_confidence = _confidence(raw, "container_confidence")
    support_confidence = _confidence(raw, "support_confidence")
    transfer_confidence = _confidence(raw, "transfer_confidence")
    evidence = raw.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Model must provide visible evidence")
    frames = set(prepared["frames"])
    facts = {fact: set() for fact in FACTS}
    for item in evidence:
        if not isinstance(item, dict) or item.get("camera") not in CAMERAS:
            raise ValueError("Model cited an unprovided camera")
        frame = item.get("frame_index")
        if type(frame) is not int or frame not in frames:
            raise ValueError("Model cited an unprovided frame")
        values = item.get("facts")
        if not isinstance(values, list) or set(values) - FACTS:
            raise ValueError("Invalid evidence facts")
        for fact in values:
            facts[fact].add(frame)
    task = prepared["task_type"]
    expected_source, expected_destination, command = (
        ("container", "gripper", 0) if task == "pick_up" else ("gripper", "container", 1)
    )
    reasons = []
    if raw["task_type"] != task:
        reasons.append("visual_task_mismatch_or_unknown")
    if (raw["source"], raw["destination"]) != (expected_source, expected_destination):
        reasons.append("source_destination_not_supported")
    if raw["container_color"] != "blue" or container_confidence < 0.85 or len(facts["blue_container"]) < 2:
        reasons.append("container_not_confirmed")
    if transfer_confidence < 0.85 or raw["visual_support"] == "insufficient":
        reasons.append("operation_not_confirmed")
    before = facts["object_in_container"] if task == "pick_up" else facts["object_held"]
    after = (
        facts["object_held"] | facts["pick_up_attempt"]
        if task == "pick_up"
        else (facts["object_released"] | facts["object_in_container"] | facts["put_down_attempt"])
    )
    matching_events = [event for event in prepared["gripper_events"] if event["command"] == command]
    temporal_match = any(
        before and after and min(before) < event["frame"] and max(after) >= event["confirmed_frame"]
        for event in matching_events
    )
    if not temporal_match:
        reasons.append("missing_temporal_operation_evidence")
    if not matching_events:
        reasons.append("gripper_operation_mismatch")
    support = raw["container_support"]
    support_fact = f"container_on_{support}"
    opposite = "container_on_metal_shelf" if support == "table" else "container_on_table"
    if support == "unknown" or support_confidence < 0.9 or len(facts.get(support_fact, ())) < 2 or facts[opposite]:
        support = "unknown"
    accepted = not reasons
    return {
        "episode_index": prepared["episode_index"],
        "canonical_prompt": CANONICAL_TASKS[task],
        "task_type": task,
        "accepted": accepted,
        "container_support": support,
        "reason_codes": reasons,
        "scene_variants": list(grounded_variants(task, support)) if accepted else [],
        "evidence": evidence,
        "model_prediction": raw,
    }


def load_scene_prompts(path: Path, repo_ids: tuple[str, ...]) -> tuple[dict[int, dict], ...]:
    asset = json.loads(path.read_text())
    if asset.get("version") != VERSION or not asset.get("complete"):
        raise ValueError("Scene prompt asset is incomplete or has an incompatible version")
    result = []
    for repo_id in repo_ids:
        root = Path(repo_id)
        if asset.get("dataset_signature") != dataset_signature(root):
            raise ValueError("Scene prompt dataset fingerprint changed; regenerate scene annotations")
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
        records = {int(key): value for key, value in asset["episodes"].items()}
        if set(records) != {row["episode_index"] for row in episodes}:
            raise ValueError("Scene prompt asset must cover every original episode")
        for row in episodes:
            record = records[row["episode_index"]]
            canonical = record.get("canonical_prompt")
            if canonical not in row["tasks"] or CANONICAL_TASKS.get(record.get("task_type")) != canonical:
                raise ValueError("Scene prompt task does not match its original episode")
            expected = (
                list(grounded_variants(record["task_type"], record["container_support"])) if record["accepted"] else []
            )
            if record.get("scene_variants") != expected:
                raise ValueError("Scene prompt strings do not match validated scene fields")
        result.append(records)
    return tuple(result)
