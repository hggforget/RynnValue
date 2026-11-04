"""Discover, filter and split fine-grained FR3 episodes before prompt augmentation."""

from collections import Counter
import dataclasses
import hashlib
import json
from pathlib import Path
import re

from openpi.policies import pku_franka_language as language


def _normalize_prompt(text: str) -> str:
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", text.lower()).split()).rstrip(".!?")


def discover_repos(roots: tuple[str, ...]) -> tuple[str, ...]:
    repos = []
    for name in roots:
        root = Path(name)
        if not root.is_dir():
            raise FileNotFoundError(f"FR3 prompt dataset is not mounted: {root}")
        paths = (
            [root / "meta/info.json"] if (root / "meta/info.json").is_file() else sorted(root.rglob("meta/info.json"))
        )
        if not paths:
            raise ValueError(f"No LeRobot datasets under {root}")
        repos.extend(str(path.parent.parent.resolve()) for path in paths)
    if len(repos) != len(set(repos)):
        raise ValueError("FR3 prompt dataset roots overlap; an episode must appear only once")
    return tuple(repos)


def episode_records(
    roots: tuple[str, ...], *, validation_fraction: float, split_seed: int, include_noted_episodes: bool
) -> tuple[language.EpisodeInstruction, ...]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    records = []
    identities = set()
    for repo in discover_repos(roots):
        root = Path(repo)
        path = root / "meta/episodes.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fine = row.get("fine_task", {})
            # In particular, do not relabel drawer_open/close or long-interaction trajectories.
            if fine.get("kind") != "pick_place":
                continue
            if fine.get("target_kind") not in {"plate", "drawer", "white_drawer"}:
                raise ValueError(f"Unsupported FR3 pick-place target in {path}: {fine}")
            if fine.get("object_color") not in language.COLORS or fine.get("object_shape") not in language.SHAPES:
                raise ValueError(f"Invalid FR3 object attributes in {path}: {fine}")
            if fine["target_kind"] == "plate" and fine.get("target_color") not in language.COLORS:
                raise ValueError(f"Invalid FR3 plate color in {path}: {fine}")
            episode = int(row["episode_index"])
            length = int(row["length"])
            if episode < 0 or length <= 0:
                raise ValueError(f"Invalid episode identity/length in {path}: {episode}, {length}")
            # Merged Oct data carries source_dataset/source_episode_index. Nested Nov data
            # uses <task>/<date>; normalize it to the same <date>/<task> identity.
            source = row.get("source_dataset") or (
                f"{root.name}/{root.parent.name}" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", root.name) else root.name
            )
            source_episode = int(row.get("source_episode_index", episode))
            identity = (source, source_episode)
            if identity in identities:
                raise ValueError(f"Duplicate source episode across FR3 datasets: {identity}")
            identities.add(identity)
            digest = hashlib.sha256(f"{split_seed}:{source}:{source_episode}".encode()).digest()
            split = "validation" if int.from_bytes(digest[:8], "big") / 2**64 < validation_fraction else "train"
            note = row.get("note") or ""
            if note.strip() and not include_noted_episodes:
                split = "excluded_note"
            date = re.search(r"\d{4}-\d{2}-\d{2}", source)
            prompts = row.get("tasks") or []
            original = row.get("prompt") or (prompts[0] if len(prompts) == 1 else "")
            # The first pass accepts checked fixed templates only. A contradictory
            # full prompt must not bypass fine_task by landing in the full-mask slot.
            if (
                original
                and split != "excluded_note"
                and _normalize_prompt(original) != _normalize_prompt(language.full_prompt_from_attributes(fine))
            ):
                raise ValueError(
                    f"Full prompt disagrees with fine_task or uses an unchecked template: {path}, episode={episode}"
                )
            records.append(
                language.EpisodeInstruction(
                    repo_id=repo,
                    episode_index=episode,
                    length=length,
                    source_dataset=source,
                    source_episode_index=source_episode,
                    date=date.group() if date else "",
                    fine_task=fine,
                    original_prompt=original,
                    note=note,
                    split=split,
                )
            )
    if not records:
        raise ValueError("No fine_task.kind=pick_place episodes found in the selected FR3 datasets")
    return tuple(records)


def save_manifest(path: Path, records: tuple[language.EpisodeInstruction, ...], **settings) -> None:
    """Keep originals and notes even for excluded/held-out episodes; never copy media."""
    summary = Counter((record.split, record.fine_task["target_kind"]) for record in records)
    content = {
        "schema_version": 1,
        "settings": settings,
        "summary": {f"{split}/{target}": count for (split, target), count in sorted(summary.items())},
        "episodes": [dataclasses.asdict(record) for record in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
