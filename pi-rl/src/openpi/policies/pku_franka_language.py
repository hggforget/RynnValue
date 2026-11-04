"""Episode-grounded FR3 instructions; no video or action duplication."""

import dataclasses
import itertools

import numpy as np

from openpi.policies import pku_franka_collect_policy as collect

COLORS = frozenset({"red", "green", "blue", "yellow"})
SHAPES = {
    "arch": "arch-shaped",
    "cylinder": "cylindrical",
    "triangular_prism": "triangular prism",
    "rectangular": "rectangular",
    "flat_rectangular": "flat rectangular",
}


@dataclasses.dataclass(frozen=True)
class EpisodeInstruction:
    repo_id: str
    episode_index: int
    length: int
    source_dataset: str
    source_episode_index: int
    date: str
    fine_task: dict
    original_prompt: str
    note: str
    split: str

    @property
    def num_attributes(self) -> int:
        return 3 if self.fine_task["target_kind"] == "plate" else 2


def validate_probabilities(probabilities: tuple[float, ...], size: int) -> None:
    values = np.asarray(probabilities)
    if len(values) != size or not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1):
        raise ValueError(f"Expected {size} finite nonnegative probabilities summing to 1, got {probabilities}")


def full_prompt_from_attributes(fine: dict) -> str:
    description = f"{fine['object_color']} {SHAPES[fine['object_shape']]} block"
    if fine["target_kind"] == "plate":
        return f"Put the {description} on the {fine['target_color']} plate."
    target = "white drawer" if fine["target_kind"] == "white_drawer" else "drawer"
    return f"Put the {description} in the {target}."


def prompt_for_mask(record: EpisodeInstruction, mask: tuple[bool, ...]) -> str:
    """Only omit attributes; every remaining attribute comes from fine_task."""
    if len(mask) != record.num_attributes:
        raise ValueError(f"Expected {record.num_attributes} attributes, got {mask}")
    if all(mask):
        return record.original_prompt or full_prompt_from_attributes(record.fine_task)
    fine = record.fine_task
    description = []
    if mask[0]:
        description.append(fine["object_color"])
    if mask[1]:
        description.append(SHAPES[fine["object_shape"]])
    description.append("block")
    if fine["target_kind"] == "plate":
        target = f"{fine['target_color']} plate" if mask[2] else "plate"
        return f"Put a {' '.join(description)} on a {target}."
    return f"Put a {' '.join(description)} in a drawer."


@dataclasses.dataclass(frozen=True)
class PkuFrankaPromptInputs(collect.PkuFrankaCollectInputs):
    """Keep original episode identities until the training-only language transform."""

    def __call__(self, data: dict) -> dict:
        result = super().__call__(data)
        if "actions" in result:
            for key in ("dataset_index", "episode_index"):
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class PkuFrankaPromptAugmentation:
    episodes: tuple[dict[int, EpisodeInstruction], ...]
    augment: bool = True
    plate_probabilities: tuple[float, ...] = (0.2, 0.3, 0.3, 0.2)
    drawer_probabilities: tuple[float, ...] = (0.2, 0.6, 0.2)

    def __post_init__(self):
        validate_probabilities(self.plate_probabilities, 4)
        validate_probabilities(self.drawer_probabilities, 3)

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        dataset = int(np.asarray(data["dataset_index"]).item())
        episode = int(np.asarray(data["episode_index"]).item())
        if not 0 <= dataset < len(self.episodes) or episode not in self.episodes[dataset]:
            raise ValueError(f"Unlisted FR3 episode: dataset={dataset}, episode={episode}")
        record = self.episodes[dataset][episode]
        n = record.num_attributes
        mask = [True] * n
        if self.augment:
            probabilities = self.plate_probabilities if n == 3 else self.drawer_probabilities
            count = int(np.random.choice(n + 1, p=probabilities))
            combinations = tuple(itertools.combinations(range(n), count))
            chosen = combinations[np.random.randint(len(combinations))]
            mask = [attribute in chosen for attribute in range(n)]
        result = {**data, "prompt": prompt_for_mask(record, tuple(mask))}
        for key in ("dataset_index", "episode_index"):
            result.pop(key)
        return result
