"""FR3 transition transforms and progress-based frame filtering."""

import dataclasses
import json
import pathlib
from typing import Literal

import numpy as np
from openpi_client import image_tools

from openpi.policies import pku_franka_collect_policy as collect


def collection_metadata(repo_id: str | None, *, require_progress: bool = True) -> tuple[dict, dict[int, int]]:
    if repo_id is None:
        return {"fps": 30}, {}
    root = pathlib.Path(repo_id)
    if not (root / "meta/info.json").exists():
        # Checkpoint serving does not require the training dataset.
        return {"fps": 30}, {}
    info = json.loads((root / "meta/info.json").read_text())
    if require_progress and "progress" not in info["features"]:
        raise ValueError("This FR3 training mode requires per-frame 'progress' in metadata and parquet files.")
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    return info, {int(ep["episode_index"]): int(ep["length"]) for ep in episodes}


def collection_task_indices(repo_id: str) -> tuple[int, ...]:
    """Use the dataset task vocabulary for critic conditioning, independently of row order."""
    root = pathlib.Path(repo_id)
    if not (root / "meta/info.json").is_file():
        # Policy serving does not use the critic or require the training dataset.
        return ()
    rows = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    indices = [row["task_index"] for row in rows]
    if not indices or any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("FR3 critic conditioning requires nonnegative integer task indices")
    if len(set(indices)) != len(indices):
        raise ValueError("FR3 critic conditioning requires unique task indices")
    return tuple(sorted(indices))


def window_advantages(progress: np.ndarray, horizon: int, discount: float) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float64).reshape(-1)
    if not len(progress) or not np.isfinite(progress).all():
        raise ValueError("progress must be nonempty and finite")
    if horizon < 1 or not 0 < discount <= 1:
        raise ValueError("Invalid advantage horizon or discount")
    rewards = np.diff(progress, append=progress[-1])
    returns = np.zeros(len(progress) + 1)
    for t in range(len(progress) - 1, -1, -1):
        returns[t] = rewards[t] + discount * returns[t + 1]
    future = np.minimum(np.arange(len(progress)) + horizon, len(progress))
    return returns[:-1] - discount**horizon * returns[future]


def filtered_frame_indices(
    repo_id: str,
    excluded_episode_ids: tuple[int, ...],
    horizon: int,
    discount: float,
    criterion: str,
    quantile: float,
) -> tuple[tuple[int, ...], float]:
    import pyarrow.parquet as pq  # Optional for NumPy-only protocol tests.

    collection_metadata(repo_id)
    if not 0 <= quantile <= 1:
        raise ValueError("advantage_quantile must be in [0, 1]")
    indices, scores = [], []
    excluded = set(excluded_episode_ids)
    for path in sorted((pathlib.Path(repo_id) / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "progress"]).to_pydict()
        if int(table["episode_index"][0]) in excluded:
            continue
        indices.append(np.asarray(table["index"], dtype=np.int64))
        scores.append(window_advantages(table["progress"], horizon, discount))
    if not scores:
        raise ValueError("No eligible FR3 episodes for Filter BC")
    values = np.concatenate(scores)
    if criterion == "zero":
        threshold = 0.0
    elif criterion == "mean":
        threshold = float(values.mean())
    elif criterion == "median":
        threshold = float(np.median(values))
    elif criterion == "quantile":
        threshold = float(np.quantile(values, quantile))
    else:
        raise ValueError(f"Unknown advantage criterion: {criterion}")
    kept = np.concatenate(indices)[values > threshold]
    if not len(kept):
        raise ValueError("Filter BC retained no frames; check progress labels and threshold")
    return tuple(int(i) for i in kept), threshold


@dataclasses.dataclass(frozen=True)
class PkuFrankaCollectRLInputs:
    episode_lengths: dict[int, int] = dataclasses.field(default_factory=dict)
    failed_episode_ids: tuple[int, ...] = ()
    discount: float = 0.99
    progress_reward_weight: float = 1.0
    failure_penalty: float = 1.0
    action_representation: collect.ActionRepresentation = "absolute"
    # Keep legacy collection PBRS as the default. Pick-and-place uses measured
    # progress increments, without replacing the last prediction by zero.
    progress_reward_mode: Literal["potential", "delta"] = "potential"
    task_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if self.progress_reward_mode not in ("potential", "delta"):
            raise ValueError("FR3 progress_reward_mode must be potential or delta")
        if not np.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("FR3 reward discount must be in (0, 1]")
        for value in (self.progress_reward_weight, self.failure_penalty):
            if not np.isfinite(value) or value < 0:
                raise ValueError("FR3 reward weights must be finite and nonnegative")
        if len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("FR3 critic task indices must be unique")

    def __call__(self, data: dict) -> dict:
        transform = collect.PkuFrankaCollectInputs(action_representation=self.action_representation)
        cartesian = np.asarray(data["observation.state.cartesian"])
        if cartesian.ndim == 1:
            # Same transform supports norm-stat scans and checkpoint inference.
            return transform(data)
        if cartesian.shape != (2, 6):
            raise ValueError(f"Expected current/next Cartesian state [2, 6], got {cartesian.shape}")
        views = []
        for offset in [0, 1]:
            item = dict(data)
            for key in [
                "observation.state.cartesian",
                "observation.state.gripper",
                "observation.state.joints",
                "observation.images.primary",
                "observation.images.wrist",
            ]:
                item[key] = np.asarray(data[key])[offset]
            views.append(transform(item))
        inputs, future = views
        episode = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        horizon = len(inputs["actions"])
        length = self.episode_lengths[episode]
        if not 0 <= frame < length or horizon < 1:
            raise ValueError("Invalid FR3 IQL frame or action horizon")
        terminal = frame + horizon >= length - 1
        steps = min(horizon, max(0, length - 1 - frame))
        progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
        if progress.shape != (horizon + 1,) or not np.isfinite(progress).all():
            raise ValueError("Expected finite progress [H+1] for FR3 IQL")
        mask = float(not terminal)
        if self.progress_reward_mode == "delta":
            # Only real transitions contribute. In particular, the final row has
            # zero reward and repeated padding cannot fabricate an improvement.
            increments = np.diff(progress)[:steps]
            shaped = self.progress_reward_weight * float(np.dot(self.discount ** np.arange(steps), increments))
        else:
            # Legacy PBRS: absorbing terminal potential is zero.
            shaped = self.progress_reward_weight * (self.discount**steps * mask * progress[-1] - progress[0])
        penalize = terminal and episode in self.failed_episode_ids
        if self.progress_reward_mode == "delta" and steps == 0:
            penalize = False
        penalty = -self.failure_penalty if penalize else 0.0
        reward = shaped + self.discount ** max(0, steps - 1) * penalty
        inputs.update(
            next_state=future["state"],
            # Model ResizeImages only visits the current image dictionary.
            next_image_base=np.asarray(image_tools.resize_with_pad(future["image"]["base_0_rgb"], 224, 224)),
            next_image_wrist=np.asarray(image_tools.resize_with_pad(future["image"]["left_wrist_0_rgb"], 224, 224)),
            mask=np.float32(mask),
            reward=np.float32(reward),
            # Already discounted; trainer must not shape it twice.
            chunk_reward=np.float32(reward),
        )
        if self.task_indices:
            task_index = int(np.asarray(data["task_index"]).item())
            if task_index not in self.task_indices:
                raise ValueError(f"Unknown FR3 critic task index: {task_index}")
            task = np.zeros(len(self.task_indices), dtype=np.float32)
            task[self.task_indices.index(task_index)] = 1.0
            # Both sides stay in measured physical units. Policy normalization
            # runs afterwards and must not alter the one-hot task vocabulary.
            inputs["iql_state"] = np.concatenate((inputs["state"], task))
            inputs["next_iql_state"] = np.concatenate((future["state"], task))
        return inputs
