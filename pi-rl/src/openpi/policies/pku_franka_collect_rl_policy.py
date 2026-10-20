"""FR3 transition transforms and progress-based frame filtering."""

import dataclasses
import json
import pathlib

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
    import pyarrow.parquet as pq  # noqa: PLC0415 - optional for NumPy-only protocol tests

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
        terminal = frame + horizon >= length - 1
        steps = min(horizon, max(0, length - 1 - frame))
        progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
        if progress.shape != (horizon + 1,) or not np.isfinite(progress).all():
            raise ValueError("Expected finite progress [H+1] for FR3 IQL")
        mask = float(not terminal)
        # Absorbing terminal potential is zero. Padded endpoints never bootstrap.
        shaped = self.progress_reward_weight * (self.discount**steps * mask * progress[-1] - progress[0])
        penalty = -self.failure_penalty if terminal and episode in self.failed_episode_ids else 0.0
        reward = shaped + self.discount ** max(0, steps - 1) * penalty
        inputs.update(
            next_state=future["state"],
            # Model ResizeImages only visits the current image dictionary.
            next_image_base=np.asarray(image_tools.resize_with_pad(future["image"]["base_0_rgb"], 224, 224)),
            next_image_wrist=np.asarray(image_tools.resize_with_pad(future["image"]["left_wrist_0_rgb"], 224, 224)),
            mask=np.float32(mask),
            reward=np.float32(reward),
            # Already discounted PBRS + terminal penalty; trainer must not shape it twice.
            chunk_reward=np.float32(reward),
        )
        return inputs
