"""Lingdong IQL transitions with explicit, frame-aligned reward labels."""

import dataclasses
import json
from pathlib import Path

import numpy as np
from openpi_client import image_tools

from openpi.policies import lingdong_language
from openpi.policies import lingdong_policy as policy

REWARD_VERSION = "lingdong_iql_rewards_v1"


def reward_metadata(
    repo_id: str,
    *,
    task: str = "all",
    reward_source: str = "progress",
    reward_labels_path: str | None = None,
    assume_success: bool = False,
) -> dict:
    """Validate labels before constructing video loaders or allocating model weights.

    A fingerprinted JSON sidecar can supply dense progress and/or episode success
    without changing source Parquets, normalization or validated scene prompts.
    Progress can also come from a real dataset ``progress`` column. Episode ends
    alone never imply success unless ``assume_success`` was explicitly enabled.
    """
    if reward_source not in ("progress", "terminal"):
        raise ValueError("Lingdong reward_source must be progress or terminal")
    root = Path(repo_id)
    policy.dataset_indices((repo_id,))
    info = json.loads((root / "meta/info.json").read_text())
    rows = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    if len(rows) != info["total_episodes"] or sum(row["length"] for row in rows) != info["total_frames"]:
        raise ValueError("Inconsistent Lingdong episode metadata")
    selected = policy.task_episode_ids((repo_id,), task)
    selected_ids = set(selected[repo_id]) if selected is not None else {row["episode_index"] for row in rows}
    lengths, tasks = {}, {}
    for row in rows:
        episode = row["episode_index"]
        if type(episode) is not int or episode < 0 or episode in lengths or row["length"] < 1:
            raise ValueError("Invalid Lingdong episode index or length")
        if row.get("tasks") not in ([prompt] for prompt in policy.TASK_PROMPTS.values()):
            raise ValueError("Lingdong IQL requires one canonical task per episode")
        lengths[episode] = row["length"]
        tasks[episode] = tuple(policy.TASK_PROMPTS.values()).index(row["tasks"][0])

    records = {}
    if reward_labels_path:
        labels = json.loads(Path(reward_labels_path).read_text())
        if labels.get("version") != REWARD_VERSION:
            raise ValueError("Unsupported Lingdong IQL reward label version")
        if labels.get("dataset_signature") != lingdong_language.dataset_signature(root):
            raise ValueError("Lingdong IQL reward dataset fingerprint changed; regenerate/reindex labels")
        records = labels.get("episodes", {})
        if not isinstance(records, dict) or set(records) - {str(index) for index in lengths}:
            raise ValueError("Lingdong IQL labels contain invalid episode IDs")
        if any(not isinstance(record, dict) for record in records.values()):
            raise ValueError("Lingdong IQL episode labels must be objects")

    progress, outcomes = {}, {}
    sidecar_progress = any(record.get("progress") is not None for record in records.values())
    dataset_progress = reward_source == "progress" and not sidecar_progress and "progress" in info["features"]
    if reward_source == "progress" and not sidecar_progress and not dataset_progress:
        raise ValueError(
            "Lingdong IQL requires real progress labels: supply --data.reward-labels-path "
            "or a dataset with a progress column. Episode length is not a progress label."
        )
    for episode in selected_ids:
        record = records.get(str(episode), {})
        if reward_source == "progress" and sidecar_progress:
            values = np.asarray(record.get("progress"), dtype=np.float32)
            if values.shape != (lengths[episode],) or not np.isfinite(values).all():
                raise ValueError(f"Expected finite per-frame progress for Lingdong episode {episode}")
            progress[episode] = values
        success = record.get("success")
        if success is not None and type(success) is not bool:
            raise ValueError(f"Lingdong success must be true/false/null: episode {episode}")
        if success is None and assume_success:
            success = True
        if reward_source == "terminal" and success is None:
            raise ValueError(
                f"Missing success/failure for Lingdong episode {episode}; provide reward labels "
                "or explicitly set --data.assume-success for an all-success demonstration baseline"
            )
        if success is not None:
            outcomes[episode] = success
    return {
        "info": info,
        "episode_lengths": lengths,
        "episode_tasks": tasks,
        "episode_ids": tuple(row["episode_index"] for row in rows if row["episode_index"] in selected_ids),
        "progress": progress,
        "outcomes": outcomes,
        "dataset_progress": dataset_progress,
    }


@dataclasses.dataclass(frozen=True)
class LingdongRightArmRLInputs:
    state_indices: tuple[int, ...] = policy.RIGHT_INDICES
    action_indices: tuple[int, ...] = policy.RIGHT_INDICES
    crop_head: bool = True
    episode_lengths: dict[int, int] = dataclasses.field(default_factory=dict, repr=False)
    episode_tasks: dict[int, int] = dataclasses.field(default_factory=dict, repr=False)
    progress: dict[int, np.ndarray] = dataclasses.field(default_factory=dict, repr=False)
    outcomes: dict[int, bool] = dataclasses.field(default_factory=dict, repr=False)
    reward_source: str = "progress"
    discount: float = 0.99
    progress_reward_weight: float = 1.0
    success_reward: float = 1.0
    failure_penalty: float = 1.0

    def __post_init__(self):
        if self.reward_source not in ("progress", "terminal"):
            raise ValueError("Lingdong reward_source must be progress or terminal")
        if not np.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("Lingdong IQL discount must be in (0, 1]")
        for value in (self.progress_reward_weight, self.success_reward, self.failure_penalty):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Lingdong IQL reward weights must be finite and nonnegative")

    def __call__(self, data: dict) -> dict:
        adapter = policy.LingdongRightArmInputs(self.state_indices, self.action_indices, crop_head=self.crop_head)
        states = np.asarray(data["observation.state"])
        if states.ndim == 1:
            # Checkpoint serving uses the ordinary SFT adapter and needs no reward data.
            return adapter(data)
        if states.ndim != 2 or states.shape[0] != 2:
            raise ValueError("Lingdong IQL requires current/next state [2, 18|8]")
        views = []
        for offset in (0, 1):
            item = dict(data)
            for key in ("observation.state", *policy.VIDEO_KEYS):
                values = np.asarray(data[key])
                if values.shape[0] != 2:
                    raise ValueError(f"Lingdong IQL requires two observations for {key}")
                item[key] = values[offset]
            views.append(adapter(item))
        inputs, next_inputs = views
        episode = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        length = self.episode_lengths[episode]
        horizon = len(inputs["actions"])
        if not 0 <= frame < length or horizon < 1:
            raise ValueError("Invalid Lingdong IQL frame or action horizon")
        steps = min(horizon, length - 1 - frame)
        terminal = frame + horizon >= length - 1
        reward = 0.0
        if self.reward_source == "progress":
            if episode in self.progress:
                indices = np.minimum(frame + np.arange(horizon + 1), length - 1)
                progress = self.progress[episode][indices]
            else:
                progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            if progress.shape != (horizon + 1,) or not np.isfinite(progress).all():
                raise ValueError("Lingdong IQL requires finite progress [H+1]")
            # Use actual progress improvements, retaining regressions and the value
            # model's scale (including negative remaining seconds). Never fabricate
            # a successful endpoint, monotonicize labels or reward padded steps.
            increments = np.diff(progress)[:steps]
            reward = self.progress_reward_weight * float(np.dot(self.discount ** np.arange(steps), increments))
        if terminal and steps > 0 and episode in self.outcomes:
            bonus = self.success_reward if self.outcomes[episode] else -self.failure_penalty
            reward += self.discount ** (steps - 1) * bonus
        if not np.isfinite(reward):
            raise ValueError("Nonfinite Lingdong IQL reward")
        task = np.eye(2, dtype=np.float32)[self.episode_tasks[episode]]
        inputs.update(
            next_state=next_inputs["state"],
            next_image_base=np.asarray(image_tools.resize_with_pad(next_inputs["image"]["base_0_rgb"], 224, 224)),
            next_image_wrist=np.asarray(
                image_tools.resize_with_pad(next_inputs["image"]["right_wrist_0_rgb"], 224, 224)
            ),
            # Separate critic inputs remain in physical units on both sides of
            # the transition; policy state/action normalization runs afterwards.
            iql_state=np.concatenate((inputs["state"], task)).astype(np.float32),
            next_iql_state=np.concatenate((next_inputs["state"], task)).astype(np.float32),
            reward=np.float32(reward),
            chunk_reward=np.float32(reward),
            mask=np.float32(not terminal),
        )
        return inputs
