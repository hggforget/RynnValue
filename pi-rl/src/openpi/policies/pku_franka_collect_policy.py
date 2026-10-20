"""Dual-camera FR3 collection with absolute targets or recorded relative actions.

Only measured state is exposed to the policy. The dataset's 20-D combined
``observation.state`` includes the current target, which equals the action
label, so neither that field nor ``observation.state.target`` is consumed.
"""

import dataclasses
import json
import pathlib
from typing import Literal

import numpy as np

COLLECT_OBJECTS_PROMPT = "place each object on the plate that matches its color."
_MISSPELLED_PROMPT = "lace each object on the plate that matches its color."
ACTION_DIM = 7
STATE_DIM = 14
ActionRepresentation = Literal["absolute", "relative"]


def action_key(action_representation: ActionRepresentation) -> str:
    if action_representation == "absolute":
        return "action"
    if action_representation == "relative":
        return "action.relative"
    raise ValueError(f"Unknown FR3 action representation: {action_representation}")


def validate_action_metadata(repo_id: str | None, action_representation: ActionRepresentation) -> None:
    """Check the selected label field when mounted; checkpoint serving needs no dataset."""
    key = action_key(action_representation)
    if repo_id is None:
        return
    info_path = pathlib.Path(repo_id) / "meta/info.json"
    if not info_path.is_file():
        return
    features = json.loads(info_path.read_text())["features"]
    if key not in features:
        raise ValueError(f"FR3 {action_representation} training requires dataset field '{key}'")
    expected = "absolute_target_pose" if action_representation == "absolute" else "relative_delta_pose"
    if features[key].get("representation", expected) != expected:
        raise ValueError(f"Dataset field '{key}' must use representation '{expected}'")


def training_episode_ids(repo_id: str, excluded_episode_ids: tuple[int, ...]) -> tuple[int, ...] | None:
    """Exclude explicitly labeled failed demos without changing dataset metadata.

    Read the episode list rather than assuming a fixed episode count, so later
    recordings remain eligible. Serving can load checkpoint assets without the
    original dataset being mounted.
    """
    if not excluded_episode_ids:
        return None
    episodes_path = pathlib.Path(repo_id) / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return None
    excluded = set(excluded_episode_ids)
    episodes = []
    for line in episodes_path.read_text().splitlines():
        if line.strip():
            episode_id = int(json.loads(line)["episode_index"])
            if episode_id not in excluded:
                episodes.append(episode_id)
    if not episodes:
        raise ValueError("No FR3 training episodes remain after excluding failed demonstrations.")
    return tuple(episodes)


def _image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image, got shape {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC or CHW RGB image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class PkuFrankaCollectInputs:
    """Map raw LeRobot fields to OpenPI; also accepts the same fields at inference.

    State: measured Cartesian pose(6), normalized gripper opening(1), joints(7).
    Absolute labels come from ``action``; relative labels come directly from
    ``action.relative`` (no differencing or accumulation). The first six values
    are Cartesian targets or increments. Both use gripper 0=closed, 1=open;
    the gripper command is never differenced.
    """

    action_representation: ActionRepresentation = "absolute"

    def __post_init__(self):
        action_key(self.action_representation)

    def __call__(self, data: dict) -> dict:
        cartesian = np.asarray(data["observation.state.cartesian"], dtype=np.float32)
        gripper = np.asarray(data["observation.state.gripper"], dtype=np.float32).reshape(-1)
        joints = np.asarray(data["observation.state.joints"], dtype=np.float32)
        if cartesian.shape != (6,) or gripper.shape != (1,) or joints.shape != (7,):
            raise ValueError(
                "Expected measured Cartesian pose(6), gripper(1), and joints(7); "
                f"got {cartesian.shape}, {gripper.shape}, and {joints.shape}"
            )
        inputs = {
            "state": np.concatenate([cartesian, gripper, joints]),
            "image": {
                "base_0_rgb": _image(data["observation.images.primary"]),
                "left_wrist_0_rgb": _image(data["observation.images.wrist"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_},
        }
        key = action_key(self.action_representation)
        if key in data:
            actions = np.asarray(data[key], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError(
                    f"Expected a finite {self.action_representation} Cartesian action chunk [T, 7], got {actions.shape}"
                )
            inputs["actions"] = actions
        elif "action" in data or "action.relative" in data:
            raise KeyError(f"Missing selected FR3 action field '{key}'; refusing to use the other representation")
        if "prompt" in data:
            prompt = np.asarray(data["prompt"]).item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            if prompt:
                inputs["prompt"] = COLLECT_OBJECTS_PROMPT if prompt == _MISSPELLED_PROMPT else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class PkuFrankaCollectOutputs:
    """Trim padding; preserve the selected action representation and 0/1 gripper."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}
