"""Dual-camera FR3 collection with absolute Cartesian target actions.

Only measured state is exposed to the policy. The dataset's 20-D combined
``observation.state`` includes the current target, which equals the action
label, so neither that field nor ``observation.state.target`` is consumed.
"""

import dataclasses
import json
import pathlib

import numpy as np

COLLECT_OBJECTS_PROMPT = "place each object on the plate that matches its color."
_MISSPELLED_PROMPT = "lace each object on the plate that matches its color."
ACTION_DIM = 7
STATE_DIM = 14


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
    Actions: absolute target [x, y, z, roll, pitch, yaw, gripper], with 0=closed
    and 1=open. ``action.relative`` is a separate recorded control signal and
    is deliberately not used by this absolute-action policy.
    """

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
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected an absolute Cartesian action chunk [T, 7], got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = np.asarray(data["prompt"]).item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            if prompt:
                inputs["prompt"] = COLLECT_OBJECTS_PROMPT if prompt == _MISSPELLED_PROMPT else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class PkuFrankaCollectOutputs:
    """Trim model padding; preserve absolute targets and the 0/1 gripper scale."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}
