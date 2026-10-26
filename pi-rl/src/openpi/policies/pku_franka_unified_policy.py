"""FR3 inputs with a required primary camera and an optional wrist camera."""

import dataclasses

import numpy as np

from openpi.policies import pku_franka_policy

IMAGE_KEY = "observation.images.primary"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action.relative"


@dataclasses.dataclass(frozen=True)
class PkuFrankaUnifiedInputs:
    """Preserve relative actions and 0=closed, 1=open gripper values.

    State is measured Cartesian pose(6) plus normalized gripper opening(1).
    No target pose, joint angles, differencing, or gripper conversion is added.
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data[STATE_KEY], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite unified state [7], got {state.shape}")
        if not 0 <= state[6] <= 1:
            raise ValueError("Unified state gripper must be normalized to [0, 1]")
        mapped = {"image": data[IMAGE_KEY], "state": state}
        if ACTION_KEY in data:
            actions = np.asarray(data[ACTION_KEY], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite relative action chunk [T, 7], got {actions.shape}")
            if np.any((actions[..., 6] < 0) | (actions[..., 6] > 1)):
                raise ValueError("Unified action gripper must be normalized to [0, 1]")
            mapped["actions"] = actions
        elif "action" in data or "actions" in data:
            raise KeyError(f"Unified training requires '{ACTION_KEY}'")
        if "prompt" in data:
            mapped["prompt"] = data["prompt"]
        inputs = pku_franka_policy.PkuFrankaInputs()(mapped)
        if WRIST_KEY in data:
            wrist = pku_franka_policy.PkuFrankaInputs()({"image": data[WRIST_KEY], "state": state})
            inputs["image"]["left_wrist_0_rgb"] = wrist["image"]["base_0_rgb"]
            inputs["image_mask"]["left_wrist_0_rgb"] = np.True_
        else:
            inputs["image"]["left_wrist_0_rgb"] = np.zeros_like(inputs["image"]["base_0_rgb"])
            inputs["image_mask"]["left_wrist_0_rgb"] = np.False_
        return inputs
