"""Single-camera PKU Franka policy with Cartesian delta actions.

State: [x, y, z, roll, pitch, yaw, gripper_width_m].
Actions: [dx, dy, dz, droll, dpitch, dyaw, gripper_command].
The recorded arm actions are already deltas; gripper commands stay in [-1, 1].
"""

import dataclasses

import numpy as np

from openpi import transforms

PKU_FRANKA_DIM = 7


@dataclasses.dataclass(frozen=True)
class PkuFrankaInputs(transforms.DataTransformFn):
    """Accept the dataset's raw image/state/actions keys for training and inference."""

    def __call__(self, data: dict) -> dict:
        image = np.asarray(data["image"])
        if image.ndim != 3:
            raise ValueError(f"Expected one RGB image, got shape {image.shape}")
        if image.shape[-1] != 3 and image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected an HWC or CHW RGB image, got shape {image.shape}")
        # LeRobot returns float CHW images in [0, 1]; inference accepts uint8 HWC.
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (PKU_FRANKA_DIM,):
            raise ValueError(f"Expected a 7-D Cartesian state, got shape {state.shape}")
        inputs = {
            "state": state,
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": np.True_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != PKU_FRANKA_DIM:
                raise ValueError(f"Expected a Cartesian action chunk [T, 7], got shape {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class PkuFrankaOutputs(transforms.DataTransformFn):
    """Remove model padding, preserving Cartesian deltas and gripper commands."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :PKU_FRANKA_DIM]}
