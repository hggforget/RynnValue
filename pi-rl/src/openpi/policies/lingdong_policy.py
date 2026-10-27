"""Tron2 right-arm joint control from the corrected, trimmed Lingdong datasets."""

import dataclasses
import json
import pathlib

import numpy as np

CONFIG_NAME = "pi05_lingdong_right_arm"
ASSET_ID = "lingdong_right_arm"
DEFAULT_DATASET_ROOT = "/mnt/rynnbot_hangzhou/hdc/pku/datasets/lingdong"
DATASET_NAMES = ("lerobot_2026-10-26_12-09-32_final",)
HEAD_IMAGE_KEY = "observation.images.cam_high"
RIGHT_IMAGE_KEY = "observation.images.cam_right_wrist"
VIDEO_KEYS = (HEAD_IMAGE_KEY, RIGHT_IMAGE_KEY)
RIGHT_NAMES = (
    "abad_R_Joint",
    "hip_R_Joint",
    "yaw_R_Joint",
    "knee_R_Joint",
    "wrist_yaw_R_Joint",
    "wrist_pitch_R_Joint",
    "wrist_roll_R_Joint",
    "right_gripper",
)
RIGHT_INDICES = (7, 8, 9, 10, 11, 12, 13, 17)
ACTION_DIM = STATE_DIM = 8


def right_indices(features: dict, key: str) -> tuple[int, ...]:
    feature = features.get(key, {})
    names = feature.get("names", [])
    if feature.get("shape") != [18] or len(names) != 18 or len(set(names)) != 18:
        raise ValueError(f"Lingdong {key} requires 18 explicitly named raw channels")
    missing = set(RIGHT_NAMES) - set(names)
    if missing:
        raise ValueError(f"Lingdong {key} is missing right-arm channels: {sorted(missing)}")
    return tuple(names.index(name) for name in RIGHT_NAMES)


def dataset_indices(
    repo_ids: tuple[str, ...], *, allow_unmounted: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve action/state by field name and reject incompatible mixed schemas."""
    available = [(pathlib.Path(repo) / "meta/info.json").is_file() for repo in repo_ids]
    if not repo_ids or len(set(repo_ids)) != len(repo_ids):
        raise ValueError("Lingdong training requires distinct dataset paths")
    for repo in repo_ids:
        for filename, operation in (
            (".merge_plan.json", "final-dataset merge"),
            (".wrist_camera_correction.json", "wrist-camera correction"),
        ):
            marker = pathlib.Path(repo) / filename
            if marker.is_file() and json.loads(marker.read_text()).get("status") != "complete":
                raise ValueError(f"Lingdong {operation} is still in progress: {repo}")
    if not any(available) and allow_unmounted:
        # Checkpoint serving can use the canonical raw schema or an already selected 8-D state.
        return RIGHT_INDICES, RIGHT_INDICES
    if not all(available):
        raise FileNotFoundError(
            f"Missing Lingdong datasets: {[r for r, ok in zip(repo_ids, available, strict=True) if not ok]}"
        )
    mappings = []
    for repo in repo_ids:
        info = json.loads((pathlib.Path(repo) / "meta/info.json").read_text())
        if info.get("codebase_version") != "v2.1" or info.get("fps") != 30:
            raise ValueError(f"Lingdong training expects LeRobot v2.1 at 30 FPS: {repo}")
        features = info["features"]
        for camera in VIDEO_KEYS:
            if features.get(camera, {}).get("dtype") != "video":
                raise ValueError(f"Missing Lingdong training camera {camera}: {repo}")
        mappings.append((right_indices(features, "observation.state"), right_indices(features, "action")))
    if len(set(mappings)) != 1:
        raise ValueError("Lingdong repos must share the same raw action/state channel order")
    return mappings[0]


def _right(values, indices: tuple[int, ...], *, sequence: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != (2 if sequence else 1) or values.shape[-1] not in (8, 18):
        raise ValueError(
            f"Expected {'an action chunk [T, 18|8]' if sequence else 'a state [18|8]'}, got {values.shape}"
        )
    selected = values if values.shape[-1] == 8 else np.take(values, indices, axis=-1)
    if not np.isfinite(selected).all():
        raise ValueError("Right-arm channels must be finite")
    return selected


def _image(value) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected one CHW or HWC RGB image, got {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating RGB images must be finite and normalized to [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("RGB images must be uint8 or normalized floating point")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class LingdongRightArmInputs:
    """Select measured right state and recorded right action; preserve gripper values.

    Raw 18-D training/serving inputs are selected by metadata-derived indices.
    Inference may instead provide an 8-D state in RIGHT_NAMES order. Wrist videos
    in these datasets are already corrected; no additional swapping is applied.
    """

    state_indices: tuple[int, ...] = RIGHT_INDICES
    action_indices: tuple[int, ...] = RIGHT_INDICES

    def __call__(self, data: dict) -> dict:
        result = {
            "state": _right(data["observation.state"], self.state_indices, sequence=False),
            "image": {"base_0_rgb": _image(data[HEAD_IMAGE_KEY]), "right_wrist_0_rgb": _image(data[RIGHT_IMAGE_KEY])},
            "image_mask": {"base_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "action" in data:
            result["actions"] = _right(data["action"], self.action_indices, sequence=True)
        if "prompt" in data:
            prompt = np.asarray(data["prompt"]).item()
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return result


@dataclasses.dataclass(frozen=True)
class LingdongRightArmOutputs:
    """Remove model padding and return only seven right joints plus right gripper."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected an action chunk with at least 8 channels, got {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
