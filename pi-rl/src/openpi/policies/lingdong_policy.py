"""Tron2 right-arm joint control from the corrected, trimmed Lingdong datasets."""

import dataclasses
import functools
import json
import pathlib

import numpy as np

CONFIG_NAME = "pi05_lingdong_right_arm"
ASSET_ID = "lingdong_right_arm"
RELATIVE_CONFIG_NAME = "pi05_lingdong_right_arm_relative"
RELATIVE_ASSET_ID = "lingdong_right_arm_relative"
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
RELATIVE_ACTION_MASK = (True,) * 7 + (False,)
GRIPPER_THRESHOLD = 0.6
GRIPPER_MIN_EXCURSION = 0.05
PROMPT_VARIANTS = {
    "pick up the object": (
        "pick up the object",
        "pick the object up",
        "lift the object",
        "lift up the object",
        "grab the object and lift it",
        "grasp the object and pick it up",
        "take hold of the object and lift it",
        "please pick up the object",
        "please lift the object",
        "pick up the item",
        "lift the item",
        "please pick the object up",
    ),
    "put down the object": (
        "put down the object",
        "put the object down",
        "set down the object",
        "set the object down",
        "place the object down",
        "lower the object and set it down",
        "please put down the object",
        "please set the object down",
        "put down the item",
        "set down the item",
        "place the item down",
        "please place the object down",
    ),
}


def gripper_binarization_spec() -> dict:
    return {
        "encoding": "binary_0_closed_1_open",
        "method": "episode_extrema_hysteresis_v1",
        "initial_threshold": GRIPPER_THRESHOLD,
        "min_excursion": GRIPPER_MIN_EXCURSION,
        "median_filter_frames": 3,
        "onset": "last_extremum_before_confirmed_motion",
        "plateaus_and_terminal": "hold_previous_command",
    }


def detect_gripper_events(gripper: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Detect whole-episode opening/closing events, retaining commands through plateaus.

    A three-frame median removes isolated interior spikes. A cumulative excursion
    of 0.05 from the last extremum confirms a reversal; its label is backdated to
    that extremum, so small initial steps belong to the same event. The 0.6
    threshold initializes only the first state, never intermediate plateaus.
    """
    gripper = np.asarray(gripper, dtype=np.float32)
    if gripper.ndim != 1 or len(gripper) == 0 or not np.isfinite(gripper).all():
        raise ValueError("Expected a nonempty finite gripper trajectory [T]")
    padded = np.pad(gripper, (1, 1), mode="edge")
    smooth = np.median(np.stack((padded[:-2], padded[1:-1], padded[2:])), axis=0)
    command = int(smooth[0] >= GRIPPER_THRESHOLD)
    labels = np.full(len(gripper), command, dtype=np.uint8)
    extremum = float(smooth[0])
    extremum_frame = 0
    events = []
    for frame, raw_value in enumerate(smooth):
        value = float(raw_value)
        if (command == 0 and value <= extremum) or (command == 1 and value >= extremum):
            extremum, extremum_frame = value, frame
            continue
        excursion = value - extremum if command == 0 else extremum - value
        if excursion < GRIPPER_MIN_EXCURSION:
            continue
        command = 1 - command
        labels[extremum_frame:] = command
        events.append(
            {
                "frame": extremum_frame,
                "confirmed_frame": frame,
                "command": command,
                "opening_before": extremum,
                "opening_at_confirmation": value,
            }
        )
        extremum, extremum_frame = value, frame
    return labels, events


def binarize_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """Copy complete selected episode actions and replace the gripper with event labels."""
    actions = np.array(actions, dtype=np.float32, copy=True)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) == 0:
        raise ValueError(f"Expected nonempty selected actions [T, 8], got {actions.shape}")
    actions[:, -1] = detect_gripper_events(actions[:, -1])[0]
    return actions


@functools.lru_cache(maxsize=4096)
def _episode_gripper_labels(repo_id: str, episode: int) -> np.ndarray:
    """Read numeric actions once per episode/worker; retain only byte-sized labels."""
    import pyarrow.parquet as pq

    root = pathlib.Path(repo_id)
    info = json.loads((root / "meta/info.json").read_text())
    path = root / info["data_path"].format(episode_index=episode, episode_chunk=episode // info["chunks_size"])
    table = pq.read_table(path, columns=["action"], use_threads=False)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    column = right_indices(info["features"], "action")[-1]
    labels = detect_gripper_events(actions[:, column])[0]
    labels.flags.writeable = False
    return labels


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
            # DeltaActions modifies its input in place, including already selected 8-D chunks.
            result["actions"] = _right(data["action"], self.action_indices, sequence=True).copy()
            for key in ("dataset_index", "episode_index", "frame_index"):
                if key in data:
                    result[key] = data[key]
        if "prompt" in data:
            prompt = np.asarray(data["prompt"]).item()
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return result


@dataclasses.dataclass(frozen=True)
class LingdongTrainingActions:
    """Slice cached whole-episode event labels using the sample's original frame index."""

    action_horizon: int | None = None
    repo_ids: tuple[str, ...] = ()

    def __call__(self, data: dict) -> dict:
        if not self.repo_ids:
            if self.action_horizon is not None:
                raise ValueError("Chunked Lingdong training requires whole-episode label sources")
            return {**data, "actions": binarize_gripper_actions(data["actions"])}
        if "episode_index" not in data or "frame_index" not in data:
            raise ValueError("Lingdong event labels require episode_index and frame_index")
        dataset = int(np.asarray(data.get("dataset_index", 0)).item())
        if not 0 <= dataset < len(self.repo_ids):
            raise ValueError(f"Invalid Lingdong dataset_index: {dataset}")
        episode = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        labels = _episode_gripper_labels(self.repo_ids[dataset], episode)
        if not 0 <= frame < len(labels):
            raise ValueError(f"Invalid Lingdong episode frame: {episode=}, {frame=}")
        actions = np.array(data["actions"], dtype=np.float32, copy=True)
        if self.action_horizon is not None and len(actions) != self.action_horizon:
            raise ValueError("Lingdong action chunk differs from the model horizon")
        indices = np.minimum(frame + np.arange(len(actions)), len(labels) - 1)
        actions[:, -1] = labels[indices]
        result = {**data, "actions": actions}
        for key in ("dataset_index", "episode_index", "frame_index"):
            result.pop(key, None)
        return result


@dataclasses.dataclass(frozen=True)
class LingdongPromptAugmentation:
    """Sample generic and validated episode-specific instructions before tokenization."""

    scene_prompts: tuple[dict[int, dict], ...] = ()

    def __call__(self, data: dict) -> dict:
        if "actions" not in data or "prompt" not in data:
            return data
        key = data["prompt"].strip().lower().rstrip(".!?")
        if (variants := PROMPT_VARIANTS.get(key)) is None:
            return data
        if self.scene_prompts:
            if "episode_index" not in data:
                raise ValueError("Scene language augmentation requires the original episode_index")
            dataset = int(np.asarray(data.get("dataset_index", 0)).item())
            episode = int(np.asarray(data["episode_index"]).item())
            if not 0 <= dataset < len(self.scene_prompts):
                raise ValueError("Invalid dataset_index for scene language augmentation")
            record = self.scene_prompts[dataset][episode]
            if record["canonical_prompt"] != key:
                raise ValueError("Scene language task differs from the sampled episode task")
            variants = (*variants, *record["scene_variants"])
        return {**data, "prompt": variants[np.random.randint(len(variants))]}


@dataclasses.dataclass(frozen=True)
class LingdongRightArmOutputs:
    """Remove model padding and return only seven right joints plus right gripper."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected an action chunk with at least 8 channels, got {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
