"""Shared constants and observation building for Lingdong Tron2 right-arm pi05 inference.

This module deliberately imports nothing from openpi or robot/camera SDKs so it can be used by the
robot-side client in any venv.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import numpy as np

CONFIG_NAME = "pi05_lingdong_right_arm"
DEFAULT_CHECKPOINT = (
    "/mnt/hdc/checkpoints/pi05-lingdong-right-arm/pi05_lingdong_right_arm/lingdong_right_arm_1027_163358/10000"
)
DEFAULT_LOG_ROOT = "/mnt/hdc/lingdong_real_logs"
DEFAULT_REPLAY_ROOT = "/mnt/hdc/lingdong_replay_data/lerobot_2026-10-26_12-09-32_subset"

CONTROL_HZ = 30.0  # dataset fps; one action step = 1/30 s
ACTION_HORIZON = 30

# Raw 18-D layout of the recorded LeRobot dataset (meta/info.json "names").
RAW_STATE_NAMES = (
    "abad_L_Joint", "hip_L_Joint", "yaw_L_Joint", "knee_L_Joint",
    "wrist_yaw_L_Joint", "wrist_pitch_L_Joint", "wrist_roll_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "yaw_R_Joint", "knee_R_Joint",
    "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint",
    "head_pitch_Joint", "head_yaw_Joint", "left_gripper", "right_gripper",
)  # fmt: skip
# Policy state/action channels, same order as openpi.policies.lingdong_policy.RIGHT_NAMES.
RIGHT_NAMES = (
    "abad_R_Joint", "hip_R_Joint", "yaw_R_Joint", "knee_R_Joint",
    "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint", "right_gripper",
)  # fmt: skip
RIGHT_INDICES = tuple(RAW_STATE_NAMES.index(n) for n in RIGHT_NAMES)  # (7..13, 17)
assert RIGHT_INDICES == (7, 8, 9, 10, 11, 12, 13, 17)

HEAD_IMAGE_KEY = "observation.images.cam_high"
RIGHT_IMAGE_KEY = "observation.images.cam_right_wrist"

# Robot joint-state message `data.q` (16): [L arm 7, R arm 7, head_pitch, head_yaw]  (robot_utils.Tron2)
Q16_LEFT = slice(0, 7)
Q16_RIGHT = slice(7, 14)
Q16_HEAD = slice(14, 16)

# Right-arm command range observed in the raw collection dataset (meta/stats.json, `action` min/max over
# all 2,8xx raw episodes). hip<=0, knee<=pi/6 and wrist pitch/roll = +-pi/4 look like teleop clamps.
# No URDF / vendor limit table was found. These are diagnostic ranges only; raw joint targets are not clipped.
DATASET_ACTION_MIN = np.array([-1.4491, -1.0082, -1.0489, -2.0864, -1.3963, -0.7854, -0.7854, 0.0])
DATASET_ACTION_MAX = np.array([0.2777, 0.0, 2.7646, 0.5236, 0.5615, 0.7854, 0.7854, 0.94])
# Training-distribution range (checkpoint norm_stats.json, state q01/q99) - used for warnings only.
NORM_STATE_Q01 = np.array([-1.0551, -0.598, -0.1094, -1.7591, -1.3686, -0.1677, -0.6747, 0.0])
NORM_STATE_Q99 = np.array([0.0194, -0.0013, 1.2359, -0.0255, -0.0394, 0.782, 0.4566, 0.94])
GRIPPER_MAX = 0.94  # recorded right_gripper max (= opening 94/100); high = open

# RealSense serials as named by the previous Tron2 deployment (tron2_old/lingdong_pi05_table_to_shelf_loop.py),
# which was trained on the RAW (uncorrected) collection data and therefore uses the collection-time names.
OLD_DEPLOY_SERIALS = {
    "cam_high": "261422302183",  # D455 head
    "cam_left_wrist": "427622271953",  # D405
    "cam_right_wrist": "427622272501",  # D405
}
# Wrist correction: in the RAW data the stream named `cam_left_wrist` shows the ACTIVE right arm (gripper over
# the bin) and `cam_right_wrist` shows the idle arm; the final training dataset exchanged the two wrist MP4s
# (no flip/rotation). So the policy's cam_right_wrist == the camera that collection called `cam_left_wrist`.
# Assuming collection used the same serial->name map as the old deployment, that is serial 427622271953.
# !!! VERIFY with `--mode camera-check` before --execute (the image must show the right gripper).
POLICY_CAMERA_SERIALS = {
    "cam_high": OLD_DEPLOY_SERIALS["cam_high"],
    "cam_right_wrist": OLD_DEPLOY_SERIALS["cam_left_wrist"],
}

# !!! The exact training prompts are in the final dataset's meta/tasks.jsonl
# (/oss/embodied-ai-data/lingdong/lerobot_2026-10-26_12-09-32_final), which is not available on this Orin.
# These are PLACEHOLDERS. Pass --prompt with the exact string from tasks.jsonl.
TASK_PROMPTS = {
    "pick_up": "pick up the object",  # task_index 0  (user-confirmed exact training string, 2026-10-28)
    "put_down": "put down the object",  # task_index 1  (UNVERIFIED placeholder)
}
# "pick up the object" confirmed by the user (2026-10-28) as the exact pick-up task string (prompt_from_task=True in
# LeRobotLingdongRightArmDataConfig; final meta/tasks.jsonl itself was not inspected here).
VERIFIED_TASK_PROMPTS: set[str] = {"pick up the object"}  # put_down still unverified


def resolve_prompt(task: str, prompt: str | None) -> tuple[str, bool]:
    """Returns (prompt, verified)."""
    if prompt:
        return prompt, True
    p = TASK_PROMPTS[task]
    verified = p in VERIFIED_TASK_PROMPTS
    if not verified:
        logging.warning(
            "!!! Using PLACEHOLDER prompt %r for task %s. The exact training prompt (final dataset "
            "meta/tasks.jsonl) was not available; pass --prompt '<exact task string>'.",
            p,
            task,
        )
    return p, verified


def build_raw_state(q16, left_opening_pct, right_opening_pct) -> np.ndarray:
    """Robot feedback -> raw 18-D state in the dataset order (by name, see RAW_STATE_NAMES)."""
    q16 = np.asarray(q16, dtype=np.float64)
    if q16.shape != (16,):
        raise ValueError(f"expected q with 16 values, got {q16.shape}")
    s = np.zeros(18, dtype=np.float32)
    s[0:7] = q16[Q16_LEFT]
    s[7:14] = q16[Q16_RIGHT]
    s[14:16] = q16[Q16_HEAD]
    s[16] = float(left_opening_pct) / 100.0
    s[17] = float(right_opening_pct) / 100.0
    return s


def select_right(state18) -> np.ndarray:
    state18 = np.asarray(state18, dtype=np.float32)
    if state18.shape[-1] != 18:
        raise ValueError(f"expected 18-D raw state, got {state18.shape}")
    return np.take(state18, RIGHT_INDICES, axis=-1)


def check_rgb(img: np.ndarray, name: str) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
        raise ValueError(f"{name}: expected HxWx3 uint8 RGB, got {img.shape} {img.dtype}")
    return np.ascontiguousarray(img)


def build_policy_obs(state8, head_rgb, right_wrist_rgb, prompt: str) -> dict:
    """Policy input exactly as LingdongRightArmInputs expects (8-D state in RIGHT_NAMES order).

    Images are passed at native resolution (480x640 RGB); the policy's own model transform performs the
    same JAX resize_with_pad(224) that was used in training.
    """
    state8 = np.asarray(state8, dtype=np.float32)
    if state8.shape != (8,) or not np.isfinite(state8).all():
        raise ValueError(f"bad state8 {state8}")
    return {
        "observation.state": state8,
        HEAD_IMAGE_KEY: check_rgb(head_rgb, "cam_high"),
        RIGHT_IMAGE_KEY: check_rgb(right_wrist_rgb, "cam_right_wrist"),
        "prompt": prompt,
    }


@dataclasses.dataclass
class Observation:
    state18: np.ndarray
    images: dict  # "cam_high", "cam_right_wrist" -> HxWx3 uint8 RGB (policy semantics)
    t_capture: float  # time.monotonic() when assembled
    ages_s: dict = dataclasses.field(default_factory=dict)  # staleness of each source
    frame_index: int | None = None  # replay only

    @property
    def state8(self) -> np.ndarray:
        return select_right(self.state18)


def ensure_dir(p: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
