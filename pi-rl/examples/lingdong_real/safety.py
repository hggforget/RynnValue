"""Raw passthrough "safety" for the Tron2 right arm (pure numpy, no hardware access).

There is deliberately NO clipping, NO rate limit, NO ramp: the policy's absolute targets are executed verbatim.
The only stop condition is a malformed chunk (wrong shape / NaN / inf). State range checks only produce warnings.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from common import DATASET_ACTION_MAX, DATASET_ACTION_MIN, NORM_STATE_Q01, NORM_STATE_Q99

RAW_BANNER = (
    "\n" + "!" * 78 + "\n"
    "!!  RAW MODE: model actions are sent to the robot VERBATIM.                          !!\n"
    "!!  No interpolation, no ramp, no per-step limit, no joint/gripper clipping,          !!\n"
    "!!  no tracking-lag / stale-state / stale-image stops. Only NaN/inf output stops.    !!\n"
    "!!  A bad chunk WILL jump the arm at full servo speed. Keep the hardware e-stop      !!\n"
    "!!  in hand. Ctrl+C = stop & hold last target (twice = abort immediately).           !!\n"
    + "!" * 78
)


class SafetyViolation(RuntimeError):
    """Raised when a chunk / state must not be executed (only: malformed or non-finite)."""


@dataclasses.dataclass
class ChunkReport:
    first_jump: float  # max |a0 - reference| over joints (informational only)
    max_step: float  # max |a[i+1] - a[i]| over joints within the chunk (informational only)
    out_of_distribution_joints: list

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ActionSafety:
    """Validates chunks and returns them unchanged."""

    def check_state(self, state8: np.ndarray, *, strict: bool = False) -> list[str]:
        """Return warnings (never blocks); raises only on non-finite state. `strict` is ignored."""
        state8 = np.asarray(state8, dtype=np.float64)
        if not np.isfinite(state8).all():
            raise SafetyViolation(f"non-finite measured state {state8}")
        msgs = []
        bad = np.where((state8[:7] < DATASET_ACTION_MIN[:7] - 0.05) | (state8[:7] > DATASET_ACTION_MAX[:7] + 0.05))[0]
        if len(bad):
            msgs.append(f"measured joints {bad.tolist()} outside the recorded command range: "
                        f"{state8[bad].round(4).tolist()} (warning only)")
        ood = np.where((state8 < NORM_STATE_Q01 - 0.1) | (state8 > NORM_STATE_Q99 + 0.1))[0]
        if len(ood):
            msgs.append(f"state channels {ood.tolist()} are outside the training q01/q99 range (+-0.1): "
                        f"{state8[ood].round(4).tolist()} (warning only; policy may behave worse from here)")
        return msgs

    def process(self, chunk: np.ndarray, reference8: np.ndarray) -> tuple[np.ndarray, ChunkReport]:
        """Return (chunk verbatim as float64 copy, report). Raises SafetyViolation only on bad shape / NaN / inf."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 8 or len(chunk) == 0:
            raise SafetyViolation(f"bad chunk shape {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise SafetyViolation("non-finite (NaN/inf) actions in chunk")
        ref = np.asarray(reference8, dtype=np.float64)
        first_jump = float(np.max(np.abs(chunk[0, :7] - ref[:7]))) if ref.shape == (8,) and np.isfinite(ref).all() else float("nan")
        max_step = float(np.abs(np.diff(chunk[:, :7], axis=0)).max()) if len(chunk) > 1 else 0.0
        ood = np.where((chunk < NORM_STATE_Q01 - 0.1) | (chunk > NORM_STATE_Q99 + 0.1))[1]
        return chunk.copy(), ChunkReport(first_jump=first_jump, max_step=max_step,
                                         out_of_distribution_joints=sorted(set(int(x) for x in ood)))
