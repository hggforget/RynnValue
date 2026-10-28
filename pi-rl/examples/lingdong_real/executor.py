"""Streams the policy's right-arm targets to the Tron2 VERBATIM (raw mode), holding left arm / head / left gripper.

One `request_servoj` per model action step at control_hz (30 Hz) -- no interpolation, no ramp, no clipping,
no rate limit, no tracking / staleness checks. Gripper: target * 100 -> right_opening (the link's protocol
rounds it to an int percent); repeated identical percents are not re-sent.
Only used with --execute (real link, armed) or --dry-run-execution (FakeLink).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from common import CONTROL_HZ, Q16_HEAD, Q16_LEFT, Q16_RIGHT
from safety import SafetyViolation

log = logging.getLogger("executor")


class RobotExecutor:
    def __init__(self, link, *, control_hz: float = CONTROL_HZ, gripper_speed: float = 100.0,
                 gripper_force: float = 50.0, servo_mode: int | None = None, on_target=None):
        if not (0 < control_hz <= 50):
            raise ValueError("control_hz must be in (0, 50]")
        self.link = link
        self.control_hz = control_hz
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self.servo_mode = servo_mode
        self.on_target = on_target
        self.held_left = None
        self.held_head = None
        self.held_left_grip = None
        self.last_target8 = None  # last commanded right-arm target (7 joints + gripper)
        self._last_grip_cmd = None
        self.started = False
        self.timing_log: list[dict] = []
        self.sent_targets: list[np.ndarray] = []  # every right8 target actually sent (for verification)

    def _q16(self, right7):
        q = np.empty(16)
        q[Q16_LEFT] = self.held_left
        q[Q16_RIGHT] = right7
        q[Q16_HEAD] = self.held_head
        return q

    def _send_gripper(self, g: float):
        pct = float(g) * 100.0
        if self._last_grip_cmd is None or round(pct) != self._last_grip_cmd:
            self.link.set_gripper(self.held_left_grip, pct, self.gripper_speed, self.gripper_force)
            self._last_grip_cmd = round(pct)

    def start(self):
        """Snapshot the joints that must be held. Sends nothing except an optional servo-mode request."""
        st = self.link.get_state(max_age=float("inf"))
        q = st["q16"]
        self.held_left = q[Q16_LEFT].copy()
        self.held_head = q[Q16_HEAD].copy()
        self.held_left_grip = float(st["left_opening"])
        self.last_target8 = np.concatenate([q[Q16_RIGHT], [st["right_opening"] / 100.0]])
        if self.servo_mode is not None:
            self.link.set_servo_mode(self.servo_mode)
            time.sleep(0.1)
        self.started = True
        log.info("executor started (RAW); holding left arm %s head %s left gripper %.0f",
                 np.round(self.held_left, 3).tolist(), np.round(self.held_head, 3).tolist(), self.held_left_grip)
        return self.last_target8.copy()

    def execute(self, targets: np.ndarray) -> dict:
        """Send each target verbatim, one servoj per step at control_hz. Returns timing stats."""
        if not self.started:
            raise RuntimeError("call start() first")
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 8:
            raise SafetyViolation(f"bad targets shape {targets.shape}")
        t0 = time.monotonic()
        late = []
        for i, tgt in enumerate(targets):
            if not np.isfinite(tgt).all():
                raise SafetyViolation(f"non-finite target at step {i}: {tgt}")
            self.link.servoj(self._q16(tgt[:7]))
            self._send_gripper(tgt[7])
            self.last_target8 = tgt.copy()
            self.sent_targets.append(tgt.copy())
            if self.on_target is not None:
                try:
                    self.on_target(tgt, time.monotonic())
                except Exception as exc:
                    log.warning("target visualization disabled: %s", exc)
                    self.on_target = None
            late.append(time.monotonic() - (t0 + i / self.control_hz))
            _sleep_until(t0 + (i + 1) / self.control_hz)
        stats = {"exec_s": time.monotonic() - t0, "n_steps": len(targets),
                 "max_send_late_ms": 1e3 * float(np.max(late)) if late else 0.0}
        self.timing_log.append(stats)
        return stats

    def hold(self, repeats: int = 3):
        """Stop streaming new targets; re-send the last commanded target so the arm stays put."""
        if not self.started or self.last_target8 is None:
            return
        try:
            for _ in range(repeats):
                self.link.servoj(self._q16(self.last_target8[:7]))
                time.sleep(1.0 / self.control_hz)
            log.warning("HOLD: right arm held at last target %s", np.round(self.last_target8, 3).tolist())
        except Exception as e:  # noqa: BLE001
            log.error("hold failed: %s  -> use the hardware e-stop if the arm is moving", e)


def _sleep_until(t: float):
    d = t - time.monotonic()
    if d > 0:
        time.sleep(d)
