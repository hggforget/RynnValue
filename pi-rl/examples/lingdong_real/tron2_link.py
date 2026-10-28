"""Minimal Tron2 WebSocket link (protocol from tron2_old/robot_utils.py) with command gating.

The link is READ-ONLY unless `arm_commands()` is called, which main.py only does after `--execute`
plus a typed confirmation. In read-only mode any non-whitelisted request raises before it is sent.
It never moves the robot to an init pose and never calls reset (unlike tron2_old Tron2/Tron2Env).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid

import numpy as np

log = logging.getLogger("tron2_link")

READ_ONLY_TITLES = frozenset({"request_get_joint_state", "request_get_limx_2fclaw_state"})
COMMAND_TITLES = frozenset({"request_servoj", "request_set_limx_2fclaw_cmd", "request_set_servo_mode"})
_ARM_TOKEN = "I_UNDERSTAND_THE_ROBOT_WILL_MOVE"


class CommandsNotArmed(RuntimeError):
    pass


class StaleState(RuntimeError):
    pass


class Tron2Link:
    def __init__(self, robot_ip: str = "10.192.1.2", port: int = 5000, poll_hz: float = 100.0,
                 open_timeout: float = 5.0):
        from websockets.sync.client import connect  # websockets>=12 (present in pi-rl and old venv)

        self.uri = f"ws://{robot_ip}:{port}"
        try:  # websockets>=15 honours *_PROXY env vars by default; the robot is on the LAN.
            self._ws = connect(self.uri, open_timeout=open_timeout, proxy=None, compression=None, max_size=None)
        except TypeError:
            self._ws = connect(self.uri, open_timeout=open_timeout, compression=None, max_size=None)
        self._armed = False
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.accid = None
        self.robot_info = None
        self._q = None
        self._q_robot_ts = None
        self._q_t = 0.0
        self._grip = None
        self._grip_t = 0.0
        self.sent_counts: dict[str, int] = {}
        self._rx = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx.start()
        self._poll_hz = poll_hz
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    # ---------------------------------------------------------------- transport
    def _send(self, title: str, data: dict | None = None) -> None:
        if title not in READ_ONLY_TITLES:
            if title not in COMMAND_TITLES:
                raise CommandsNotArmed(f"title {title!r} is not allowed by this client")
            if not self._armed:
                raise CommandsNotArmed(f"refusing to send {title!r}: link is read-only (dry-run)")
        msg = {"accid": self.accid, "title": title, "timestamp": int(time.time() * 1000),
               "guid": str(uuid.uuid4()), "data": data or {}}
        with self._send_lock:
            self._ws.send(json.dumps(msg))
        self.sent_counts[title] = self.sent_counts.get(title, 0) + 1

    def _recv_loop(self):
        try:
            for raw in self._ws:
                try:
                    root = json.loads(raw)
                except Exception:
                    continue
                title = root.get("title", "")
                self.accid = root.get("accid", self.accid)
                data = root.get("data", {}) or {}
                now = time.monotonic()
                if title == "response_get_joint_state":
                    q = data.get("q", [])
                    if len(q) >= 16:
                        with self._lock:
                            self._q = np.asarray(q[:16], dtype=np.float64)
                            self._q_robot_ts = root.get("timestamp")
                            self._q_t = now
                elif title == "response_get_limx_2fclaw_state":
                    with self._lock:
                        self._grip = (float(data.get("left_opening", -1)), float(data.get("right_opening", -1)))
                        self._grip_t = now
                elif title == "notify_robot_info":
                    self.robot_info = data
        except Exception as e:  # connection closed
            if not self._stop.is_set():
                log.error("robot websocket receive loop ended: %s", e)

    def _poll_loop(self):
        period = 1.0 / self._poll_hz
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                self._send("request_get_joint_state")
                self._send("request_get_limx_2fclaw_state")
            except Exception as e:
                if not self._stop.is_set():
                    log.error("poll send failed: %s", e)
                    time.sleep(0.5)
            time.sleep(max(0.0, period - (time.monotonic() - t)))

    # ---------------------------------------------------------------- state
    def wait_ready(self, timeout: float = 5.0) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._q is not None and self._grip is not None:
                    return
            time.sleep(0.01)
        raise StaleState(f"no joint/gripper state from {self.uri} within {timeout}s")

    def get_state(self, max_age: float = 0.2) -> dict:
        now = time.monotonic()
        with self._lock:
            if self._q is None or self._grip is None:
                raise StaleState("no state yet")
            q, grip, qa, ga, ts = self._q.copy(), self._grip, now - self._q_t, now - self._grip_t, self._q_robot_ts
        if qa > max_age or ga > max_age:
            raise StaleState(f"state too old (joint {qa*1e3:.0f} ms, gripper {ga*1e3:.0f} ms)")
        if grip[0] < 0 or grip[1] < 0:
            raise StaleState(f"invalid gripper state {grip}")
        return {"q16": q, "left_opening": grip[0], "right_opening": grip[1], "joint_age_s": qa,
                "gripper_age_s": ga, "robot_ts_ms": ts}

    # ---------------------------------------------------------------- commands (gated)
    def arm_commands(self, token: str) -> None:
        if token != _ARM_TOKEN:
            raise CommandsNotArmed("bad arm token")
        self._armed = True
        log.warning("Tron2 link ARMED: motion commands enabled")

    def disarm(self) -> None:
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    def set_servo_mode(self, mode: int) -> None:
        self._send("request_set_servo_mode", {"mode": int(mode)})

    def servoj(self, q16) -> None:
        q16 = [float(x) for x in q16]
        if len(q16) != 16 or not np.isfinite(q16).all():
            raise ValueError("servoj needs 16 finite values")
        self._send("request_servoj", {"q": q16, "filter_ratio": 1.0})

    def set_gripper(self, left_opening: float, right_opening: float, speed: float = 100, force: float = 50) -> None:
        c = lambda v: int(np.clip(round(v), 0, 100))  # noqa: E731
        self._send("request_set_limx_2fclaw_cmd", {
            "left_opening": c(left_opening), "left_speed": c(speed), "left_force": c(force),
            "right_opening": c(right_opening), "right_speed": c(speed), "right_force": c(force)})

    def close(self) -> None:
        self._armed = False
        self._stop.set()
        try:
            self._ws.close()
        except Exception:
            pass


class FakeLink:
    """In-memory stand-in for dry-run execution tests: records commands, follows them perfectly."""

    def __init__(self, q16=None, right_opening: float = 0.0, left_opening: float = 0.0, lag_steps: int = 0):
        self._q = np.zeros(16) if q16 is None else np.asarray(q16, dtype=np.float64).copy()
        self._grip = [float(left_opening), float(right_opening)]
        self._armed = False
        self.commands: list[tuple[float, str, object]] = []
        self.sent_counts: dict[str, int] = {}
        self.accid = "FAKE"

    def wait_ready(self, timeout=5.0):
        return

    def get_state(self, max_age=0.2):
        return {"q16": self._q.copy(), "left_opening": self._grip[0], "right_opening": self._grip[1],
                "joint_age_s": 0.0, "gripper_age_s": 0.0, "robot_ts_ms": None}

    def set_state(self, q16=None, right_opening=None):
        if q16 is not None:
            self._q = np.asarray(q16, dtype=np.float64).copy()
        if right_opening is not None:
            self._grip[1] = float(right_opening)

    def arm_commands(self, token):
        if token != _ARM_TOKEN:
            raise CommandsNotArmed("bad arm token")
        self._armed = True

    def disarm(self):
        self._armed = False

    @property
    def armed(self):
        return self._armed

    def _rec(self, title, data):
        if not self._armed:
            raise CommandsNotArmed(f"refusing to send {title!r}: link is read-only (dry-run)")
        self.commands.append((time.monotonic(), title, data))
        self.sent_counts[title] = self.sent_counts.get(title, 0) + 1

    def set_servo_mode(self, mode):
        self._rec("request_set_servo_mode", mode)

    def servoj(self, q16):
        self._rec("request_servoj", np.asarray(q16, dtype=np.float64).copy())
        self._q = np.asarray(q16, dtype=np.float64).copy()

    def set_gripper(self, left_opening, right_opening, speed=100, force=50):
        self._rec("request_set_limx_2fclaw_cmd", (left_opening, right_opening))
        self._grip = [float(left_opening), float(right_opening)]

    def close(self):
        self._armed = False
