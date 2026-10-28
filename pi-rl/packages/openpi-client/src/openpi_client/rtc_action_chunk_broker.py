"""An opt-in asynchronous RTC executor, independent of ActionChunkBroker.

Call ``warmup(obs)`` while the robot is stationary, then call ``infer(obs)`` once
per policy control tick. Inference runs in one serialized worker. Only the
control thread merges chunks; reset generations invalidate outstanding results.
"""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
import math
import threading
import time

import numpy as np

from openpi_client import base_policy


class RTCExecutionError(RuntimeError):
    """Execution must stop and the broker must be reset before restarting."""


@dataclasses.dataclass
class _Pending:
    future: Future
    start: int
    delay: int
    generation: int
    overlap: int


class RTCActionChunkBroker(base_policy.BasePolicy):
    def __init__(self, policy, *, execution_horizon=10, inference_delay=3):
        metadata_fn = getattr(policy, "get_server_metadata", None)
        metadata = metadata_fn() if callable(metadata_fn) else getattr(policy, "metadata", {})
        if metadata.get("inference_mode") != "rtc":
            raise ValueError("RTCActionChunkBroker requires a server explicitly advertising inference_mode='rtc'")
        if metadata.get("action_dim") != 8 or metadata.get("action_representation") != "recorded_joint_targets":
            raise ValueError("RTCActionChunkBroker currently supports Lingdong 8-D joint targets")
        self._horizon = int(metadata["action_horizon"])
        self._fps = float(metadata["fps"])
        if not math.isfinite(self._fps) or self._fps <= 0:
            raise ValueError("RTC policy fps must be positive")
        for value in (execution_horizon, inference_delay):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("RTC horizons and delays must be integers")
        if not 0 < execution_horizon < self._horizon or inference_delay < 0:
            raise ValueError("RTC execution_horizon must fit the prediction horizon; delay must be nonnegative")
        self._policy = policy
        self._execution_horizon = execution_horizon
        self._initial_delay = inference_delay
        self._delay = inference_delay
        self._check_capacity(inference_delay)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-inference")
        self._lock = threading.RLock()
        self._generation = 0
        self._step = 0
        self._chunk = None
        self._chunk_start = 0
        self._last_request = 0
        self._pending = None
        self._warmed_up = False
        self._warming = False
        self._closed = False
        self._failed = False
        self._task = None

    def _check_capacity(self, delay):
        if delay > self._execution_horizon or self._execution_horizon + delay > self._horizon:
            raise RTCExecutionError(
                f"RTC delay {delay} exceeds queue capacity: require delay <= execution_horizon "
                f"<= action_horizon - delay (H={self._horizon}, s={self._execution_horizon}). "
                "Adjust the execution horizon or reduce inference latency."
            )

    def _check_open(self):
        if self._closed:
            raise RTCExecutionError("RTC broker is closed")
        if self._failed:
            raise RTCExecutionError("RTC broker failed; reset before restarting execution")

    def _rpc(self, obs):
        started = time.monotonic()
        result = self._policy.infer(obs)
        duration = time.monotonic() - started
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (self._horizon, 8) or not np.isfinite(actions).all():
            raise RTCExecutionError("RTC server returned an invalid action chunk")
        guided = result.get("rtc", {}).get("guided")
        if guided is not ("rtc" in obs):
            raise RTCExecutionError("Server did not acknowledge the requested RTC inference mode")
        if "rtc" in obs and result["rtc"].get("inference_delay") != obs["rtc"]["inference_delay"]:
            raise RTCExecutionError("Server acknowledged a different RTC prefix length")
        return actions.copy(), duration

    def warmup(self, obs):
        """Compile both paths and measure one warm RTC RPC before any motion."""
        with self._lock:
            self._check_open()
            if self._warming or self._chunk is not None or self._pending is not None:
                raise RTCExecutionError("Warm up RTC before execution, after reset")
            self._warming = True
            generation = self._generation
        try:
            raw = copy.deepcopy(obs)
            raw.pop("rtc", None)
            first, _ = self._executor.submit(self._rpc, raw).result()
            request = copy.deepcopy(raw)
            request["rtc"] = {
                "previous_actions": first[: self._horizon - self._execution_horizon],
                "inference_delay": self._initial_delay,
            }
            self._executor.submit(self._rpc, request).result()  # Compile guided sampling.
            _, duration = self._executor.submit(self._rpc, request).result()  # Warm latency only.
            delay = max(self._initial_delay, math.ceil(duration * self._fps))
            self._check_capacity(delay)
            with self._lock:
                self._check_open()
                if generation != self._generation:
                    raise RTCExecutionError("RTC was reset during warmup; use a fresh observation")
                self._delay = delay
                self._warmed_up = True
            return {"rpc_ms": duration * 1000, "inference_delay": delay}
        finally:
            with self._lock:
                self._warming = False

    @staticmethod
    def _prompt(obs):
        value = obs.get("prompt")
        if value is None:
            return None
        value = np.asarray(value).item()
        return value.decode() if isinstance(value, bytes) else str(value)

    def _remaining(self):
        if self._chunk is None:
            return np.empty((0, 8), dtype=np.float32)
        return self._chunk[self._step - self._chunk_start :]

    def _merge_completed(self):
        pending = self._pending
        if pending is None or not pending.future.done():
            return
        self._pending = None
        if pending.generation != self._generation:
            return
        actions, duration = pending.future.result()
        elapsed = self._step - pending.start
        delay = max(self._delay, elapsed, math.ceil(duration * self._fps))
        self._check_capacity(delay)
        remaining = self._remaining()
        # If the result arrives early, keep executing the whole committed prefix.
        keep = max(0, pending.start + pending.delay - self._step)
        offset = elapsed + keep
        if offset >= self._horizon or keep > len(remaining) or elapsed > pending.overlap:
            raise RTCExecutionError("RTC result no longer overlaps the executable queue")
        self._chunk = np.concatenate([remaining[:keep], actions[offset:]], axis=0)
        self._chunk_start = self._step
        self._delay = delay

    def infer(self, obs):
        """Return one action. Only the initial, fresh chunk blocks after warmup."""
        prompt = self._prompt(obs)
        with self._lock:
            self._check_open()
            if not self._warmed_up or self._warming:
                raise RTCExecutionError("Call warmup(obs) while stationary before RTC execution")
            if self._chunk is not None and prompt != self._task:
                self.reset()
            generation = self._generation
            initial = self._chunk is None
            if initial:
                raw = copy.deepcopy(obs)
                raw.pop("rtc", None)
                future = self._executor.submit(self._rpc, raw)
        try:
            if initial:
                actions, _ = future.result()
                with self._lock:
                    self._check_open()
                    if generation != self._generation:
                        raise RTCExecutionError("RTC reset during initial inference; use a fresh observation")
                    self._chunk = actions
                    self._chunk_start = self._step
                    self._task = prompt
            with self._lock:
                self._check_open()
                self._merge_completed()
                remaining = self._remaining()
                if len(remaining) == 0:
                    raise RTCExecutionError("RTC action queue exhausted before inference completed")
                if self._pending is None and self._step - self._last_request >= self._execution_horizon:
                    if len(remaining) < self._delay:
                        raise RTCExecutionError("RTC queue cannot cover the measured inference delay")
                    raw = copy.deepcopy(obs)
                    raw["rtc"] = {"previous_actions": remaining.copy(), "inference_delay": self._delay}
                    self._pending = _Pending(
                        self._executor.submit(self._rpc, raw), self._step, self._delay, self._generation, len(remaining)
                    )
                    self._last_request = self._step
                result = remaining[0].copy()
                self._step += 1
                return {"actions": result}
        except Exception as exc:
            with self._lock:
                if generation == self._generation:
                    self._failed = True
            if isinstance(exc, RTCExecutionError):
                raise
            raise RTCExecutionError("RTC inference failed; reset before restarting execution") from exc

    def reset(self):
        """Invalidate old chunks; serialize transport reset behind outstanding RPCs."""
        with self._lock:
            if self._closed:
                return
            self._generation += 1
            if self._pending is not None:
                self._pending.future.cancel()
            self._pending = None
            self._step = self._chunk_start = self._last_request = 0
            self._chunk = None
            self._task = None
            self._failed = False
            self._executor.submit(self._policy.reset)

    def close(self):
        """Stop execution and interrupt pending socket I/O when supported."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._chunk = None
            if self._pending is not None:
                self._pending.future.cancel()
        close = getattr(self._policy, "close", None)
        try:
            if callable(close):
                close()
        finally:
            self._executor.shutdown(wait=False)
