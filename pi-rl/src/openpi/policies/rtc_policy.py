"""Explicit Lingdong RTC policy; ordinary Policy never imports RTC code."""

import dataclasses
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import rtc
from openpi.policies import policy
from openpi.shared import nnx_utils


class RTCPolicy(policy.Policy):
    """Stateless guided requests, with independent ordinary sampling for cold starts.

    ``obs['rtc']`` contains physical ``previous_actions[T, 8]`` and an integer
    ``inference_delay``. The prefix is normalized through the same input pipeline
    as training targets. No prefix is kept on the server or shared between clients.
    """

    def __init__(self, model, *, rtc_config=None, **kwargs):
        if kwargs.get("is_pytorch", False) or not isinstance(model, pi0.Pi0):
            raise ValueError("RTC currently supports JAX Pi0/Pi05 checkpoints only")
        metadata = kwargs.get("metadata") or {}
        if metadata.get("action_representation") != "recorded_joint_targets" or metadata.get("action_dim") != 8:
            raise ValueError("RTC currently supports the Lingdong 8-D recorded-joint-target policy only")
        super().__init__(model, **kwargs)
        self._rtc_config = rtc_config or rtc.RTCConfig()
        self._rtc_horizon = model.action_horizon
        self._rtc_model_dim = model.action_dim
        self._rtc_sampler = nnx_utils.module_jit(rtc.RTCSampler(model, self._rtc_config, action_dim=8).sample_actions)
        self._metadata = {
            **self._metadata,
            "inference_mode": "rtc",
            "action_horizon": self._rtc_horizon,
            "rtc": {"method": "guided", **dataclasses.asdict(self._rtc_config)},
        }
        unknown = set(self._sample_kwargs) - {"num_steps"}
        if unknown:
            raise ValueError(f"Unsupported RTC sample options: {sorted(unknown)}")
        steps = self._sample_kwargs.get("num_steps", 10)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("RTC num_steps must be a positive integer")

    def infer(self, obs, *, noise=None):
        raw = dict(obs)
        request = raw.pop("rtc", None)
        if request is None:
            result = super().infer(raw, noise=noise)
            result["rtc"] = {"guided": False, "inference_delay": 0, "overlap_steps": 0}
            return result
        if not isinstance(request, dict):
            raise ValueError("rtc must be an object with previous_actions and inference_delay")
        delay = request.get("inference_delay")
        if isinstance(delay, bool | np.bool_) or not isinstance(delay, int | np.integer):
            raise ValueError("RTC inference_delay must be an integer")
        previous = np.asarray(request.get("previous_actions"), dtype=np.float32)
        if previous.ndim != 2 or previous.shape[1] != 8 or not 0 < len(previous) <= self._rtc_horizon:
            raise ValueError(f"RTC previous_actions must have shape [1..{self._rtc_horizon}, 8]")
        if not np.isfinite(previous).all():
            raise ValueError("RTC previous_actions must be finite")
        if not 0 <= delay <= len(previous) or delay >= self._rtc_horizon:
            raise ValueError("RTC inference_delay must fit the available overlap and be smaller than the horizon")

        # Holding the last value avoids treating a raw zero as a normalized zero.
        raw["action"] = np.pad(previous, ((0, self._rtc_horizon - len(previous)), (0, 0)), mode="edge")
        inputs = self._input_transform(jax.tree.map(lambda x: x, raw))
        prefix = np.asarray(inputs.pop("actions"), dtype=np.float32)
        if prefix.shape != (self._rtc_horizon, self._rtc_model_dim) or not np.isfinite(prefix).all():
            raise ValueError("RTC input transforms must normalize and pad action prefixes to the model shape")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self._rng, key = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.ndim == 2:
                noise = noise[None, ...]
            if noise.shape != (1, self._rtc_horizon, self._rtc_model_dim) or not np.isfinite(noise).all():
                raise ValueError("RTC noise must be finite and match the model action shape")
            sample_kwargs["noise"] = jnp.asarray(noise)
        started = time.monotonic()
        actions = self._rtc_sampler(
            key,
            model_lib.Observation.from_dict(inputs),
            previous_actions=jnp.asarray(prefix)[None, ...],
            inference_delay=jnp.asarray(delay, dtype=jnp.int32),
            overlap_steps=jnp.asarray(len(previous), dtype=jnp.int32),
            **sample_kwargs,
        )
        actions = np.asarray(actions[0])  # Synchronize before recording inference time.
        infer_ms = (time.monotonic() - started) * 1000
        if not np.isfinite(actions).all() or not math.isfinite(infer_ms):
            raise RuntimeError("RTC sampling produced nonfinite actions")
        result = self._output_transform({"state": np.asarray(inputs["state"][0]), "actions": actions})
        result["policy_timing"] = {"infer_ms": infer_ms}
        result["rtc"] = {"guided": True, "inference_delay": int(delay), "overlap_steps": len(previous)}
        return result
