"""Inference-time RTC for JAX Pi0/Pi05; the original sampler is unchanged.

Reference: https://arxiv.org/abs/2506.07339. This module uses OpenPI's
noise-at-one convention, so guidance is subtracted from the velocity.
"""

import dataclasses
import math
from typing import Literal

import einops
from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models import pi0


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    max_guidance_weight: float = 10.0
    prefix_schedule: Literal["exp", "linear", "hard"] = "exp"

    def __post_init__(self):
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight < 0:
            raise ValueError("RTC max_guidance_weight must be finite and nonnegative")
        if self.prefix_schedule not in ("exp", "linear", "hard"):
            raise ValueError(f"Unknown RTC prefix schedule: {self.prefix_schedule}")


def prefix_weights(horizon, delay, overlap, schedule):
    """Unit weight on committed actions, taper on editable overlap, zero outside it."""
    delay = jnp.minimum(delay, overlap)
    positions = jnp.arange(horizon)
    if schedule == "hard":
        weights = (positions < delay).astype(jnp.float32)
    else:
        weights = jnp.clip((overlap - positions) / (overlap - delay + 1), 0.0, 1.0)
        if schedule == "exp":
            weights *= jnp.expm1(weights) / jnp.expm1(1.0)
    return jnp.where(positions < overlap, weights, 0.0)


def guided_velocity(velocity_fn, x, time, previous_actions, weights, max_guidance_weight):
    """One denoising forward/VJP, differentiating actions only (never a full Jacobian)."""

    def denoise(value):
        velocity = velocity_fn(value, time)
        return value - time * velocity, velocity

    clean, pullback, velocity = jax.vjp(denoise, x, has_aux=True)
    correction = pullback((previous_actions - clean) * weights)[0]
    gain = jnp.minimum(
        max_guidance_weight,
        (time**2 + (1 - time) ** 2) / jnp.maximum(time * (1 - time), 1e-6),
    )
    return velocity - gain * correction


class RTCSampler(nnx.Module):
    """A separate sampler sharing an existing model's read-only parameters."""

    def __init__(self, model: pi0.Pi0, config: RTCConfig, *, action_dim: int):
        if not 0 < action_dim <= model.action_dim:
            raise ValueError("RTC physical action dimension must fit the model")
        self.model = model
        self.config = config
        self.action_dim = action_dim

    def sample_actions(
        self, rng, observation, *, previous_actions, inference_delay, overlap_steps, num_steps=10, noise=None
    ):
        if self.config.max_guidance_weight == 0:
            return self.model.sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        model = self.model
        observation = model_lib.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        shape = (batch_size, model.action_horizon, model.action_dim)
        if previous_actions.shape != shape:
            raise ValueError(f"Expected RTC previous_actions {shape}, got {previous_actions.shape}")
        if noise is None:
            noise = jax.random.normal(rng, shape)
        elif noise.shape != shape:
            raise ValueError(f"Expected RTC noise {shape}, got {noise.shape}")

        # Images and language are encoded once, outside the differentiated velocity function.
        tokens, prefix_mask, ar_mask = model.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, cache = model.PaliGemma.llm(
            [tokens, None], mask=pi0.make_attn_mask(prefix_mask, ar_mask), positions=positions
        )

        def velocity_fn(x, time):
            suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(
                observation, x, jnp.broadcast_to(time, (batch_size,))
            )
            prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1])
            mask = jnp.concatenate([prefix_attention, pi0.make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, out), _ = model.PaliGemma.llm(
                [None, suffix], mask=mask, positions=positions, kv_cache=cache, adarms_cond=[None, cond]
            )
            return model.action_out_proj(out[:, -model.action_horizon :])

        temporal = prefix_weights(model.action_horizon, inference_delay, overlap_steps, self.config.prefix_schedule)
        physical = jnp.arange(model.action_dim) < self.action_dim
        weights = temporal[None, :, None] * physical[None, None, :]
        dt = -1.0 / num_steps

        def step(carry):
            x, time = carry
            velocity = guided_velocity(velocity_fn, x, time, previous_actions, weights, self.config.max_guidance_weight)
            return x + dt * velocity, time + dt

        result, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))
        return result
