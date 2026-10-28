import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import rtc
from openpi.shared import nnx_utils


class _TinyLLM(nnx.Module):
    def __call__(self, embedded, *, kv_cache=None, **kwargs):
        del kwargs
        prefix, suffix = embedded
        if suffix is None:
            return (prefix, None), prefix * 0.02
        return (None, jnp.tanh(suffix) * 0.2 + kv_cache), kv_cache


class TinyPi0(pi0.Pi0):
    """Exercise the unmodified Pi0 sampler with a tiny deterministic action expert."""

    def __init__(self):
        self.action_dim = 32
        self.action_horizon = 6
        self.PaliGemma = nnx.Dict(llm=_TinyLLM())
        self.action_out_proj = nnx.Linear(32, 32, rngs=nnx.Rngs(0))

    def embed_prefix(self, obs):
        return obs.state[:, None, :], jnp.ones((len(obs.state), 1), dtype=bool), jnp.array([False])

    def embed_suffix(self, obs, noisy_actions, timestep):
        del obs, timestep
        return (
            noisy_actions,
            jnp.ones(noisy_actions.shape[:2], dtype=bool),
            jnp.array([True] + [False] * (self.action_horizon - 1)),
            None,
        )


def tiny_observation():
    return model_lib.Observation(images={}, image_masks={}, state=jnp.ones((1, 32)))


def test_prefix_weights_preserve_committed_steps_and_ignore_unavailable_tail():
    weights = rtc.prefix_weights(10, 2, 6, "linear")
    np.testing.assert_allclose(weights, [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0])
    exponential = np.asarray(rtc.prefix_weights(10, 2, 6, "exp"))
    np.testing.assert_array_equal(exponential[:2], 1)
    assert np.all(exponential[2:6] < np.asarray(weights)[2:6])
    np.testing.assert_array_equal(rtc.prefix_weights(10, 2, 6, "hard"), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("time", [1.0, 0.5, 0.001])
def test_guidance_has_correct_noise_at_one_sign_and_finite_clipped_gain(time):
    x = jnp.zeros((1, 4, 3))
    target = jnp.ones_like(x)
    weights = jnp.array([1, 1, 0, 0])[None, :, None] * jnp.array([1, 1, 0])[None, None, :]
    velocity = rtc.guided_velocity(lambda value, time: jnp.zeros_like(value), x, time, target, weights, 2.0)
    np.testing.assert_array_equal(velocity, -2 * weights)
    updated = x - 0.1 * velocity
    assert np.linalg.norm((updated - target) * weights) < np.linalg.norm((x - target) * weights)
    assert np.isfinite(velocity).all()


@pytest.mark.parametrize(("guidance", "overlap"), [(0.0, 4), (10.0, 0)])
def test_disabled_guidance_matches_original_pi0_sampler_and_preserves_parameters(guidance, overlap):
    model = TinyPi0()
    sampler = rtc.RTCSampler(model, rtc.RTCConfig(guidance), action_dim=8)
    before = jax.tree.map(np.asarray, nnx.state(model))
    noise = jax.random.normal(jax.random.key(1), (1, 6, 32))
    ordinary = model.sample_actions(jax.random.key(2), tiny_observation(), noise=noise, num_steps=3)
    output = sampler.sample_actions(
        jax.random.key(2),
        tiny_observation(),
        previous_actions=jnp.ones_like(noise),
        inference_delay=jnp.array(0),
        overlap_steps=jnp.array(overlap),
        noise=noise,
        num_steps=3,
    )
    np.testing.assert_array_equal(ordinary, output)
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(model)), strict=True):
        np.testing.assert_array_equal(old, new)


def test_guided_sampling_jits_with_dynamic_delay_and_overlap():
    sampler = rtc.RTCSampler(TinyPi0(), rtc.RTCConfig(1.0), action_dim=8)
    sample = nnx_utils.module_jit(sampler.sample_actions)
    results = []
    for delay, overlap in [(1, 4), (2, 3)]:
        result = sample(
            jax.random.key(1),
            tiny_observation(),
            previous_actions=jnp.ones((1, 6, 32)),
            inference_delay=jnp.array(delay),
            overlap_steps=jnp.array(overlap),
            num_steps=3,
        )
        assert result.shape == (1, 6, 32)
        assert np.isfinite(result).all()
        results.append(result)
    assert not np.allclose(*results)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_guidance_strength_is_rejected(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rtc.RTCConfig(weight)


def test_guidance_runs_through_real_bfloat16_gemma_bridge_and_pi05_suffix(monkeypatch):
    # Retain real attention, adaRMS, KV caching, and the Linen/NNX bridge, while
    # shrinking dimensions and vocabulary so the test needs neither GPU nor weights.
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 64)
    config = gemma.Config(width=32, depth=1, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16)

    class TinyGemmaPi0(TinyPi0):
        def __init__(self):
            super().__init__()
            self.pi05 = True
            llm = nnx_bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="bfloat16", adarms=True))
            llm.lazy_init(rngs=nnx.Rngs(1), method="init", use_adarms=[False, True])
            self.PaliGemma = nnx.Dict(llm=llm)
            self.action_in_proj = nnx.Linear(32, 32, rngs=nnx.Rngs(2))
            self.time_mlp_in = nnx.Linear(32, 32, rngs=nnx.Rngs(3))
            self.time_mlp_out = nnx.Linear(32, 32, rngs=nnx.Rngs(4))

        def embed_suffix(self, obs, noisy_actions, timestep):
            return pi0.Pi0.embed_suffix(self, obs, noisy_actions, timestep)

    model = TinyGemmaPi0()
    before = jax.tree.map(np.asarray, nnx.state(model))
    sampler = nnx_utils.module_jit(rtc.RTCSampler(model, rtc.RTCConfig(1.0), action_dim=8).sample_actions)
    output = sampler(
        jax.random.key(1),
        tiny_observation(),
        previous_actions=jnp.ones((1, 6, 32)),
        inference_delay=jnp.array(2),
        overlap_steps=jnp.array(4),
        num_steps=2,
    )
    assert output.shape == (1, 6, 32)
    assert np.isfinite(output).all()
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(model)), strict=True):
        np.testing.assert_array_equal(old, new)
