import dataclasses

import jax
import numpy as np
import pytest

from openpi.training import config

from . import train_iql


def batch():
    return {
        "image": {
            "base_0_rgb": np.full((2, 32, 32, 3), 60, dtype=np.uint8),
            "right_wrist_0_rgb": np.full((2, 32, 32, 3), 120, dtype=np.uint8),
        },
        "image_mask": {"base_0_rgb": np.ones(2, dtype=bool), "right_wrist_0_rgb": np.ones(2, dtype=bool)},
        "state": np.zeros((2, 32), dtype=np.float32),
        "actions": np.zeros((2, 3, 32), dtype=np.float32),
        "tokenized_prompt": np.zeros((2, 20), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((2, 20), dtype=bool),
        "next_image_base": np.full((2, 32, 32, 3), 80, dtype=np.uint8),
        "next_image_wrist": np.full((2, 32, 32, 3), 180, dtype=np.uint8),
        "iql_state": np.array([[0.1] * 8 + [1, 0], [0.2] * 8 + [0, 1]], dtype=np.float32),
        "next_iql_state": np.array([[0.3] * 8 + [1, 0], [0.4] * 8 + [0, 1]], dtype=np.float32),
        "chunk_reward": np.array([0.2, -0.3], dtype=np.float32),
        "reward": np.array([99, 99], dtype=np.float32),
        "mask": np.array([1, 0], dtype=np.float32),
    }


def kwargs():
    preset = config.get_config("pi05_lingdong_right_arm_crop_iql")
    return {
        "image_keys": preset.iql.image_keys,
        "next_image_keys": preset.iql.next_image_keys,
        "proprio_key": preset.iql.proprio_key,
        "next_proprio_key": preset.iql.next_proprio_key,
        "action_dim": preset.iql.action_dim,
    }


def test_critic_receives_physical_actions_proprioception_and_task_before_policy_image_conversion():
    raw = batch()
    _, actions, critic = train_iql.split_batch(raw, **kwargs(), step_reward_discount=0.99)
    assert actions.shape == (2, 3, 32)
    assert critic["actions"].shape == (2, 3, 8)
    assert critic["observations"]["pixels"].shape == (2, 32, 32, 6, 1)
    assert critic["observations"]["pixels"].dtype == np.uint8
    assert (critic["observations"]["pixels"][..., :3, :] == 60).all()
    np.testing.assert_array_equal(critic["observations"]["proprio"], raw["iql_state"])
    np.testing.assert_array_equal(critic["next_observations"]["proprio"], raw["next_iql_state"])
    np.testing.assert_array_equal(critic["rewards"], raw["chunk_reward"])


def test_critic_initialization_and_training_use_identical_observation_contracts():
    raw = batch()
    view = train_iql._build_iql_view(raw, **kwargs())  # noqa: SLF001
    keys = kwargs()
    observations, actions = train_iql.host_sample_for_iql_init(
        raw,
        image_keys=keys["image_keys"],
        proprio_key=keys["proprio_key"],
        action_dim=keys["action_dim"],
    )
    assert set(observations) == set(view["observations"]) == {"pixels", "proprio"}
    for key in observations:
        np.testing.assert_array_equal(observations[key], view["observations"][key][:1])
    np.testing.assert_array_equal(actions, view["actions"][:1])


def test_legacy_iql_preserves_pixel_only_and_padded_action_behavior():
    keys = kwargs()
    view = train_iql._build_iql_view(  # noqa: SLF001
        batch(),
        image_keys=keys["image_keys"],
        next_image_keys=keys["next_image_keys"],
    )
    assert set(view["observations"]) == {"pixels"}
    assert view["actions"].shape == (2, 3, 32)


@pytest.mark.manual
def test_small_critic_updates_advantages_and_restores_conditioned_parameters():
    if jax.default_backend() != "cpu":
        pytest.skip("Run this smoke test with JAX_PLATFORMS=cpu")
    raw = batch()
    keys = kwargs()
    preset = config.get_config("pi05_lingdong_right_arm_crop_iql")
    preset = dataclasses.replace(
        preset,
        model=dataclasses.replace(preset.model, action_horizon=3),
        iql=dataclasses.replace(
            preset.iql,
            encoder_type="small",
            hidden_dims=(16, 16),
            cnn_features=(8, 8),
            cnn_strides=(2, 2),
            latent_dim=8,
        ),
    )
    observations, actions = train_iql.host_sample_for_iql_init(
        raw,
        image_keys=keys["image_keys"],
        proprio_key=keys["proprio_key"],
        action_dim=keys["action_dim"],
    )
    agent = train_iql.init_pi_iql(preset, observations, actions)
    view = train_iql._build_iql_view(raw, **keys)  # noqa: SLF001
    before = agent.iql_save_pytree()
    info = agent.update(view)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(info))
    assert agent.bellman_discount == pytest.approx(0.99**3)
    _, weights, _, qs, values, advantages = agent.compute_advs(view)
    assert weights.shape == values.shape == advantages.shape == (2,)
    assert qs.shape == (2, 2)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert (weights <= 100).all()
    after = agent.iql_save_pytree()
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True))
    agent.iql_load_pytree(before)
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(agent.iql_save_pytree()), strict=True):
        np.testing.assert_array_equal(expected, actual)
