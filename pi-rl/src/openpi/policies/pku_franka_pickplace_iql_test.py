"""Reward, task conditioning and serving contracts for merged FR3 pick-and-place."""

import dataclasses
import json

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import pku_franka_collect_rl_policy as rl
from openpi.training import config


def transition(frame=0, task=8):
    return {
        "observation.images.primary": np.zeros((2, 3, 8, 12), dtype=np.float32),
        "observation.images.wrist": np.ones((2, 8, 12, 3), dtype=np.uint8),
        "observation.state.cartesian": np.arange(12, dtype=np.float32).reshape(2, 6),
        "observation.state.gripper": np.array([[1], [0]], dtype=np.float32),
        "observation.state.joints": np.arange(14, dtype=np.float32).reshape(2, 7),
        "observation.state": np.full((2, 20), np.nan),
        "observation.state.target": np.full((2, 6), np.nan),
        "action": np.arange(21, dtype=np.float32).reshape(3, 7),
        "progress": np.array([-5, -4, -4.5, -2], dtype=np.float32),
        "episode_index": 0,
        "frame_index": frame,
        "task_index": task,
        "prompt": "Put the red block on the blue plate.",
    }


def adapter(length=8):
    return rl.PkuFrankaCollectRLInputs(
        episode_lengths={0: length},
        discount=0.9,
        progress_reward_mode="delta",
        task_indices=(3, 8),
    )


def test_progress_increments_retain_regressions_and_original_action_labels():
    raw = transition()
    result = adapter()(raw)
    assert result["mask"] == 1
    # +1 second improvement, -0.5 regression, then +2.5 improvement.
    np.testing.assert_allclose(result["chunk_reward"], 1 - 0.9 * 0.5 + 0.9**2 * 2.5)
    np.testing.assert_array_equal(result["actions"], raw["action"])
    assert result["prompt"] == raw["prompt"]
    assert np.isfinite(result["state"]).all()  # Target-containing fields were ignored.


@pytest.mark.parametrize(
    ("frame", "progress", "reward"),
    [
        (4, [-5, -4, -4.5, -2], 1 - 0.9 * 0.5 + 0.9**2 * 2.5),  # Exactly reaches terminal.
        (6, [-5, -4, 100, 200], 1.0),  # Padded entries cannot contribute reward.
        (7, [-5, 100, 200, 300], 0.0),  # No transition remains at the last frame.
    ],
)
def test_terminal_chunks_preserve_endpoint_residual_and_ignore_padding(frame, progress, reward):
    raw = transition(frame)
    raw["progress"] = np.asarray(progress, dtype=np.float32)
    result = adapter()(raw)
    assert result["mask"] == 0
    np.testing.assert_allclose(result["chunk_reward"], reward)


def test_both_critic_views_distinguish_goals_and_keep_measured_state_units():
    transform = adapter()
    first = transform(transition(task=3))
    second = transform(transition(task=8))
    np.testing.assert_array_equal(first["iql_state"][-2:], [1, 0])
    np.testing.assert_array_equal(second["iql_state"][-2:], [0, 1])
    np.testing.assert_array_equal(second["next_iql_state"][-2:], [0, 1])
    np.testing.assert_array_equal(first["iql_state"][:14], first["state"])
    np.testing.assert_array_equal(first["next_iql_state"][:14], first["next_state"])
    normalized = transforms.Normalize({"state": transforms.NormStats(mean=np.ones(14), std=np.ones(14) * 2)})(first)
    np.testing.assert_array_equal(normalized["iql_state"], first["iql_state"])
    np.testing.assert_array_equal(normalized["next_iql_state"], first["next_iql_state"])
    with pytest.raises(ValueError, match="Unknown FR3 critic task"):
        transform(transition(task=99))


def test_serving_and_norm_scan_need_no_reward_task_or_episode_labels():
    raw = transition()
    for key in list(raw):
        if key.startswith("observation."):
            raw[key] = raw[key][0]
    for key in ("progress", "episode_index", "frame_index", "task_index"):
        del raw[key]
    norm = adapter()(raw)
    np.testing.assert_array_equal(norm["actions"], raw.pop("action"))
    serving = adapter()(raw)
    np.testing.assert_array_equal(serving["state"], norm["state"])
    assert "iql_state" not in serving
    assert "actions" not in serving


def test_preset_wires_tasks_rewards_and_sft_contract(tmp_path, monkeypatch):
    preset = config.get_config("pi05_pku_franka_pickplace_iql")
    sft = config.get_config("pi05_pku_franka_pickplace")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 30, "features": {"action": {}, "progress": {}}}))
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 8}) + "\n")
    (meta / "tasks.jsonl").write_text("\n".join(json.dumps({"task_index": i}) for i in (8, 3)))
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    factory = dataclasses.replace(preset.data, repo_id=str(tmp_path))
    data = factory.create(tmp_path / "assets", preset.model)
    assert isinstance(data, config.RLDataConfig)
    assert data.prompt_from_task
    assert data.episode_ids is None
    assert data.asset_id == sft.data.assets.asset_id == "pku_franka_pickplace"
    assert preset.assets_dirs != sft.assets_dirs  # Independent normalization files.
    assert data.action_sequence_keys == ("action",)
    assert data.extra_delta_timestamps["observation.state.cartesian"] == (0.0, 1.0)
    assert data.extra_delta_timestamps["observation.images.primary"] == (0.0, 1.0)
    assert data.extra_delta_timestamps["progress"] == tuple(t / 30 for t in range(31))
    transform = data.data_transforms.inputs[0]
    assert transform.task_indices == (3, 8)
    assert transform.failed_episode_ids == ()
    assert transform.progress_reward_mode == "delta"
    assert transform.discount == preset.iql.discount
    assert preset.model == sft.model
    assert preset.iql.action_dim == 7
    assert preset.iql.proprio_key == "iql_state"
    assert preset.iql.next_proprio_key == "next_iql_state"
    assert preset.iql.critic_warmup_steps == 1_000
    (meta / "tasks.jsonl").write_text('{"task_index": 3}\n{"task_index": 3}\n')
    with pytest.raises(ValueError, match="unique task indices"):
        factory.create(tmp_path / "assets", preset.model)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"discount": 0}, "discount"),
        ({"progress_reward_weight": np.nan}, "reward weights"),
        ({"progress_reward_mode": "unknown"}, "progress_reward_mode"),
    ],
)
def test_invalid_reward_settings_fail_before_training(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rl.PkuFrankaCollectRLInputs(**kwargs)
