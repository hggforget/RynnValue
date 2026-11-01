import dataclasses
import json

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import lingdong_language
from openpi.policies import lingdong_policy as policy
from openpi.policies import lingdong_rl_policy as rl
from openpi.training import config


@pytest.fixture
def reward_dataset(tmp_path):
    repo = tmp_path / policy.DATASET_NAMES[0]
    meta = repo / "meta"
    meta.mkdir(parents=True)
    names = [f"unused_{index}" for index in range(18)]
    for index, name in zip(policy.RIGHT_INDICES, policy.RIGHT_NAMES, strict=True):
        names[index] = name
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 3,
        "total_frames": 18,
        "features": {
            "observation.state": {"shape": [18], "names": names},
            "action": {"shape": [18], "names": names},
            **{key: {"dtype": "video"} for key in policy.VIDEO_KEYS},
        },
    }
    rows = [
        {"episode_index": index, "length": 6, "tasks": [policy.TASK_PROMPTS[task]]}
        for index, task in enumerate(("put_down", "pick_up", "put_down"))
    ]
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (meta / "tasks.jsonl").write_text("{}\n")
    labels = {
        "version": rl.REWARD_VERSION,
        "dataset_signature": lingdong_language.dataset_signature(repo),
        "episodes": {str(index): {"progress": [-5, -4, -3, -3.5, -1, 0], "success": index != 2} for index in range(3)},
    }
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps(labels))
    return repo, path, labels


def transition(frame=0, episode=1):
    state = np.arange(36, dtype=np.float32).reshape(2, 18) / 20
    return {
        "observation.state": state,
        policy.HEAD_IMAGE_KEY: np.stack(
            [
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.full((480, 640, 3), 80, dtype=np.uint8),
            ]
        ),
        policy.RIGHT_IMAGE_KEY: np.stack(
            [
                np.full((3, 12, 16), 0.2, dtype=np.float32),
                np.full((3, 12, 16), 0.8, dtype=np.float32),
            ]
        ),
        "action": np.ones((3, 18), dtype=np.float32),
        "progress": np.array([-5, -4, -3, -3.5], dtype=np.float32),
        "episode_index": episode,
        "frame_index": frame,
        "prompt": policy.TASK_PROMPTS["pick_up"],
    }


def test_progress_reward_retains_regressions_and_conditions_both_critic_views():
    raw = transition()
    transform = rl.LingdongRightArmRLInputs(episode_lengths={1: 8}, episode_tasks={1: 0}, discount=0.9)
    result = transform(raw)
    np.testing.assert_allclose(result["chunk_reward"], 1 + 0.9 - 0.9**2 * 0.5, rtol=1e-6)
    assert result["mask"] == 1
    assert result["image"]["base_0_rgb"].shape == (320, 320, 3)
    assert result["next_image_base"].shape == result["next_image_wrist"].shape == (224, 224, 3)
    assert (result["next_image_base"] == 80).all()
    np.testing.assert_array_equal(result["iql_state"][:8], raw["observation.state"][0, list(policy.RIGHT_INDICES)])
    np.testing.assert_array_equal(result["next_iql_state"][:8], raw["observation.state"][1, list(policy.RIGHT_INDICES)])
    np.testing.assert_array_equal(result["iql_state"][-2:], [1, 0])
    np.testing.assert_array_equal(result["next_iql_state"][-2:], [1, 0])
    assert raw["action"].shape == (3, 18)


@pytest.mark.parametrize("frame", [2, 3, 5])
def test_exact_and_padded_terminals_ignore_padding_and_do_not_invent_success(frame):
    values = np.array([-5, -4, -3, -3.5, -1, -0.5], dtype=np.float32)
    transform = rl.LingdongRightArmRLInputs(
        episode_lengths={1: 6},
        episode_tasks={1: 1},
        progress={1: values},
        discount=0.9,
    )
    result = transform(transition(frame))
    steps = min(3, 5 - frame)
    expected = sum(0.9**offset * (values[frame + offset + 1] - values[frame + offset]) for offset in range(steps))
    np.testing.assert_allclose(result["chunk_reward"], expected, rtol=1e-6)
    assert result["mask"] == 0
    np.testing.assert_array_equal(result["iql_state"][-2:], [0, 1])


@pytest.mark.parametrize(("success", "bonus"), [(True, 1.0), (False, -1.0)])
def test_sparse_success_failure_rewards_have_correct_terminal_discount(success, bonus):
    transform = rl.LingdongRightArmRLInputs(
        reward_source="terminal",
        episode_lengths={1: 6},
        episode_tasks={1: 0},
        outcomes={1: success},
        discount=0.9,
    )
    assert transform(transition(0))["chunk_reward"] == 0
    np.testing.assert_allclose(transform(transition(2))["chunk_reward"], 0.9**2 * bonus, rtol=1e-6)
    np.testing.assert_allclose(transform(transition(3))["chunk_reward"], 0.9 * bonus, rtol=1e-6)
    assert transform(transition(5))["chunk_reward"] == 0  # Already at the absorbing endpoint.


def test_raw_dataset_requires_explicit_reward_labels_and_terminal_assumptions(reward_dataset):
    repo, _, _ = reward_dataset
    with pytest.raises(ValueError, match="requires real progress"):
        rl.reward_metadata(str(repo))
    with pytest.raises(ValueError, match="Missing success/failure"):
        rl.reward_metadata(str(repo), reward_source="terminal")
    metadata = rl.reward_metadata(str(repo), reward_source="terminal", assume_success=True)
    assert metadata["outcomes"] == {0: True, 1: True, 2: True}


@pytest.mark.parametrize("failure", ["fingerprint", "short_progress", "nonfinite", "bad_outcome", "unknown_episode"])
def test_reward_sidecar_rejects_stale_or_invalid_labels(reward_dataset, failure):
    repo, path, labels = reward_dataset
    if failure == "fingerprint":
        labels["dataset_signature"] = "stale"
    elif failure == "short_progress":
        labels["episodes"]["1"]["progress"] = [0]
    elif failure == "nonfinite":
        labels["episodes"]["1"]["progress"][0] = float("nan")
    elif failure == "bad_outcome":
        labels["episodes"]["1"]["success"] = "true"
    else:
        labels["episodes"]["99"] = {}
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="fingerprint|finite per-frame|true/false/null|invalid episode"):
        rl.reward_metadata(str(repo), reward_labels_path=str(path))


@pytest.mark.parametrize("task", ["all", "pick_up", "put_down"])
@pytest.mark.parametrize("representation", ["absolute", "relative"])
def test_iql_presets_preserve_task_filters_gripper_labels_and_sft_assets(reward_dataset, monkeypatch, task, representation):
    repo, path, _ = reward_dataset
    names = policy.SINGLE_TASK_RELATIVE_CROP_CONFIGS if representation == "relative" else policy.SINGLE_TASK_CROP_CONFIGS
    joint = policy.RELATIVE_CROP_CONFIG_NAME if representation == "relative" else policy.CROP_CONFIG_NAME
    name = joint if task == "all" else names[task]
    preset = config.get_config(f"{name}_iql")
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    factory = dataclasses.replace(
        preset.data,
        dataset_root=str(repo.parent),
        reward_labels_path=str(path),
        augment_prompts=False,
    )
    data = factory.create(preset.assets_dirs, preset.model)
    assert isinstance(data, config.RLDataConfig)
    assert data.repo_id == str(repo)
    assert data.episode_ids == {"all": (0, 1, 2), "pick_up": (1,), "put_down": (0, 2)}[task]
    assert data.asset_id == config.get_config(name).data.assets.asset_id
    assert preset.data.action_representation == representation
    assert data.video_keys == policy.VIDEO_KEYS
    assert data.extra_delta_timestamps["observation.state"] == (0.0, 1.0)
    assert "progress" not in data.extra_delta_timestamps  # The sidecar supplies it.
    assert any(isinstance(t, policy.LingdongTrainingActions) for t in data.training_transforms.inputs)
    assert preset.iql.image_keys[-1] == "image.right_wrist_0_rgb"
    assert preset.iql.action_dim == 8
    assert preset.iql.proprio_key == "iql_state"


def test_iql_serving_uses_the_same_adapter_without_training_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    preset = config.get_config(f"{policy.CROP_CONFIG_NAME}_iql")
    data = dataclasses.replace(preset.data, dataset_root=str(tmp_path)).create_for_inference(
        preset.assets_dirs, preset.model
    )
    raw = transition()
    for key in ("observation.state", *policy.VIDEO_KEYS):
        raw[key] = raw[key][0]
    del raw["action"], raw["progress"]
    actual = transforms.compose(data.data_transforms.inputs)(raw)
    expected = policy.LingdongRightArmInputs(crop_head=True)(raw)
    np.testing.assert_array_equal(actual["state"], expected["state"])
    np.testing.assert_array_equal(actual["image"]["base_0_rgb"], expected["image"]["base_0_rgb"])


def test_iql_serving_does_not_require_rewards_even_when_raw_training_data_is_mounted(reward_dataset, monkeypatch):
    repo, _, _ = reward_dataset
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    preset = config.get_config(f"{policy.CROP_CONFIG_NAME}_iql")
    factory = dataclasses.replace(preset.data, dataset_root=str(repo.parent), augment_prompts=False)
    with pytest.raises(ValueError, match="requires real progress"):
        factory.create(preset.assets_dirs, preset.model)
    data = factory.create_for_inference(preset.assets_dirs, preset.model)
    assert isinstance(data.data_transforms.inputs[0], policy.LingdongRightArmInputs)


def test_dataset_progress_column_is_requested_when_no_sidecar_is_used(reward_dataset, monkeypatch):
    repo, _, _ = reward_dataset
    path = repo / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["progress"] = {"dtype": "float32", "shape": [1]}
    path.write_text(json.dumps(info))
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    preset = config.get_config(f"{policy.RELATIVE_CROP_CONFIG_NAME}_iql")
    factory = dataclasses.replace(preset.data, dataset_root=str(repo.parent), augment_prompts=False)
    data = factory.create(preset.assets_dirs, preset.model)
    assert data.extra_delta_timestamps["progress"] == tuple(offset / 30 for offset in range(31))
    assert any(isinstance(t, transforms.DeltaActions) for t in data.data_transforms.inputs)
