import dataclasses
import json
import types

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import lingdong_policy as policy
from openpi.training import config
from openpi.training import data_loader
from openpi.training import lerobot_patched


def raw_sample():
    state = np.full(18, np.nan, dtype=np.float32)
    state[list(policy.RIGHT_INDICES)] = np.arange(8) / 10
    action = np.full((30, 18), np.nan, dtype=np.float32)
    action[:, list(policy.RIGHT_INDICES)] = np.arange(8) / 10
    return {
        "observation.state": state,
        "action": action,
        policy.HEAD_IMAGE_KEY: np.full((8, 12, 3), 10, dtype=np.uint8),
        policy.RIGHT_IMAGE_KEY: np.ones((3, 8, 12), dtype=np.float32),
        "observation.images.cam_left_wrist": object(),
        "prompt": b"pick up the object",
    }


def metadata():
    names = [f"{name}_L_Joint" for name in ("abad", "hip", "yaw", "knee", "wrist_yaw", "wrist_pitch", "wrist_roll")]
    names += [*policy.RIGHT_NAMES[:-1], "head_pitch_Joint", "head_yaw_Joint", "left_gripper", "right_gripper"]
    return {
        "codebase_version": "v2.1",
        "fps": 30,
        "features": {
            "observation.state": {"shape": [18], "names": names},
            "action": {"shape": [18], "names": names},
            **{key: {"dtype": "video", "shape": [480, 640, 3]} for key in policy.VIDEO_KEYS},
        },
    }


def test_only_right_channels_and_two_physical_cameras_reach_policy():
    raw = raw_sample()
    result = policy.LingdongRightArmInputs()(raw)
    np.testing.assert_array_equal(result["state"], raw["observation.state"][list(policy.RIGHT_INDICES)])
    np.testing.assert_array_equal(result["actions"], raw["action"][:, list(policy.RIGHT_INDICES)])
    assert result["state"].shape == (8,)
    assert result["actions"].shape == (30, 8)
    assert set(result["image"]) == {"base_0_rgb", "right_wrist_0_rgb"}
    assert result["image"]["right_wrist_0_rgb"].shape == (8, 12, 3)
    assert (result["image"]["right_wrist_0_rgb"] == 255).all()
    assert all(result["image_mask"].values())
    assert result["prompt"] == "pick up the object"
    padded = np.pad(result["actions"], ((0, 0), (0, 24)))
    np.testing.assert_array_equal(policy.LingdongRightArmOutputs()({"actions": padded})["actions"], result["actions"])


def test_inference_accepts_selected_state_without_left_arm_or_action():
    raw = raw_sample()
    raw["observation.state"] = raw["observation.state"][list(policy.RIGHT_INDICES)]
    del raw["action"]
    result = policy.LingdongRightArmInputs()(raw)
    assert "actions" not in result
    assert result["state"].shape == (8,)


@pytest.mark.parametrize("selected", [False, True])
def test_gripper_events_preserve_state_joints_and_source(selected):
    raw = raw_sample()
    raw["action"][:, 17] = np.repeat([0.0, 0.2, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4], 3)
    if selected:
        raw["action"] = raw["action"][:, list(policy.RIGHT_INDICES)]
    original = raw["action"].copy()
    result = policy.LingdongTrainingActions()(policy.LingdongRightArmInputs()(raw))
    np.testing.assert_array_equal(result["actions"][:, -1], np.r_[np.zeros(2), np.ones(9), np.zeros(19)])
    expected_joints = original[:, :7] if selected else original[:, list(policy.RIGHT_INDICES[:7])]
    np.testing.assert_array_equal(result["actions"][:, :7], expected_joints)
    np.testing.assert_array_equal(result["state"], raw["observation.state"][list(policy.RIGHT_INDICES)])
    np.testing.assert_array_equal(raw["action"], original)
    assert result["actions"].dtype == np.float32


def test_events_backdate_small_initial_steps_and_hold_across_plateaus():
    raw = np.repeat([0.4, 0.43, 0.63, 0.94, 0.94, 0.91, 0.71, 0.48, 0.48], 3)
    labels, events = policy.detect_gripper_events(raw)
    np.testing.assert_array_equal(labels, np.r_[np.zeros(2), np.ones(12), np.zeros(13)])
    assert [(e["frame"], e["confirmed_frame"], e["command"]) for e in events] == [(2, 6, 1), (14, 18, 0)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([0.4] * 5 + [0.94] + [0.4] * 5 + [0.42] * 3 + [0.39] * 3, 0),
        ([0.94] * 5 + [0.92] * 5 + [0.94] * 5, 1),
    ],
)
def test_isolated_spikes_and_small_reversals_do_not_change_commands(raw, expected):
    labels, events = policy.detect_gripper_events(raw)
    np.testing.assert_array_equal(labels, expected)
    assert not events


def test_slow_cumulative_motion_and_high_blocked_grasp_are_detected():
    raw = np.repeat([0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.94, 0.92, 0.79, 0.79], 3)
    labels, events = policy.detect_gripper_events(raw)
    assert [(e["frame"], e["command"]) for e in events] == [(2, 1), (23, 0)]
    np.testing.assert_array_equal(labels[2:23], 1)
    np.testing.assert_array_equal(labels[23:], 0)  # Closing holds even when the grasp stops above 0.6.


def test_chunk_binarization_requires_whole_episode_sources():
    training = policy.LingdongTrainingActions(action_horizon=30)
    with pytest.raises(ValueError, match="whole-episode label sources"):
        training({"actions": np.zeros((30, 8))})


@pytest.mark.parametrize("prompt", list(policy.PROMPT_VARIANTS))
def test_prompt_augmentation_samples_within_the_same_task(monkeypatch, prompt):
    raw = raw_sample()
    raw["prompt"] = prompt.encode()
    inputs = policy.LingdongRightArmInputs()(raw)
    variants = policy.PROMPT_VARIANTS[prompt]
    assert len(variants) == len(set(variants)) == 12
    other_variants = set().union(*(v for key, v in policy.PROMPT_VARIANTS.items() if key != prompt))
    assert not set(variants) & other_variants
    augmentation = policy.LingdongPromptAugmentation()
    for index, expected in enumerate(variants):
        monkeypatch.setattr(np.random, "randint", lambda n, index=index: index)
        result = augmentation(inputs)
        assert result["prompt"] == expected
        assert inputs["prompt"] == prompt
        np.testing.assert_array_equal(result["actions"], inputs["actions"])


def test_prompt_augmentation_preserves_serving_and_unknown_prompts(monkeypatch):
    def unexpected_random(n):
        pytest.fail("Inference and custom prompts must not be augmented")

    monkeypatch.setattr(np.random, "randint", unexpected_random)
    augmentation = policy.LingdongPromptAugmentation()
    for data in (
        {"prompt": "pick up the object"},
        {"prompt": "pick up the red cup", "actions": np.zeros((30, 8))},
        {"actions": np.zeros((30, 8))},
    ):
        assert augmentation(data) is data


@pytest.mark.parametrize(
    ("key", "value"),
    [("observation.state", np.zeros(14)), ("action", np.zeros((30, 14))), ("observation.state", np.full(8, np.nan))],
)
def test_incompatible_or_nonfinite_selected_channels_fail(key, value):
    raw = raw_sample()
    raw[key] = value
    with pytest.raises(ValueError, match="Expected|finite"):
        policy.LingdongRightArmInputs()(raw)


def test_field_names_determine_selection_even_if_state_order_differs():
    info = metadata()
    info["features"]["observation.state"]["names"] = list(reversed(info["features"]["observation.state"]["names"]))
    state_ids = policy.right_indices(info["features"], "observation.state")
    action_ids = policy.right_indices(info["features"], "action")
    assert state_ids != action_ids
    assert action_ids == policy.RIGHT_INDICES
    raw = raw_sample()
    raw["observation.state"] = raw["observation.state"][::-1]
    result = policy.LingdongRightArmInputs(state_ids, action_ids)(raw)
    np.testing.assert_array_equal(result["state"], np.arange(8, dtype=np.float32) / 10)


@pytest.mark.parametrize("journal", [".wrist_camera_correction.json", ".merge_plan.json"])
@pytest.mark.parametrize("has_metadata", [True, False])
def test_dataset_loading_waits_for_dataset_updates_to_finish(tmp_path, journal, has_metadata):
    meta = tmp_path / "meta"
    meta.mkdir()
    if has_metadata:
        (meta / "info.json").write_text(json.dumps(metadata()))
    marker = tmp_path / journal
    marker.write_text(json.dumps({"status": "in_progress"}))
    with pytest.raises(ValueError, match="still in progress"):
        policy.dataset_indices((str(tmp_path),), allow_unmounted=True)
    marker.write_text(json.dumps({"status": "complete"}))
    assert policy.dataset_indices((str(tmp_path),), allow_unmounted=True) == (
        policy.RIGHT_INDICES,
        policy.RIGHT_INDICES,
    )


@pytest.mark.parametrize("config_name", [policy.CONFIG_NAME, policy.RELATIVE_CONFIG_NAME])
def test_config_reads_final_dataset_and_keeps_both_tasks_and_two_cameras(tmp_path, monkeypatch, config_name):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    for name in policy.DATASET_NAMES:
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps(metadata()))
    preset = config.get_config(config_name)
    factory = dataclasses.replace(preset.data, dataset_root=str(tmp_path))
    data = factory.create(tmp_path / "assets", preset.model)
    assert isinstance(data, config.MultiDataConfig)
    assert data.repo_ids == (str(tmp_path / policy.DATASET_NAMES[0]),)
    assert data.video_keys == policy.VIDEO_KEYS
    assert data.action_sequence_keys == ("action",)
    assert data.prompt_from_task
    assert data.video_backend == "pyav"
    assert any(isinstance(t, policy.LingdongPromptAugmentation) for t in data.training_transforms.inputs)
    assert not any(isinstance(t, policy.LingdongPromptAugmentation) for t in data.data_transforms.inputs)
    plain = dataclasses.replace(factory, augment_prompts=False).create(tmp_path / "assets", preset.model)
    assert not any(isinstance(t, policy.LingdongPromptAugmentation) for t in plain.training_transforms.inputs)
    monkeypatch.setattr(data_loader, "create_multi_torch_dataset", lambda data, horizon, model: horizon)
    assert data_loader.make_torch_dataset(data, preset.model.action_horizon, preset.model) == 30
    monkeypatch.setattr(policy, "_episode_gripper_labels", lambda repo, episode: np.ones(30, dtype=np.uint8))
    raw = raw_sample()
    raw.update(dataset_index=0, episode_index=0, frame_index=0)
    adapted = data_loader.transform_dataset([raw], data, skip_norm_stats=True)[0]
    assert adapted["actions"].shape == (30, 8)
    np.testing.assert_array_equal(adapted["actions"][:, -1], 1)
    assert preset.model.action_dim == 32
    assert preset.model.action_horizon == 30
    assert preset.policy_metadata["state_dim"] == preset.policy_metadata["action_dim"] == 8
    prompts = transforms.PromptFromMultiLeRobotTask(({0: "pick up the object", 1: "put down the object"},))
    assert (
        prompts({"dataset_index": 0, "task_index": 0})["prompt"]
        != prompts({"dataset_index": 0, "task_index": 1})["prompt"]
    )


@pytest.mark.parametrize("selected", [False, True])
def test_relative_chunks_use_current_state_and_serving_restores_absolute_targets(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    monkeypatch.setattr(policy, "_episode_gripper_labels", lambda repo, episode: np.ones(30, dtype=np.uint8))
    preset = config.get_config(policy.RELATIVE_CONFIG_NAME)
    absolute = config.get_config(policy.CONFIG_NAME)
    assert preset.model == absolute.model
    assert preset.lr_schedule == absolute.lr_schedule
    assert preset.num_train_steps == absolute.num_train_steps
    assert preset.data.assets.asset_id == policy.RELATIVE_ASSET_ID
    assert preset.assets_dirs != absolute.assets_dirs
    factory = dataclasses.replace(preset.data, dataset_root=str(tmp_path), augment_prompts=False, state_dropout_prob=1)
    data = factory.create(tmp_path / "assets", preset.model)
    assert data.asset_id == policy.RELATIVE_ASSET_ID
    raw = raw_sample()
    raw["action"][:, list(policy.RIGHT_INDICES[:7])] += np.arange(30, dtype=np.float32)[:, None] / 10 + 0.25
    if selected:
        raw["action"] = raw["action"][:, list(policy.RIGHT_INDICES)]
    original = raw["action"].copy()
    raw.update(dataset_index=0, episode_index=0, frame_index=0)
    result = data_loader.transform_dataset([raw], data, skip_norm_stats=True)[0]
    targets = original if selected else original[:, list(policy.RIGHT_INDICES)]
    state = raw["observation.state"][list(policy.RIGHT_INDICES)]
    np.testing.assert_allclose(result["actions"][:, :7], targets[:, :7] - state[:7], atol=1e-7)
    np.testing.assert_array_equal(result["actions"][:, -1], 1)
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(raw["action"], original)
    assert result["_drop_state_input"]

    # Simulate policy output after unnormalizing padded model state/actions.
    output = transforms.compose(data.data_transforms.outputs)(
        {"actions": np.pad(result["actions"], ((0, 0), (0, 24))), "state": np.pad(state, (0, 24))}
    )
    assert output["actions"].shape == (30, 8)
    np.testing.assert_allclose(output["actions"][:, :7], targets[:, :7], atol=1e-7)
    np.testing.assert_array_equal(output["actions"][:, -1], 1)
    serving = transforms.compose(data.data_transforms.inputs)(
        {key: value for key, value in raw_sample().items() if key != "action"}
    )
    np.testing.assert_array_equal(serving["state"], state)


def test_state_dropout_omits_training_tokens_but_keeps_serving_state(tmp_path, monkeypatch):
    tokenized_states = []

    def tokenize(prompt, state):
        tokenized_states.append(state)
        return np.array([1, 2]), np.array([True, True])

    tokenizer = types.SimpleNamespace(tokenize=tokenize)
    monkeypatch.setattr(
        config.ModelTransformFactory,
        "__call__",
        lambda self, model: transforms.Group(
            inputs=[transforms.TokenizePrompt(tokenizer, discrete_state_input=model.discrete_state_input)]
        ),
    )
    for name in policy.DATASET_NAMES:
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps(metadata()))
    preset = config.get_config(policy.CONFIG_NAME)
    factory = dataclasses.replace(
        preset.data, dataset_root=str(tmp_path), augment_prompts=False, state_dropout_prob=1.0
    )
    data = factory.create(tmp_path / "assets", preset.model)
    monkeypatch.setattr(policy, "_episode_gripper_labels", lambda repo, episode: np.ones(30, dtype=np.uint8))
    raw = raw_sample()
    raw.update(dataset_index=0, episode_index=0, frame_index=0)
    training = data_loader.transform_dataset([raw], data, skip_norm_stats=True)[0]
    assert tokenized_states[-1] is None
    assert "_drop_state_input" not in training
    np.testing.assert_array_equal(training["state"], np.arange(8, dtype=np.float32) / 10)
    np.testing.assert_array_equal(training["actions"][:, -1], 1)

    # Policy loading uses these two groups, without dataset-only training transforms.
    serving_transform = transforms.compose([*data.data_transforms.inputs, *data.model_transforms.inputs])
    serving = {key: value for key, value in raw_sample().items() if key != "action"}
    output = serving_transform(serving)
    np.testing.assert_array_equal(tokenized_states[-1], training["state"])
    np.testing.assert_array_equal(output["state"], training["state"])
    assert "_drop_state_input" not in output

    default = dataclasses.replace(factory, state_dropout_prob=0.0).create(tmp_path / "assets", preset.model)
    assert not any(isinstance(t, transforms.StateInputDropout) for t in default.training_transforms.inputs)
    no_state_model = dataclasses.replace(preset.model, discrete_state_input=False)
    with pytest.raises(ValueError, match="discrete_state_input=True"):
        factory.create(tmp_path / "assets", no_state_model)


def test_left_video_is_filtered_before_decoder(monkeypatch, tmp_path):
    seen = []

    def decode(self, timestamps, episode):
        seen.extend(timestamps)
        return timestamps

    monkeypatch.setattr(lerobot_patched._ld.LeRobotDataset, "_query_videos", decode)  # noqa: SLF001
    dataset = object.__new__(lerobot_patched.PatchedLeRobotDataset)
    dataset._selected_video_keys = policy.VIDEO_KEYS  # noqa: SLF001
    dataset.root = tmp_path
    dataset.meta = types.SimpleNamespace(get_video_file_path=lambda episode, camera: f"{camera}/{episode}.mp4")
    dataset.video_backend = "pyav"
    dataset.tolerance_s = 1e-4
    timestamps = {key: [0.0] for key in (*policy.VIDEO_KEYS, "observation.images.cam_left_wrist")}
    dataset._query_videos(timestamps, 0)  # noqa: SLF001
    assert tuple(seen) == policy.VIDEO_KEYS


@pytest.mark.parametrize("episodes", [None, [2]])
def test_parquet_loading_uses_exact_metadata_paths_without_directory_discovery(tmp_path, monkeypatch, episodes):
    calls, transforms = [], []
    loaded = types.SimpleNamespace(set_transform=transforms.append)

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return loaded

    monkeypatch.setattr(lerobot_patched._ld, "load_dataset", load)  # noqa: SLF001
    dataset = object.__new__(lerobot_patched.PatchedLeRobotDataset)
    dataset.root = tmp_path
    dataset.episodes = episodes
    dataset.meta = types.SimpleNamespace(
        episodes={0: {}, 2: {}},
        get_data_file_path=lambda index: f"data/chunk-000/episode_{index:06d}.parquet",
    )
    assert dataset.load_hf_dataset() is loaded
    expected = [0, 2] if episodes is None else episodes
    assert calls == [
        (
            "parquet",
            {
                "data_files": [str(tmp_path / f"data/chunk-000/episode_{index:06d}.parquet") for index in expected],
                "split": "train",
            },
        )
    ]
    assert transforms == [lerobot_patched._ld.hf_transform_to_torch]  # noqa: SLF001
