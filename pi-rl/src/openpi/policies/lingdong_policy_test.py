import dataclasses
import json
import types

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import lingdong_policy as policy
from openpi.training import config
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
    assert policy.dataset_indices((str(tmp_path),), allow_unmounted=True) == (policy.RIGHT_INDICES, policy.RIGHT_INDICES)


def test_config_reads_final_dataset_and_keeps_both_tasks_and_two_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    for name in policy.DATASET_NAMES:
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps(metadata()))
    preset = config.get_config(policy.CONFIG_NAME)
    factory = dataclasses.replace(preset.data, dataset_root=str(tmp_path))
    data = factory.create(tmp_path / "assets", preset.model)
    assert isinstance(data, config.MultiDataConfig)
    assert data.repo_ids == (str(tmp_path / policy.DATASET_NAMES[0]),)
    assert data.video_keys == policy.VIDEO_KEYS
    assert data.action_sequence_keys == ("action",)
    assert data.prompt_from_task
    assert data.video_backend == "pyav"
    assert preset.model.action_dim == 32
    assert preset.model.action_horizon == 30
    assert preset.policy_metadata["state_dim"] == preset.policy_metadata["action_dim"] == 8
    prompts = transforms.PromptFromMultiLeRobotTask(({0: "pick up the object", 1: "put down the object"},))
    assert (
        prompts({"dataset_index": 0, "task_index": 0})["prompt"]
        != prompts({"dataset_index": 0, "task_index": 1})["prompt"]
    )


def test_left_video_is_filtered_before_decoder(monkeypatch):
    seen = []

    def decode(self, timestamps, episode):
        seen.extend(timestamps)
        return timestamps

    monkeypatch.setattr(lerobot_patched._ld.LeRobotDataset, "_query_videos", decode)  # noqa: SLF001
    dataset = object.__new__(lerobot_patched.PatchedLeRobotDataset)
    dataset._selected_video_keys = policy.VIDEO_KEYS  # noqa: SLF001
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
            {"data_files": [str(tmp_path / f"data/chunk-000/episode_{index:06d}.parquet") for index in expected], "split": "train"},
        )
    ]
    assert transforms == [lerobot_patched._ld.hf_transform_to_torch]  # noqa: SLF001
