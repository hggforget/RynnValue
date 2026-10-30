import copy
import dataclasses
import json

import numpy as np
import pytest

from openpi.policies import lingdong_language as language
from openpi.policies import lingdong_policy as policy


def sample(task="pick_up", support="table", episode=0):
    before = "object_in_container" if task == "pick_up" else "object_held"
    after = "object_held" if task == "pick_up" else "object_released"
    prepared = {
        "episode_index": episode,
        "task_type": task,
        "frames": [0, 20, 40, 60, 80, 99],
        "gripper_events": [{"command": 0 if task == "pick_up" else 1, "frame": 50, "confirmed_frame": 60}],
    }
    raw = {
        "task_type": task,
        "container_color": "blue",
        "container_support": support,
        "source": "container" if task == "pick_up" else "gripper",
        "destination": "gripper" if task == "pick_up" else "container",
        "container_confidence": 0.99,
        "support_confidence": 0.99,
        "transfer_confidence": 0.95,
        "visual_support": "object_transfer",
        "notes": "Visible manipulation of an object in a blue tray.",
        "evidence": [
            {"camera": "head", "frame_index": 0, "facts": ["blue_container", f"container_on_{support}", before]},
            {"camera": "head", "frame_index": 99, "facts": ["blue_container", f"container_on_{support}", after]},
        ],
    }
    return raw, prepared


@pytest.mark.parametrize("task", ["pick_up", "put_down"])
@pytest.mark.parametrize("support", ["table", "metal_shelf"])
def test_verified_scenes_generate_only_matching_locations_and_directions(task, support):
    raw, prepared = sample(task, support)
    result = language.validate_prediction(raw, prepared)
    assert result["accepted"]
    variants = result["scene_variants"]
    assert any(f"on the {'table' if support == 'table' else 'metal shelf'}" in text for text in variants)
    assert not any(("metal shelf" if support == "table" else "table") in text for text in variants)
    if task == "pick_up":
        assert all("from the blue" in text or "out of the blue" in text or "in the blue" in text for text in variants)
        assert not any("from the gripper" in text for text in variants)
    else:
        assert any("from the gripper to the blue" in text for text in variants)


@pytest.mark.parametrize("conflicting", [False, True])
def test_uncertain_or_contradictory_support_keeps_only_container_context(conflicting):
    raw, prepared = sample()
    if conflicting:
        raw["evidence"][0]["facts"].append("container_on_metal_shelf")
    else:
        raw["support_confidence"] = 0.5
    result = language.validate_prediction(raw, prepared)
    assert result["accepted"]
    assert result["container_support"] == "unknown"
    assert not any("table" in text or "shelf" in text for text in result["scene_variants"])


@pytest.mark.parametrize("problem", ["task", "source", "container", "time", "gripper"])
def test_unsupported_episode_relationships_fall_back_to_generic_language(problem):
    raw, prepared = sample()
    if problem == "task":
        raw["task_type"] = "put_down"
    elif problem == "source":
        raw["source"] = "gripper"
    elif problem == "container":
        raw["container_confidence"] = 0.3
    elif problem == "time":
        raw["evidence"][0]["facts"][-1] = "object_held"
        raw["evidence"][1]["facts"][-1] = "object_in_container"
    else:
        prepared["gripper_events"] = [{"command": 1, "frame": 50, "confirmed_frame": 60}]
    result = language.validate_prediction(raw, prepared)
    assert not result["accepted"]
    assert result["reason_codes"]
    assert result["scene_variants"] == []


@pytest.mark.parametrize(("field", "value"), [("frame_index", 42), ("camera", "left_wrist")])
def test_model_cannot_cite_unprovided_evidence(field, value):
    raw, prepared = sample()
    raw["evidence"][0][field] = value
    with pytest.raises(ValueError, match="unprovided"):
        language.validate_prediction(raw, prepared)


def test_empty_observations_add_no_scene_evidence():
    raw, prepared = sample()
    original = language.validate_prediction(raw, prepared)
    raw["evidence"].append({"camera": "right_wrist", "frame_index": 20, "facts": []})
    assert language.validate_prediction(raw, prepared)["scene_variants"] == original["scene_variants"]
    raw["evidence"][0]["facts"] = []
    rejected = language.validate_prediction(raw, prepared)
    assert not rejected["accepted"]
    assert rejected["scene_variants"] == []


def test_original_episode_identity_selects_the_correct_scene(monkeypatch):
    raw, prepared = sample(episode=0)
    table = language.validate_prediction(raw, prepared)
    raw, prepared = sample(support="metal_shelf", episode=1)
    shelf = language.validate_prediction(raw, prepared)
    augmentation = policy.LingdongPromptAugmentation(({0: table, 1: shelf},))
    monkeypatch.setattr(np.random, "randint", lambda n: len(policy.PROMPT_VARIANTS["pick up the object"]) + 6)
    data = {"actions": np.zeros((30, 8)), "prompt": "pick up the object", "dataset_index": 0}
    assert "on the table" in augmentation({**data, "episode_index": 0})["prompt"]
    assert "on the metal shelf" in augmentation({**data, "episode_index": 1})["prompt"]
    with pytest.raises(ValueError, match="episode_index"):
        augmentation(data)
    wrong = {**data, "episode_index": 0, "prompt": "put down the object"}
    with pytest.raises(ValueError, match="differs"):
        augmentation(wrong)


def test_dataset_fingerprint_and_episode_tasks_are_required_when_loading_assets(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    (meta / "tasks.jsonl").write_text("{}\n")
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["pick up the object"]}) + "\n")
    raw, prepared = sample()
    record = language.validate_prediction(raw, prepared)
    asset = {
        "version": language.VERSION,
        "complete": True,
        "dataset_signature": language.dataset_signature(root),
        "episodes": {"0": record},
    }
    path = tmp_path / "asset.json"
    path.write_text(json.dumps(asset))
    assert language.load_scene_prompts(path, (str(root),))[0][0] == record
    wrong = copy.deepcopy(asset)
    wrong["episodes"]["0"]["task_type"] = "put_down"
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="task"):
        language.load_scene_prompts(path, (str(root),))
    path.write_text(json.dumps(asset))
    (meta / "info.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="fingerprint"):
        language.load_scene_prompts(path, (str(root),))


@pytest.mark.parametrize("config_name", [policy.CONFIG_NAME, policy.RELATIVE_CONFIG_NAME])
def test_training_factory_automatically_loads_validated_episode_scenes(tmp_path, monkeypatch, config_name):
    from openpi import transforms
    from openpi.training import config

    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    repo = tmp_path / "datasets" / policy.DATASET_NAMES[0]
    meta = repo / "meta"
    meta.mkdir(parents=True)
    names = [f"unused_{i}" for i in range(18)]
    for index, name in zip(policy.RIGHT_INDICES, policy.RIGHT_NAMES, strict=True):
        names[index] = name
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "features": {
            "observation.state": {"shape": [18], "names": names},
            "action": {"shape": [18], "names": names},
            **{key: {"dtype": "video"} for key in policy.VIDEO_KEYS},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "tasks.jsonl").write_text("{}\n")
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["pick up the object"]}) + "\n")
    raw, prepared = sample()
    record = language.validate_prediction(raw, prepared)
    asset = {
        "version": language.VERSION,
        "complete": True,
        "dataset_signature": language.dataset_signature(repo),
        "episodes": {"0": record},
    }
    assets_root = tmp_path / "assets"
    path = assets_root / policy.CONFIG_NAME / policy.ASSET_ID / language.ASSET_NAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(asset))
    preset = config.get_config(config_name)
    factory = dataclasses.replace(preset.data, dataset_root=str(tmp_path / "datasets"))
    data = factory.create(assets_root / config_name, preset.model)
    augmentation = next(t for t in data.training_transforms.inputs if isinstance(t, policy.LingdongPromptAugmentation))
    assert augmentation.scene_prompts[0][0] == record
    assert isinstance(data.training_transforms.inputs[0], policy.LingdongPromptAugmentation)
