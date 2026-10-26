import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.models import pi0
from openpi.policies import pku_franka_unified_policy as policy
from openpi.training import config


def sample():
    return {
        policy.IMAGE_KEY: np.zeros((3, 8, 12), dtype=np.float32),
        policy.STATE_KEY: np.array([0.35, 0.02, 0.23, -3.1, 0, 0, 0.4], dtype=np.float32),
        policy.ACTION_KEY: np.tile([0.01, -0.02, 0, 0, 0, 0.001, 0.75], (10, 1)).astype(np.float32),
        "prompt": b"Open the white drawer.",
    }


def test_unified_values_are_preserved_without_second_gripper_or_delta_conversion():
    raw = sample()
    output = policy.PkuFrankaUnifiedInputs()(raw)
    np.testing.assert_array_equal(output["state"], raw[policy.STATE_KEY])
    np.testing.assert_array_equal(output["actions"], raw[policy.ACTION_KEY])
    assert output["image"]["base_0_rgb"].shape == (8, 12, 3)
    assert set(output["image"]) == {"base_0_rgb", "left_wrist_0_rgb"}
    assert not output["image_mask"]["left_wrist_0_rgb"]
    assert not output["image"]["left_wrist_0_rgb"].any()
    assert output["prompt"] == "Open the white drawer."


def test_inference_needs_no_action_label():
    raw = sample()
    del raw[policy.ACTION_KEY]
    assert "actions" not in policy.PkuFrankaUnifiedInputs()(raw)
    raw["action"] = np.zeros((10, 7))
    with pytest.raises(KeyError, match="action.relative"):
        policy.PkuFrankaUnifiedInputs()(raw)


@pytest.mark.parametrize("chw", [False, True])
@pytest.mark.parametrize("floating", [False, True])
def test_real_wrist_image_is_preserved(chw, floating):
    raw = sample()
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 1] = 255
    wrist = image.astype(np.float32) / 255 if floating else image
    raw[policy.WRIST_KEY] = np.moveaxis(wrist, -1, 0) if chw else wrist
    output = policy.PkuFrankaUnifiedInputs()(raw)
    np.testing.assert_array_equal(output["image"]["left_wrist_0_rgb"], image)
    assert output["image_mask"]["left_wrist_0_rgb"]


def test_single_and_dual_camera_batch_masks_reach_model_attention():
    single = policy.PkuFrankaUnifiedInputs()(sample())
    dual_raw = sample()
    dual_raw[policy.WRIST_KEY] = np.zeros((3, 8, 12), dtype=np.float32)
    dual = policy.PkuFrankaUnifiedInputs()(dual_raw)
    # A real black image is valid; availability must not be inferred from pixels.
    assert dual["image_mask"]["left_wrist_0_rgb"]
    batch = {
        "image": {key: np.stack([single["image"][key], dual["image"][key]]) for key in single["image"]},
        "image_mask": {key: np.array([single["image_mask"][key], dual["image_mask"][key]]) for key in single["image_mask"]},
        "state": np.stack([single["state"], dual["state"]]),
    }
    observation = model.Observation.from_dict(batch)
    # Exercise the actual pi0 prefix/attention masking without loading model weights.
    encoder = SimpleNamespace(PaliGemma=SimpleNamespace(img=lambda image, train: (jnp.ones((2, 4, 8)), None)))
    _, mask, ar_mask = pi0.Pi0.embed_prefix(encoder, observation)
    np.testing.assert_array_equal(mask[0], [True] * 4 + [False] * 4)
    np.testing.assert_array_equal(mask[1], [True] * 8)
    attention = np.asarray(pi0.make_attn_mask(mask, ar_mask))
    assert not attention[0, :, 4:].any()
    assert attention[1].all()


@pytest.mark.parametrize(("key", "value"), [
    (policy.STATE_KEY, np.zeros(20)),
    (policy.STATE_KEY, np.full(7, np.nan)),
    (policy.ACTION_KEY, np.zeros((10, 14))),
    (policy.ACTION_KEY, np.tile([0, 0, 0, 0, 0, 0, -1], (10, 1))),
])
def test_reject_incompatible_raw_formats(key, value):
    raw = sample()
    raw[key] = value
    with pytest.raises(ValueError, match="finite|normalized"):
        policy.PkuFrankaUnifiedInputs()(raw)


def write_metadata(root, name, fps=5):
    meta = root / name / "meta"
    meta.mkdir(parents=True)
    features = {
        policy.IMAGE_KEY: {"dtype": "video", "shape": [480, 640, 3]},
        policy.STATE_KEY: {"shape": [7], "gripper_convention": "0_closed_1_open"},
        policy.ACTION_KEY: {"shape": [7], "representation": "relative_delta_pose", "gripper_convention": "0_closed_1_open"},
    }
    (meta / "info.json").write_text(json.dumps({"fps": fps, "features": features}))
    (meta / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": i}) + "\n" for i in range(106)))


def test_config_selects_cleaned_variant_and_filters_only_failed_collection_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    preset = config.get_config("pi05_pku_franka_unified_relative")
    names = ("fr3_long", "fr3_long_sft_clean", "fr3_collect_objects_20261017")
    for name in names:
        write_metadata(tmp_path, name)
    factory = dataclasses.replace(preset.data, dataset_root=str(tmp_path), dataset_names=names)
    data = factory.create(tmp_path / "assets", preset.model)
    assert isinstance(data, config.MixedMultiDataConfig)
    assert tuple(path.rsplit("/", 1)[-1] for path in data.repo_ids) == names[1:]
    collection = str(tmp_path / names[-1])
    assert len(data.episode_ids[collection]) == 98
    assert 71 in data.episode_ids[collection]  # Retried grasps alone do not mean a failed episode.
    assert 78 not in data.episode_ids[collection]
    assert data.action_sequence_keys == (policy.ACTION_KEY,)
    assert data.prompt_from_task
    assert data.video_backend == "pyav"
    assert preset.model.action_horizon == 10
    raw = sample()
    transformed = transforms.compose(data.data_transforms.inputs)(raw)
    padded = np.pad(transformed["actions"], ((0, 0), (0, preset.model.action_dim - 7)))
    restored = transforms.compose(data.data_transforms.outputs)({"actions": padded})
    np.testing.assert_array_equal(restored["actions"], raw[policy.ACTION_KEY])


def test_config_rejects_mixed_fps(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    preset = config.get_config("pi05_pku_franka_unified_relative")
    write_metadata(tmp_path, "wrong_fps", fps=30)
    factory = dataclasses.replace(preset.data, dataset_root=str(tmp_path), repo_ids=(str(tmp_path / "wrong_fps"),))
    with pytest.raises(ValueError, match="5 Hz"):
        factory.create(tmp_path / "assets", preset.model)
