import numpy as np
import pytest

from openpi import transforms
from openpi.policies import pku_franka_policy
from openpi.training import config as _config


@pytest.mark.parametrize("config_name", ["pi05_pku_franka", "pi05_pku_franka_lora"])
def test_recorded_actions_stay_deltas(config_name, tmp_path, monkeypatch):
    # Test the robot transforms without downloading a language tokenizer.
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    config = _config.get_config(config_name)
    factory = _config.LeRobotPkuFrankaDataConfig(repo_id=str(tmp_path))
    data_config = factory.create(tmp_path, config.model)
    state = np.array([0.35, 0.02, 0.23, -3.15, 0.01, 0.0, 0.08], dtype=np.float32)
    actions = np.tile([0.003, -0.002, 0, 0, 0, 0.01, -1], (config.model.action_horizon, 1)).astype(np.float32)
    inputs = transforms.compose(data_config.data_transforms.inputs)(
        {"image": np.zeros((3, 8, 12), dtype=np.float32), "state": state, "actions": actions}
    )
    np.testing.assert_array_equal(inputs["state"], state)
    np.testing.assert_array_equal(inputs["actions"], actions)
    padded_actions = np.pad(inputs["actions"], ((0, 0), (0, config.model.action_dim - 7)))
    outputs = transforms.compose(data_config.data_transforms.outputs)({"actions": padded_actions, "state": state})
    np.testing.assert_array_equal(outputs["actions"], actions)
    assert data_config.action_sequence_keys == ("actions",)
    assert data_config.video_backend == "pyav"
    assert config.data.base_config.prompt_from_task


@pytest.mark.parametrize("chw", [False, True])
@pytest.mark.parametrize("floating", [False, True])
def test_single_camera_inputs(chw, floating):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 0] = 255
    source_image = image.astype(np.float32) / 255 if floating else image
    if chw:
        source_image = np.moveaxis(source_image, -1, 0)
    inputs = pku_franka_policy.PkuFrankaInputs()(
        {"image": source_image, "state": np.zeros(7), "prompt": b"put the cube onto the plate"}
    )
    assert set(inputs["image"]) == {"base_0_rgb"}
    np.testing.assert_array_equal(inputs["image"]["base_0_rgb"], image)
    assert inputs["image"]["base_0_rgb"].dtype == np.uint8
    assert inputs["image_mask"]["base_0_rgb"]
    assert inputs["prompt"] == "put the cube onto the plate"
    assert "actions" not in inputs


def test_reject_joint_space_state():
    with pytest.raises(ValueError, match="7-D Cartesian state"):
        pku_franka_policy.PkuFrankaInputs()({"image": np.zeros((8, 12, 3)), "state": np.zeros(8)})


def test_reject_joint_space_actions():
    with pytest.raises(ValueError, match="action chunk"):
        pku_franka_policy.PkuFrankaInputs()(
            {"image": np.zeros((8, 12, 3)), "state": np.zeros(7), "actions": np.zeros((10, 8))}
        )
