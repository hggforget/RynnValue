import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openpi_client.franka_collect.client import prepare_observation


def observation():
    return {
        "observation.images.primary": np.full((8, 12, 3), [255, 0, 50], dtype=np.uint8),
        "observation.images.wrist": np.full((8, 12, 3), [0, 120, 255], dtype=np.uint8),
        "observation.state.cartesian": np.arange(6, dtype=np.float32),
        "observation.state.gripper": 0.25,
        "observation.state.joints": np.arange(7, dtype=np.float32),
    }


def test_observation_preserves_cameras_and_recorded_gripper_without_target_leakage():
    raw = observation()
    raw["observation.state.target"] = np.ones(6) * 999
    raw["observation.state"] = np.ones(20) * 999
    raw["action"] = np.ones((30, 7)) * 999
    prepared = prepare_observation(raw)
    assert set(prepared) == set(observation()) | {"prompt"}
    for key in ("observation.images.primary", "observation.images.wrist"):
        np.testing.assert_array_equal(prepared[key], raw[key])
    np.testing.assert_array_equal(prepared["observation.state.gripper"], [0.25])
    assert prepared["observation.state.joints"].shape == (7,)


def test_observation_round_trips_through_native_msgpack_protocol():
    from openpi_client import msgpack_numpy

    prepared = prepare_observation(observation())
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(prepared))
    assert decoded["prompt"] == prepared["prompt"]
    for key, value in prepared.items():
        if isinstance(value, np.ndarray):
            assert decoded[key].dtype == value.dtype
            np.testing.assert_array_equal(decoded[key], value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("observation.state.cartesian", np.zeros(7)),
        ("observation.state.gripper", [0.0, 1.0]),
        ("observation.state.joints", np.full(7, np.nan)),
        ("observation.images.wrist", np.zeros((8, 12), dtype=np.uint8)),
        ("observation.images.primary", np.zeros((8, 12, 3), dtype=np.float32)),
    ],
)
def test_invalid_observations_are_rejected(key, value):
    raw = observation()
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        prepare_observation(raw)


@pytest.mark.parametrize("module", ["openpi_client.franka_collect.client"])
def test_cli_help_is_available_without_loading_model_or_robot_dependencies(module):
    root = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
sys.argv = [sys.argv[1], '--help']
try:
    runpy.run_module(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not {'jax', 'torch', 'openpi', 'rclpy', 'crisp_gym'} & sys.modules.keys()
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, "-c", script, module],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
