from types import SimpleNamespace

import numpy as np
import pytest

from openpi_client.franka_collect.live_observation import LiveObservationReader


@pytest.fixture
def reader(monkeypatch):
    clock = [100.0]

    def monotonic():
        clock[0] += 0.001
        return clock[0]

    monkeypatch.setattr("openpi_client.franka_collect.live_observation.time.monotonic", monotonic)
    instance = object.__new__(LiveObservationReader)
    instance.max_age, instance.max_skew = 0.25, 0.1
    instance._last_returned = 98.0
    instance._errors = {}
    instance.topics = {
        key: "/test/" + key
        for key in (
            "observation.state.cartesian",
            "observation.state.joints",
            "observation.state.gripper",
            "observation.images.primary",
            "observation.images.wrist",
        )
    }
    instance._samples = {
        key: (np.array([index]), 99.99, 99.99) for index, key in enumerate(instance.topics)
    }
    instance.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)),
    )
    instance.executor = SimpleNamespace(spin_once=lambda **kwargs: None)
    return instance


@pytest.mark.parametrize(
    "stamp,received",
    [
        (0.0, 99.99),
        (50.0, 99.99),
        (200.0, 99.99),
        (99.8, 99.99),
        (99.99, 99.0),
    ],
)
def test_camera_delay_and_timestamp_skew_do_not_block_observations(reader, stamp, received):
    reader._samples["observation.images.primary"] = (np.array([3]), received, stamp)
    observation = reader.get_obs(timeout=0.05)
    assert set(observation) == set(reader.topics)
    np.testing.assert_array_equal(observation["observation.images.primary"], [3])


def test_stale_robot_feedback_is_still_rejected(reader):
    reader._samples["observation.state.cartesian"] = (np.array([0]), 99.99, 99.0)
    with pytest.raises(TimeoutError, match="observation.state.cartesian"):
        reader.get_obs(timeout=0.05)


def test_missing_camera_input_is_still_reported(reader):
    reader._samples.pop("observation.images.primary")
    with pytest.raises(TimeoutError, match="observation.images.primary"):
        reader.get_obs(timeout=0.05)


def test_operator_input_can_interrupt_waiting_for_observations(reader):
    reader._samples.pop("observation.images.primary")
    assert reader.get_obs(timeout=0.05, should_stop=lambda: True) is None
