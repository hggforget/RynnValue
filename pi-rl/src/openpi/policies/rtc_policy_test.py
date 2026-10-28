# ruff: noqa: SLF001
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest

from openpi import transforms
from openpi.models.rtc_test import TinyPi0
from openpi.policies import lingdong_policy
from openpi.policies import policy
from openpi.policies import rtc_policy
from openpi.shared.normalize import NormStats


def observation():
    return {
        "observation.state": np.arange(8, dtype=np.float32),
        lingdong_policy.HEAD_IMAGE_KEY: np.zeros((4, 4, 3), dtype=np.uint8),
        lingdong_policy.RIGHT_IMAGE_KEY: np.zeros((4, 4, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def policy_arguments():
    stats = {"actions": NormStats(mean=np.zeros(8), std=np.ones(8), q01=np.zeros(8), q99=np.full(8, 2))}
    return {
        "transforms": [
            lingdong_policy.LingdongRightArmInputs(),
            transforms.Normalize(stats, use_quantiles=True),
            transforms.PadStatesAndActions(32),
            transforms.TokenizePrompt(
                SimpleNamespace(tokenize=lambda prompt, state: (np.array([1], dtype=np.int32), np.array([True])))
            ),
        ],
        "output_transforms": [
            transforms.Unnormalize(stats, use_quantiles=True),
            lingdong_policy.LingdongRightArmOutputs(),
        ],
        "metadata": {"action_representation": "recorded_joint_targets", "action_dim": 8, "fps": 30},
        "sample_kwargs": {"num_steps": 3},
    }


def test_rtc_without_prefix_uses_original_sampler_with_identical_actions():
    model = TinyPi0()
    original = policy.Policy(model, **policy_arguments())
    optional = rtc_policy.RTCPolicy(model, **policy_arguments())
    optional._rtc_sampler = lambda *args, **kwargs: pytest.fail("Guided sampler must not run for a plain request")
    noise = np.ones((6, 32), dtype=np.float32)
    expected = original.infer(observation(), noise=noise)
    actual = optional.infer(observation(), noise=noise)
    np.testing.assert_array_equal(expected["actions"], actual["actions"])
    assert actual["rtc"]["guided"] is False
    assert "rtc" not in expected


def test_physical_prefix_uses_training_quantile_normalization_and_padding():
    optional = rtc_policy.RTCPolicy(TinyPi0(), **policy_arguments())
    captured = {}

    def sample(key, obs, **kwargs):
        del key, obs
        captured.update(kwargs)
        return kwargs["previous_actions"]

    optional._rtc_sampler = sample
    previous = np.full((2, 8), 0.5, dtype=np.float32)
    raw = observation()
    raw["rtc"] = {"previous_actions": previous, "inference_delay": 1}
    result = optional.infer(raw)
    prefix = np.asarray(captured["previous_actions"])
    assert prefix.shape == (1, 6, 32)
    np.testing.assert_allclose(prefix[:, :, :8], -0.5, atol=1e-6)
    np.testing.assert_array_equal(prefix[:, :, 8:], 0)
    np.testing.assert_allclose(result["actions"][:2], previous, atol=1e-6)
    assert int(captured["overlap_steps"]) == 2
    assert result["rtc"]["guided"] is True
    assert "action" not in raw


@pytest.mark.parametrize("delay", [-1, 3, 6, True, 1.5])
def test_invalid_or_uncovered_delays_fail_before_sampling(delay):
    optional = rtc_policy.RTCPolicy(TinyPi0(), **policy_arguments())
    optional._rtc_sampler = lambda *args, **kwargs: pytest.fail("Invalid request reached sampler")
    raw = observation()
    raw["rtc"] = {"previous_actions": np.ones((2, 8)), "inference_delay": delay}
    with pytest.raises(ValueError, match="inference_delay"):
        optional.infer(raw)


def test_actual_guided_policy_returns_finite_physical_actions():
    optional = rtc_policy.RTCPolicy(TinyPi0(), **policy_arguments())
    raw = observation()
    raw["rtc"] = {"previous_actions": np.ones((3, 8)), "inference_delay": 1}
    result = optional.infer(raw, noise=np.zeros((6, 32), dtype=np.float32))
    assert result["actions"].shape == (6, 8)
    assert np.isfinite(result["actions"]).all()
    assert result["policy_timing"]["infer_ms"] > 0
    assert optional.metadata["inference_mode"] == "rtc"


def test_rtc_request_and_response_round_trip_existing_websocket_codec():
    optional = rtc_policy.RTCPolicy(TinyPi0(), **policy_arguments())
    raw = observation()
    raw["rtc"] = {"previous_actions": np.ones((3, 8), dtype=np.float32), "inference_delay": 1}
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(raw))
    result = optional.infer(decoded, noise=np.zeros((6, 32), dtype=np.float32))
    received = msgpack_numpy.unpackb(msgpack_numpy.packb(result))
    assert received["actions"].shape == (6, 8)
    assert received["rtc"] == {"guided": True, "inference_delay": 1, "overlap_steps": 3}
    np.testing.assert_array_equal(received["actions"], result["actions"])
