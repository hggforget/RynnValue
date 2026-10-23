from types import SimpleNamespace

import numpy as np
import pytest

from openpi_client.franka_collect.action_execution import plan_targets
from openpi_client.franka_collect.action_test import (
    ScriptedActionPolicy,
    check_robot_service,
    create_run_directory,
)


@pytest.mark.parametrize("distance_m", [0.005, 0.1])
@pytest.mark.parametrize("representation", ["absolute", "relative"])
@pytest.mark.parametrize("follow", [True, False])
def test_assessment_requires_measured_outbound_and_return_motion(
    representation, follow, distance_m
):
    policy = ScriptedActionPolicy(representation, distance_m=distance_m, execute=True)
    start = np.array([0.35, 0.03, 0.23, 3.1, 0.01, -0.02])
    measured, commanded = start.copy(), start.copy()
    executions = []
    for _ in range(policy.requests):
        actions = policy.infer(
            {
                "observation.state.cartesian": measured,
                "observation.state.gripper": np.array([0.3]),
            }
        )["actions"]
        assert actions.shape == (30, 7)
        targets = plan_targets(actions, commanded, representation, steps=policy.steps)
        trajectory = np.vstack([commanded[:3], targets[:, :3]])
        assert np.max(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) <= 0.0010001
        commanded = targets[-1, :6]
        if follow:
            measured = commanded.copy()
        executions.append(
            {
                "execution": {
                    "steps_executed": policy.steps,
                    "gripper_commands_enabled": False,
                }
            }
        )
    assessment = policy.assessment({"requests": executions})
    assert assessment["status"] == ("passed" if follow else "failed")
    if follow:
        assert assessment["measured_peak_z_mm"] == pytest.approx(distance_m * 1000, abs=0.001)
        np.testing.assert_allclose(measured, start, atol=1e-7)
        assert assessment["return_max_error_mm"] < 0.001


def test_dry_run_reports_no_physical_execution_and_incomplete_run_is_rejected():
    policy = ScriptedActionPolicy(execute=False)
    with pytest.raises(RuntimeError, match="Incomplete test"):
        policy.assessment({"requests": []})
    for _ in range(policy.requests):
        policy.infer(
            {
                "observation.state.cartesian": np.array([0.35, 0.03, 0.23, 3.1, 0, 0]),
                "observation.state.gripper": np.array([0.0]),
            }
        )
    report = {
        "requests": [
            {
                "execution": {
                    "steps_executed": 0,
                    "gripper_commands_enabled": False,
                }
            }
        ]
        * policy.requests
    }
    assert policy.assessment(report)["status"] == "dry_run"


def test_feedback_outside_small_envelope_stops_further_commands():
    policy = ScriptedActionPolicy(execute=True)
    pose = np.array([0.35, 0.03, 0.23, 3.1, 0, 0])
    policy.infer({"observation.state.cartesian": pose, "observation.state.gripper": np.array([0])})
    pose[0] += 0.02
    with pytest.raises(RuntimeError, match="left test envelope"):
        policy.infer(
            {
                "observation.state.cartesian": pose,
                "observation.state.gripper": np.array([0]),
            }
        )


@pytest.mark.parametrize("running", [True, False])
def test_preflight_requires_running_driver_before_arming(running, monkeypatch):
    result = SimpleNamespace(stdout='{"robot":{"running":' + str(running).lower() + "}}")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: result)
    if running:
        assert check_robot_service("test-host")["running"] is True
    else:
        with pytest.raises(RuntimeError, match="Robot driver is stopped.*No actions sent"):
            check_robot_service("test-host")


def test_repeated_output_directory_preserves_prior_reports(tmp_path):
    first = create_run_directory(tmp_path / "run")
    (first / "report.json").write_text("previous run")
    second = create_run_directory(tmp_path / "run")
    assert second != first and second.parent == first
    assert (first / "report.json").read_text() == "previous run"
