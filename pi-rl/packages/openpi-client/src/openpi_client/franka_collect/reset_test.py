import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi_client.franka_collect.action_execution import ActionExecutor, pose_error

HOME = np.array([0.3550, 0.0277, 0.2356, -np.pi, 0, 0])


@pytest.fixture
def virtual_executor(monkeypatch):
    state = {"pose": HOME + [-0.015, 0, 0.012, 0.08, 0, 0], "follow": True, "clock": 0.0}
    events = []
    config = SimpleNamespace(
        robot_config=SimpleNamespace(target_pose_topic="target_pose"),
        gripper_config=SimpleNamespace(command_topic="gripper_action"),
        safety_box=None,
    )

    def spin(duration):
        state["clock"] += max(duration, 0.001)

    reader = SimpleNamespace(
        node=SimpleNamespace(
            resolve_topic_name=lambda name: "/test/" + name, count_publishers=lambda topic: 1
        ),
        config=config,
        get_sample=lambda key, **kwargs: state["pose"].copy(),
        spin_for=spin,
    )
    executor = ActionExecutor(reader, "relative", steps=10)

    def prepare(measured):
        events.append(("prepare", None))
        executor.publisher = object()
        executor.started = True
        executor.last_target = measured.copy()

    def publish(pose):
        events.append(("pose", np.array(pose)))
        if state["follow"]:
            state["pose"] = np.array(pose)

    monkeypatch.setattr(executor, "_prepare", prepare)
    monkeypatch.setattr(executor, "_check_controller", lambda: None)
    monkeypatch.setattr(executor, "_publish_pose", publish)
    monkeypatch.setattr(executor, "_send_gripper", lambda value: events.append(("gripper", value)))
    monkeypatch.setattr(
        "openpi_client.franka_collect.action_execution.time.monotonic", lambda: state["clock"]
    )
    return executor, state, events


def test_reset_opens_gripper_then_reaches_fixed_pose_with_bounded_steps(virtual_executor):
    executor, state, events = virtual_executor
    initial = state["pose"].copy()
    result = executor.reset_to_pose(HOME)
    assert result["reached"] and not result["dry_run"]
    assert pose_error(state["pose"], HOME)[0] <= 0.02
    assert pose_error(state["pose"], HOME)[1] <= np.deg2rad(2)
    first_open = next(
        i for i, (kind, value) in enumerate(events) if kind == "gripper" and value == 1
    )
    first_move = next(
        i
        for i, (kind, value) in enumerate(events)
        if kind == "pose" and np.linalg.norm(value[:3] - initial[:3]) > 1e-6
    )
    assert first_open < first_move
    previous = initial
    for kind, value in events:
        if kind == "pose":
            distance, angle = pose_error(value, previous)
            assert distance <= 0.003 + 1e-9 and angle <= 0.02 + 1e-9
            previous = value
    assert executor.representation == "relative" and executor.steps == 10


def test_reset_already_at_home_still_opens_gripper(virtual_executor):
    executor, state, events = virtual_executor
    state["pose"] = HOME.copy()
    assert executor.reset_to_pose(HOME)["reached"]
    assert ("gripper", 1.0) in [(kind, value) for kind, value in events if kind == "gripper"]


@pytest.mark.parametrize("distance,angle", [(0.0191, 0.0293), (0.0199, np.deg2rad(1.99))])
def test_reset_accepts_measured_pose_within_relaxed_tolerances(virtual_executor, distance, angle):
    executor, state, events = virtual_executor
    state["pose"] = HOME + [distance, 0, 0, angle, 0, 0]
    state["follow"] = False
    report = executor.reset_to_pose(HOME, timeout=0.5)
    assert report["reached"]
    np.testing.assert_allclose(report["final_pose"], state["pose"])
    assert ("gripper", 1.0) in [(kind, value) for kind, value in events if kind == "gripper"]


@pytest.mark.parametrize("distance,angle", [(0.0201, 0), (0, np.deg2rad(2.01))])
def test_reset_still_rejects_feedback_outside_either_tolerance(virtual_executor, distance, angle):
    executor, state, events = virtual_executor
    state["pose"] = HOME + [distance, 0, 0, angle, 0, 0]
    state["follow"] = False
    with pytest.raises(TimeoutError, match="pose error"):
        executor.reset_to_pose(HOME, timeout=0.5)


def test_reset_dry_run_does_not_acquire_control_or_move(virtual_executor):
    executor, state, events = virtual_executor
    executor.dry_run = True
    initial = state["pose"].copy()
    report = executor.reset_to_pose(HOME)
    assert report["dry_run"] and not report["reached"]
    assert events == [] and executor.publisher is None
    np.testing.assert_array_equal(state["pose"], initial)


def test_reset_refuses_disabled_gripper_before_any_motion(virtual_executor):
    executor, state, events = virtual_executor
    executor.control_gripper = False
    with pytest.raises(ValueError, match="gripper"):
        executor.reset_to_pose(HOME)
    assert events == []


def test_reset_stuck_feedback_times_out_instead_of_claiming_arrival(virtual_executor):
    executor, state, events = virtual_executor
    state["follow"] = False
    with pytest.raises(TimeoutError, match="reset"):
        executor.reset_to_pose(HOME, timeout=0.5)


@pytest.mark.parametrize(
    "target,timeout", [(np.zeros(5), 30), (np.full(6, np.nan), 30), (HOME, 0), (HOME, float("nan"))]
)
def test_invalid_reset_is_rejected_before_controlling(virtual_executor, target, timeout):
    executor, state, events = virtual_executor
    with pytest.raises(ValueError):
        executor.reset_to_pose(target, timeout=timeout)
    assert events == []


def test_pause_callback_stops_the_remaining_chunk(virtual_executor):
    executor, state, events = virtual_executor
    calls = []

    def should_pause():
        calls.append(1)
        return len(calls) == 4

    report = executor.execute(
        np.tile([0.001, 0, 0, 0, 0, 0, 1], (30, 1)), should_pause=should_pause
    )
    assert report["paused"] and report["steps_executed"] == 3
    assert len([value for kind, value in events if kind == "pose"]) == 3


def test_hold_keeps_measured_pose_and_reseeds_next_actions(virtual_executor):
    executor, state, events = virtual_executor
    executor.publisher, executor.started = object(), True
    executor.last_target = HOME.copy()
    executor.hold()
    np.testing.assert_array_equal(executor.last_target, state["pose"])
    np.testing.assert_array_equal(events[-1][1], state["pose"])


def test_reset_cli_help_does_not_import_robot_or_model():
    source = Path(__file__).resolve().parents[2]
    script = """
import runpy, sys
sys.argv = ['reset', '--help']
try:
    runpy.run_module('openpi_client.franka_collect.reset', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not {'rclpy', 'crisp_gym', 'torch', 'jax'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(source)},
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--execute" in result.stdout


@pytest.mark.parametrize(
    "settings,execute",
    [
        ({"DRY_RUN": "1"}, False),
        ({"DRY_RUN": "0", "EXECUTE": "1"}, True),
        ({"DRY_RUN": "0", "EXECUTE": "0"}, False),
    ],
)
def test_reset_shell_routes_to_local_reset_without_a_policy_server(tmp_path, settings, execute):
    root = Path(__file__).resolve().parents[5]
    runtime = tmp_path / "runtime.sh"
    runtime.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
    result = subprocess.run(
        ["bash", str(root / "examples/pku_franka/run_franka.sh"), "reset"],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "ROBOT_RUNTIME": str(runtime),
            **settings,
            "RUN_DIR": str(tmp_path / "result"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "openpi_client.franka_collect.reset" in result.stdout
    assert "--host" not in result.stdout and "--port" not in result.stdout
    assert ("--execute\n" in result.stdout) == execute
    assert ("openpi_client.franka_collect.test_supervisor" in result.stdout) == execute


def test_reset_uses_settings_edited_at_the_top_of_run_franka(tmp_path):
    root = Path(__file__).resolve().parents[5]
    scripts = tmp_path / "examples/pku_franka"
    scripts.mkdir(parents=True)
    launcher = (root / "examples/pku_franka/run_franka.sh").read_text()
    launcher = launcher.replace("DRY_RUN:-0", "DRY_RUN:-1").replace(
        "TARGET_NAMESPACE:-left", "TARGET_NAMESPACE:-custom_arm"
    )
    (scripts / "run_franka.sh").write_text(launcher)
    (scripts / "run_reset.sh").write_text((root / "examples/pku_franka/run_reset.sh").read_text())
    runtime = tmp_path / "runtime.sh"
    runtime.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DRY_RUN", "EXECUTE", "TARGET_NAMESPACE"}
    }
    result = subprocess.run(
        ["bash", str(scripts / "run_franka.sh"), "reset"],
        capture_output=True,
        text=True,
        timeout=10,
        env={**environment, "ROBOT_RUNTIME": str(runtime), "RUN_DIR": str(tmp_path / "result")},
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run-execution\n" in result.stdout and "--execute\n" not in result.stdout
    assert "--namespace\ncustom_arm\n" in result.stdout
