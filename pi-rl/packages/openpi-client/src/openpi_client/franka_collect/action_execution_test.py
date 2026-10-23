import sys
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi_client.franka_collect.action_execution import (
    ActionExecutor,
    ExecutionLimits,
    plan_targets,
    pose_error,
)


def test_relative_translation_is_in_base_frame_and_rotation_is_left_multiplied():
    seed = np.array([0.3, 0.1, 0.2, 0.25, -0.15, 0.4])
    actions = np.array([[0.002, -0.001, 0.003, 0.01, -0.02, 0.015, 1.02]] * 2)
    limits = ExecutionLimits(max_step_m=1, max_step_rad=1)
    targets = plan_targets(actions, seed, "relative", 2, limits)
    np.testing.assert_allclose(targets[-1, :3], seed[:3] + 2 * actions[0, :3])
    expected = Rotation.from_euler("xyz", actions[0, 3:6]) ** 2 * Rotation.from_euler(
        "xyz", seed[3:]
    )
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", targets[-1, 3:6]).as_matrix(), expected.as_matrix()
    )
    np.testing.assert_array_equal(targets[:, 6], [1, 1])


def test_absolute_targets_do_not_accumulate_and_equivalent_euler_angles_do_not_jump():
    seed = np.array([0.3, 0.1, 0.2, np.pi - 0.001, 0, 0])
    action = np.r_[seed[:3] + [0.001, 0, 0], -np.pi + 0.001, 0, 0, 0.5]
    targets = plan_targets(np.tile(action, (3, 1)), seed, "absolute", 3)
    np.testing.assert_allclose(targets[:, :3], np.tile(action[:3], (3, 1)))
    assert pose_error(targets[0, :6], seed)[1] == pytest.approx(0.002)
    np.testing.assert_allclose(targets[0], targets[2])


def test_targets_are_limited_per_step():
    seed = np.array([0.3, 0.1, 0.2, 0, 0, 0])
    actions = np.tile([0.35, 0.14, 0.22, 0.15, 0, 0, -0.1], (4, 1))
    targets = plan_targets(actions, seed, "absolute", 4)
    previous = seed
    for target in targets:
        distance, angle = pose_error(target[:6], previous)
        assert distance <= 0.005 + 1e-9 and angle <= 0.03 + 1e-9
        previous = target[:6]
    np.testing.assert_array_equal(targets[:, 6], 0)


def test_zero_relative_action_holds_the_limited_target_without_catching_up():
    seed = np.array([0.3, 0.1, 0.2, 0, 0, 0])
    actions = np.array([[0.02, 0, 0, 0, 0, 0.1, 1], [0, 0, 0, 0, 0, 0, 1]])
    targets = plan_targets(actions, seed, "relative", 2)
    assert targets[0, 0] == pytest.approx(0.305)
    assert targets[0, 5] == pytest.approx(0.03)
    np.testing.assert_allclose(targets[0], targets[1])


@pytest.mark.parametrize("case", ["nan", "wrong_shape", "far_goal", "workspace", "bad_steps"])
def test_bad_chunks_are_rejected_before_execution(case):
    seed = np.array([0.3, 0.1, 0.2, 0, 0, 0])
    actions = np.tile(np.r_[seed, 1.0], (3, 1))
    options = {"steps": 3}
    if case == "nan":
        actions[2, 0] = np.nan
    elif case == "wrong_shape":
        actions = actions[:, :6]
    elif case == "far_goal":
        actions[2, 0] += 0.5
    elif case == "workspace":
        options["upper"] = [0.2, 1, 1]
    else:
        options["steps"] = 4
    with pytest.raises(ValueError):
        plan_targets(actions, seed, "absolute", **options)


def test_dry_run_never_creates_publishers_or_changes_the_virtual_robot_target():
    seed = np.array([0.3, 0.1, 0.2, 0, 0, 0])
    config = SimpleNamespace(
        robot_config=SimpleNamespace(target_pose_topic="target_pose"),
        gripper_config=SimpleNamespace(command_topic="gripper_action"),
        safety_box=None,
    )
    reader = SimpleNamespace(
        node=SimpleNamespace(resolve_topic_name=lambda name: "/test/" + name),
        config=config,
        get_sample=lambda key: seed.copy(),
    )
    with ActionExecutor(reader, "relative", steps=3, dry_run=True) as executor:
        report = executor.execute(np.tile([0.001, 0, 0, 0, 0, 0, 1], (30, 1)))
        assert report["dry_run"] and report["steps_executed"] == 0
        assert executor.publisher is None and executor.last_target is None
        assert len(report["planned_targets"]) == 3


def test_execution_ownership_excludes_a_second_client(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    first = object.__new__(ActionExecutor)
    second = object.__new__(ActionExecutor)
    first.topics = second.topics = {"pose": "/test/target_pose"}
    first._acquire_ownership()
    try:
        with pytest.raises(RuntimeError, match="Another Pi0.5 executor"):
            second._acquire_ownership()
    finally:
        first.ownership.close()
    second._acquire_ownership()
    second.ownership.close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pose_publisher_uses_python_floats_in_ros_double_fields(dtype, monkeypatch):
    messages = ModuleType("geometry_msgs.msg")
    messages.PoseStamped = lambda: SimpleNamespace(
        header=SimpleNamespace(),
        pose=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", messages)
    executor = object.__new__(ActionExecutor)
    executor.config = SimpleNamespace(robot_config=SimpleNamespace(base_frame="base"))
    executor.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
    )
    sent = []
    executor.publisher = SimpleNamespace(publish=sent.append)
    pose = np.array([0.35, 0.03, 0.23, 3.12, 0.01, -0.02], dtype=dtype)
    executor._publish_pose(pose)
    message = sent[0]
    position = vars(message.pose.position)
    assert all(type(value) is float for value in position.values())
    assert all(type(value) is float for value in vars(message.pose.orientation).values())
    np.testing.assert_array_equal(list(position.values()), pose[:3].astype(np.float64))
    assert message.header.frame_id == "base" and message.header.stamp == "stamp"


class FakeGoalHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancels = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancels += 1
        return Future()


@pytest.fixture
def gripper_executor(monkeypatch):
    parent = ModuleType("control_msgs")
    actions = ModuleType("control_msgs.action")
    actions.GripperCommand = SimpleNamespace(
        Goal=lambda: SimpleNamespace(command=SimpleNamespace())
    )
    parent.action = actions
    monkeypatch.setitem(sys.modules, "control_msgs", parent)
    monkeypatch.setitem(sys.modules, "control_msgs.action", actions)
    parent = ModuleType("action_msgs")
    messages = ModuleType("action_msgs.msg")
    messages.GoalStatus = SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_CANCELED=5)
    parent.msg = messages
    monkeypatch.setitem(sys.modules, "action_msgs", parent)
    monkeypatch.setitem(sys.modules, "action_msgs.msg", messages)
    config = SimpleNamespace(
        robot_config=SimpleNamespace(target_pose_topic="target_pose"),
        gripper_config=SimpleNamespace(
            command_topic="gripper_action", min_value=0.0, max_value=0.038, max_effort=20.0
        ),
        safety_box=None,
    )
    reader = SimpleNamespace(
        node=SimpleNamespace(resolve_topic_name=lambda name: "/test/" + name), config=config
    )
    executor = ActionExecutor(reader, "absolute", dry_run=True)
    sent = []

    def send(goal):
        future = Future()
        sent.append((goal, future))
        return future

    executor.gripper = SimpleNamespace(send_goal_async=send)
    return executor, sent


def test_gripper_serializes_complete_goals_not_just_acknowledgements(gripper_executor):
    executor, sent = gripper_executor
    executor._send_gripper(1.0)
    handle = FakeGoalHandle()
    sent[0][1].set_result(handle)
    executor._send_gripper(0.0)
    assert len(sent) == 1 and executor.gripper_pending
    assert sent[0][0].command.position == 0.038
    handle.result.set_result(SimpleNamespace(status=4, result=SimpleNamespace(reached_goal=True)))
    executor._send_gripper(0.0)
    assert len(sent) == 2 and sent[1][0].command.position == 0.0


@pytest.mark.parametrize("status", [4, 6])
def test_gripper_failure_stops_execution_even_when_goal_was_accepted(gripper_executor, status):
    executor, sent = gripper_executor
    executor._send_gripper(0.0)
    handle = FakeGoalHandle()
    sent[0][1].set_result(handle)
    handle.result.set_result(
        SimpleNamespace(status=status, result=SimpleNamespace(reached_goal=False))
    )
    with pytest.raises(RuntimeError, match="Gripper action failed"):
        executor._check_gripper()


def test_gripper_ack_timeout_is_bounded(gripper_executor):
    executor, sent = gripper_executor
    executor._send_gripper(0.0)
    executor.gripper_sent_at -= 3
    with pytest.raises(TimeoutError, match="2s"):
        executor._check_gripper()


def test_gripper_completion_timeout_is_bounded(gripper_executor):
    executor, sent = gripper_executor
    executor._send_gripper(0.0)
    sent[0][1].set_result(FakeGoalHandle())
    executor.gripper_sent_at -= 11
    with pytest.raises(TimeoutError, match="10s"):
        executor._check_gripper()


def test_gripper_rejection_stops_execution(gripper_executor):
    executor, sent = gripper_executor
    executor._send_gripper(0.0)
    sent[0][1].set_result(FakeGoalHandle(accepted=False))
    with pytest.raises(RuntimeError, match="was rejected"):
        executor._check_gripper()


def test_late_gripper_ack_is_canceled_during_shutdown(gripper_executor):
    executor, sent = gripper_executor
    executor._send_gripper(0.0)
    executor.closing = True
    handle = FakeGoalHandle()
    sent[0][1].set_result(handle)
    assert handle.cancels == 1
    handle.result.set_result(SimpleNamespace(status=5, result=SimpleNamespace(reached_goal=False)))
    assert not executor.gripper_pending and executor.error is None


def test_aborted_gripper_is_not_silently_treated_as_cancellation_on_exit(gripper_executor):
    executor, sent = gripper_executor
    executor._send_gripper(0.0)
    handle = FakeGoalHandle()
    sent[0][1].set_result(handle)
    executor.closing = True
    handle.result.set_result(SimpleNamespace(status=6, result=SimpleNamespace(reached_goal=False)))
    with pytest.raises(RuntimeError, match="Gripper action failed"):
        executor._check_gripper()


def test_failed_gripper_send_does_not_leave_a_permanent_pending_request(gripper_executor):
    executor, sent = gripper_executor

    def failed_send(goal):
        raise OSError("send failed")

    executor.gripper.send_goal_async = failed_send
    with pytest.raises(OSError, match="send failed"):
        executor._send_gripper(0.0)
    assert not executor.gripper_pending
