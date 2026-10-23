"""Plan bounded Cartesian targets and execute them with the existing CRISP controller."""

import fcntl
import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ExecutionLimits:
    max_step_m: float = 0.005
    max_step_rad: float = 0.03

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("Execution limits must be finite and positive")


def pose_error(first, second):
    translation = float(np.linalg.norm(np.asarray(first)[:3] - np.asarray(second)[:3]))
    rotation = (
        Rotation.from_euler("xyz", first[3:6]) * Rotation.from_euler("xyz", second[3:6]).inv()
    )
    return translation, float(rotation.magnitude())


def plan_targets(actions, seed, representation, steps=10, limits=None, lower=None, upper=None):
    """Convert the selected chunk prefix to absolute targets, preserving base-frame deltas."""
    limits = limits or ExecutionLimits()
    actions = np.asarray(actions, dtype=np.float64)
    seed = np.asarray(seed, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Expected finite actions [T, 7]")
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError("Expected a finite measured/commanded seed pose [6]")
    if not 1 <= steps <= len(actions) or representation not in ("absolute", "relative"):
        raise ValueError("Invalid chunk prefix length or action representation")
    lower = np.full(3, -np.inf) if lower is None else np.asarray(lower)
    upper = np.full(3, np.inf) if upper is None else np.asarray(upper)
    desired_position = seed[:3].copy()
    desired_rotation = Rotation.from_euler("xyz", seed[3:])
    position, rotation = desired_position.copy(), desired_rotation
    targets = []
    for action in actions[:steps]:
        action_rotation = Rotation.from_euler("xyz", action[3:6])
        if representation == "relative":
            # Match env.step(): add each delta to the target actually committed last step.
            # Discard any part removed by limiting; a following zero action must hold position.
            desired_position = position + action[:3]
            desired_rotation = action_rotation * rotation
        else:
            desired_position, desired_rotation = action[:3], action_rotation
        delta = desired_position - position
        rotation_delta = (desired_rotation * rotation.inv()).as_rotvec()
        distance, angle = np.linalg.norm(delta), np.linalg.norm(rotation_delta)
        if np.any(desired_position < lower) or np.any(desired_position > upper):
            raise ValueError(
                f"Model target outside configured workspace: {desired_position.tolist()}"
            )
        position = position + delta * min(1, limits.max_step_m / max(distance, 1e-12))
        rotation = (
            Rotation.from_rotvec(rotation_delta * min(1, limits.max_step_rad / max(angle, 1e-12)))
            * rotation
        )
        targets.append(np.r_[position, rotation.as_euler("xyz"), np.clip(action[6], 0, 1)])
    return np.asarray(targets)


class ActionExecutor:
    """Own target publishing only when execution is explicitly enabled."""

    def __init__(
        self,
        reader,
        representation,
        steps=10,
        control_hz=30,
        dry_run=False,
        limits=None,
        *,
        control_gripper=True,
    ):
        if not 1 <= steps <= 30 or not np.isfinite(control_hz) or not 0 < control_hz <= 50:
            raise ValueError("Use 1..30 execution steps and a control frequency in (0, 50] Hz")
        self.reader, self.node = reader, reader.node
        self.config = reader.config
        self.representation, self.steps = representation, steps
        self.control_hz, self.dry_run = control_hz, dry_run
        self.control_gripper = control_gripper
        self.limits = limits or ExecutionLimits()
        self.last_target = None
        self.publisher = None
        self.gripper = None
        self.gripper_goal = None
        self.gripper_pending = False
        self.gripper_sent_at = None
        self.gripper_cancel_future = None
        self.gripper_cancel_requested = False
        self.closing = False
        self.last_gripper = None
        self.error = None
        self.started = False
        self.ownership = None
        self.topics = {
            "pose": self.node.resolve_topic_name(self.config.robot_config.target_pose_topic),
            "gripper": self.node.resolve_topic_name(self.config.gripper_config.command_topic),
        }
        # Use the same effective box as ManipulatorCartesianEnv.clip_position_for_safety().
        # Reconstructing it from min_z changes the semantics of this deployed config.
        box = self.config.safety_box
        self.lower = np.full(3, -np.inf) if box is None else np.asarray(box["lower"])
        self.upper = np.full(3, np.inf) if box is None else np.asarray(box["upper"])
        self.listing = None

    def _acquire_ownership(self):
        ident = hashlib.sha256(self.topics["pose"].encode()).hexdigest()[:16]
        path = Path(tempfile.gettempdir()) / f"openpi-franka-execution-{ident}.lock"
        self.ownership = path.open("a")
        try:
            fcntl.flock(self.ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.ownership.close()
            self.ownership = None
            raise RuntimeError(f"Another Pi0.5 executor owns {self.topics['pose']}") from None

    def _call(self, client, request):
        if not client.wait_for_service(timeout_sec=3):
            raise TimeoutError(f"Controller service unavailable: {client.srv_name}")
        future = client.call_async(request)
        self.reader.executor.spin_until_future_complete(future, timeout_sec=3)
        if not future.done():
            raise TimeoutError(f"Controller service timed out: {client.srv_name}")
        return future.result()

    def _prepare(self, measured):
        from control_msgs.action import GripperCommand
        from controller_manager_msgs.srv import ListControllers, SwitchController
        from geometry_msgs.msg import PoseStamped
        from rclpy.action import ActionClient

        self._acquire_ownership()
        if self.node.count_publishers(self.topics["pose"]):
            raise RuntimeError(
                f"Another command publisher owns {self.topics['pose']}; stop that client first"
            )
        if self.control_gripper and not self.config.gripper_config.use_gripper_command_action:
            raise ValueError("This Franka executor requires a GripperCommand action server")
        self.listing = self.node.create_client(
            ListControllers, "controller_manager/list_controllers"
        )
        switching = self.node.create_client(
            SwitchController, "controller_manager/switch_controller"
        )
        if self.control_gripper:
            self.gripper = ActionClient(self.node, GripperCommand, self.topics["gripper"])
            if not self.gripper.wait_for_server(timeout_sec=3):
                raise TimeoutError(f"Gripper action server unavailable: {self.topics['gripper']}")
        controllers = self._call(self.listing, ListControllers.Request()).controller
        required = self.config.robot_config.cartesian_impedance_controller_name
        if required not in {controller.name for controller in controllers}:
            raise RuntimeError(f"Cartesian controller is not loaded: {required}")
        command_controllers = {
            required,
            "gravity_compensation",
            "joint_trajectory_controller",
            self.config.robot_config.joint_trajectory_controller_name,
            self.config.robot_config.cartesian_admittance_controller_name,
        }
        active = {controller.name for controller in controllers if controller.state == "active"}
        unknown = [
            controller.name
            for controller in controllers
            if controller.state == "active"
            and controller.claimed_interfaces
            and controller.name not in command_controllers
        ]
        if unknown:
            raise RuntimeError(f"Unexpected active command controllers: {unknown}")
        self.publisher = self.node.create_publisher(PoseStamped, self.topics["pose"], 1)
        deadline = time.monotonic() + 3
        while not self.publisher.get_subscription_count():
            if time.monotonic() > deadline:
                raise TimeoutError(f"No controller subscribes to {self.topics['pose']}")
            self.reader.spin_for(0.05)
        measured = self.reader.get_sample("observation.state.cartesian")
        self._publish_pose(measured)
        self.last_target = measured.copy()
        self.started = True
        if required not in active:
            request = SwitchController.Request()
            request.activate_controllers = [required]
            request.deactivate_controllers = sorted((active & command_controllers) - {required})
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout.sec = 3
            if not self._call(switching, request).ok:
                raise RuntimeError(f"Could not activate {required}")
        self._check_controller()
        print(f"Execution ready: {required}, {self.topics}", flush=True)

    def _check_controller(self):
        from controller_manager_msgs.srv import ListControllers

        required = self.config.robot_config.cartesian_impedance_controller_name
        controllers = self._call(self.listing, ListControllers.Request()).controller
        if not any(c.name == required and c.state == "active" for c in controllers):
            raise RuntimeError(f"Cartesian controller is no longer active: {required}")

    def _publish_pose(self, pose):
        from geometry_msgs.msg import PoseStamped

        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite Cartesian pose [6] before ROS publishing")
        message = PoseStamped()
        message.header.frame_id = self.config.robot_config.base_frame
        message.header.stamp = self.node.get_clock().now().to_msg()
        # ROS double fields must hold Python floats. On this deployment, assigning
        # np.float32 is accepted but rclpy serializes its memory as a double, corrupting XYZ.
        message.pose.position.x, message.pose.position.y, message.pose.position.z = map(
            float, pose[:3]
        )
        quaternion = Rotation.from_euler("xyz", pose[3:6]).as_quat()
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = map(float, quaternion)
        self.publisher.publish(message)

    def _send_gripper(self, value):
        from control_msgs.action import GripperCommand

        self._check_gripper()
        if (
            self.gripper_pending
            or self.last_gripper is not None
            and abs(value - self.last_gripper) < 0.05
        ):
            return
        goal = GripperCommand.Goal()
        config = self.config.gripper_config
        goal.command.position = float(
            config.min_value + value * (config.max_value - config.min_value)
        )
        goal.command.max_effort = float(config.max_effort)
        self.gripper_pending = True
        self.gripper_sent_at = time.monotonic()

        def accepted(future):
            try:
                handle = future.result()
                if not handle.accepted:
                    raise RuntimeError("Gripper command was rejected")
                self.gripper_goal = handle
                self.last_gripper = value
                handle.get_result_async().add_done_callback(self._gripper_completed)
                if self.closing or self.gripper_cancel_requested:
                    self._cancel_gripper()
            except Exception as exc:
                self.gripper_pending = False
                self.error = exc

        try:
            self.gripper.send_goal_async(goal).add_done_callback(accepted)
        except Exception as exc:
            self.gripper_pending = False
            self.error = exc
            raise

    def _gripper_completed(self, future):
        from action_msgs.msg import GoalStatus

        try:
            response = future.result()
            canceled_by_client = (
                self.closing or self.gripper_cancel_requested
            ) and response.status == GoalStatus.STATUS_CANCELED
            if not canceled_by_client and (
                response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.reached_goal
            ):
                raise RuntimeError(
                    f"Gripper action failed: status={response.status}, "
                    f"reached_goal={response.result.reached_goal}"
                )
        except Exception as exc:
            self.error = exc
        finally:
            self.gripper_goal = None
            self.gripper_pending = False
            self.gripper_cancel_future = None
            self.gripper_cancel_requested = False

    def _check_gripper(self):
        if self.error:
            raise self.error
        if self.gripper_pending:
            timeout = 2 if self.gripper_goal is None else 10
            if time.monotonic() - self.gripper_sent_at > timeout:
                raise TimeoutError(f"Gripper action timed out after {timeout}s")

    def _cancel_gripper(self):
        if self.gripper_pending:
            self.gripper_cancel_requested = True
        if self.gripper_goal is not None and self.gripper_cancel_future is None:
            self.gripper_cancel_future = self.gripper_goal.cancel_goal_async()

    def hold(self):
        """Hold measured pose and stop unfinished gripper motion, retaining ownership."""
        if self.publisher is not None:
            measured = self.reader.get_sample("observation.state.cartesian")
            self._publish_pose(measured)
            self.last_target = measured.copy()
        self._cancel_gripper()
        deadline = time.monotonic() + 2
        while self.gripper_pending and time.monotonic() < deadline:
            self.reader.spin_for(0.02)
        if self.gripper_pending:
            raise TimeoutError("Gripper pause was not confirmed")
        self._check_gripper()
        self.last_gripper = None

    def _execute_targets(self, targets, *, control_hz=None, should_pause=None):
        frequency = self.control_hz if control_hz is None else control_hz
        executed = 0
        for target in targets:
            if should_pause is not None and should_pause():
                return executed, True
            start = time.monotonic()
            self._check_gripper()
            if self.node.count_publishers(self.topics["pose"]) > 1:
                raise RuntimeError("Another target publisher appeared during execution")
            self.reader.get_sample("observation.state.cartesian")
            self._publish_pose(target[:6])
            self.last_target = target[:6].copy()
            if self.control_gripper:
                self._send_gripper(float(target[6]))
            executed += 1
            self.reader.spin_for(max(0, 1 / frequency - (time.monotonic() - start)))
        self._check_gripper()
        return executed, False

    def reset_to_pose(self, target_pose, timeout=30.0, observation_timeout=10.0):
        """Open the gripper, then return to an absolute pose using measured arrival."""
        if not self.control_gripper:
            raise ValueError("Reset requires gripper control; omit --disable-gripper")
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (6,) or not np.isfinite(target_pose).all():
            raise ValueError("Expected a finite reset pose [6]")
        if any(not np.isfinite(value) or value <= 0 for value in (timeout, observation_timeout)):
            raise ValueError("Reset timeouts must be finite and positive")
        measured = self.reader.get_sample(
            "observation.state.cartesian", timeout=observation_timeout
        )
        limits = ExecutionLimits(max_step_m=0.003, max_step_rad=0.02)
        action = np.r_[target_pose, 1.0][None, :]
        preview = plan_targets(action, measured, "absolute", 1, limits, self.lower, self.upper)
        report = {
            "target_pose": target_pose.tolist(),
            "initial_pose": measured.tolist(),
            "dry_run": self.dry_run,
            "reached": False,
            "control_hz": 10,
            "planned_targets": preview.tolist(),
            "steps_executed": 0,
        }
        if self.dry_run:
            return report
        started = time.monotonic()
        deadline = started + timeout
        if self.publisher is None:
            self._prepare(measured)
        else:
            self.hold()
            self._check_controller()
        measured = self.reader.get_sample("observation.state.cartesian")
        self._execute_targets(np.r_[measured, 1.0][None, :], control_hz=10)
        while self.gripper_pending:
            self._check_gripper()
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out opening gripper for reset")
            self.reader.spin_for(0.02)
        while True:
            self._check_gripper()
            measured = self.reader.get_sample("observation.state.cartesian")
            distance, angle = pose_error(measured, target_pose)
            if distance <= 0.02 and angle <= np.deg2rad(2):
                report.update(
                    reached=True,
                    final_pose=measured.tolist(),
                    elapsed_s=round(time.monotonic() - started, 3),
                )
                return report
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out during reset: pose error {distance:.4f} m / {angle:.4f} rad"
                )
            self._check_controller()
            targets = plan_targets(
                action, self.last_target, "absolute", 1, limits, self.lower, self.upper
            )
            self._execute_targets(targets, control_hz=10)
            report["steps_executed"] += 1

    def execute(self, actions, *, should_pause=None):
        measured = self.reader.get_sample("observation.state.cartesian")
        seed = measured if self.last_target is None else self.last_target
        targets = plan_targets(
            actions, seed, self.representation, self.steps, self.limits, self.lower, self.upper
        )
        executed, paused = 0, False
        if not self.dry_run:
            if self.publisher is None:
                self._prepare(measured)
                targets = plan_targets(
                    actions,
                    self.last_target,
                    self.representation,
                    self.steps,
                    self.limits,
                    self.lower,
                    self.upper,
                )
            else:
                self._check_controller()
            executed, paused = self._execute_targets(targets, should_pause=should_pause)
        return {
            "dry_run": self.dry_run,
            "steps_executed": executed,
            "paused": paused,
            "planned_targets": targets.tolist(),
            "control_hz": self.control_hz,
            "gripper_commands_enabled": self.control_gripper,
            "topics": self.topics,
            "workspace_lower": [float(x) if np.isfinite(x) else None for x in self.lower],
            "workspace_upper": [float(x) if np.isfinite(x) else None for x in self.upper],
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.closing = True
        try:
            if self.started:
                try:
                    try:
                        hold = self.reader.get_sample("observation.state.cartesian")
                    except TimeoutError:
                        hold = self.last_target
                    if hold is not None:
                        self._publish_pose(hold)
                finally:
                    self._cancel_gripper()
                    deadline = time.monotonic() + 2
                    while self.gripper_pending and time.monotonic() < deadline:
                        self.reader.spin_for(0.02)
                    self.reader.spin_for(0.05)
                    if self.gripper_pending:
                        raise TimeoutError("Gripper stop was not confirmed before shutdown")
        finally:
            if self.gripper is not None:
                self.gripper.destroy()
            if self.ownership is not None:
                self.ownership.close()
        if exc_type is None and self.error is not None:
            raise self.error
