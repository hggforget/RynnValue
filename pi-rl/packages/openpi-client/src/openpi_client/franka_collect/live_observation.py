"""Read the collection observation schema from ROS without creating control publishers."""

import functools
import time

import numpy as np


class LiveObservationReader:
    """Subscribe to the same pose, joints, gripper and cameras used during collection."""

    def __init__(
        self,
        namespace="left",
        env_config="fr3_dual_cam_640x480",
        max_age=0.25,
        max_skew=0.1,
        *,
        include_cameras=True,
    ):
        import rclpy
        from crisp_gym.envs.manipulator_env_config import make_env_config
        from crisp_py.robot import Pose
        from geometry_msgs.msg import PoseStamped
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, JointState

        config = make_env_config(env_config)
        self.config = config
        self.joint_names = config.robot_config.joint_names
        self.gripper_config = config.gripper_config
        self.max_age = max_age
        self.max_skew = max_skew
        self._pose_type = Pose
        self._samples = {}
        self._errors = {}
        self._last_returned = 0.0

        def topic(name):
            return name if name.startswith("/") else f"/{namespace.strip('/')}/{name}"

        self.topics = {
            "observation.state.cartesian": topic(config.robot_config.current_pose_topic),
            "observation.state.joints": topic(config.robot_config.current_joint_topic),
            "observation.state.gripper": topic(config.gripper_config.joint_state_topic),
        }
        if include_cameras:
            cameras = {camera.camera_name: camera for camera in config.camera_configs}
            if not {"primary", "wrist"}.issubset(cameras):
                raise ValueError(
                    "Pi0.5 collect-objects inference requires primary and wrist cameras"
                )
            self.topics.update(
                {
                    f"observation.images.{name}": topic(cameras[name].camera_color_image_topic)
                    + "/compressed"
                    for name in ("primary", "wrist")
                }
            )
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node(
            "pi05_observation_reader", namespace=namespace, context=self.context
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for key, name in self.topics.items():
            message_type = (
                CompressedImage
                if ".images." in key
                else (PoseStamped if key.endswith("cartesian") else JointState)
            )
            self.node.create_subscription(
                message_type, name, functools.partial(self._receive, key), qos
            )

    def spin_for(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=min(0.02, max(0, deadline - time.monotonic())))

    def get_sample(self, key, timeout=0.25):
        """Read a fresh measured value while servicing control callbacks."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.005)
            for _ in self.topics:
                self.executor.spin_once(timeout_sec=0)
            sample = self._samples.get(key)
            if sample is not None:
                value, received, stamp = sample
                ros_now = self.node.get_clock().now().nanoseconds * 1e-9
                if (
                    stamp > 0
                    and time.monotonic() - received <= self.max_age
                    and 0 <= ros_now - stamp <= self.max_age
                ):
                    return value.copy()
        raise TimeoutError(f"No fresh control feedback: {key}")

    def _receive(self, key, message):
        try:
            if key.endswith("cartesian"):
                value = (
                    self._pose_type.from_ros_msg(message).to_pos_euler_array().astype(np.float32)
                )
            elif key.endswith("joints"):
                positions = dict(zip(message.name, message.position, strict=True))
                value = np.asarray([positions[name] for name in self.joint_names], dtype=np.float32)
            elif key.endswith("gripper"):
                config = self.gripper_config
                opening = np.clip(
                    (message.position[config.index] - config.min_value)
                    / (config.max_value - config.min_value),
                    0,
                    1,
                )
                # Preserve ManipulatorBaseEnv._get_obs(): the recorded state is 1 - gripper.value.
                value = np.asarray([1 - opening], dtype=np.float32)
            else:
                import cv2

                bgr = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Cannot decode compressed camera image")
                value = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite observation")
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self._samples[key] = (value, time.monotonic(), stamp)
            self._errors.pop(key, None)
        except Exception as exc:
            self._samples.pop(key, None)
            self._errors[key] = str(exc)

    def get_obs(self, timeout=10.0, *, should_stop=None):
        deadline = time.monotonic() + timeout
        fresh = {}
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                return None
            self.executor.spin_once(timeout_sec=min(0.03, max(0, deadline - time.monotonic())))
            # Drain queued sensor messages before taking a snapshot.
            for _ in self.topics:
                self.executor.spin_once(timeout_sec=0)
            now = time.monotonic()
            ros_now = self.node.get_clock().now().nanoseconds * 1e-9
            fresh = {}
            stamps = []
            for key, (value, received, stamp) in self._samples.items():
                if received <= self._last_returned:
                    continue
                # Camera transport latency does not gate inference. Require an image
                # update, but apply age and timestamp synchronization only to robot state.
                if ".images." not in key:
                    if now - received > self.max_age:
                        continue
                    if stamp <= 0 or not 0 <= ros_now - stamp <= self.max_age:
                        continue
                    stamps.append(stamp)
                fresh[key] = value
            if len(fresh) == len(self.topics) and max(stamps) - min(stamps) <= self.max_skew:
                self._last_returned = now
                return {key: value.copy() for key, value in fresh.items()}
        if should_stop is not None and should_stop():
            return None
        missing = [f"{key} ({topic})" for key, topic in self.topics.items() if key not in fresh]
        if not missing:
            missing = [f"robot state timestamps differ by more than {self.max_skew}s"]
        raise TimeoutError(
            f"Missing/stale ROS observations after {timeout}s: {missing}; errors: {self._errors}"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()
