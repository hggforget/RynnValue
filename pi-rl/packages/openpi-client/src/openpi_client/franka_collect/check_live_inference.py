"""Check live observation handling using recorded data on isolated ROS test topics."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, JointState

from openpi_client.franka_collect.client import load_dataset_observation
from openpi_client.franka_collect.live_observation import LiveObservationReader


def main():
    directory = Path(
        os.environ.get("PI05_RUNTIME_DIR", str(Path.cwd() / ".runtime/pku_franka_collect_objects"))
    )
    dataset = Path.home() / "workspace/DATA/local/fr3_collect_objects_20261017"
    observation = load_dataset_observation(dataset, 1, 0)
    namespace = f"pi05_observation_test_{os.getpid()}"
    with LiveObservationReader(namespace) as reader:
        node = rclpy.create_node("recorded_observation_fixture", context=reader.context)
        reader.executor.add_node(node)
        messages = {}
        pose = PoseStamped()
        state = observation["observation.state.cartesian"]
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, state[:3])
        quaternion = Rotation.from_euler("xyz", state[3:]).as_quat()
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = quaternion
        pose.header.frame_id = "base"
        messages["observation.state.cartesian"] = pose
        joints = JointState()
        joints.name = reader.joint_names[::-1]
        joints.position = list(map(float, observation["observation.state.joints"][::-1]))
        messages["observation.state.joints"] = joints
        gripper = JointState()
        config = reader.gripper_config
        gripper.position = [0.0] * (config.index + 1)
        recorded_gripper = float(
            np.asarray(observation["observation.state.gripper"]).reshape(-1)[0]
        )
        gripper.position[config.index] = float(
            config.min_value + (1 - recorded_gripper) * (config.max_value - config.min_value)
        )
        messages["observation.state.gripper"] = gripper
        for key in ("observation.images.primary", "observation.images.wrist"):
            message = CompressedImage()
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR))
            assert ok
            message.format = "jpeg"
            message.data = encoded.tobytes()
            messages[key] = message
        publishers = {
            key: node.create_publisher(type(message), reader.topics[key], qos_profile_sensor_data)
            for key, message in messages.items()
        }
        stale = [False]

        def publish():
            stamp_ns = node.get_clock().now().nanoseconds - (10_000_000_000 if stale[0] else 0)
            for key, message in messages.items():
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(
                    stamp_ns, 1_000_000_000
                )
                publishers[key].publish(message)

        timer = node.create_timer(0.05, publish)
        try:
            live = reader.get_obs(timeout=5)
            for key in (
                "observation.state.cartesian",
                "observation.state.joints",
                "observation.state.gripper",
            ):
                np.testing.assert_allclose(live[key], observation[key], atol=1e-6)
            for key in ("observation.images.primary", "observation.images.wrist"):
                assert live[key].shape == (480, 640, 3)
                assert live[key].dtype == np.uint8
            print(
                "PASS: ROS pose, named joint ordering, recorded gripper convention, dual RGB",
                flush=True,
            )
            stale[0] = True
            try:
                reader.get_obs(timeout=0.5)
            except TimeoutError:
                print("PASS: stale ROS timestamps rejected", flush=True)
            else:
                raise AssertionError("Accepted stale ROS observations")
            stale[0] = False
            worker = threading.Thread(target=reader.executor.spin, daemon=True)
            worker.start()
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "openpi_client.franka_collect.client",
                        "--mode",
                        "inference",
                        "--namespace",
                        namespace,
                        "--port",
                        "19080",
                        "--repeat",
                        "3",
                        "--output",
                        str(directory / "results/inference_ros_test_actions.npy"),
                        "--report",
                        str(directory / "results/inference_ros_test.json"),
                    ],
                    check=True,
                    timeout=30,
                )
            finally:
                reader.executor.shutdown()
                worker.join(timeout=5)
            report = json.loads((directory / "results/inference_ros_test.json").read_text())
            assert report["mode"] == "inference"
            assert len(report["requests"]) == 3
            print(
                "PASS: inference mode → native WebSocket → 4090 Pi0.5 → finite [30,7] actions",
                flush=True,
            )
        finally:
            timer.cancel()
            reader.executor.remove_node(node)
            node.destroy_node()


if __name__ == "__main__":
    main()
