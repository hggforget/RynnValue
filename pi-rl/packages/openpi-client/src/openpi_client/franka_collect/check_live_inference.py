"""Check live observation handling using recorded data on isolated ROS test topics."""

import argparse
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


def check_supervisor_paths(command, directory, node, namespace, faults):
    from rcl_interfaces.msg import Log
    from rclpy.qos import QoSProfile

    from openpi_client.franka_collect.action_test import ScriptedActionPolicy, run_client_session

    topic = f"/{namespace}/supervisor_fault"
    faults["supervisor_log_publisher"] = node.create_publisher(
        Log, topic, QoSProfile(depth=100)
    )
    try:
        for case in ("normal", "hardware_error", "interrupt"):
            faults["supervisor_error"] = case == "hardware_error"
            policy = ScriptedActionPolicy(execute=True)
            output = directory / "results" / f"supervisor_{case}"
            output.mkdir(parents=True, exist_ok=True)
            client_report = output / "client_report.json"

            def arguments(port):
                supervisor = [
                    sys.executable,
                    "-m",
                    "openpi_client.franka_collect.test_supervisor",
                    "--rosout-topic",
                    topic,
                    "--",
                    *command[3:],
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--disable-gripper",
                    "--execute-steps",
                    str(policy.steps),
                    "--repeat",
                    str(policy.requests),
                    "--report",
                    str(client_report),
                ]
                if case == "interrupt":
                    supervisor = ["timeout", "--signal=INT", "--kill-after=8s", "4s", *supervisor]
                return supervisor

            if case == "normal":
                run_client_session(policy, arguments, output, "127.0.0.1", 0)
                result = policy.assessment(json.loads(client_report.read_text()))
                assert result["status"] == "passed", result
            else:
                try:
                    run_client_session(policy, arguments, output, "127.0.0.1", 0)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Supervisor did not interrupt the client")
                log = (output / "client.log").read_text()
                expected = (
                    "Hardware error:" if case == "hardware_error" else "Interrupted by signal 2"
                )
                assert expected in log and "SUPERVISOR: stopping inference client:" in log, log
            print(f"PASS: isolated supervisor {case}", flush=True)
    finally:
        node.destroy_publisher(faults.pop("supervisor_log_publisher"))


def check_failure_paths(command, faults, directory):
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    from openpi_client import msgpack_numpy

    def predictions(connection):
        connection.send(msgpack_numpy.packb({"config": "pi05_pku_franka_collect_objects"}))
        try:
            for message in connection:
                observation = msgpack_numpy.unpackb(message)
                pose = np.asarray(observation["observation.state.cartesian"], dtype=np.float32)
                actions = np.tile(np.r_[pose, np.float32(1)], (30, 1))
                actions[:, 0] += 0.001
                connection.send(msgpack_numpy.packb({"actions": actions}))
        except ConnectionClosed:
            pass

    # Exercise error handling with fixed, nearby targets so model sampling cannot mask a fault.
    with serve(predictions, "127.0.0.1", 0, compression=None, max_size=None) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            arguments = command + [
                "--host",
                "127.0.0.1",
                "--port",
                str(server.socket.getsockname()[1]),
            ]
            for fault, expected in (
                ("gripper_abort", "Gripper action failed"),
                ("controller_drop", "Cartesian controller is no longer active"),
            ):
                faults[fault] = True
                faults["controller_reads"] = 0
                result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
                output = result.stdout + result.stderr
                assert result.returncode != 0 and expected in output, output
                (directory / f"results/{fault}.log").write_text(output)
                faults[fault] = False
                print(f"PASS: {fault} stops execution with an explicit error", flush=True)
        finally:
            server.shutdown()
            worker.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-execution", action="store_true", help="Test execution on isolated fake controllers"
    )
    parser.add_argument(
        "--check-failures", action="store_true", help="Inject isolated gripper/controller failures"
    )
    parser.add_argument(
        "--check-scripted-actions",
        action="store_true",
        help="Check deterministic up/return actions and measured motion on isolated controllers",
    )
    parser.add_argument("--check-supervisor", action="store_true")
    args = parser.parse_args()
    if args.check_failures and not args.with_execution:
        parser.error("--check-failures requires --with-execution")
    if args.check_scripted_actions and not args.with_execution:
        parser.error("--check-scripted-actions requires --with-execution")
    if args.check_supervisor and not args.with_execution:
        parser.error("--check-supervisor requires --with-execution")
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
        commands, gripper_goals = [], []
        controller_active = [False]
        faults = {"gripper_abort": False, "controller_drop": False, "controller_reads": 0}
        action_server = None
        if args.with_execution:
            from control_msgs.action import GripperCommand
            from controller_manager_msgs.msg import ControllerState
            from controller_manager_msgs.srv import ListControllers, SwitchController
            from rclpy.action import ActionServer

            def list_controllers(request, response):
                faults["controller_reads"] += 1
                state = ControllerState()
                state.name = "cartesian_impedance_controller"
                state.state = "active" if controller_active[0] else "inactive"
                if faults["controller_drop"] and faults["controller_reads"] >= 3:
                    state.state = "inactive"
                response.controller = [state]
                return response

            def switch_controllers(request, response):
                assert request.activate_controllers == ["cartesian_impedance_controller"]
                assert not request.deactivate_controllers
                assert request.activate_asap
                controller_active[0] = True
                response.ok = True
                return response

            def follow_target(message):
                assert message.header.frame_id == "base"
                commands.append(message)
                pose.pose = message.pose
                if faults.get("supervisor_error"):
                    from rcl_interfaces.msg import Log

                    faults["supervisor_error"] = False
                    log = Log()
                    log.stamp = node.get_clock().now().to_msg()
                    log.level = Log.ERROR
                    log.name = "FrankaHardwareInterface"
                    log.msg = "Injected hardware error on isolated test topics"
                    faults["supervisor_log_publisher"].publish(log)

            def move_gripper(handle):
                command = handle.request.command
                gripper_goals.append(command)
                result = GripperCommand.Result()
                result.position = command.position
                if faults["gripper_abort"]:
                    handle.abort()
                    result.reached_goal = False
                else:
                    gripper.position[config.index] = command.position
                    handle.succeed()
                    result.reached_goal = True
                return result

            node.create_service(
                ListControllers,
                f"/{namespace}/controller_manager/list_controllers",
                list_controllers,
            )
            node.create_service(
                SwitchController,
                f"/{namespace}/controller_manager/switch_controller",
                switch_controllers,
            )
            node.create_subscription(PoseStamped, f"/{namespace}/target_pose", follow_target, 10)
            action_server = ActionServer(
                node, GripperCommand, f"/{namespace}/franka_gripper/gripper_action", move_gripper
            )
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
                        "8000",
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
                if args.with_execution:
                    execution_command = [
                        sys.executable,
                        "-m",
                        "openpi_client.franka_collect.client",
                        "--mode",
                        "inference",
                        "--namespace",
                        namespace,
                        "--execute",
                        "--execute-steps",
                        "3",
                        "--repeat",
                        "2",
                    ]
                    subprocess.run(
                        execution_command
                        + ["--report", str(directory / "results/execution_ros_test.json")],
                        check=True,
                        timeout=30,
                    )
                    execution = json.loads(
                        (directory / "results/execution_ros_test.json").read_text()
                    )
                    assert execution["execution_enabled"]
                    assert all(
                        item["execution"]["steps_executed"] == 3 for item in execution["requests"]
                    )
                    assert controller_active[0] and len(commands) >= 8 and gripper_goals
                    for command in gripper_goals:
                        assert config.min_value <= command.position <= config.max_value
                        assert command.max_effort == config.max_effort
                    print(
                        "PASS: isolated activation, pose targets, gripper goals and hold on exit",
                        flush=True,
                    )
                    if args.check_scripted_actions:
                        from openpi_client.franka_collect.action_test import (
                            ScriptedActionPolicy,
                            run_client_session,
                        )

                        for representation in ("absolute", "relative"):
                            policy = ScriptedActionPolicy(representation, execute=True)
                            output = directory / "results" / f"scripted_{representation}"
                            output.mkdir(parents=True, exist_ok=True)
                            client_report = output / "client_report.json"
                            gripper_before = len(gripper_goals)

                            def command(port):
                                return [
                                    sys.executable,
                                    "-m",
                                    "openpi_client.franka_collect.client",
                                    "--mode",
                                    "inference",
                                    "--namespace",
                                    namespace,
                                    "--host",
                                    "127.0.0.1",
                                    "--port",
                                    str(port),
                                    "--action-representation",
                                    representation,
                                    "--execute",
                                    "--disable-gripper",
                                    "--execute-steps",
                                    str(policy.steps),
                                    "--repeat",
                                    str(policy.requests),
                                    "--report",
                                    str(client_report),
                                ]

                            run_client_session(policy, command, output, "127.0.0.1", 0)
                            assessment = policy.assessment(json.loads(client_report.read_text()))
                            assert assessment["status"] == "passed", assessment
                            assert len(gripper_goals) == gripper_before
                            (output / "report.json").write_text(
                                json.dumps(
                                    {
                                        "assessment": assessment,
                                        "records": policy.records,
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
                            print(
                                f"PASS: scripted {representation} moves up and returns; "
                                "no gripper goals",
                                flush=True,
                            )
                    if args.check_failures:
                        check_failure_paths(execution_command, faults, directory)
                    if args.check_supervisor:
                        check_supervisor_paths(
                            execution_command, directory, node, namespace, faults
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
            if action_server is not None:
                action_server.destroy()
            reader.executor.remove_node(node)
            node.destroy_node()


if __name__ == "__main__":
    main()
