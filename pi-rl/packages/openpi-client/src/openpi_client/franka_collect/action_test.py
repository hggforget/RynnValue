"""Run on the inference host: send a small Z excursion through the existing Franka client."""

import argparse
import json
import math
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from openpi_client import msgpack_numpy
from openpi_client.franka_collect.action_execution import pose_error


def check_robot_service(robot_host):
    response = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            robot_host,
            "curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:8766/api/status",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    robot = json.loads(response.stdout).get("robot", {})
    if robot.get("running") is not True:
        raise RuntimeError(
            "Robot driver is stopped; start robot in data_collection_web before retrying. "
            "A failed/interrupted execution test stops this driver. No actions sent."
        )
    return robot


def create_run_directory(requested):
    directory = requested.resolve()
    if directory.exists() and any(directory.iterdir()):
        directory = directory / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class ScriptedActionPolicy:
    """Fixed hold → +Z → hold → return → hold, with independent measured feedback."""

    steps = 10
    settle_requests = 3
    hold_requests = 6

    def __init__(self, representation="absolute", distance_m=0.005, execute=False):
        if representation not in ("absolute", "relative") or not 0.003 <= distance_m <= 0.1:
            raise ValueError("Use absolute/relative actions and a Z excursion of 3..100 mm")
        self.representation, self.distance_m, self.execute = representation, distance_m, execute
        # Spread long excursions across chunks: never ask for more than 1 mm per step.
        self.ramp_requests = max(1, math.ceil(distance_m / (self.steps * 0.001)))
        self.phase_requests = self.ramp_requests + self.hold_requests - 1
        self.requests = self.settle_requests + 2 * self.phase_requests + 1
        self.start = None
        self.commanded = None
        self.records = []
        self.error = None

    @property
    def metadata(self):
        return {
            "config": "pi05_pku_franka_collect_objects"
            + ("_relative" if self.representation == "relative" else ""),
            "policy_type": "scripted_hardware_execution_test",
            "distance_m": self.distance_m,
            "gripper_commands_enabled": False,
        }

    def infer(self, observation):
        pose = np.asarray(observation["observation.state.cartesian"], dtype=np.float64)
        gripper = np.asarray(observation["observation.state.gripper"], dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all() or gripper.shape != (1,):
            raise ValueError("Invalid measured pose/gripper")
        if not np.isfinite(gripper).all():
            raise ValueError("Invalid measured gripper")
        index = len(self.records)
        if index >= self.requests:
            raise RuntimeError("Unexpected extra action request")
        if self.start is None:
            self.start = pose.copy()
            self.commanded = pose.copy()
        distance, rotation = pose_error(pose, self.start)
        if distance > self.distance_m + 0.008 or rotation > 0.08:
            raise RuntimeError(f"Measured pose left test envelope: {distance} m / {rotation} rad")
        out = self.settle_requests
        back = out + self.phase_requests
        if index < out:
            phase = "initial_hold"
        elif index < back:
            phase = "up"
        elif index < self.requests - 1:
            phase = "return"
        else:
            phase = "final_feedback"
        goal = self.start.copy()
        if phase == "up":
            goal[2] += self.distance_m * min((index - out + 1) / self.ramp_requests, 1)
        elif phase == "return":
            goal[2] += self.distance_m * max(1 - (index - back + 1) / self.ramp_requests, 0)
        actions = np.zeros((30, 7), dtype=np.float32)
        # State uses 1-opening; action uses opening. No gripper goals are sent in this test.
        actions[:, 6] = np.clip(1 - gripper[0], 0, 1)
        if self.representation == "absolute":
            actions[:, :6] = goal
            actions[: self.steps, :3] = np.linspace(self.commanded[:3], goal[:3], self.steps + 1)[
                1:
            ]
        else:
            actions[: self.steps, :3] = (goal[:3] - self.commanded[:3]) / self.steps
        self.records.append(
            {
                "index": index,
                "phase": phase,
                "time": time.time(),
                "measured_pose": pose.tolist(),
                "measured_gripper": float(gripper[0]),
                "target_pose": goal.tolist(),
                "actions": actions.tolist(),
            }
        )
        self.commanded = goal
        print(
            f"{index + 1}/{self.requests} {phase}: "
            f"measured delta mm={((pose[:3] - self.start[:3]) * 1000).round(3).tolist()}, "
            f"target delta mm={((goal[:3] - self.start[:3]) * 1000).round(3).tolist()}",
            flush=True,
        )
        return {"actions": actions}

    def assessment(self, client_report):
        if len(self.records) != self.requests:
            raise RuntimeError(f"Incomplete test: {len(self.records)}/{self.requests} observations")
        executions = [item["execution"] for item in client_report["requests"]]
        if len(executions) != self.requests:
            raise RuntimeError("Incomplete client execution report")
        if any(item["gripper_commands_enabled"] for item in executions):
            raise RuntimeError("The arm test must disable gripper commands")
        expected_steps = self.steps if self.execute else 0
        if any(item["steps_executed"] != expected_steps for item in executions):
            raise RuntimeError("Client did not execute the requested prefix")
        out = [r for r in self.records if r["phase"] == "up"][-3:]
        returned = self.records[-3:]
        goal = self.start.copy()
        goal[2] += self.distance_m
        out_error = max(pose_error(r["measured_pose"], goal)[0] for r in out)
        return_error = max(pose_error(r["measured_pose"], self.start)[0] for r in returned)
        peak_z = max(r["measured_pose"][2] - self.start[2] for r in self.records)
        rotation = max(pose_error(r["measured_pose"], self.start)[1] for r in self.records)
        gripper_drift = max(
            abs(r["measured_gripper"] - self.records[0]["measured_gripper"]) for r in self.records
        )
        result = {
            "status": "dry_run" if not self.execute else "failed",
            "requested_z_mm": self.distance_m * 1000,
            "measured_peak_z_mm": peak_z * 1000,
            "outbound_max_error_mm": out_error * 1000,
            "return_max_error_mm": return_error * 1000,
            "max_rotation_error_rad": rotation,
            "max_gripper_state_drift": gripper_drift,
            "steps_executed": sum(item["steps_executed"] for item in executions),
        }
        if self.execute:
            result["status"] = (
                "passed"
                if (
                    out_error <= 0.0015
                    and return_error <= 0.0015
                    and peak_z >= self.distance_m * 0.6
                    and rotation <= 0.05
                    and gripper_drift <= 0.05
                )
                else "failed"
            )
        return result


def run_client_session(policy, command, directory, host, port, timeout=100):
    """Serve native OpenPI messages while the unmodified inference loop runs in a subprocess."""
    directory.mkdir(parents=True, exist_ok=True)
    connected = threading.Event()
    connections = []

    def predictions(connection):
        if connected.is_set():
            connection.close(code=1008, reason="Only one test client is allowed")
            return
        connected.set()
        connections.append(connection)
        connection.send(msgpack_numpy.packb(policy.metadata))
        try:
            for message in connection:
                connection.send(msgpack_numpy.packb(policy.infer(msgpack_numpy.unpackb(message))))
        except ConnectionClosed:
            pass
        except Exception as exc:
            policy.error = str(exc)
            connection.close(code=1011, reason="Action test stopped; inspect the 4090 report")

    with serve(predictions, host, port, compression=None, max_size=None) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with (directory / "client.log").open("w") as log:
                arguments = (
                    command(server.socket.getsockname()[1]) if callable(command) else command
                )
                process = subprocess.Popen(
                    arguments,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )

                def copy_output():
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if "SUPERVISOR:" in line or line.startswith(
                            (
                                "Execution ready:",
                                "TimeoutError:",
                                "RuntimeError:",
                                "ValueError:",
                                "ImportError:",
                                "ModuleNotFoundError:",
                            )
                        ):
                            print(line.rstrip(), flush=True)

                output_worker = threading.Thread(target=copy_output, daemon=True)
                output_worker.start()
                try:
                    return_code = process.wait(timeout=timeout)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    output_worker.join(timeout=3)
            if policy.error:
                raise RuntimeError(policy.error)
            if return_code:
                raise RuntimeError(
                    f"Franka client exited with code {return_code}; see {directory / 'client.log'}"
                )
        finally:
            for connection in connections:
                connection.close(code=1001, reason="Action test finished or stopped")
            server.shutdown()
            worker.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Move the real arm +Z and back")
    mode.add_argument("--dry-run", action="store_true", help="Plan only (default)")
    parser.add_argument("--robot-host", default="pku-franka-hdc")
    parser.add_argument("--remote-root", default="/home/hdc/workspace/RynnValue/pi-rl")
    parser.add_argument("--host", default="219.223.195.77", help="4090 address reachable by Franka")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--namespace", default="left")
    parser.add_argument(
        "--action-representation", choices=["absolute", "relative"], default="absolute"
    )
    parser.add_argument("--distance-mm", type=float, default=5)
    parser.add_argument("--control-hz", type=float, default=10)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.distance_mm) or not 3 <= args.distance_mm <= 100:
        parser.error("--distance-mm must be finite and in [3, 100]")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if not np.isfinite(args.control_hz) or not 5 <= args.control_hz <= 30:
        parser.error("--control-hz must be in [5, 30]")
    if not 0 <= args.countdown <= 60:
        parser.error("--countdown must be in [0, 60]")
    if args.distance_mm > 10 and args.control_hz != 10:
        parser.error("Excursions above 10 mm require --control-hz 10 (at most 10 mm/s)")
    directory = args.output or Path(".runtime/pku_franka_collect_objects/results") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_action_test_" + args.action_representation
    )
    directory = create_run_directory(directory)
    policy = ScriptedActionPolicy(args.action_representation, args.distance_mm / 1000, args.execute)
    remote_report = str(
        Path(args.remote_root)
        / ".runtime/pku_franka_collect_objects/results"
        / (directory.name + "_client.json")
    )
    client = [
        "bash",
        str(
            Path(args.remote_root)
            / "examples/pku_franka"
            / ("run_supervised_test.sh" if args.execute else "run_inference.sh")
        ),
        "--mode",
        "inference",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--namespace",
        args.namespace,
        "--action-representation",
        args.action_representation,
        "--repeat",
        str(policy.requests),
        "--execute-steps",
        str(policy.steps),
        "--control-hz",
        str(args.control_hz),
        "--disable-gripper",
        "--report",
        remote_report,
        "--execute" if args.execute else "--dry-run-execution",
    ]
    # Remote timeout bounds robot ownership even if the SSH connection disappears.
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        args.robot_host,
        "exec " + shlex.join(["timeout", "--signal=INT", "--kill-after=5s", "90s", *client]),
    ]
    report = {
        "execute": args.execute,
        "robot_host": args.robot_host,
        "endpoint": f"ws://{args.host}:{args.port}",
        "namespace": args.namespace,
        "representation": args.action_representation,
        "client_command": client,
        "metadata": policy.metadata,
    }
    armed = False

    def stop_remote_robot():
        nonlocal armed
        if not armed:
            return
        armed = False
        print("STOP: requesting Franka robot driver shutdown", flush=True)
        response = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=3",
                args.robot_host,
                "curl --noproxy '*' --max-time 8 -fsS -X POST "
                "-H 'Content-Type: application/json' -d '{}' "
                "http://127.0.0.1:8766/api/service/robot/stop",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=12,
        )
        result = json.loads(response.stdout)
        if result.get("running") is not False:
            raise RuntimeError(f"Robot driver stop was not confirmed: {result}")
        report["robot_driver_stopped"] = True
        print("STOP: Franka robot driver stopped", flush=True)

    def interrupted(signum, _):
        stop_remote_robot()
        raise KeyboardInterrupt(f"Stopping action test on signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    print(
        f"REAL EXECUTION: {args.execute}; +Z {args.distance_mm} mm then return; gripper disabled",
        flush=True,
    )
    print(
        f"Ramp: {policy.ramp_requests} chunks each way, <=1 mm/step; "
        f"one way >= {policy.ramp_requests * policy.steps / args.control_hz:.1f}s",
        flush=True,
    )
    print(f"4090 test service: {report['endpoint']}; results: {directory}", flush=True)
    try:
        report["robot_service_before_test"] = check_robot_service(args.robot_host)
        if args.execute:
            print("Ctrl+C stops the test AND the robot driver", flush=True)
            for remaining in range(args.countdown, 0, -1):
                print(f"Starting real robot test in {remaining}s", flush=True)
                time.sleep(1)
            armed = True
        run_client_session(policy, command, directory, args.host, args.port)
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.robot_host, "cat " + shlex.quote(remote_report)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        client_report = json.loads(result.stdout)
        (directory / "client_report.json").write_text(json.dumps(client_report, indent=2) + "\n")
        report["assessment"] = policy.assessment(client_report)
        print(json.dumps(report["assessment"], indent=2), flush=True)
        if report["assessment"]["status"] == "failed":
            raise RuntimeError("Measured motion did not meet the execution test tolerances")
    except BaseException as exc:
        report["error"] = str(exc)
        stop_remote_robot()
        raise
    finally:
        report["records"] = policy.records
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
