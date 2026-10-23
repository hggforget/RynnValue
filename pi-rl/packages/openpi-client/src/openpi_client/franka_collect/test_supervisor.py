"""Supervise a hardware test and stop the managed robot driver on faults or interruption."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def hardware_fault(message, started_ns):
    stamp_ns = message.stamp.sec * 1_000_000_000 + message.stamp.nanosec
    name = message.name.lower()
    return (
        stamp_ns >= started_ns
        and message.level >= 40
        and ("frankahardwareinterface" in name or name.endswith("controller_manager"))
    )


def stop_robot(url):
    """Use the existing web service to stop the entire driver, including auto recovery."""
    request = urllib.request.Request(
        url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=8) as response:
        result = json.load(response)
    if result.get("name") != "robot" or result.get("running") is not False:
        raise RuntimeError(f"Robot driver stop was not confirmed: {result}")
    print("SUPERVISOR: robot driver stopped", flush=True)
    return result


def main():
    import rclpy
    from rcl_interfaces.msg import Log
    from rclpy.qos import QoSProfile
    from rclpy.signals import SignalHandlerOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-url", default="http://127.0.0.1:8766/api/service/robot/stop")
    parser.add_argument("--rosout-topic", default="/rosout")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("client_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.client_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or not 0 < args.timeout <= 120:
        parser.error("Provide client arguments after -- and a timeout in (0, 120]")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("pi05_hardware_test_supervisor")
    started_ns = node.get_clock().now().nanoseconds
    reason = []

    def received(message):
        if hardware_fault(message, started_ns) and not reason:
            reason.append(f"Hardware error: {message.name}: {message.msg}")

    def interrupted(signum, _):
        if not reason:
            reason.append(f"Interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    node.create_subscription(Log, args.rosout_topic, received, QoSProfile(depth=100))
    process = None
    failed = False
    try:
        # Establish ROS discovery before the child can activate the arm.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not reason:
            rclpy.spin_once(node, timeout_sec=0.02)
        if reason:
            raise RuntimeError(reason[0])
        print("SUPERVISOR: hardware error monitor ready; Ctrl+C stops the robot driver", flush=True)
        process = subprocess.Popen(
            [sys.executable, "-m", "openpi_client.franka_collect.client", *arguments],
            start_new_session=True,
        )
        deadline = time.monotonic() + args.timeout
        while process.poll() is None and not reason:
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() >= deadline:
                reason.append("Hardware test timed out")
        if reason:
            raise RuntimeError(reason[0])
        if process.returncode:
            raise RuntimeError(f"Test client exited with code {process.returncode}")
    except BaseException as exc:
        failed = True
        print(f"SUPERVISOR: STOP: {exc}", flush=True)
        # Stop the driver before waiting on child cleanup. Closing the WebSocket
        # alone does not stop a controller or its automatic error-recovery loop.
        stop_robot(args.stop_url)
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        node.destroy_node()
        rclpy.shutdown()
        if not failed:
            print("SUPERVISOR: client completed normally", flush=True)


if __name__ == "__main__":
    main()
