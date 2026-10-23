"""Monitor hardware faults and bound the inference client lifetime."""

import argparse
import os
import signal
import subprocess
import sys
import time
from math import isfinite


def hardware_fault(message, started_ns):
    stamp_ns = message.stamp.sec * 1_000_000_000 + message.stamp.nanosec
    name = message.name.lower()
    return (
        stamp_ns >= started_ns
        and message.level >= 40
        and ("frankahardwareinterface" in name or name.endswith("controller_manager"))
    )


def read_pause_state(channel, paused):
    """Drain ordered child notifications; the most recent event defines the state."""
    while True:
        try:
            events = os.read(channel, 4096)
        except BlockingIOError:
            return paused
        if not events:
            return paused
        paused = events[-1:] == b"p"


def main():
    import rclpy
    from rcl_interfaces.msg import Log
    from rclpy.qos import QoSProfile
    from rclpy.signals import SignalHandlerOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rosout-topic", default="/rosout")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Optional active runtime limit in seconds; 0 disables it",
    )
    parser.add_argument(
        "--client-module",
        default="openpi_client.franka_collect.client",
        choices=("openpi_client.franka_collect.client", "openpi_client.franka_collect.reset"),
    )
    parser.add_argument("client_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.client_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or not isfinite(args.timeout) or args.timeout < 0:
        parser.error(
            "Provide client arguments after -- and a finite, nonnegative timeout (0 disables it)"
        )
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("pi05_hardware_test_supervisor")
    started_ns = node.get_clock().now().nanoseconds
    reason = []
    paused = False

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
    pause_read = pause_write = None
    failed = False
    try:
        # Establish ROS discovery before the child can activate the arm.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not reason:
            rclpy.spin_once(node, timeout_sec=0.02)
        if reason:
            raise RuntimeError(reason[0])
        print(
            "SUPERVISOR: hardware error monitor ready; Ctrl+C stops the inference client",
            flush=True,
        )
        pause_read, pause_write = os.pipe()
        os.set_blocking(pause_read, False)
        process = subprocess.Popen(
            [sys.executable, "-m", args.client_module, *arguments],
            start_new_session=True,
            env={**os.environ, "PI05_SUPERVISOR_FD": str(pause_write)},
            pass_fds=(pause_write,),
        )
        os.close(pause_write)
        pause_write = None
        remaining = args.timeout
        previous = time.monotonic()
        while process.poll() is None and not reason:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            paused = read_pause_state(pause_read, paused)
            if args.timeout > 0 and not paused:
                remaining -= now - previous
            previous = now
            if args.timeout > 0 and remaining <= 0:
                reason.append("Hardware test timed out")
        if reason:
            raise RuntimeError(reason[0])
        if process.returncode:
            raise RuntimeError(f"Test client exited with code {process.returncode}")
    except BaseException as exc:
        failed = True
        print(f"SUPERVISOR: stopping inference client: {exc}", flush=True)
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
        for channel in (pause_read, pause_write):
            if channel is not None:
                os.close(channel)
        if not failed:
            print("SUPERVISOR: client completed normally", flush=True)


if __name__ == "__main__":
    main()
