"""Return Franka to the fixed collection start without a policy server or cameras."""

import argparse
import json
import pathlib
import signal

import numpy as np

START_POSE = (0.3550, 0.0277, 0.2356, -np.pi, 0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="left")
    parser.add_argument("--env-config", default="fr3_dual_cam_640x480")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--observation-timeout", type=float, default=10.0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Move the robot to the fixed start")
    modes.add_argument("--dry-run-execution", action="store_true", help="Preview without commands")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    if any(
        not np.isfinite(value) or value <= 0 for value in (args.timeout, args.observation_timeout)
    ):
        parser.error("Timeouts must be finite and positive")

    def interrupted(signum, _):
        raise KeyboardInterrupt(f"Stopping reset on signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    from openpi_client.franka_collect.action_execution import ActionExecutor
    from openpi_client.franka_collect.live_observation import LiveObservationReader

    with LiveObservationReader(args.namespace, args.env_config, include_cameras=False) as reader:
        with ActionExecutor(reader, "absolute", dry_run=not args.execute) as executor:
            report = executor.reset_to_pose(START_POSE, args.timeout, args.observation_timeout)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
