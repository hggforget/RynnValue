"""Request Pi0.5 actions using replayed dataset observations or live ROS observations."""

import argparse
import contextlib
import json
import os
import pathlib
import signal
import sys
import time
import urllib.parse

import numpy as np

PROMPT = "place each object on the plate that matches its color."
STATE_DIMS = {
    "observation.state.cartesian": 6,
    "observation.state.gripper": 1,
    "observation.state.joints": 7,
}
IMAGE_KEYS = ("observation.images.primary", "observation.images.wrist")


def create_client(host: str, port: int, *, retry=True):
    """Connect directly to the policy host, bypassing inherited HTTP/SOCKS proxies."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    hostname = urllib.parse.urlsplit(
        host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
    ).hostname
    if not hostname:
        raise ValueError(f"Invalid policy host: {host}")
    for name in ("NO_PROXY", "no_proxy"):
        previous = os.environ.get(name, "")
        os.environ[name] = f"{previous},{hostname}" if previous else hostname
    return WebsocketClientPolicy(host=host, port=port, retry=retry)


def prepare_observation(raw: dict, prompt: str = PROMPT) -> dict:
    """Keep the measured fields used in training, preserving the recorded gripper convention."""
    observation = {"prompt": prompt}
    for key, dimension in STATE_DIMS.items():
        value = np.asarray(raw[key], dtype=np.float32)
        if dimension == 1:
            value = value.reshape(-1)
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(
                f"Expected {key} to contain {dimension} finite values; got {value.shape}"
            )
        observation[key] = value
    for key in IMAGE_KEYS:
        value = np.asarray(raw[key])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"Expected {key} as HWC uint8 RGB; got {value.shape}, {value.dtype}")
        observation[key] = value
    return observation


def load_dataset_observation(root: pathlib.Path, episode: int, frame: int) -> dict:
    """Decode matching Parquet and video frame indices from a LeRobot v2 dataset."""
    return next(iter_dataset_observations(root, episode, frame, 1, 1))


def iter_dataset_observations(root: pathlib.Path, episode: int, frame: int, count: int, step: int):
    """Stream matching state and video frames; decode each video once."""
    import av
    import pyarrow.parquet as parquet

    info = json.loads((root / "meta/info.json").read_text())
    if not 0 <= episode < info["total_episodes"]:
        raise ValueError(f"Episode must be in [0, {info['total_episodes']})")
    indices = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
    table = parquet.read_table(root / info["data_path"].format(**indices), columns=list(STATE_DIMS))
    final_frame = frame + (count - 1) * step
    if not 0 <= frame <= final_frame < table.num_rows:
        raise ValueError(
            f"Requested frames {frame}..{final_frame}; episode has {table.num_rows} frames"
        )
    with contextlib.ExitStack() as stack:
        decoders = {}
        for key in IMAGE_KEYS:
            video = root / info["video_path"].format(video_key=key, **indices)
            container = stack.enter_context(av.open(str(video)))
            decoders[key] = container.decode(video=0)
        for index in range(final_frame + 1):
            try:
                images = {key: next(decoder) for key, decoder in decoders.items()}
            except StopIteration as exc:
                raise ValueError(f"Video ended before requested frame {index}") from exc
            if index >= frame and (index - frame) % step == 0:
                observation = table.slice(index, 1).to_pylist()[0]
                observation.update(
                    {key: image.to_ndarray(format="rgb24") for key, image in images.items()}
                )
                yield observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replay", "inference"], default="replay")
    parser.add_argument("--host", default="219.223.195.77")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--action-representation", choices=["absolute", "relative"], default="absolute"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=pathlib.Path, help="Local LeRobot v2 dataset root")
    source.add_argument(
        "--observation", type=pathlib.Path, help="NPZ with the five raw observation fields"
    )
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--frame-step", type=int, default=1, help="Frame interval between replay requests"
    )
    parser.add_argument("--namespace", default="left", help="ROS namespace in inference mode")
    parser.add_argument("--env-config", default="fr3_dual_cam_640x480")
    parser.add_argument("--observation-timeout", type=float, default=10.0)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--attempts",
        type=int,
        default=0,
        help="Operator-judged attempts; 0 uses the legacy request loop",
    )
    parser.add_argument(
        "--attempt-timeout",
        type=float,
        default=300,
        help="Maximum elapsed seconds per evaluation attempt",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute", action="store_true", help="Execute predictions on the real robot"
    )
    execution.add_argument(
        "--dry-run-execution",
        action="store_true",
        help="Plan bounded targets without publishing commands",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=10,
        help="Chunk prefix to execute before observing again",
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--disable-gripper",
        action="store_true",
        help="Execute arm targets without gripper commands",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, help="Save the last returned action chunk as NPY"
    )
    parser.add_argument(
        "--report", type=pathlib.Path, help="Save validation and timing information as JSON"
    )
    args = parser.parse_args()
    if args.attempts < 0:
        parser.error("--attempts must be nonnegative")
    if not np.isfinite(args.attempt_timeout) or args.attempt_timeout <= 0:
        parser.error("attempt timeout must be positive and finite")
    if args.attempts:
        if args.mode != "inference" or not args.execute:
            parser.error("evaluation requires live execution (--mode inference --execute)")
        if args.disable_gripper:
            parser.error("evaluation requires gripper control for reset; omit --disable-gripper")
        if not sys.stdin.isatty():
            parser.error("evaluation requires a terminal for s/f/r input; use SSH -tt")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.frame_step < 1 or args.observation_timeout <= 0:
        parser.error("--frame-step and --observation-timeout must be positive")
    if args.mode == "replay" and not (args.dataset or args.observation):
        parser.error("replay mode requires --dataset or --observation")
    if args.mode == "inference" and (args.dataset or args.observation):
        parser.error("inference mode reads ROS observations; omit --dataset and --observation")
    if (args.execute or args.dry_run_execution) and args.mode != "inference":
        parser.error("action execution requires live inference; replay never controls the robot")
    if (
        not 1 <= args.execute_steps <= 30
        or not np.isfinite(args.control_hz)
        or not 0 < args.control_hz <= 50
    ):
        parser.error("--execute-steps must be 1..30 and --control-hz must be in (0, 50]")
    if args.execute:

        def interrupted(signum, _):
            raise KeyboardInterrupt(f"Stopping robot execution on signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
    if args.attempts:
        from openpi_client.franka_collect.evaluation import run_evaluation

        run_evaluation(args)
        return
    client = create_client(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    expected = "pi05_pku_franka_collect_objects" + (
        "_relative" if args.action_representation == "relative" else ""
    )
    if metadata.get("config") != expected:
        raise ValueError(
            f"Expected server config {expected!r}, received {metadata.get('config')!r}"
        )
    report = {
        "server": metadata,
        "endpoint": {"host": args.host, "port": args.port},
        "mode": args.mode,
        "action_representation": args.action_representation,
        "dataset": str(args.dataset) if args.dataset else None,
        "observation_file": str(args.observation) if args.observation else None,
        "episode": args.episode if args.dataset else None,
        "frame": args.frame if args.dataset else None,
        "execution_enabled": args.execute,
        "execution_dry_run": args.dry_run_execution,
        "gripper_commands_enabled": not args.disable_gripper,
        "requests": [],
    }
    with contextlib.ExitStack() as stack:
        executor = None
        controls = None
        if args.mode == "inference":
            from openpi_client.franka_collect.live_observation import LiveObservationReader

            reader = stack.enter_context(LiveObservationReader(args.namespace, args.env_config))
            report["topics"] = reader.topics
            if args.execute or args.dry_run_execution:
                from openpi_client.franka_collect.action_execution import ActionExecutor

                executor = stack.enter_context(
                    ActionExecutor(
                        reader,
                        args.action_representation,
                        args.execute_steps,
                        args.control_hz,
                        dry_run=args.dry_run_execution,
                        control_gripper=not args.disable_gripper,
                    )
                )
                if args.execute:
                    from openpi_client.franka_collect.terminal_control import TerminalControl

                    controls = stack.enter_context(TerminalControl(reader))
            observations = (reader.get_obs(args.observation_timeout) for _ in range(args.repeat))
        elif args.dataset:
            observations = iter_dataset_observations(
                args.dataset.expanduser(), args.episode, args.frame, args.repeat, args.frame_step
            )
            stack.callback(observations.close)
        else:
            with np.load(args.observation.expanduser(), allow_pickle=False) as arrays:
                raw = dict(arrays)
            observations = (raw for _ in range(args.repeat))
        for index, raw in enumerate(observations):
            if controls is not None and controls.poll():
                controls.wait(executor)
                raw = reader.get_obs(args.observation_timeout)
            observation = prepare_observation(raw, args.prompt)
            report["inputs"] = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in observation.items()
                if isinstance(value, np.ndarray)
            }
            start = time.perf_counter()
            result = client.infer(observation)
            elapsed_ms = (time.perf_counter() - start) * 1000
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (30, 7) or not np.isfinite(actions).all():
                raise ValueError(
                    f"Invalid action chunk: expected finite [30,7], received {actions.shape}"
                )
            timing = {
                "round_trip_ms": round(elapsed_ms, 3),
                "shape": list(actions.shape),
                "finite": True,
            }
            if args.dataset:
                timing["frame"] = args.frame + index * args.frame_step
            timing["server_infer_ms"] = round(result.get("server_timing", {}).get("infer_ms", 0), 3)
            if executor is not None:
                timing["execution"] = executor.execute(
                    actions, should_pause=controls.poll if controls is not None else None
                )
            report["requests"].append(timing)
            print(json.dumps(timing), flush=True)
            if controls is not None and timing["execution"]["paused"]:
                controls.wait(executor)
    report["first_action"] = actions[0].tolist()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
