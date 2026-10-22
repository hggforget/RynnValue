"""Request Pi0.5 actions using replayed dataset observations or live ROS observations."""

import argparse
import contextlib
import json
import pathlib
import time

import numpy as np

PROMPT = "place each object on the plate that matches its color."
STATE_DIMS = {
    "observation.state.cartesian": 6,
    "observation.state.gripper": 1,
    "observation.state.joints": 7,
}
IMAGE_KEYS = ("observation.images.primary", "observation.images.wrist")


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19080)
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
        "--output", type=pathlib.Path, help="Save the last returned action chunk as NPY"
    )
    parser.add_argument(
        "--report", type=pathlib.Path, help="Save validation and timing information as JSON"
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.frame_step < 1 or args.observation_timeout <= 0:
        parser.error("--frame-step and --observation-timeout must be positive")
    if args.mode == "replay" and not (args.dataset or args.observation):
        parser.error("replay mode requires --dataset or --observation")
    if args.mode == "inference" and (args.dataset or args.observation):
        parser.error("inference mode reads ROS observations; omit --dataset and --observation")
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=args.host, port=args.port)
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
        "mode": args.mode,
        "action_representation": args.action_representation,
        "dataset": str(args.dataset) if args.dataset else None,
        "observation_file": str(args.observation) if args.observation else None,
        "episode": args.episode if args.dataset else None,
        "frame": args.frame if args.dataset else None,
        "requests": [],
    }
    with contextlib.ExitStack() as stack:
        if args.mode == "inference":
            from openpi_client.franka_collect.live_observation import LiveObservationReader

            reader = stack.enter_context(LiveObservationReader(args.namespace, args.env_config))
            report["topics"] = reader.topics
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
            report["requests"].append(timing)
            print(json.dumps(timing), flush=True)
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
