"""Build a non-destructive, audited FR3 relative-action dataset collection.

Run with the project's pi-rl Python environment. Without --execute, only print
the conversion plan. New-source controller semantics remain explicitly unverified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

OLD = "fr3_collect_objects_20261017"
NAMES = (
    "fr3_drawer_multitask_90", "fr3_long", "fr3_long_sft_clean", "fr3_open",
    "fr3_pick", "fr3_pick_spacemouse", "fr3_put_on_plate_cotrain_0712",
    "fr3_put_on_plate_cotrain_0712-1", "fr3_put_on_plate_cotrain_0713",
    "fr3_put_on_plate_cotrain_0905", OLD,
)
IMAGE = "observation.images.primary"
STATE = "observation.state"
ACTION = "action.relative"
FPS = 5
FORMAT_VERSION = "fr3-relative-unified-v1"
SCHEMA = pa.schema([
    pa.field(STATE, pa.list_(pa.float32(), 7)),
    pa.field(ACTION, pa.list_(pa.float32(), 7)),
    pa.field("timestamp", pa.float32()),
    *[pa.field(key, pa.int64()) for key in (
        "frame_index", "episode_index", "index", "task_index",
        "source_frame_start", "source_frame_end",
    )],
    pa.field("action_duration_s", pa.float32()),
])


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def text_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_checked(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream, target.open("xb") as output:
        while block := stream.read(4 * 1024 * 1024):
            output.write(block)
            digest.update(block)
            size += len(block)
    if target.stat().st_size != size or checksum(target) != digest.hexdigest():
        raise RuntimeError(f"Copy verification failed: {target}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def convert_arrays(state, actions, stride, old):
    """Normalize state and compose consecutive base-frame target increments."""
    if state.ndim != 2 or state.shape[1] != 7 or actions.shape != state.shape:
        raise ValueError("Expected matching [frames, 7] state/action arrays")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Source contains nonfinite state/action values")
    starts = np.arange(0, len(state), stride)
    ends = np.minimum(starts + stride, len(state))
    normalized = state[starts].astype(np.float64, copy=True)
    normalized[:, 3:6] = Rotation.from_euler("xyz", normalized[:, 3:6]).as_euler("xyz")
    normalized[:, 6] = np.clip(1 - normalized[:, 6] if old else normalized[:, 6] / 0.08, 0, 1)
    output = np.empty((len(starts), 7), dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        chunk = actions[start:end]
        output[i, :3] = chunk[:, :3].sum(axis=0)
        rotation = Rotation.identity()
        for delta in Rotation.from_euler("xyz", chunk[:, 3:6]):
            rotation = delta * rotation
        output[i, 3:6] = rotation.as_euler("xyz")
        output[i, 6] = np.clip(chunk[-1, 6] if old else (chunk[-1, 6] + 1) / 2, 0, 1)
    return normalized.astype(np.float32), output.astype(np.float32), starts, ends


def stats_for(data):
    result = {}
    for key, values in data.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        result[key] = {
            "min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(), "std": array.std(axis=0).tolist(),
            "count": [len(array)],
        }
    return result


def fixed_features(videos):
    return {
        **videos,
        STATE: {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_opening"], "units": ["m"] * 3 + ["rad"] * 3 + ["normalized"], "gripper_convention": "0_closed_1_open"},
        ACTION: {"dtype": "float32", "shape": [7], "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_command"], "representation": "relative_delta_pose", "coordinate_frame": "robot_base", "rotation_representation": "euler_xyz", "units": ["m"] * 3 + ["rad"] * 3 + ["normalized"], "gripper_convention": "0_closed_1_open"},
        **{field.name: {"dtype": "float32" if pa.types.is_floating(field.type) else "int64", "shape": [1], "names": None} for field in SCHEMA if field.name not in (STATE, ACTION)},
    }


def prepare(source_root, destination):
    datasets, jobs = {}, []
    for name in NAMES:
        source = source_root / name
        info = json.loads((source / "meta/info.json").read_text())
        episodes = read_jsonl(source / "meta/episodes.jsonl")
        source_tasks = read_jsonl(source / "meta/tasks.jsonl")
        old = name == OLD
        source_fps = info["fps"]
        stride = round(source_fps / FPS)
        if stride < 1 or source_fps != FPS * stride:
            raise ValueError(f"Cannot exactly resample {name}: {source_fps} Hz")
        video_keys = {
            (IMAGE if key == "image" else key): key
            for key, feature in info["features"].items() if feature["dtype"] == "video"
        }
        videos = {}
        for target_key, source_key in video_keys.items():
            video = copy.deepcopy(info["features"][source_key])
            for key in ("info", "video_info"):
                video.setdefault(key, {})["video.fps"] = source_fps
            videos[target_key] = video
        canonical_tasks, task_map, text_indices = [], {}, {}
        for task in source_tasks:
            text = task["task"]
            if text.startswith("lace each object"):
                text = "p" + text
            if text not in text_indices:
                text_indices[text] = len(canonical_tasks)
                canonical_tasks.append({"task_index": len(canonical_tasks), "task": text})
            task_map[task["task_index"]] = text_indices[text]
        semantics = {
            "source": str(source), "source_fps": source_fps, "output_fps": FPS,
            "state_gripper_transform": "clip(1 - source_gripper, 0, 1)" if old else "clip(source_width_m / 0.08, 0, 1)",
            "action_gripper_transform": "identity" if old else "(source_command + 1) / 2",
            "relative_translation": "sum source base-frame deltas in each time interval",
            "relative_rotation": "left-compose source Euler XYZ delta rotations in time order",
            "action_gripper_resampling": "last command in each interval",
            "euler_state": "canonical extrinsic XYZ radians via scipy Rotation",
            "action_reference": "previous_control_target" if old else "source_recorded_relative_delta; reference_not_source_verified",
            "source_controller_semantics_verified": old,
            "unverified_items": [] if old else ["reference pose for recorded delta", "TCP/frame calibration", "rotation convention from collection source", "observation/action recording alignment"],
            "coordinate_frame_status": "source metadata and target trajectory verified" if old else "base-frame translation supported by motion correlation; source code unavailable",
            "video_storage": "source bitstream retained; query video by timestamp; encoded fps may differ from dataset fps",
            "image_statistics": "not copied; numeric episode statistics recomputed from output rows",
            "camera_keys": list(videos),
        }
        output_info = {
            "codebase_version": "v2.1", "robot_type": "franka", "total_episodes": len(episodes),
            "total_frames": sum((episode["length"] + stride - 1) // stride for episode in episodes),
            "total_tasks": len(canonical_tasks), "total_videos": len(episodes) * len(videos), "total_chunks": 1,
            "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": fixed_features(videos), "unification": semantics,
        }
        if old:
            output_info["auxiliary_progress_path"] = "annotations/progress/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        row_start = 0
        for episode in episodes:
            frames = (episode["length"] + stride - 1) // stride
            jobs.append({"name": name, "episode": episode, "row_start": row_start, "source": source, "destination": destination / name, "old": old, "source_fps": source_fps, "stride": stride, "video_keys": video_keys, "task_map": task_map, "tasks": canonical_tasks})
            row_start += frames
        datasets[name] = {"info": output_info, "source_info": info, "source_tasks": source_tasks, "source_episodes": episodes, "tasks": canonical_tasks}
    return datasets, jobs


def convert_episode(job):
    source, destination, episode = job["source"], job["destination"], job["episode"]
    episode_id = episode["episode_index"]
    chunk = episode_id // 1000
    file = f"episode_{episode_id:06d}"
    table = pq.read_table(source / f"data/chunk-{chunk:03d}/{file}.parquet")
    def array(key):
        return np.asarray(table[key].to_pylist())
    if job["old"]:
        state = np.column_stack((array("observation.state.cartesian"), array("observation.state.gripper").reshape(-1)))
        actions = array(ACTION)
    else:
        state, actions = array("state"), array("actions")
    if len(state) != episode["length"]:
        raise ValueError(f"Episode length changed: {job['name']} / {episode_id}")
    normalized, deltas, starts, ends = convert_arrays(state, actions, job["stride"], job["old"])
    count = len(starts)
    data = {
        STATE: normalized, ACTION: deltas,
        "timestamp": (np.arange(count) / FPS).astype(np.float32),
        "frame_index": np.arange(count, dtype=np.int64),
        "episode_index": np.full(count, episode_id, dtype=np.int64),
        "index": np.arange(job["row_start"], job["row_start"] + count, dtype=np.int64),
        "task_index": np.array([job["task_map"][int(value)] for value in array("task_index")[starts]], dtype=np.int64),
        "source_frame_start": starts.astype(np.int64), "source_frame_end": ends.astype(np.int64),
        "action_duration_s": ((ends-starts) / job["source_fps"]).astype(np.float32),
    }
    artifacts = {}
    with tempfile.TemporaryDirectory(prefix="fr3-unify-", dir="/tmp") as stage:
        staged = Path(stage) / "episode.parquet"
        output_table = pa.Table.from_pydict({key: value.tolist() for key, value in data.items()}, schema=SCHEMA)
        pq.write_table(output_table, staged, compression="zstd")
        relative = f"data/chunk-{chunk:03d}/{file}.parquet"
        artifacts[relative] = copy_checked(staged, destination / relative)
        if job["old"] and "progress" in table.column_names:
            annotation = {"frame_index": data["frame_index"], "timestamp": data["timestamp"]}
            for key in ("progress", "remaining_time_s"):
                if key in table.column_names:
                    annotation[key] = array(key)[starts].astype(np.float32)
            pq.write_table(pa.Table.from_pydict(annotation), staged, compression="zstd")
            relative = f"annotations/progress/chunk-{chunk:03d}/{file}.parquet"
            artifacts[relative] = copy_checked(staged, destination / relative)
    for target_key, source_key in job["video_keys"].items():
        relative = f"videos/chunk-{chunk:03d}/{target_key}/{file}.mp4"
        source_video = source / f"videos/chunk-{chunk:03d}/{source_key}/{file}.mp4"
        artifacts[relative] = copy_checked(source_video, destination / relative)
    output_episode = copy.deepcopy(episode)
    output_episode["length"] = count
    output_episode["source_length"] = episode["length"]
    output_episode["tasks"] = [job["tasks"][int(index)]["task"] for index in np.unique(data["task_index"])]
    output_episode["prompt"] = output_episode["tasks"][0] if len(output_episode["tasks"]) == 1 else " ".join(output_episode["tasks"])
    return {"name": job["name"], "episode": output_episode, "stats": {"episode_index": episode_id, "stats": stats_for(data)}, "artifacts": artifacts, "frames": count}


def replace_text(path, content):
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    text_write(temporary, content)
    temporary.replace(path)


def add_missing_cameras(root, workers, execute):
    """Copy optional source cameras into an existing conversion, keeping numeric data intact."""
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("conversion_complete"):
        raise ValueError("Finish the initial conversion before adding cameras")
    updated, jobs = {}, []
    for dataset in manifest["datasets_summary"]:
        name = dataset["name"]
        base = root / name
        info = json.loads((base / "meta/info.json").read_text())
        source_info = json.loads((base / "provenance/source_info.json").read_text())
        episodes = read_jsonl(base / "meta/episodes.jsonl")
        source = Path(info["unification"]["source"])
        for key, feature in source_info["features"].items():
            if feature["dtype"] != "video" or key in (IMAGE, "image"):
                continue
            video = copy.deepcopy(feature)
            for section in ("info", "video_info"):
                video.setdefault(section, {})["video.fps"] = source_info["fps"]
            info["features"][key] = video
            for episode in episodes:
                eid = episode["episode_index"]
                chunk = eid // info["chunks_size"]
                relative = info["video_path"].format(episode_chunk=chunk, episode_index=eid, video_key=key)
                source_relative = source_info["video_path"].format(
                    episode_chunk=eid // source_info["chunks_size"], episode_index=eid, video_key=key,
                )
                jobs.append((source / source_relative, base / relative, f"{name}/{relative}"))
        cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        info["total_videos"] = len(episodes) * len(cameras)
        info["unification"]["camera_keys"] = cameras
        updated[name] = info
    log(f"CAMERA PLAN: {len(jobs)} optional-camera videos, destination={root}")
    if not execute:
        return
    pending = root / "camera_update.pending.json"
    if pending.exists():
        raise RuntimeError(f"A previous camera update is pending; inspect {pending}")
    text_write(pending, json_text({"videos": len(jobs), "started": time.time()}))
    backup = root / "provenance" / f"camera_update_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    for relative in ("manifest.json", "SHA256SUMS", "convert.py", *(f"{name}/meta/info.json" for name in updated)):
        copy_checked(root / relative, backup / relative)
    checksums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(None, 1)
        checksums[relative] = digest
    added_bytes = 0

    def copy_camera(job):
        source, target, relative = job
        if target.exists():
            digest = checksum(source)
            if checksum(target) != digest:
                raise RuntimeError(f"Existing optional video differs: {target}")
            result = {"sha256": digest, "bytes": target.stat().st_size}
        else:
            result = copy_checked(source, target)
        return relative, result

    completed = 0
    last_report = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_camera, job) for job in jobs]
        for future in as_completed(futures):
            relative, artifact = future.result()
            if relative not in checksums:
                added_bytes += artifact["bytes"]
            checksums[relative] = artifact["sha256"]
            completed += 1
            if time.monotonic() - last_report > 10:
                status = {"videos_done": completed, "videos_total": len(jobs), "bytes_added": added_bytes}
                Path("/tmp/fr3_camera_update_status.json").write_text(json_text(status))
                log("CAMERA PROGRESS " + json.dumps(status))
                last_report = time.monotonic()
    for name, info in updated.items():
        replace_text(root / name / "meta/info.json", json_text(info))
    replace_text(root / "SHA256SUMS", "".join(f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items())))
    replace_text(root / "convert.py", Path(__file__).read_text())
    manifest["artifact_files_verified"] = len(checksums)
    manifest["bytes_copied"] += added_bytes
    manifest["camera_update_complete"] = True
    manifest["camera_input_policy"] = "primary required; wrist optional; missing wrist uses a false image mask"
    manifest["camera_metadata_backup"] = str(backup.relative_to(root))
    for dataset in manifest["datasets_summary"]:
        info = updated[dataset["name"]]
        dataset["camera_keys"] = info["unification"]["camera_keys"]
        dataset["total_videos"] = info["total_videos"]
    replace_text(root / "manifest.json", json_text(manifest))
    pending.unlink()
    log(f"CAMERA COMPLETE: {completed} optional videos verified, {added_bytes:,} bytes added")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/oss/embodied-ai-data/pku"))
    parser.add_argument("--output-root", type=Path, default=Path("/oss/embodied-ai-data/pku/fr3_unified_5hz_relative"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--add-missing-cameras", action="store_true")
    args = parser.parse_args()
    if args.add_missing_cameras:
        add_missing_cameras(args.output_root, args.workers, args.execute)
        return
    datasets, jobs = prepare(args.source_root, args.output_root)
    summary = {"format_version": FORMAT_VERSION, "output": str(args.output_root), "datasets": len(datasets), "episodes": len(jobs), "frames": sum(data["info"]["total_frames"] for data in datasets.values()), "fps": FPS, "action_key": ACTION, "source_controller_semantics_verified_for_all": False, "mixed_policy_training_ready": False, "unresolved": "New sources need collection code to verify delta reference/TCP/rotation/time alignment", "derived_overlap": {"fr3_long_sft_clean": "derived from fr3_long; avoid counting both as independent demos"}}
    log("PLAN " + json.dumps(summary, ensure_ascii=False))
    if not args.execute:
        return
    if args.output_root.exists():
        raise FileExistsError(f"Will not overwrite output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    text_write(args.output_root / "manifest.pending.json", json_text(summary))
    for name, dataset in datasets.items():
        base = args.output_root / name
        for filename, value in (("meta/info.json", dataset["info"]), ("provenance/source_info.json", dataset["source_info"])):
            text_write(base / filename, json_text(value))
        for filename, rows in (("meta/tasks.jsonl", dataset["tasks"]), ("provenance/source_tasks.jsonl", dataset["source_tasks"]), ("provenance/source_episodes.jsonl", dataset["source_episodes"])):
            text_write(base / filename, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    results, failures = [], []
    last_report = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_episode, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"dataset": job["name"], "episode": job["episode"]["episode_index"], "error": str(exc)})
                log("FAILED " + json.dumps(failures[-1], ensure_ascii=False))
            now = time.monotonic()
            if now-last_report > 10:
                status = {"episodes_done": len(results), "episodes_total": len(jobs), "frames_done": sum(row["frames"] for row in results), "failures": failures}
                Path("/tmp/fr3_unification_status.json").write_text(json_text(status))
                log("PROGRESS " + json.dumps(status, ensure_ascii=False))
                last_report = now
    if failures:
        text_write(args.output_root / "failures.json", json_text(failures))
        raise RuntimeError(f"{len(failures)} episode conversions failed; output is marked pending")
    checksum_lines = []
    for name in datasets:
        rows = sorted((row for row in results if row["name"] == name), key=lambda row: row["episode"]["episode_index"])
        base = args.output_root / name
        for filename, key in (("meta/episodes.jsonl", "episode"), ("meta/episodes_stats.jsonl", "stats")):
            text_write(base / filename, "".join(json.dumps(row[key], ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
        for row in rows:
            for relative, artifact in row["artifacts"].items():
                checksum_lines.append(f"{artifact['sha256']}  {name}/{relative}\n")
    for filename in ("fr3_semantics_audit.json", "fr3_new_cumulative_audit.json"):
        source = Path("/tmp") / filename
        if source.exists():
            copy_checked(source, args.output_root / filename)
    copy_checked(Path(__file__), args.output_root / "convert.py")
    text_write(args.output_root / "SHA256SUMS", "".join(sorted(checksum_lines)))
    summary["conversion_complete"] = True
    summary["artifact_files_verified"] = len(checksum_lines)
    summary["bytes_copied"] = sum(artifact["bytes"] for row in results for artifact in row["artifacts"].values())
    summary["datasets_summary"] = [{"name": name, "episodes": data["info"]["total_episodes"], "frames": data["info"]["total_frames"], "encoded_video_fps": data["info"]["unification"]["source_fps"]} for name, data in datasets.items()]
    text_write(args.output_root / "manifest.json", json_text(summary))
    (args.output_root / "manifest.pending.json").unlink()
    Path("/tmp/fr3_unification_status.json").write_text(json_text(summary))
    log("COMPLETE " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
