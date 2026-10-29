"""Exact frame-weighted 8-D Lingdong normalization from Parquet; no video decoding."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.policies import lingdong_policy as policy
from openpi.shared import normalize

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "assets" / policy.CONFIG_NAME / policy.ASSET_ID


def dataset_manifest(dataset_root: Path) -> tuple[dict, list[tuple]]:
    repos = tuple(str((dataset_root / name).resolve()) for name in policy.DATASET_NAMES)
    state_ids, action_ids = policy.dataset_indices(repos)
    sources, jobs = [], []
    for repo in repos:
        root = Path(repo)
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
        if len(episodes) != info["total_episodes"] or sum(e["length"] for e in episodes) != info["total_frames"]:
            raise ValueError(f"Inconsistent dataset metadata: {repo}")
        hashes = {}
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "file_checksums.json"):
            path = root / "meta" / name
            if path.is_file():
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(
            {"repo_id": repo, "episodes": len(episodes), "frames": info["total_frames"], "metadata_sha256": hashes}
        )
        for episode in episodes:
            index = episode["episode_index"]
            path = root / info["data_path"].format(episode_index=index, episode_chunk=index // info["chunks_size"])
            jobs.append((path, episode["length"], state_ids, action_ids))
    value = {
        "version": 2,
        "sources": sources,
        "channel_names": policy.RIGHT_NAMES,
        "state_indices": state_ids,
        "action_indices": action_ids,
        "action_gripper": policy.gripper_binarization_spec(),
        "statistics": "Each retained frame contributes once; exact pooled 1% and 99% quantiles.",
    }
    value["signature"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value, jobs


def validate(output: Path, manifest: dict) -> None:
    saved = json.loads((output / "norm_stats_manifest.json").read_text())
    if saved["signature"] != manifest["signature"]:
        raise ValueError("Normalization sources or action preprocessing changed; recompute Lingdong norms")
    stats = normalize.load(output)
    for key in ("state", "actions"):
        for name in ("mean", "std", "q01", "q99"):
            values = np.asarray(getattr(stats[key], name))
            if values.shape != (8,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid 8-D normalization: {key}.{name}")
        if np.any(stats[key].std < 0) or np.any(stats[key].q99 < stats[key].q01):
            raise ValueError(f"Invalid normalization ranges: {key}")


def read_episode(job):
    path, length, state_ids, action_ids = job
    table = pq.read_table(path, columns=["observation.state", "action"], use_threads=False)
    if len(table) != length:
        raise ValueError(f"Episode length differs from metadata: {path}")
    result = {}
    for source, target, ids in (("observation.state", "state", state_ids), ("action", "actions", action_ids)):
        values = np.asarray(table[source].to_pylist(), dtype=np.float32)
        if values.shape != (length, 18):
            raise ValueError(f"Invalid {source} shape: {path}")
        values = values[:, ids]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite right-arm channels: {path}")
        if target == "actions":
            values = policy.binarize_gripper_actions(values)
        result[target] = values
    return result


def compute(dataset_root: Path, output: Path, workers: int = 16, *, check: bool = False):
    manifest, jobs = dataset_manifest(dataset_root)
    if check:
        validate(output, manifest)
        print(f"Valid 8-D Lingdong norms: {output}")
        return
    values = {"state": [], "actions": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(read_episode, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for key, batches in values.items():
                batches.append(result[key])
            if n % 500 == 0 or n == len(jobs):
                print(f"Read {n}/{len(jobs)} episodes", flush=True)
    stats = {}
    for key, batches in values.items():
        array = np.concatenate(batches)
        if len(array) != sum(source["frames"] for source in manifest["sources"]):
            raise ValueError("Normalization did not cover all retained frames")
        q01, q99 = np.quantile(array, [0.01, 0.99], axis=0)
        stats[key] = normalize.NormStats(
            mean=np.mean(array, axis=0, dtype=np.float64), std=np.std(array, axis=0, dtype=np.float64), q01=q01, q99=q99
        )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "norm_stats.json.tmp"
    temporary.write_text(normalize.serialize_json(stats))
    temporary.replace(output / "norm_stats.json")
    (output / "norm_stats_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    validate(output, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(jobs),
                "frames": sum(s["frames"] for s in manifest["sources"]),
                "state_dim": 8,
                "action_dim": 8,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(policy.DEFAULT_DATASET_ROOT))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("workers must be 1..64")
    compute(args.dataset_root, args.output, args.workers, check=args.check)
