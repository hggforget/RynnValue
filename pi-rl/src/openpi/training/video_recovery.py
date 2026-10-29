"""Bounded recovery of transient video reads without dropping training samples."""

from collections.abc import Callable
import hashlib
import io
import json
import logging
from pathlib import Path
import time

import av
import numpy as np
import torch

logger = logging.getLogger(__name__)
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


class VideoReadError(RuntimeError):
    """A video still cannot be read after bounded retries and verified fallback."""


def decode_snapshot(path: Path, timestamps, tolerance_s: float, root: Path) -> torch.Tensor:
    """Read and verify the complete object, then decode sequentially in RAM.

    This avoids filesystem range reads during decoding and seeking into a broken
    AV1 temporal unit. It neither re-encodes video nor changes target timestamps.
    """
    size = path.stat().st_size
    if size > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"Video exceeds the {MAX_SNAPSHOT_BYTES}-byte recovery memory limit")
    data = path.read_bytes()
    if len(data) != size:
        raise OSError(f"Incomplete video read: expected {size} bytes, received {len(data)}")
    inventory = root / "meta/file_checksums.json"
    if inventory.is_file():
        relative = path.relative_to(root).as_posix()
        entry = json.loads(inventory.read_text())[relative]
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or digest != entry["sha256"]:
            raise OSError(f"Video checksum mismatch: {relative}; actual_sha256={digest}")
    queries = np.asarray(timestamps, dtype=np.float64)
    if queries.ndim != 1 or len(queries) == 0 or not np.isfinite(queries).all():
        raise ValueError("Video timestamps must be a nonempty finite vector")
    order = np.argsort(queries, kind="stable")
    frames = [None] * len(queries)
    next_query = 0

    def retain(query_index, frame):
        error = abs(float(frame.time) - queries[query_index])
        if error >= tolerance_s:
            raise ValueError(f"No video frame within tolerance: requested={queries[query_index]}, error={error}")
        rgb = frame.to_ndarray(format="rgb24")
        frames[query_index] = torch.from_numpy(rgb).permute(2, 0, 1)

    with av.open(io.BytesIO(data), metadata_errors="ignore") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        previous = None
        for current in container.decode(video=0):
            while next_query < len(order) and queries[order[next_query]] <= current.time:
                index = int(order[next_query])
                closest = current
                if previous is not None and abs(previous.time - queries[index]) <= abs(current.time - queries[index]):
                    closest = previous
                retain(index, closest)
                next_query += 1
            previous = current
            if next_query == len(order):
                break
        while next_query < len(order) and previous is not None:
            retain(int(order[next_query]), previous)
            next_query += 1
    if next_query != len(order):
        raise ValueError("Video has no decodable frames")
    return torch.stack(frames).to(torch.float32) / 255


def read_video_with_recovery(
    read: Callable, *, path: Path, timestamps, tolerance_s: float, backend, root: Path, episode: int, camera: str
):
    """Retry the same sample three times; never skip it or substitute frames."""
    details = {
        "path": str(path),
        "episode": int(episode),
        "camera": camera,
        "timestamps": [float(t) for t in timestamps],
        "backend": backend,
    }
    for attempt in range(1, 4):
        try:
            result = read()
            if attempt > 1:
                logger.warning("Video read recovered on attempt %d: %s", attempt, json.dumps(details))
            return result
        except (av.error.FFmpegError, OSError, AssertionError) as exc:
            details["last_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Video read failed on attempt %d/3: %s", attempt, json.dumps(details))
            if attempt < 3:
                time.sleep(0.25 * attempt)
    if backend == "pyav":
        try:
            result = decode_snapshot(path, timestamps, tolerance_s, root)
            logger.warning("Video read recovered by sequential RAM decode: %s", json.dumps(details))
            return result.squeeze(0)
        except (av.error.FFmpegError, OSError, ValueError, KeyError) as exc:
            details["snapshot_error"] = f"{type(exc).__name__}: {exc}"
    raise VideoReadError("VIDEO_READ_FAILED " + json.dumps(details, ensure_ascii=False))
