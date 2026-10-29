import hashlib
import json

import av
from lerobot.common.datasets.video_utils import decode_video_frames
import numpy as np
import pytest
import torch

from openpi.training import video_recovery as recovery


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "videos/tiny.mp4"
    path.parent.mkdir()
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(8):
            pixels = np.full((24, 32, 3), index * 25, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    (tmp_path / "meta").mkdir()
    inventory = {
        "videos/tiny.mp4": {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    }
    (tmp_path / "meta/file_checksums.json").write_text(json.dumps(inventory))
    return tmp_path, path


def arguments(root, path):
    return {
        "path": path,
        "timestamps": [0.1],
        "tolerance_s": 1e-4,
        "backend": "pyav",
        "root": root,
        "episode": 17,
        "camera": "right_wrist",
    }


def test_transient_error_retries_same_sample_without_fallback(video, monkeypatch):
    root, path = video
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)
    expected = torch.ones(3, 24, 32)
    calls = []

    def read():
        calls.append(1)
        if len(calls) < 3:
            raise av.error.InvalidDataError(1094995529, "temporary AV1 read error")
        return expected

    monkeypatch.setattr(recovery, "decode_snapshot", lambda *a: pytest.fail("Unnecessary snapshot fallback"))
    assert recovery.read_video_with_recovery(read, **arguments(root, path)) is expected
    assert len(calls) == 3


def test_verified_sequential_snapshot_matches_normal_decoder_and_query_order(video):
    root, path = video
    timestamps = [7 / 30, 0, 3 / 30, 3 / 30]
    expected = decode_video_frames(path, timestamps, 1e-4, "pyav")
    actual = recovery.decode_snapshot(path, timestamps, 1e-4, root)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_repeated_seek_failure_recovers_without_changing_frame(video, monkeypatch):
    root, path = video
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)

    def read():
        raise av.error.InvalidDataError(1094995529, "bad range read")

    actual = recovery.read_video_with_recovery(read, **arguments(root, path))
    expected = decode_video_frames(path, [0.1], 1e-4, "pyav").squeeze(0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_checksum_mismatch_is_not_silently_accepted(video, monkeypatch):
    root, path = video
    path.write_bytes(path.read_bytes() + b"corrupted")
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)

    def read():
        raise OSError("short read")

    with pytest.raises(recovery.VideoReadError, match="checksum mismatch") as error:
        recovery.read_video_with_recovery(read, **arguments(root, path))
    assert str(path) in str(error.value)
    assert '"episode": 17' in str(error.value)
    assert '"timestamps": [0.1]' in str(error.value)


def test_snapshot_does_not_substitute_a_different_timestamp(video):
    root, path = video
    with pytest.raises(ValueError, match="within tolerance"):
        recovery.decode_snapshot(path, [10.0], 1e-4, root)
