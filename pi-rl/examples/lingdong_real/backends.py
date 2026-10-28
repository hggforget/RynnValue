"""Observation sources: live Tron2 (read-only) and LeRobot episode replay."""

from __future__ import annotations

import json
import logging
import pathlib
import time

import numpy as np

from common import Observation, build_raw_state

log = logging.getLogger("backends")


class Tron2Backend:
    """Live robot state + RealSense images. Reading only; commands go through RobotExecutor."""

    def __init__(self, link, cameras, *, head_cam="cam_high", right_wrist_cam="cam_right_wrist",
                 state_max_age=0.2, image_max_age=0.15):
        self.link = link
        self.cameras = cameras
        self.head_cam = head_cam
        self.right_wrist_cam = right_wrist_cam
        self.state_max_age = state_max_age
        self.image_max_age = image_max_age

    def observe(self) -> Observation:
        head, a_head = self.cameras.get(self.head_cam, self.image_max_age)
        wrist, a_wrist = self.cameras.get(self.right_wrist_cam, self.image_max_age)
        st = self.link.get_state(self.state_max_age)
        state18 = build_raw_state(st["q16"], st["left_opening"], st["right_opening"])
        return Observation(state18=state18, images={"cam_high": head, "cam_right_wrist": wrist},
                           t_capture=time.monotonic(),
                           ages_s={"cam_high": a_head, "cam_right_wrist": a_wrist, "joint": st["joint_age_s"],
                                   "gripper": st["gripper_age_s"]})

    def advance(self, n: int):
        pass

    def close(self):
        pass


def _decode_video(path: pathlib.Path) -> np.ndarray:
    import av

    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack(frames)


class ReplayBackend:
    """Replays one recorded episode (open loop): observation at a frame cursor, advanced by executed steps.

    wrist_source: which recorded stream is fed as the policy's cam_right_wrist.
      * RAW collection data (this subset): 'cam_left_wrist'  (raw streams are left/right exchanged; the final
        training dataset swapped them, so final cam_right_wrist == raw cam_left_wrist)
      * already-corrected final dataset: 'cam_right_wrist'
    """

    def __init__(self, root, episode: int, *, wrist_source="cam_left_wrist", start: str | int = "auto"):
        import pyarrow.parquet as pq

        root = pathlib.Path(root)
        info = json.loads((root / "meta/info.json").read_text())
        names = info["features"]["observation.state"]["names"]
        from common import RAW_STATE_NAMES

        if tuple(names) != RAW_STATE_NAMES:
            raise ValueError(f"unexpected state names {names}")
        chunk = episode // info.get("chunks_size", 1000)
        t = pq.read_table(root / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet")
        self.state18 = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        self.action18 = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        self.task_index = int(t.column("task_index")[0].as_py())
        vid = root / f"videos/chunk-{chunk:03d}"
        t0 = time.monotonic()
        self.head = _decode_video(vid / "observation.images.cam_high" / f"episode_{episode:06d}.mp4")
        self.wrist = _decode_video(vid / f"observation.images.{wrist_source}" / f"episode_{episode:06d}.mp4")
        n = min(len(self.state18), len(self.head), len(self.wrist))
        if n != len(self.state18):
            log.warning("length mismatch: parquet %d head %d wrist %d", len(self.state18), len(self.head),
                        len(self.wrist))
        self.n = n
        if start == "auto":  # skip leading all-zero startup frames (removed by trimming in the final dataset)
            nz = np.where(np.abs(self.state18[:n, 7:14]).sum(1) > 1e-6)[0]
            start = int(nz[0]) if len(nz) else 0
        self.cursor = int(start)
        self.start = self.cursor
        self.episode = episode
        self.wrist_source = wrist_source
        log.info("replay ep %d: %d frames (start %d), task_index %d, wrist<-%s, decoded in %.1fs", episode, n,
                 self.cursor, self.task_index, wrist_source, time.monotonic() - t0)

    @property
    def done(self) -> bool:
        return self.cursor >= self.n - 1

    def observe(self) -> Observation:
        i = min(self.cursor, self.n - 1)
        return Observation(state18=self.state18[i].copy(), images={"cam_high": self.head[i],
                           "cam_right_wrist": self.wrist[i]}, t_capture=time.monotonic(), frame_index=i)

    def recorded_actions8(self, i: int, horizon: int) -> np.ndarray:
        from common import RIGHT_INDICES

        a = self.action18[i:min(i + horizon, self.n), :][:, list(RIGHT_INDICES)]
        return a

    def advance(self, n: int):
        self.cursor += int(n)

    def close(self):
        pass
