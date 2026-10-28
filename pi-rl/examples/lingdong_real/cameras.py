"""Read-only RealSense capture (same settings as tron2_old/realsense_image.py: 640x480 bgr8 @30, + z16 depth)."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("cameras")


class RealSenseCameras:
    def __init__(self, serials: dict[str, str], width=640, height=480, fps=30, enable_depth=True,
                 start_timeout=10.0):
        import pyrealsense2 as rs

        self._rs = rs
        found = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
                 for d in rs.context().query_devices()}
        missing = {n: s for n, s in serials.items() if s not in found}
        if missing:
            raise RuntimeError(f"RealSense cameras not found: {missing}; connected: {found}")
        self.serials = dict(serials)
        self._pipes = {}
        self._latest: dict[str, tuple[np.ndarray, float, float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        for name, serial in serials.items():
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if enable_depth:
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            pipe = rs.pipeline()
            pipe.start(cfg)
            self._pipes[name] = pipe
            th = threading.Thread(target=self._loop, args=(name, pipe), daemon=True)
            th.start()
            self._threads.append(th)
            log.info("camera %s = %s (%s)", name, serial, found[serial])
        t0 = time.monotonic()
        while time.monotonic() - t0 < start_timeout:
            with self._lock:
                if all(n in self._latest for n in serials):
                    return
            time.sleep(0.02)
        raise RuntimeError("cameras did not deliver frames in time")

    def _loop(self, name, pipe):
        while not self._stop.is_set():
            try:
                frames = pipe.wait_for_frames(1000)
            except Exception as e:  # noqa: BLE001
                if not self._stop.is_set():
                    log.warning("camera %s: %s", name, e)
                continue
            c = frames.get_color_frame()
            if not c:
                continue
            bgr = np.asanyarray(c.get_data())
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB, as in data collection
            with self._lock:
                self._latest[name] = (rgb, time.monotonic(), c.get_timestamp())

    def get(self, name: str, max_age: float = 0.15) -> tuple[np.ndarray, float]:
        with self._lock:
            if name not in self._latest:
                raise RuntimeError(f"no frame from {name}")
            img, t, _ = self._latest[name]
        age = time.monotonic() - t
        if age > max_age:
            raise RuntimeError(f"camera {name} frame is {age*1e3:.0f} ms old")
        return img, age

    def close(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
        for p in self._pipes.values():
            try:
                p.stop()
            except Exception:
                pass
