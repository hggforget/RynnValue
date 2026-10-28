"""Read-only live video / action-state viewer. Control callbacks only enqueue targets.

The sampler reads existing camera/state caches; it never opens a robot connection or sends commands.
JPEG encoding, plotting, video recording and HTTP requests run outside the control loop.
"""
from __future__ import annotations

import collections
import csv
import io
import json
import logging
import os
import pathlib
import queue
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np

from common import RIGHT_NAMES, build_raw_state, select_right
from viewer_server import atomic_write, health

log = logging.getLogger("live_view")
ACTION_COLOR = "#ed9234"
STATE_COLOR = "#35bba1"


class LiveView:
    def __init__(self, out_dir, *, link=None, cameras=None, host="0.0.0.0", port=8001,
                 fps=8.0, execution="real", record=True):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.link, self.cameras = link, cameras
        self.host, self.port, self.fps = host, port, fps
        self.execution, self.record = execution, record
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending = queue.Queue(maxsize=4096)
        self._events = collections.deque(maxlen=12000)
        self._seq = 0
        self._images = {}
        self._jpeg = {}
        self._status = {"phase": "initializing", "cycle": None, "execution": execution}
        self._video_offset = None
        self._video_error = None
        self._dropped = 0
        self._http = None
        self._threads = []
        self._trace = None
        self._closed = False

    def start(self):
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                u = urlsplit(self.path)
                if u.path in ("/", "/index.html"):
                    self.reply(viewer._page("live").encode(), "text/html; charset=utf-8")
                elif u.path == "/api":
                    try:
                        after = int(parse_qs(u.query).get("after", ["0"])[0])
                    except ValueError:
                        self.send_error(400)
                        return
                    with viewer._lock:
                        events = [e for e in viewer._events if e["seq"] > after]
                        payload = {"events": events, "seq": viewer._seq,
                                   "status": dict(viewer._status), "now": viewer.elapsed(),
                                   "dropped": viewer._dropped, "video_error": viewer._video_error}
                    self.reply(json.dumps(payload).encode(), "application/json")
                elif u.path in ("/head.jpg", "/wrist.jpg"):
                    key = "cam_high" if u.path == "/head.jpg" else "cam_right_wrist"
                    with viewer._lock:
                        data = viewer._jpeg.get(key)
                    if data:
                        self.reply(data, "image/jpeg")
                    else:
                        self.send_error(503, "Waiting for camera frames")
                else:
                    self.send_error(404)

            def reply(self, body, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

        # Reuse the persistent file-backed viewer, with in-process HTTP as a test/embedded fallback.
        external = health(self.host, self.port)
        targets = []
        if not external:
            self._http = ThreadingHTTPServer((self.host, self.port), Handler)
            self._http.daemon_threads = True
            self.port = self._http.server_address[1]
            targets.append(("viz-http", self._http.serve_forever))
        self._trace = (self.out_dir / "telemetry.jsonl").open("w", buffering=1)
        self._publish_session(True)
        targets.extend((("viz-sample", self._sample_loop), ("viz-video", self._video_loop)))
        for name, target in targets:
            thread = threading.Thread(name=name, target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        log.info("LIVE video + action/state: http://10.192.1.155:%d (localhost: http://127.0.0.1:%d)",
                 self.port, self.port)
        log.info("viewer recordings: %s", self.out_dir)
        return self

    def _publish_session(self, active):
        atomic_write(self.out_dir / "viewer-session.json", json.dumps({
            "pid": os.getpid(), "active": active, "start_monotonic": self.t0,
            "execution": self.execution, "feedback_source": "robot" if self.link is not None else "replay",
            "dropped": self._dropped, "video_error": self._video_error}).encode())

    def elapsed(self):
        return time.monotonic() - self.t0

    def _put(self, event):
        if self._closed:
            return
        try:
            self._pending.put_nowait(event)
        except queue.Full:
            self._dropped += 1

    def record_target(self, target, timestamp=None):
        """Called only AFTER a successful model-step send; no encoding, disk I/O or waits."""
        self._put({"t": (timestamp if timestamp is not None else time.monotonic()) - self.t0,
                   "kind": "action", "values": np.asarray(target, dtype=float).tolist()})

    def record_observation(self, obs):
        # Live feedback is sampled independently even while inference is blocked.
        if self.link is None:
            self._put({"t": self.elapsed(), "kind": "state", "values": obs.state8.tolist()})
            with self._lock:
                self._images = dict(obs.images)

    def set_status(self, phase, **kwargs):
        self._put({"t": self.elapsed(), "kind": "status", "phase": phase, **kwargs})

    def _accept(self, event):
        with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._events.append(event)
            if event["kind"] == "status":
                self._status.update({k: v for k, v in event.items() if k not in ("kind", "seq", "t")})
        self._trace.write(json.dumps(event, allow_nan=False) + "\n")

    def _drain(self):
        while True:
            try:
                event = self._pending.get_nowait()
            except queue.Empty:
                return
            self._accept(event)

    def _sample_loop(self):
        last_warning = 0.0
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                self._drain()
                if self.link is not None:
                    st = self.link.get_state(max_age=float("inf"))
                    state = select_right(build_raw_state(st["q16"], st["left_opening"], st["right_opening"]))
                    self._accept({"t": self.elapsed(), "kind": "state", "values": state.tolist(),
                                  "joint_age_s": st.get("joint_age_s"),
                                  "gripper_age_s": st.get("gripper_age_s")})
            except Exception as exc:  # visualization must not stop robot control
                if t - last_warning > 5:
                    log.warning("viewer telemetry: %s", exc)
                    last_warning = t
            self._stop.wait(max(0.0, 0.05 - (time.monotonic() - t)))
        self._drain()

    def _video_loop(self):
        from PIL import Image

        writer = stream = None
        video_failed = False
        last_warning = 0.0
        try:
            while not self._stop.is_set():
                t = time.monotonic()
                try:
                    if self.cameras is not None:
                        images = {k: self.cameras.get(k, float("inf"))[0]
                                  for k in ("cam_high", "cam_right_wrist")}
                    else:
                        with self._lock:
                            images = dict(self._images)
                    if len(images) == 2:
                        jpegs = {}
                        for key, rgb in images.items():
                            buf = io.BytesIO()
                            Image.fromarray(rgb).save(buf, format="JPEG", quality=78)
                            jpegs[key] = buf.getvalue()
                            filename = "head.jpg" if key == "cam_high" else "wrist.jpg"
                            atomic_write(self.out_dir / filename, jpegs[key])
                        with self._lock:
                            self._jpeg = jpegs
                            events = list(self._events)
                            status = dict(self._status)
                        if self.record and not video_failed:
                            try:
                                import av

                                now = self.elapsed()
                                frame = render_frame(images, events, status, now)
                                if writer is None:
                                    writer = av.open(str(self.out_dir / "dashboard.mp4"), "w")
                                    stream = writer.add_stream("libx264", rate=Fraction(str(self.fps)))
                                    stream.width, stream.height = frame.size
                                    stream.pix_fmt = "yuv420p"
                                    stream.options = {"preset": "ultrafast", "crf": "25", "tune": "zerolatency"}
                                    stream.time_base = Fraction(1, 1000)
                                    self._video_offset = now
                                vf = av.VideoFrame.from_image(frame)
                                vf.pts = round((now - self._video_offset) * 1000)
                                vf.time_base = Fraction(1, 1000)
                                for packet in stream.encode(vf):
                                    writer.mux(packet)
                            except Exception as exc:
                                video_failed = True
                                self._video_error = str(exc)
                                log.warning("video recording disabled; live page continues: %s", exc)
                except Exception as exc:
                    if t - last_warning > 5:
                        log.warning("viewer camera: %s", exc)
                        last_warning = t
                self._stop.wait(max(0.0, 1.0 / self.fps - (time.monotonic() - t)))
        finally:
            if writer is not None:
                try:
                    if stream is not None and not video_failed:
                        for packet in stream.encode():
                            writer.mux(packet)
                    writer.close()
                except Exception as exc:
                    self._video_error = str(exc)
                    log.warning("video finalize: %s", exc)

    def _page(self, mode, events=None):
        config = {"mode": mode, "names": list(RIGHT_NAMES), "execution": self.execution,
                  "feedback_source": "robot" if self.link is not None else "replay",
                  "session": self.out_dir.name, "video_offset": self._video_offset or 0,
                  "events": events or []}
        template = pathlib.Path(__file__).with_name("live_view.html").read_text()
        return template.replace("/*VIEW_CONFIG*/{}", json.dumps(config).replace("</", "<\\/"))

    def close(self):
        if self._closed:
            return
        self.set_status("finished")
        self._closed = True
        self._stop.set()
        for thread in self._threads:
            if thread.name != "viz-http":
                thread.join()  # motion has already stopped; finish telemetry and the MP4 trailer
        self._publish_session(False)
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        if self._trace is not None:
            self._trace.close()
            events = [json.loads(s) for s in (self.out_dir / "telemetry.jsonl").read_text().splitlines()]
            (self.out_dir / "visualization.html").write_text(self._page("recorded", events))
            with (self.out_dir / "telemetry.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "kind"] + [f"action.{n}" for n in RIGHT_NAMES]
                           + [f"state.{n}" for n in RIGHT_NAMES])
                for e in events:
                    if e["kind"] in ("action", "state"):
                        values = e["values"]
                        w.writerow([e["t"], e["kind"]] + (values + [""] * 8 if e["kind"] == "action"
                                                         else [""] * 8 + values))
            (self.out_dir / "visualization_meta.json").write_text(json.dumps({
                "video_offset_s": self._video_offset, "video_error": self._video_error,
                "dropped_events": self._dropped, "execution": self.execution,
                "event_count": len(events), "sample_hz": 20 if self.link is not None else "per_cycle",
                "feedback_source": "robot" if self.link is not None else "replay",
                "video_fps": self.fps}, indent=2))
            log.info("saved video + curves: %s/visualization.html ; dashboard.mp4 ; telemetry.csv", self.out_dir)


def render_frame(images, events, status, now, window=20.0):
    """Compose camera views and timestamp-aligned curves into an exportable MP4 frame."""
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (1280, 896), "#101824")
    d = ImageDraw.Draw(frame)
    d.text((14, 9), f"Tron2 right arm  |  {status.get('execution', '')}  |  {status.get('phase', '')}  "
           f"cycle {status.get('cycle', '-')}  |  t={now:.2f}s  |  ACTION orange / STATE green", fill="white")
    for i, key in enumerate(("cam_high", "cam_right_wrist")):
        frame.paste(Image.fromarray(images[key]).resize((640, 480)), (640 * i, 32))
        d.rectangle((640 * i, 32, 640 * i + 240, 53), fill="#101824")
        d.text((640 * i + 10, 37), key, fill="white")
    left, right = max(0, now - window), max(window, now)
    # Keep the last target before the visible window to show the held command while inferring.
    series = {}
    for kind in ("action", "state"):
        all_values = sorted((e for e in events if e["kind"] == kind), key=lambda e: e["t"])
        prior = [e for e in all_values if e["t"] < left]
        series[kind] = prior[-1:] + [e for e in all_values if left <= e["t"] <= right]
    for j, name in enumerate(RIGHT_NAMES):
        x, y = (j % 4) * 320, 522 + (j // 4) * 184
        d.text((x + 10, y + 4), name.replace("_Joint", "") + (" [0..1]" if j == 7 else " [rad]"), fill="#d7e0ed")
        box = (x + 45, y + 27, x + 310, y + 153)
        vals = [e["values"][j] for a in series.values() for e in a]
        lo, hi = (min(vals), max(vals)) if vals else (-1.0, 1.0)
        margin = max((hi - lo) * 0.12, 0.03)
        lo, hi = lo - margin, hi + margin
        if j == 7:
            lo, hi = min(-0.05, lo), max(1.05, hi)
        for k in range(4):
            yy = box[1] + k * (box[3] - box[1]) / 3
            d.line((box[0], yy, box[2], yy), fill="#2a3747")
        d.text((x + 2, box[1]), f"{hi:.2f}", fill="#9caec2")
        d.text((x + 2, box[3] - 8), f"{lo:.2f}", fill="#9caec2")
        d.text((box[0], box[3] + 7), f"{left:.1f}s", fill="#9caec2")
        d.text((box[2] - 36, box[3] + 7), f"{right:.1f}s", fill="#9caec2")
        for kind, color in (("state", STATE_COLOR), ("action", ACTION_COLOR)):
            pts = []
            for e in series[kind]:
                px = box[0] + (max(left, e["t"]) - left) / (right - left) * (box[2] - box[0])
                py = box[3] - (e["values"][j] - lo) / (hi - lo) * (box[3] - box[1])
                if pts and kind == "action":
                    pts.append((px, pts[-1][1]))
                pts.append((px, py))
            if pts and kind == "action":
                px = box[0] + (now - left) / (right - left) * (box[2] - box[0])
                pts.append((px, pts[-1][1]))
            if len(pts) > 1:
                d.line(pts, fill=color, width=2)
    return frame
