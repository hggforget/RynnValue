"""Persistent, read-only web viewer. Only reads inference log files; never opens hardware."""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from common import RIGHT_NAMES

SERVICE = "lingdong-live-view-v1"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def atomic_write(path, data):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def health(host, port):
    if not port:
        return {}
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{port}/health", timeout=0.3) as r:
            result = json.load(r)
        return result if result.get("service") == SERVICE else {}
    except (OSError, ValueError):
        return {}


def ensure_server(root, host="0.0.0.0", port=8001):
    """Start a detached viewer if necessary. Called BEFORE robot/camera initialization."""
    root = pathlib.Path(root).resolve()
    existing = health(host, port)
    if existing:
        if existing["root"] != str(root):
            raise RuntimeError(f"viewer port {port} serves {existing['root']}; choose another --viz-port")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".viewer-{port}.log").open("ab") as output:
        process = subprocess.Popen(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--root", str(root),
             "--host", host, "--port", str(port)], stdin=subprocess.DEVNULL,
            stdout=output, stderr=output, start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        result = health(host, port)
        if result:
            if result["root"] != str(root):
                raise RuntimeError(f"viewer port {port} is already used for another log directory")
            return result
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError(f"could not start viewer on {host}:{port}; see {root}/.viewer-{port}.log")


class DiskView:
    def __init__(self, root):
        self.root = pathlib.Path(root).resolve()
        self.lock = threading.Lock()
        self.run_dir = None
        self.events = collections.deque(maxlen=12000)
        self.offset = 0
        self.partial = b""
        self.last_scan = -1

    def update(self):
        now = time.monotonic()
        if now - self.last_scan > 0.4:
            dirs = [p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")]
            newest = max(dirs, key=lambda p: p.stat().st_mtime, default=None)
            if newest != self.run_dir:
                self.run_dir = newest
                self.events.clear()
                self.offset, self.partial = 0, b""
            self.last_scan = now
        if self.run_dir is None:
            return
        trace = self.run_dir / "telemetry.jsonl"
        try:
            if trace.stat().st_size < self.offset:
                self.events.clear()
                self.offset, self.partial = 0, b""
            with trace.open("rb") as f:
                f.seek(self.offset)
                block = f.read()
            self.offset += len(block)
            lines = (self.partial + block).split(b"\n")
            self.partial = lines.pop()
            for line in lines:
                try:
                    self.events.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass

    def snapshot(self, after=0, session=""):
        with self.lock:
            self.update()
            name = self.run_dir.name if self.run_dir else "等待推理启动"
            metadata = read_json(self.run_dir / "viewer-session.json") if self.run_dir else {}
            result = read_json(self.run_dir / "run_meta.json") if self.run_dir else {}
            status = {"phase": "等待推理启动", "execution": metadata.get("execution", "dry")}
            for e in self.events:
                if e["kind"] == "status":
                    status.update({k: v for k, v in e.items() if k not in ("kind", "t", "seq")})
            active = metadata.get("active", False)
            if active:
                try:
                    os.kill(metadata["pid"], 0)
                except (OSError, KeyError):
                    active = False
            error = result.get("stop_reason")
            if error:
                status["phase"] = "启动或运行失败"
            elif metadata and not active:
                status["phase"] = "推理已结束"
            elif self.run_dir and not metadata:
                status["phase"] = "等待相机和模型初始化"
            seq = self.events[-1]["seq"] if self.events else 0
            if session != name or after > seq:
                after = 0
            elapsed = self.events[-1]["t"] if self.events else 0
            if active:
                elapsed = max(elapsed, time.monotonic() - metadata.get("start_monotonic", time.monotonic()))
            return {"session": name, "events": [e for e in self.events if e["seq"] > after], "seq": seq,
                    "status": status, "now": elapsed, "error": error,
                    "feedback_source": metadata.get("feedback_source", "robot"),
                    "dropped": metadata.get("dropped", 0), "video_error": metadata.get("video_error")}

    def page(self):
        snapshot = self.snapshot()
        config = {"mode": "live", "names": list(RIGHT_NAMES), "session": snapshot["session"],
                  "execution": snapshot["status"]["execution"],
                  "feedback_source": snapshot["feedback_source"], "events": []}
        template = pathlib.Path(__file__).with_name("live_view.html").read_text()
        return template.replace("/*VIEW_CONFIG*/{}", json.dumps(config).replace("</", "<\\/"))


def make_server(root, host="0.0.0.0", port=8001):
    view = DiskView(root)
    view.root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlsplit(self.path)
            if u.path == "/health":
                self.reply(json.dumps({"service": SERVICE, "root": str(view.root)}).encode(), "application/json")
            elif u.path in ("/", "/index.html"):
                self.reply(view.page().encode(), "text/html; charset=utf-8")
            elif u.path == "/api":
                args = parse_qs(u.query)
                try:
                    after = int(args.get("after", ["0"])[0])
                except ValueError:
                    self.send_error(400)
                    return
                data = view.snapshot(after, args.get("session", [""])[0])
                self.reply(json.dumps(data).encode(), "application/json")
            elif u.path in ("/head.jpg", "/wrist.jpg"):
                with view.lock:
                    view.update()
                    path = view.run_dir / u.path[1:] if view.run_dir else None
                    try:
                        data = path.read_bytes() if path else None
                    except OSError:
                        data = None
                if data:
                    self.reply(data, "image/jpeg")
                else:
                    # A valid neutral placeholder keeps the waiting page usable before camera initialization.
                    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">'
                           '<rect width="640" height="480" fill="#101824"/>'
                           '<text x="320" y="240" fill="#94a6bf" font-size="20" text-anchor="middle">'
                           'Waiting for camera / inference</text></svg>')
                    self.reply(svg.encode(), "image/svg+xml")
            else:
                self.send_error(404)

        def reply(self, data, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/hdc/lingdong_real_logs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    make_server(args.root, args.host, args.port).serve_forever()
