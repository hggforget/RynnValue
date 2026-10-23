import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from openpi_client.franka_collect.test_supervisor import hardware_fault, stop_robot


@pytest.mark.parametrize(
    "name,level,stamp,expected",
    [
        ("FrankaHardwareInterface", 40, 11, True),
        ("left.controller_manager", 40, 11, True),
        ("FrankaHardwareInterface", 30, 11, False),
        ("FrankaHardwareInterface", 40, 9, False),
        ("camera.primary", 40, 11, False),
    ],
)
def test_hardware_monitor_rejects_stale_and_unrelated_errors(name, level, stamp, expected):
    message = SimpleNamespace(name=name, level=level, stamp=SimpleNamespace(sec=stamp, nanosec=0))
    assert hardware_fault(message, 10_000_000_000) == expected


@pytest.mark.parametrize("running", [True, False])
def test_driver_stop_requires_confirmed_shutdown_and_bypasses_proxies(running, monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"name": "robot", "running": running}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/service/robot/stop"
        if running:
            with pytest.raises(RuntimeError, match="stop was not confirmed"):
                stop_robot(url)
        else:
            assert stop_robot(url)["running"] is False
        assert received == [("/api/service/robot/stop", b"{}")]
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()
