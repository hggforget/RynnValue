"""No robot/camera access: verify the viewer records commands and feedback independently."""
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from live_view import LiveView
from tron2_link import FakeLink
from executor import RobotExecutor
import executor as E
import threading
from viewer_server import make_server, health


def test_command_feedback_and_http(tmp_path):
    q = np.zeros(16)
    q[7:14] = 0.25
    link = FakeLink(q, right_opening=40)
    view = LiveView(tmp_path, link=link, port=0, record=False).start()
    try:
        target = np.full(8, 0.8)
        view.record_target(target)
        target[:] = -5  # the queued event must own its values
        view.set_status("executing", cycle=2, infer_ms=456)
        time.sleep(0.12)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{view.port}/api?after=0", timeout=2) as r:
            payload = json.load(r)
        actions = [e for e in payload["events"] if e["kind"] == "action"]
        states = [e for e in payload["events"] if e["kind"] == "state"]
        assert len(actions) == 1 and actions[0]["values"] == [0.8] * 8
        assert states and states[-1]["values"][:7] == [0.25] * 7
        assert states[-1]["values"][7] == pytest.approx(0.4)
        assert payload["status"]["cycle"] == 2
        with opener.open(f"http://127.0.0.1:{view.port}/api?after={payload['seq']}", timeout=2) as r:
            assert all(e["seq"] > payload["seq"] for e in json.load(r)["events"])
        assert not link.sent_counts  # viewer never commands or polls the robot transport
    finally:
        view.close()
    assert "action.abad_R_Joint" in (tmp_path / "telemetry.csv").read_text()
    html = (tmp_path / "visualization.html").read_text()
    assert '"mode": "recorded"' in html and "/*VIEW_CONFIG*/" not in html
    assert json.loads((tmp_path / "visualization_meta.json").read_text())["dropped_events"] == 0


def test_executor_observer_cannot_change_commands(monkeypatch):
    monkeypatch.setattr(E, "_sleep_until", lambda _: None)
    link = FakeLink(np.zeros(16))
    link.arm_commands("I_UNDERSTAND_THE_ROBOT_WILL_MOVE")
    observed = []
    ex = RobotExecutor(link, on_target=lambda t, stamp: observed.append((t.copy(), stamp)))
    ex.start()
    targets = np.arange(24, dtype=float).reshape(3, 8) / 20
    ex.execute(targets)
    commands = np.array([c[2][7:14] for c in link.commands if c[1] == "request_servoj"])
    assert np.array_equal(commands, targets[:, :7])
    assert np.array_equal([x[0] for x in observed], targets)
    assert all(b[1] >= a[1] for a, b in zip(observed, observed[1:]))
    ex.on_target = lambda *_: (_ for _ in ()).throw(RuntimeError("viewer failed"))
    ex.execute(targets)
    assert ex.on_target is None and len(ex.sent_targets) == 6


def test_video_encodes_cameras_and_curves(tmp_path):
    av = pytest.importorskip("av")

    class CachedCameras:
        def get(self, name, max_age):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :, 0 if name == "cam_high" else 2] = 160
            return img, 0.01

    view = LiveView(tmp_path, link=FakeLink(np.zeros(16)), cameras=CachedCameras(),
                    port=0, fps=8, execution="fake").start()
    try:
        view.record_target(np.arange(8) / 10)
        time.sleep(0.65)
    finally:
        view.close()
    meta = json.loads((tmp_path / "visualization_meta.json").read_text())
    assert meta["video_error"] is None
    with av.open(str(tmp_path / "dashboard.mp4")) as f:
        frames = list(f.decode(video=0))
    assert len(frames) >= 2
    assert (frames[0].width, frames[0].height) == (1280, 896)
    assert frames[-1].pts > frames[0].pts


def test_persistent_viewer_survives_run_and_shows_initialization_error(tmp_path):
    server = make_server(tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        run1 = tmp_path / "run1"
        view = LiveView(run1, link=FakeLink(np.zeros(16)), host="127.0.0.1", port=port,
                        record=False).start()
        try:
            view.record_target(np.full(8, 0.7))
            time.sleep(0.12)
        finally:
            view.close()
        assert health("127.0.0.1", port)["root"] == str(tmp_path)
        with opener.open(f"http://127.0.0.1:{port}/api?session=run1", timeout=2) as r:
            d = json.load(r)
        assert d["status"]["phase"] == "推理已结束"
        assert len([e for e in d["events"] if e["kind"] == "action"]) == 1
        # A new run can fail before LiveView ever starts (e.g. missing head camera).
        run2 = tmp_path / "run2"
        run2.mkdir()
        (run2 / "run_meta.json").write_text(json.dumps({"stop_reason": "D455 camera missing"}))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with opener.open(f"http://127.0.0.1:{port}/api?after=999&session=run1", timeout=2) as r:
                d = json.load(r)
            if d["session"] == "run2":
                break
            time.sleep(0.05)
        assert d["session"] == "run2" and d["events"] == [] and d["seq"] == 0
        assert d["error"] == "D455 camera missing"
        assert d["status"]["phase"] == "启动或运行失败"
        with opener.open(f"http://127.0.0.1:{port}/", timeout=2) as r:
            assert "Tron2" in r.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
