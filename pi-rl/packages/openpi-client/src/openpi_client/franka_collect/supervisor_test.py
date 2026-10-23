import json
import os
import signal
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from openpi_client.franka_collect import action_test, test_supervisor


def test_supervisor_pause_events_preserve_resume_then_pause_order():
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    try:
        os.write(writer, b"cp")
        paused = test_supervisor.read_pause_state(reader, False)
        assert paused
        assert test_supervisor.read_pause_state(reader, paused)
        os.write(writer, b"c")
        assert not test_supervisor.read_pause_state(reader, paused)
    finally:
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize(
    "cause", ["client_failure", "timeout", "interrupt", "hardware_error", "long_run"]
)
def test_supervisor_failure_cleans_up_client_without_stopping_driver(monkeypatch, cause):
    requests, client_signals, handlers, subscriptions = [], [], {}, []
    state = {"spawned": False, "notified": False, "clock": 0.0, "shutdown": False, "spins": 0}
    process = SimpleNamespace(pid=12345, returncode=1 if cause == "client_failure" else None)
    process.poll = lambda: process.returncode
    process.wait = lambda timeout: setattr(process, "returncode", -signal.SIGINT)

    def spawn(*args, **kwargs):
        state["spawned"] = True
        return process

    def monotonic():
        state["clock"] += 60 if cause == "long_run" else 0.2
        return state["clock"]

    def spin_once(*args, **kwargs):
        if cause == "long_run" and state["spawned"]:
            state["spins"] += 1
            if state["spins"] >= 3:
                process.returncode = 0
            return
        if not state["spawned"] or state["notified"]:
            return
        state["notified"] = True
        if cause == "interrupt":
            handlers[signal.SIGINT](signal.SIGINT, None)
        elif cause == "hardware_error":
            subscriptions[0](
                SimpleNamespace(
                    stamp=SimpleNamespace(sec=2, nanosec=0),
                    level=40,
                    name="FrankaHardwareInterface",
                    msg="hardware fault",
                )
            )

    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        create_subscription=lambda kind, topic, callback, qos: subscriptions.append(callback),
        destroy_node=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        SimpleNamespace(
            init=lambda **kwargs: None,
            create_node=lambda name: node,
            spin_once=spin_once,
            shutdown=lambda: state.update(shutdown=True),
        ),
    )
    monkeypatch.setitem(sys.modules, "rclpy.qos", SimpleNamespace(QoSProfile=lambda **kwargs: None))
    monkeypatch.setitem(
        sys.modules,
        "rclpy.signals",
        SimpleNamespace(
            SignalHandlerOptions=SimpleNamespace(NO=0),
        ),
    )
    monkeypatch.setitem(sys.modules, "rcl_interfaces.msg", SimpleNamespace(Log=object))
    monkeypatch.setattr(test_supervisor.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        test_supervisor.os, "killpg", lambda pid, signum: client_signals.append((pid, signum))
    )
    monkeypatch.setattr(
        test_supervisor.signal, "signal", lambda signum, handler: handlers.update({signum: handler})
    )
    monkeypatch.setattr(test_supervisor.time, "monotonic", monotonic)

    def open_request(request, **kwargs):
        requests.append(request.full_url)
        return BytesIO(b'{"name":"robot","running":false}')

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *args: SimpleNamespace(open=open_request)
    )
    arguments = ["supervisor"]
    if cause != "long_run":
        arguments += ["--timeout", "0.1"]
    monkeypatch.setattr(sys, "argv", arguments + ["--", "--execute"])
    expected = {
        "client_failure": "Test client exited with code 1",
        "timeout": "Hardware test timed out",
        "interrupt": "Interrupted by signal 2",
        "hardware_error": "Hardware error:",
    }
    if cause == "long_run":
        test_supervisor.main()
        assert state["clock"] > 120
    else:
        with pytest.raises(RuntimeError, match=expected[cause]):
            test_supervisor.main()
    assert requests == []
    assert client_signals == (
        [] if cause in ("client_failure", "long_run") else [(12345, signal.SIGINT)]
    )
    assert state["shutdown"]


def test_remote_action_test_failure_does_not_request_driver_shutdown(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        sys, "argv", ["action_test", "--execute", "--countdown", "0", "--output", str(tmp_path)]
    )
    monkeypatch.setattr(action_test, "check_robot_service", lambda host: {"running": True})
    monkeypatch.setattr(action_test.signal, "signal", lambda *args: None)

    def fail_session(*args):
        raise RuntimeError("inference failed")

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({"name": "robot", "running": False}))

    monkeypatch.setattr(action_test, "run_client_session", fail_session)
    monkeypatch.setattr(action_test.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="inference failed"):
        action_test.main()
    assert commands == []
    assert json.loads((tmp_path / "report.json").read_text())["error"] == "inference failed"
