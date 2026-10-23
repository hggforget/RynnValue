import importlib
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


def evaluator_type():
    return importlib.import_module("openpi_client.franka_collect.evaluation").Evaluator


def make_observation():
    return {
        "observation.state.cartesian": np.array([0.355, 0.0277, 0.2356, -np.pi, 0, 0]),
        "observation.state.gripper": np.array([0.0]),
        "observation.state.joints": np.zeros(7),
        "observation.images.primary": np.zeros((8, 12, 3), np.uint8),
        "observation.images.wrist": np.zeros((8, 12, 3), np.uint8),
    }


def test_judgments_are_saved_before_reset_and_final_attempt_also_resets(tmp_path):
    path = tmp_path / "report.json"
    sequence = iter(
        [
            ("start", "start"),
            ("running", "success"),
            ("reset", "reset"),
            ("start", "start"),
            ("running", "failure"),
            ("reset", "reset"),
        ]
    )
    calls, clears = [], []

    def event(phase):
        expected, result = next(sequence)
        assert phase == expected
        if phase == "reset":
            saved = json.loads(path.read_text())
            assert saved["attempts"][-1]["result"] in ("success", "failure")
            assert len(calls) == len(saved["attempts"]) - 1
        return result

    executor = SimpleNamespace(
        hold=lambda: None, reset_to_pose=lambda pose: calls.append(pose) or {"reached": True}
    )
    reader = SimpleNamespace(spin_for=lambda duration: None)
    control = SimpleNamespace(next_event=event, discard_pending_input=lambda: clears.append(1))
    evaluation = evaluator_type()(
        reader, executor, control, lambda: None, attempts=2, report_path=path
    )
    report = evaluation.run()
    assert len(calls) == 2
    assert len(clears) == 4
    assert [entry["result"] for entry in report["attempts"]] == ["success", "failure"]
    assert report["summary"] == {"completed": 2, "successes": 1, "failures": 1, "success_rate": 0.5}


def test_timeout_aborts_stalled_inference_marks_failure_and_waits_for_reset(tmp_path):
    clock = [0.0]
    stopped = threading.Event()
    resets = []
    closed = []

    class Policy:
        def get_server_metadata(self):
            return {"config": "pi05_pku_franka_collect_objects"}

        def infer(self, observation):
            stopped.wait(3)
            raise OSError("request aborted")

        def close(self):
            closed.append(1)
            stopped.set()

    def spin(duration):
        clock[0] += 0.1
        time.sleep(0.001)

    def event(phase):
        if phase == "start":
            return "start"
        if phase == "reset":
            report = json.loads((tmp_path / "report.json").read_text())
            assert report["attempts"][-1]["reason"] == "timeout"
            assert not resets
            return "reset"
        return None

    reader = SimpleNamespace(spin_for=spin, get_obs=lambda timeout, **kwargs: make_observation())
    executor = SimpleNamespace(
        hold=lambda: None, reset_to_pose=lambda pose: resets.append(pose) or {"reached": True}
    )
    report = evaluator_type()(
        reader,
        executor,
        SimpleNamespace(next_event=event, discard_pending_input=lambda: None),
        Policy,
        attempts=1,
        timeout=0.5,
        clock=lambda: clock[0],
        report_path=tmp_path / "report.json",
    ).run()
    assert report["attempts"][0]["result"] == "failure"
    assert report["attempts"][0]["reason"] == "timeout"
    assert report["attempts"][0]["requests"] == 1
    assert closed and stopped.is_set() and len(resets) == 1


def test_quit_while_waiting_for_reset_preserves_judgment(tmp_path):
    sequence = iter(["start", "success", "quit"])
    executor = SimpleNamespace(
        hold=lambda: None, reset_to_pose=lambda pose: pytest.fail("reset without command")
    )
    path = tmp_path / "report.json"
    runner = evaluator_type()(
        SimpleNamespace(spin_for=lambda duration: None),
        executor,
        SimpleNamespace(
            next_event=lambda phase: next(sequence), discard_pending_input=lambda: None
        ),
        lambda: None,
        attempts=3,
        report_path=path,
    )
    report = runner.run()
    assert len(report["attempts"]) == 1 and report["summary"]["successes"] == 1
    assert json.loads(path.read_text())["attempts"][0]["reset_status"] == "waiting"


def test_success_input_during_observation_wait_is_not_replaced_by_timeout():
    sequence = iter(["start", None, "success", "reset"])

    def get_obs(timeout, *, should_stop):
        assert should_stop()
        return None

    runner = evaluator_type()(
        SimpleNamespace(get_obs=get_obs, spin_for=lambda duration: None),
        SimpleNamespace(hold=lambda: None, reset_to_pose=lambda pose: {"reached": True}),
        SimpleNamespace(
            next_event=lambda phase: next(sequence), discard_pending_input=lambda: None
        ),
        lambda: None,
        attempts=1,
    )
    report = runner.run()
    assert report["attempts"][0]["result"] == "success"
    assert report["attempts"][0]["reason"] == "manual"


def test_pause_counts_toward_deadline_but_reset_does_not_reduce_next_attempt():
    clock, resets = [0.0], []

    def event(phase):
        if phase == "start":
            return "start"
        if phase == "reset":
            return "reset"
        if phase == "running":
            return "success" if resets else "pause"
        return None

    def reset(pose):
        resets.append(1)
        clock[0] += 1000
        return {"reached": True}

    reader = SimpleNamespace(spin_for=lambda duration: clock.__setitem__(0, clock[0] + 0.1))
    runner = evaluator_type()(
        reader,
        SimpleNamespace(hold=lambda: None, reset_to_pose=reset),
        SimpleNamespace(next_event=event, discard_pending_input=lambda: None),
        lambda: None,
        attempts=2,
        timeout=0.5,
        clock=lambda: clock[0],
    )
    report = runner.run()
    assert [entry["reason"] for entry in report["attempts"]] == ["timeout", "manual"]
    assert report["attempts"][1]["result"] == "success"


def test_cleanup_error_does_not_overwrite_an_operator_success(monkeypatch):
    module = importlib.import_module("openpi_client.franka_collect.evaluation")
    state = {"pending": False, "holds": 0}

    class Pending:
        def __init__(self, *args):
            state["pending"] = True
            self.done = threading.Event()

        def abort(self):
            pass

    def event(phase):
        if phase == "start":
            return "start"
        if phase == "reset":
            return "reset"
        return "success" if state["pending"] else None

    def hold():
        state["holds"] += 1
        if state["holds"] == 1:
            raise RuntimeError("Gripper pause was not confirmed")

    monkeypatch.setattr(module, "_Request", Pending)
    runner = module.Evaluator(
        SimpleNamespace(
            get_obs=lambda **kwargs: make_observation(), spin_for=lambda duration: None
        ),
        SimpleNamespace(hold=hold, reset_to_pose=lambda pose: {"reached": True}),
        SimpleNamespace(next_event=event, discard_pending_input=lambda: None),
        lambda: None,
        attempts=1,
    )
    report = runner.run()
    entry = report["attempts"][0]
    assert entry["result"] == "success" and entry["reason"] == "manual"
    assert "Gripper pause" in entry["cleanup_error"]
    assert state["holds"] >= 2


def test_each_attempt_waits_for_space_before_creation_and_deadline_starts(tmp_path):
    clock, polls, resets = [0.0], [0], []
    path = tmp_path / "report.json"

    def event(phase):
        if phase == "start":
            # Waiting longer than five minutes must neither create nor time out a trial.
            saved = json.loads(path.read_text())
            assert len(saved["attempts"]) == len(resets)
            polls[0] += 1
            return "start" if polls[0] % 3 == 0 else None
        if phase == "running":
            return "success"
        assert phase == "reset"
        return "reset"

    runner = evaluator_type()(
        SimpleNamespace(spin_for=lambda duration: clock.__setitem__(0, clock[0] + 301)),
        SimpleNamespace(
            hold=lambda: None, reset_to_pose=lambda pose: resets.append(pose) or {"reached": True}
        ),
        SimpleNamespace(next_event=event, discard_pending_input=lambda: None),
        lambda: pytest.fail("Policy requested before Space"),
        attempts=2,
        timeout=300,
        clock=lambda: clock[0],
        report_path=path,
    )
    report = runner.run()
    assert polls[0] == 6 and len(resets) == 2
    assert [entry["reason"] for entry in report["attempts"]] == ["manual", "manual"]
    assert [entry["duration_s"] for entry in report["attempts"]] == [0, 0]


def test_quit_before_first_space_does_not_count_an_attempt():
    def event(phase):
        assert phase == "start"
        return "quit"

    runner = evaluator_type()(
        SimpleNamespace(spin_for=lambda duration: None),
        SimpleNamespace(hold=lambda: pytest.fail("Not started")),
        SimpleNamespace(next_event=event, discard_pending_input=lambda: None),
        lambda: pytest.fail("Not started"),
    )
    report = runner.run()
    assert report["attempts"] == [] and report["summary"]["completed"] == 0


def test_failed_reset_requires_retry_before_next_space_prompt(tmp_path):
    path = tmp_path / "report.json"
    phases = iter(
        [
            ("start", "start"),
            ("running", "success"),
            ("reset", "reset"),
            ("reset", "reset"),
            ("start", "quit"),
        ]
    )
    resets = []

    def event(phase):
        expected, result = next(phases)
        assert phase == expected
        if phase == "start" and resets:
            saved = json.loads(path.read_text())
            assert len(saved["attempts"]) == 1
            assert saved["attempts"][0]["reset_status"] == "completed"
        return result

    def reset(pose):
        resets.append(pose)
        if len(resets) == 1:
            raise TimeoutError("Pose error 0.0191 m")
        return {"reached": True}

    runner = evaluator_type()(
        SimpleNamespace(spin_for=lambda duration: None),
        SimpleNamespace(hold=lambda: None, reset_to_pose=reset),
        SimpleNamespace(next_event=event, discard_pending_input=lambda: None),
        lambda: pytest.fail("Unexpected policy request"),
        attempts=2,
        report_path=path,
    )
    report = runner.run()
    assert len(resets) == 2 and report["summary"]["successes"] == 1


def test_policy_close_interrupts_a_blocked_native_request():
    from websockets.sync.server import serve

    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    received, release, done = threading.Event(), threading.Event(), threading.Event()

    def server_handler(connection):
        connection.send(msgpack_numpy.packb({"config": "pi05_pku_franka_collect_objects"}))
        connection.recv()
        received.set()
        release.wait(3)

    with serve(server_handler, "127.0.0.1", 0) as server:
        server_worker = threading.Thread(target=server.serve_forever, daemon=True)
        server_worker.start()
        policy = WebsocketClientPolicy("127.0.0.1", server.socket.getsockname()[1])

        def infer():
            try:
                policy.infer({"test": 1})
            except Exception:
                pass
            finally:
                done.set()

        worker = threading.Thread(target=infer, daemon=True)
        worker.start()
        try:
            assert received.wait(1)
            policy.close()
            assert done.wait(1), "Canceled request did not unblock"
        finally:
            release.set()
            server.shutdown()
            server_worker.join(timeout=2)
            worker.join(timeout=2)
