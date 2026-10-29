import json

import pytest

from . import monitor_lingdong_training as monitor


def checkpoint(root, step, *, complete=True, optimizer=True):
    path = root / str(step)
    (path / "params").mkdir(parents=True)
    (path / "params/_METADATA").write_text("{}")
    if optimizer:
        (path / "train_state").mkdir()
        (path / "train_state/_METADATA").write_text("{}")
    (path / "assets/lingdong_right_arm").mkdir(parents=True)
    (path / "assets/lingdong_right_arm/norm_stats.json").write_text("{}")
    (path / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1 if complete else None}))
    return path


def test_only_committed_checkpoints_with_optimizer_state_are_resumed(tmp_path):
    valid = checkpoint(tmp_path, 1000)
    checkpoint(tmp_path, 2000, complete=False)
    checkpoint(tmp_path, 3000, optimizer=False)
    (tmp_path / "4000.orbax-checkpoint-tmp-0").mkdir()
    assert monitor.complete_checkpoints(tmp_path) == [(1000, valid)]


def test_resume_preserves_training_parameters_and_experiment_without_copying_weights():
    original = [
        "python3",
        "scripts/train.py",
        monitor.CONFIG_NAME,
        "--exp-name",
        "old",
        "--batch-size",
        "128",
        "--data.dataset-root",
        "/oss/data",
        "--fsdp-devices",
        "8",
    ]
    actual = monitor.resume_argv(original, "new", "/python")
    assert monitor.option(actual, "--exp-name") == "new"
    assert monitor.option(actual, "--batch-size") == "128"
    assert monitor.option(actual, "--data.dataset-root") == "/oss/data"
    assert monitor.option(actual, "--fsdp-devices") == "8"
    assert actual.count("--resume") == 1
    assert original[0] == "python3"


def test_overwrite_is_never_used_for_automatic_resume():
    with pytest.raises(ValueError, match="overwrite"):
        monitor.resume_argv(["python", "train.py", monitor.CONFIG_NAME, "--overwrite"], "x", "python")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Traceback\nInvalidDataError: Error parsing OBU data", "recoverable_error"),
        ("libdav1d error\nTraceback\nKeyboardInterrupt", "manual_stop"),
        ("Traceback\nValueError: incompatible model", "needs_attention"),
        ("Waiting for checkpoint manager to finish\nSaved successfully", "complete"),
        ("Waiting for checkpoint manager to finish\nTraceback\nRuntimeError: save failed", "needs_attention"),
    ],
)
def test_manual_stops_and_unknown_errors_do_not_trigger_blind_restarts(text, expected):
    assert monitor.classify_exit(text) == expected


def test_progress_parser_handles_tmux_ansi_and_carriage_returns():
    assert monitor.progress_step("Step 100: loss=.1\r  7%|abc| 1366/20000 [2:48:00<30:00:00]\x1b[A") == 1366
    assert monitor.progress_step("loading parameters") is None


def test_resume_arguments_support_equals_syntax_and_no_duplicate_flags():
    argv = ["python", "train.py", monitor.CONFIG_NAME, "--exp-name=old", "--resume"]
    result = monitor.resume_argv(argv, "new", "python")
    assert monitor.option(result, "--exp-name") == "new"
    assert result.count("--resume") == 1


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    inventory = dataset / "lerobot_2026-10-26_12-09-32_final/meta/file_checksums.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("{}")
    state = tmp_path / "monitor"
    state.mkdir()
    (state / "recovery_ready").touch()
    config = {
        "pane": "%0",
        "initial_pid": 42,
        "cwd": str(tmp_path),
        "python": "/python",
        "environment": {},
        "argv": [
            "python",
            "scripts/train.py",
            monitor.CONFIG_NAME,
            "--exp-name",
            "current",
            "--data.dataset-root",
            str(dataset),
            "--checkpoint-base-dir",
            "checkpoints",
        ],
    }
    instance = monitor.Supervisor(config, state)
    checkpoint(instance.checkpoint_root / "current", 1000)
    monkeypatch.setattr(monitor, "training_in_pane", lambda _: None)
    return instance


def test_healthy_training_is_observed_without_sending_commands(supervisor, monkeypatch):
    monkeypatch.setattr(monitor.subprocess, "check_output", lambda *a, **kw: "99\n")
    monkeypatch.setattr(monitor, "training_in_pane", lambda _: {"pid": 42, "argv": supervisor.config["argv"]})

    def refuse_mutation(*args, **kwargs):
        pytest.fail("Healthy training must never receive an injected command")

    monkeypatch.setattr(monitor.subprocess, "run", refuse_mutation)
    supervisor.log_path.write_text("Step 1200: loss=0.1\n")
    assert supervisor.poll()
    assert supervisor.status == "running"
    assert supervisor.last_seen_step == 1200
    assert supervisor.saved_step == 1000
    assert monitor.read_json(supervisor.directory / "status.json")["automatic_restarts"] == 0


def test_recoverable_failure_resumes_same_experiment_once_from_complete_checkpoint(supervisor, monkeypatch):
    responses = iter(["99\n", "bash\n", "root@host:~#\n"])
    monkeypatch.setattr(monitor.subprocess, "check_output", lambda *a, **kw: next(responses))
    sent = []
    monkeypatch.setattr(monitor.subprocess, "run", lambda argv, **kw: sent.append(argv))
    supervisor.log_path.write_text("Traceback\nav.error.InvalidDataError: Error parsing OBU data\n")
    assert supervisor.poll()
    plan = monitor.read_json(supervisor.directory / "resume_1.json")
    assert plan["checkpoint"].endswith("/current/1000")
    assert monitor.option(plan["argv"], "--exp-name") == "current"
    assert "--resume" in plan["argv"]
    assert "--run-plan" in sent[0][-1]
    assert sent[1][-1] == "Enter"
    assert supervisor.restarts == 1


def test_waiting_for_idle_shell_does_not_consume_retry_budget(supervisor, monkeypatch):
    supervisor.last_resume_step = 1000
    monkeypatch.setattr(monitor.subprocess, "check_output", lambda *a, **kw: "vim\n")
    for _ in range(5):
        supervisor.restart(99)
    assert supervisor.status == "waiting_for_idle_pane"
    assert supervisor.restarts == 0
    assert supervisor.failures_without_progress == 0


def test_manual_stop_leaves_training_stopped(supervisor, monkeypatch):
    monkeypatch.setattr(monitor.subprocess, "check_output", lambda *a, **kw: "99\n")
    supervisor.log_path.write_text("Traceback\nKeyboardInterrupt\n")
    assert supervisor.poll()
    assert supervisor.status == "manual_stop"
    assert supervisor.restarts == 0
