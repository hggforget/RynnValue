"""Check launcher routing and Ctrl+C logging without starting ROS or a robot client."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[5] / "examples/pku_franka/run_franka.sh"


def launcher_environment(tmp_path):
    runtime = tmp_path / "runtime.sh"
    runtime.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    env = os.environ.copy()
    for name in (
        "MODE",
        "EXECUTE",
        "DRY_RUN",
        "ACTION_REPRESENTATION",
        "PORT",
        "REPEAT",
        "EXECUTE_STEPS",
        "CONTROL_HZ",
        "INFERENCE_HOST",
    ):
        env.pop(name, None)
    capture = tmp_path / "arguments.txt"
    env.update(
        PYTHON="",
        ROBOT_RUNTIME=str(runtime),
        CAPTURE_ARGS=str(capture),
        RUN_DIR=str(tmp_path / "results"),
        PI05_RUNTIME_DIR=str(tmp_path / "runtime"),
    )
    return env, capture


@pytest.mark.parametrize(
    "arguments,settings,module,mode,execute,dry,port",
    [
        ([], {}, "test_supervisor", "inference", True, False, "8000"),
        (["replay"], {}, "client", "replay", False, False, "8000"),
        ([], {"MODE": "replay"}, "client", "replay", False, False, "8000"),
        ([], {"EXECUTE": "0"}, "client", "inference", False, False, "8000"),
        ([], {"DRY_RUN": "1"}, "client", "inference", False, True, "8000"),
        (
            [],
            {"ACTION_REPRESENTATION": "relative"},
            "test_supervisor",
            "inference",
            True,
            False,
            "8001",
        ),
    ],
)
def test_default_execution_and_read_only_overrides(
    tmp_path, arguments, settings, module, mode, execute, dry, port
):
    env, capture = launcher_environment(tmp_path)
    env.update(settings)
    result = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    received = capture.read_text().splitlines()
    assert f"openpi_client.franka_collect.{module}" in received
    assert ("--execute" in received) == execute
    assert ("--dry-run-execution" in received) == dry
    assert "--disable-gripper" not in received
    assert received[received.index("--mode") + 1] == mode
    assert received[received.index("--port") + 1] == port
    assert received[received.index("--repeat") + 1] == "10000"
    if mode == "inference":
        assert received[received.index("--execute-steps") + 1] == "10"
        assert received[received.index("--control-hz") + 1] == "30"


@pytest.mark.parametrize(
    "arguments,settings",
    [
        (["replay"], {"EXECUTE": "1"}),
        ([], {"EXECUTE": "1", "DRY_RUN": "1"}),
    ],
)
def test_invalid_execution_combinations_do_not_launch(tmp_path, arguments, settings):
    env, capture = launcher_environment(tmp_path)
    env.update(settings)
    result = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert not capture.exists()


def test_ctrl_c_keeps_log_pipe_open_for_supervisor_shutdown_output(tmp_path):
    env, _ = launcher_environment(tmp_path)
    ready = tmp_path / "ready"
    stub = tmp_path / "signal_stub.py"
    stub.write_text(
        "import signal,time\nfrom pathlib import Path\n"
        "def stop(*_):\n    print('STUB: driver stop confirmed',flush=True)\n"
        "    raise SystemExit(130)\n"
        "signal.signal(signal.SIGINT,stop)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(.1)\n"
    )
    Path(env["ROBOT_RUNTIME"]).write_text(
        '#!/usr/bin/env bash\nexec "$CAPTURE_PYTHON" "$CAPTURE_SCRIPT"\n'
    )
    env.update(CAPTURE_PYTHON=sys.executable, CAPTURE_SCRIPT=str(stub))
    process = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None
            if time.monotonic() >= deadline:
                raise AssertionError("Signal stub did not start")
            time.sleep(0.01)
        os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 130, output
        assert "STUB: driver stop confirmed" in output
        assert "STUB: driver stop confirmed" in (tmp_path / "results/client.log").read_text()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
