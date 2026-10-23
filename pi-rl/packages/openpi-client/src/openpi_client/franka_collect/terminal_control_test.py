import importlib
import io
import os
import pty
import select
import termios
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def terminal():
    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r")
    original = termios.tcgetattr(slave)
    try:
        yield master, slave, stream, original
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def control_type():
    return importlib.import_module("openpi_client.franka_collect.terminal_control").TerminalControl


def test_pause_reset_remains_paused_until_continue_and_restores_terminal(terminal):
    master, slave, stream, original = terminal
    events = []
    reader = SimpleNamespace(spin_for=lambda duration: None)
    executor = SimpleNamespace(
        hold=lambda: events.append("hold"),
        reset_to_pose=lambda pose: events.append(np.asarray(pose)),
    )
    output = io.StringIO()
    with control_type()(reader, stream=stream, output=output) as control:
        os.write(master, b"prc")
        assert select.select([slave], [], [], 1)[0]
        assert control.poll()
        control.wait(executor)
        assert not control.paused
        assert events[0] == "hold" and len(events) == 2
        np.testing.assert_allclose(events[1], [0.3550, 0.0277, 0.2356, -np.pi, 0, 0])
    assert termios.tcgetattr(slave) == original
    assert "r" in output.getvalue() and "c" in output.getvalue()


def test_quit_from_paused_menu_restores_terminal(terminal):
    master, slave, stream, original = terminal
    reader = SimpleNamespace(spin_for=lambda duration: None)
    executor = SimpleNamespace(hold=lambda: None)
    with pytest.raises(KeyboardInterrupt):
        with control_type()(reader, stream=stream, output=io.StringIO()) as control:
            os.write(master, b" q")
            assert select.select([slave], [], [], 1)[0]
            assert control.poll()
            control.wait(executor)
    assert termios.tcgetattr(slave) == original


def test_noninteractive_input_does_not_enable_keyboard_control():
    with control_type()(None, stream=io.StringIO(), output=io.StringIO()) as control:
        assert not control.enabled and not control.poll()


def test_terminal_is_restored_when_startup_output_fails(terminal):
    master, slave, stream, original = terminal

    class ClosedOutput:
        def write(self, value):
            raise BrokenPipeError("output closed")

    with pytest.raises(BrokenPipeError):
        with control_type()(None, stream=stream, output=ClosedOutput()):
            pass
    assert termios.tcgetattr(slave) == original
