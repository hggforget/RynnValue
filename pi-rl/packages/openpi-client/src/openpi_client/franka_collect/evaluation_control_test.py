import importlib
import io
import os
import pty
import select

import pytest


@pytest.fixture
def control():
    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r")
    cls = importlib.import_module(
        "openpi_client.franka_collect.evaluation_control"
    ).EvaluationControl
    try:
        with cls(None, stream=stream, output=io.StringIO()) as instance:
            yield master, slave, instance
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize("command,event", [("s", "success"), ("f", "failure")])
def test_running_judgment_is_a_single_key_without_enter(control, command, event):
    master, slave, instance = control
    os.write(master, command.encode())
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("running") == event


def test_retry_key_is_only_allowed_after_judgment(control):
    master, slave, instance = control
    os.write(master, b"r")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("running") is None
    os.write(master, b"r")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("reset") == "reset"


def test_pause_and_continue_remain_available(control):
    master, slave, instance = control
    os.write(master, b"p")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("running") == "pause"
    os.write(master, b"c")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("paused") == "resume"


def test_reset_prompt_discards_retry_keys_from_before_the_verdict(control):
    master, slave, instance = control
    os.write(master, b"pr")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("running") == "pause"
    os.write(master, b"r")
    assert select.select([slave], [], [], 1)[0]
    instance.discard_pending_input()
    assert instance.next_event("reset") is None
    os.write(master, b"r")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("reset") == "reset"


def test_each_start_requires_space_and_ignores_judgment_and_retry_keys(control):
    master, slave, instance = control
    os.write(master, b"sfrpc")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("start") is None
    os.write(master, b" ")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("start") == "start"


def test_start_prompt_discards_spaces_entered_during_reset(control):
    master, slave, instance = control
    os.write(master, b"r ")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("reset") == "reset"
    instance.discard_pending_input()
    assert instance.next_event("start") is None
    os.write(master, b" ")
    assert select.select([slave], [], [], 1)[0]
    assert instance.next_event("start") == "start"
