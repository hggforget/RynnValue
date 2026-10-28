# ruff: noqa: SLF001
import threading
import time

import numpy as np
import pytest

from openpi_client.action_chunk_broker import ActionChunkBroker
from openpi_client.rtc_action_chunk_broker import RTCActionChunkBroker
from openpi_client.rtc_action_chunk_broker import RTCExecutionError


class FakePolicy:
    metadata = {
        "inference_mode": "rtc",
        "action_horizon": 12,
        "action_dim": 8,
        "action_representation": "recorded_joint_targets",
        "fps": 30,
    }

    def __init__(self):
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.block_all = False
        self.bad = False
        self.active = 0
        self.max_active = 0
        self.reset_calls = 0

    def infer(self, obs):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            self.requests.append(obs)
            guided = "rtc" in obs
            if self.block and (guided or self.block_all):
                self.started.set()
                assert self.release.wait(3), "Test failed to release a blocked fake RPC"
            actions = np.broadcast_to(np.arange(12, dtype=np.float32)[:, None], (12, 8)).copy()
            if guided:
                actions += 1000
            if self.bad:
                actions = actions[:1]
            return {
                "actions": actions,
                "rtc": {"guided": guided, "inference_delay": obs.get("rtc", {}).get("inference_delay", 0)},
            }
        finally:
            self.active -= 1

    def reset(self):
        assert self.active == 0, "Transport reset raced with an RPC"
        self.reset_calls += 1

    def close(self):
        self.release.set()


@pytest.fixture
def ready_broker():
    policy = FakePolicy()
    broker = RTCActionChunkBroker(policy, execution_horizon=4, inference_delay=2)
    broker.warmup({"prompt": "pick up"})
    policy.requests.clear()
    try:
        yield broker, policy
    finally:
        broker.close()


def test_ordinary_broker_remains_independent_of_rtc_requests():
    policy = FakePolicy()
    broker = ActionChunkBroker(policy, action_horizon=4)
    actual = [broker.infer({})["actions"][0] for _ in range(5)]
    assert actual == [0, 1, 2, 3, 0]
    assert len(policy.requests) == 2 and all("rtc" not in r for r in policy.requests)


def test_rtc_rejects_standard_server_and_requires_stationary_warmup():
    policy = FakePolicy()
    policy.metadata = {**policy.metadata, "inference_mode": "standard"}
    with pytest.raises(ValueError, match="explicitly advertising"):
        RTCActionChunkBroker(policy)
    policy.metadata = {**policy.metadata, "inference_mode": "rtc"}
    broker = RTCActionChunkBroker(policy, execution_horizon=4, inference_delay=2)
    try:
        with pytest.raises(RTCExecutionError, match="warmup"):
            broker.infer({})
    finally:
        broker.close()


def test_control_continues_while_rpc_runs_and_late_actions_are_skipped(ready_broker):
    broker, policy = ready_broker
    obs = {"prompt": "pick up"}
    assert [broker.infer(obs)["actions"][0] for _ in range(4)] == [0, 1, 2, 3]
    policy.block = True
    assert broker.infer(obs)["actions"][0] == 4
    assert policy.started.wait(1)
    started = time.monotonic()
    assert broker.infer(obs)["actions"][0] == 5
    assert time.monotonic() - started < 0.2
    policy.release.set()
    broker._pending.future.result(timeout=2)
    assert broker.infer(obs)["actions"][0] == 1002
    prefix = policy.requests[-1]["rtc"]["previous_actions"]
    np.testing.assert_array_equal(prefix[:, 0], np.arange(4, 12))


def test_early_result_keeps_remaining_committed_prefix(ready_broker):
    broker, _ = ready_broker
    obs = {"prompt": "pick up"}
    for _ in range(5):
        broker.infer(obs)
    broker._pending.future.result(timeout=2)
    assert broker.infer(obs)["actions"][0] == 5
    assert broker.infer(obs)["actions"][0] == 1002


def test_reset_discards_inflight_results_and_serializes_transport_reset(ready_broker):
    broker, policy = ready_broker
    obs = {"prompt": "pick up"}
    policy.block = True
    for _ in range(5):
        broker.infer(obs)
    assert policy.started.wait(1)
    broker.reset()
    policy.release.set()
    assert broker.infer(obs)["actions"][0] == 0
    assert policy.reset_calls == 1 and policy.max_active == 1


def test_prompt_change_clears_old_queue(ready_broker):
    broker, policy = ready_broker
    broker.infer({"prompt": "pick up"})
    assert broker.infer({"prompt": "put down"})["actions"][0] == 0
    assert policy.reset_calls == 1


def test_reset_during_initial_rpc_does_not_poison_new_generation(ready_broker):
    broker, policy = ready_broker
    policy.block = policy.block_all = True
    failures = []

    def control():
        try:
            broker.infer({"prompt": "pick up"})
        except RTCExecutionError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=control)
    thread.start()
    assert policy.started.wait(1)
    broker.reset()
    policy.release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert "reset during initial" in failures[0]
    assert broker.infer({"prompt": "pick up"})["actions"][0] == 0


def test_queue_exhaustion_surfaces_instead_of_replaying_stale_actions(ready_broker):
    broker, policy = ready_broker
    policy.block = True
    for _ in range(12):
        broker.infer({"prompt": "pick up"})
    with pytest.raises(RTCExecutionError, match="queue exhausted"):
        broker.infer({"prompt": "pick up"})
    with pytest.raises(RTCExecutionError, match="reset"):
        broker.infer({"prompt": "pick up"})


def test_background_errors_surface_on_control_thread(ready_broker):
    broker, policy = ready_broker
    for _ in range(4):
        broker.infer({"prompt": "pick up"})
    policy.bad = True
    broker.infer({"prompt": "pick up"})
    assert broker._pending.future.exception(timeout=2) is not None
    with pytest.raises(RTCExecutionError, match="invalid action chunk"):
        broker.infer({"prompt": "pick up"})
