"""Hardware-free tests: python -m pytest examples/lingdong_real/test_lingdong_real.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

import common as C
import executor as E
from executor import RobotExecutor
from safety import ActionSafety, SafetyViolation
from tron2_link import CommandsNotArmed, FakeLink, Tron2Link

REF = np.array([-0.4, -0.25, 0.2, -1.0, -0.35, 0.3, 0.0, 0.5])


def test_state_by_name():
    q16 = np.arange(16) * 0.01
    s18 = C.build_raw_state(q16, 30, 94)
    s8 = C.select_right(s18)
    assert np.allclose(s8[:7], q16[7:14]) and np.isclose(s8[7], 0.94)


def test_process_is_verbatim():
    s = ActionSafety()
    rng = np.random.default_rng(0)
    chunk = REF + rng.normal(0, 1.0, (10, 8))  # huge jumps, out of range joints, gripper > 1 and < 0
    chunk[0, 3] += 3.0
    chunk[:, 7] = np.linspace(-0.5, 1.5, 10)
    out, rep = s.process(chunk, REF)
    assert np.array_equal(out, chunk) and rep.first_jump > 2.0


def test_nan_inf_and_shape_stop():
    s = ActionSafety()
    for bad_val in (np.nan, np.inf, -np.inf):
        bad = np.tile(REF, (5, 1)); bad[2, 0] = bad_val
        with pytest.raises(SafetyViolation):
            s.process(bad, REF)
    with pytest.raises(SafetyViolation):
        s.process(np.zeros((5, 7)), REF)


def test_state_warnings_never_block():
    s = ActionSafety()
    st = REF.copy(); st[5] = -0.745; st[0] = -5.0
    msgs = s.check_state(st, strict=True)
    assert msgs and all("warning only" in m for m in msgs)
    with pytest.raises(SafetyViolation):
        s.check_state(np.full(8, np.nan))


def _executor(ref8):
    q16 = np.zeros(16); q16[:7] = 0.11; q16[7:14] = ref8[:7]; q16[14:] = [0.2, -0.1]
    link = FakeLink(q16, right_opening=ref8[7] * 100, left_opening=30)
    link.arm_commands("I_UNDERSTAND_THE_ROBOT_WILL_MOVE")
    ex = RobotExecutor(link, control_hz=30)
    ex.start()
    return link, ex


def test_executor_sends_model_output_verbatim(monkeypatch):
    monkeypatch.setattr(E, "_sleep_until", lambda t: None)
    ref = REF.copy(); ref[5] = -0.745
    chunk = np.tile(ref, (10, 1)); chunk[:, 5] += 0.578 + np.linspace(0, 0.4, 10)  # 0.58 rad first jump
    chunk[3, 0] = 2.5  # far outside the recorded range, single-step spike
    chunk[:, 7] = np.linspace(0.0, 1.2, 10)  # gripper beyond 0.94
    link, ex = _executor(ref)
    out, _ = ActionSafety().process(chunk, ref)
    ex.execute(out)
    sj = np.array([c[2] for c in link.commands if c[1] == "request_servoj"])
    assert len(sj) == len(chunk)  # one servoj per model step: no interpolation, no ramp
    assert np.max(np.abs(sj[:, 7:14] - chunk[:, :7])) == 0.0
    assert np.allclose(sj[:, :7], 0.11) and np.allclose(sj[:, 14:], [0.2, -0.1])  # left arm / head held
    gr = [c[2] for c in link.commands if c[1] == "request_set_limx_2fclaw_cmd"]
    assert [g[0] for g in gr] == [30.0] * len(gr)
    assert np.allclose([g[1] for g in gr], chunk[:, 7] * 100)  # every value differs by >= 1 % -> all sent, unclipped
    assert np.max(np.abs(np.asarray(ex.sent_targets) - chunk)) == 0.0
    with pytest.raises(SafetyViolation):
        bad = chunk.copy(); bad[4, 2] = np.nan
        ex.execute(bad)


def test_fake_link_gated():
    link = FakeLink(np.zeros(16), right_opening=50)
    with pytest.raises(CommandsNotArmed):
        link.servoj(np.zeros(16))


def test_real_link_rejects_unknown_titles():
    link = Tron2Link.__new__(Tron2Link)
    link._armed = False
    for t in ("request_servoj", "request_movej", "request_emgy_stop", "request_set_limx_2fclaw_cmd"):
        with pytest.raises(CommandsNotArmed):
            link._send(t, {})
    link._armed = True
    with pytest.raises(CommandsNotArmed):
        link._send("request_movej", {})


def test_cli_and_confirm(monkeypatch):
    import main as M
    a = M.parse_args([])
    assert not a.yes and not hasattr(a, "ramp_speed") and not hasattr(a, "max_joint_step")
    for gone in ("--ramp-speed", "--max-step-delta", "--max-lag", "--servo-hz", "--max-state-age-ms"):
        with pytest.raises(SystemExit):
            M.parse_args([gone, "1"])
    assert M._confirm("x", assume_yes=True)
    monkeypatch.setattr(M.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert M._confirm("x")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert not M._confirm("x")
