# Franka Reset Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add an independently invoked, fixed-start robot reset.
**Architecture:** Reuse ActionExecutor for bounded motion and gripper ownership;
read state without cameras; launch a dedicated reset CLI under the existing supervisor.
**Tech Stack:** Python, NumPy/SciPy, ROS 2, Bash, pytest.
**Spec:** ../specs/2026-10-23-franka-reset-design.md

## Global Constraints

- XYZ `[0.3550, 0.0277, 0.2356]` m, Euler `[-pi, 0, 0]` rad, gripper open.
- 10 Hz, 3 mm/0.02 rad maximum command steps, 30 s timeout, 20 mm/2 degree arrival.
- No camera or policy-server dependency, robot driver stop, or automatic reset.
- Preserve unrelated local changes; this checkout has no Git metadata.

## Review Focus

- Already at target still opens the gripper.
- Stationary/stale feedback must not report successful arrival.
- Dry run must not publish commands or acquire ownership.
- Invalid/nonfinite settings must fail before controlling hardware.
- Ctrl+C must retain client cleanup and never stop the driver.

### Task 1: Reset executor and command

Files: modify `action_execution.py`, `live_observation.py`, `test_supervisor.py`,
`examples/pku_franka/run_franka.sh`, `examples/pku_franka/inference.md`;
create `reset.py`, `reset_test.py`, `examples/pku_franka/run_reset.sh` in their existing directories.

Interface: `ActionExecutor.reset_to_pose(target_pose, timeout=30.0,
observation_timeout=10.0)` uses absolute bounded targets regardless of model settings;
returns a reset report. `hold()` holds measured pose and cancels pending gripper motion.
`LiveObservationReader(..., include_cameras=False)` subscribes only to state.
Supervisor `--client-module` selects the existing client or the reset client.

- [x] Write behavioral tests for bounded convergence, opening first, already-at-target,
  stuck feedback timeout, dry run, invalid inputs, state-only reader and shell routing.
- [x] Run tests and observe expected failures before implementing.
- [x] Implement the local reset and shell entry; preserve all existing inference settings.
- [x] Run unit, CLI, shell and lint checks; exercise actual ROS using isolated fake controllers.
- [x] Review final changes and record pre-existing full-suite failures; deliver usage.

### Task 2: Pause and resume

Create `terminal_control.py` and behavioral tests; modify `client.py` and `action_execution.py`.
`TerminalControl.poll()` detects pause/quit without blocking; `wait(executor)` handles
hold/reset/resume. `execute(actions, should_pause=None)` checks the callback between
targets and returns actual executed count plus `paused`. The client discards remaining
actions after a pause and requests a fresh observation/prediction on resume.

- [x] Test pause during a chunk, hold/cancel, reset while remaining paused, resume,
  quit, non-TTY mode, and terminal restoration on exceptions.
- [x] Implement and run unit plus isolated ROS validation for reset and pause.
- [x] Update command help and usage documentation.

## Execution ledger

Ruling: implement inline in the existing non-Git deployment directory, with a local
backup, because the user requested this installed script and already selected the target.
Baseline: full-suite collection lacks `tree`; existing `far_goal` test expects an error
that the unchanged planner no longer raises.
Ruling: user added interactive pause while reset work was in progress; reuse the same
executor for paused reset so ownership is retained and both absolute/relative models work.
Review fixes: forward resolved script-local reset settings; reject reset before motion when
gripper control is disabled; use an ordered inherited pipe for supervisor pause events;
restore terminal settings if initialization output fails. Added regression tests for each.
Minor limitation: q is serviced by the paused menu after synchronous reset; Ctrl+C can abort
the reset immediately. Documented in inference.md.

Final verification: isolated ROS reset with no cameras, PTY pause/held targets, pause
countdown suspension, paused reset, fresh observation on resume, and terminal restoration
all passed. Ruff and shell syntax passed. Broad tests excluding msgpack_numpy_test.py:
64 passed; unchanged test_bad_chunks_are_rejected_before_execution[far_goal] failed.
Full collection remains blocked by missing tree dependency in msgpack_numpy_test.py.
No real robot commands were published.
