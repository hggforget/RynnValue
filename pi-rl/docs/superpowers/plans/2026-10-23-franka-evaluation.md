# Franka Evaluation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Configurable attempts, manual success/failure, explicit reset and automatic
failure after 300 seconds per attempt.
**Architecture:** Dedicated evaluation runner and single-key terminal atop the
existing executor; cancellable policy worker keeps ROS and operator input responsive.
**Tech Stack:** Python, ROS 2, native WebSocket, NumPy, Bash, pytest and PTYs.
**Spec:** ../specs/2026-10-23-franka-evaluation-design.md

## Global Constraints

- ATTEMPTS=10 configurable; ATTEMPT_TIMEOUT=300 seconds configurable.
- Pause counts toward attempt deadline; reset/waiting does not.
- Judgment is saved before waiting for reset; final attempt also resets.
- Fixed start and robot-driver/camera behavior remain as previously requested.
- Existing deployment has no Git metadata. Preserve other local edits.

## Review Focus

- Single-key judgments must be recognized without Enter.
- Stalled inference must not defeat the deadline or hold transition.
- Repeated/early commands must not record an attempt twice or bypass reset.
- Quit/reset failure must preserve already recorded results.
- New attempts must use post-reset feedback and no stale worker result.
- Every attempt requires a fresh Space; waiting must not consume time or attempts.

### Task 1: Terminal and evaluation engine

Create evaluation_control.py, evaluation.py and their behavioral tests.
Extend TerminalControl instruction prompts and WebsocketClientPolicy cancellation/
single-connect behavior without changing defaults for existing consumers.

- [x] Write failing PTY, fake-clock lifecycle and local WebSocket cancellation tests.
- [x] Implement running/paused/reset command handling, attempt lifecycle and atomic report.
- [x] Verify success/failure/timeout, explicit final reset, blocked I/O, and quit persistence.

### Task 2: Client integration and launch settings

Modify client.py CLI, run_franka.sh and inference.md; add CLI/routing tests.

- [x] Add --attempts (0 keeps legacy mode) and --attempt-timeout=300 validation.
- [x] Enable attempts only for normal live execution from run_franka.sh.
- [x] Verify full workflow on unique fake ROS topics and a local policy server.
- [x] Run regression/lint/shell checks and obtain a read-only code review.

## Ledger

Ruling: default 10 attempts is an implementation choice, announced to the user.
The user's explicit 5-minute request supersedes the earlier removed global 90-second limit.
No additional design approval is needed for this requested extension.

Review fixes: cooperative sensor-wait cancellation, preserve judgments after cleanup
errors, flush pending TTY input on state transitions, and count dispatched requests.
Regression tests observed failing before each correction.

Final verification: evaluation PTY + isolated ROS passed manual success, manual failure,
explicit r before each reset, fresh post-reset observation, stalled native policy timeout,
final reset, aggregate results and terminal restoration. Legacy reset/pause ROS test also
passed. Broad tests: 83 passed, existing far_goal failure unchanged. Full collection still
requires missing tree dependency. Ruff and shell syntax checks passed; no real motion.

### Follow-up: Space to start every round

- [x] Reproduce missing start gate with failing PTY and fake-clock lifecycle tests.
- [x] Wait for fresh Space before creating each attempt; preserve reset/final-exit flow.
- [x] Update prompts and operator guide.
- [x] Verify initial/post-reset idle, manual judgments, timeout, final reset on isolated ROS.

Space-start verification passed: initial and post-reset idle waited longer than the
test's 3-second attempt budget without model requests, pose commands or new attempts.
Manual success/failure and blocked-policy timeout completed all three rounds with
explicit r and Space, final reset, normal supervisor exit and restored terminal.
Fake-clock tests waited over 300 seconds before each Space without consuming budget.
Final follow-up checks: 87 regression tests passed, excluding the unchanged far_goal
baseline failure; Ruff and Bash syntax passed. Independent read-only review found no
issues in the start gate. Reset retry failure also retains the current round correctly.

Read-only investigation: latest hardware log is from evaluation mode, not the pasted
older global-timeout traceback. Its resets failed at approximately 19 mm / 0.029 rad
measured error. The evaluator correctly retained the reset stage. Actual hardware
arrival has not been verified; do not bypass arrival checks to advance a round.
