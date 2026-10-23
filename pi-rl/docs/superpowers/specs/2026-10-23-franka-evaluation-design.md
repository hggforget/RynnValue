# Franka repeated evaluation

The user requests two running judgments, s=success and f=failure, followed by an explicit
r=reset/retry command. Timeout is failure. Set a configurable number of attempts and a
five-minute maximum per attempt.

The normal executing run_franka.sh entry enables evaluation with ATTEMPTS=10
(operator configurable) and ATTEMPT_TIMEOUT=300 seconds. Existing replay,
prediction-only, dry-run and legacy direct clients remain available. Evaluation
requires an interactive terminal and gripper control because every attempt must
be followed by the previously selected fixed-start reset.

Operator presses s or f without Enter. End the current action prefix,
hold measured pose, cancel unfinished gripper motion, save the judgment, then
wait for the single key r entered after the reset prompt. Never move automatically
while waiting. Reset returns to XYZ [.3550,.0277,.2356], Euler [-pi,0,0], gripper
open. Every attempt, including the first and each post-reset attempt, requires a
fresh Space entered after the ready prompt. Waiting must not create an attempt,
request a policy or execute targets. Reset after the final attempt before ending.
Preserve pause/continue and manual quit. During evaluation, paused reset is gated
by a judgment so no attempt can be silently discarded.

300 seconds is elapsed wall time for each running attempt, including pause and
inference I/O, starting only after Space. Waiting for Space/reset and reset motion
are outside that budget. Timeout
must hold and record failure, not exit the evaluator. A network request must not
block judgment input or the deadline: run policy I/O in a cancellable daemon worker,
abort its socket and discard stale results when the attempt ends or pauses.

Save each attempt and reset outcome atomically as it happens, including result,
reason, duration and request count, and aggregate successes/failures/success rate.
Keep robot driver running and do not reintroduce camera age checks. No real robot
motion in verification.

Approach: extend the existing local client with a dedicated evaluation runner and
single-key terminal control. A supervisor-wide timeout would terminate the client and prevent
the required reset/next-attempt workflow, so retain deadlines inside the evaluator.
