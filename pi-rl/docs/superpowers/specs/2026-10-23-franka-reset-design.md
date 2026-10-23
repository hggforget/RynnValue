# Franka fixed-start reset

The user selected the existing collection start: base-frame XYZ
`[0.3550, 0.0277, 0.2356]` metres and Euler XYZ `[-pi, 0, 0]` radians,
with the gripper open. Expose it as an independent `run_franka.sh reset` command.

Reuse ActionExecutor ownership, controller activation, bounded Cartesian targets,
gripper completion, and measured-pose hold on exit. A local reset command reads
robot feedback without requiring cameras or a policy server. It moves at 10 Hz,
at most 3 mm and 0.02 rad per target, and confirms measured arrival within 20 mm
and 2 degrees within 30 seconds. These arrival tolerances were relaxed at the user's
request after observing residual errors of 19.1 mm / 0.0293 rad. Open the gripper
before moving. Dry run previews
the target without acquiring control. Errors and interruption terminate only the
client; no robot driver stop requests or automatic reset on inference exit.

Approaches: calling the recording UI requires the teleoperation recorder to own
the arm, which conflicts with inference. The existing inference executor already
provides the required ownership and control path, so use it for the selected target.
Testing uses virtual feedback and isolated ROS topics, never real motion.

During interactive real-time execution, `p` or Space pauses between published
targets. Hold the measured arm pose and cancel unfinished gripper motion. A paused
menu accepts `r` (fixed-start reset, then remain paused), `c` (resume using a new
observation and prediction), or `q` (quit). Reset reuses the current executor,
including its ownership lock, regardless of the model action representation.
Restore terminal settings on every exit. Noninteractive input keeps existing behavior.
