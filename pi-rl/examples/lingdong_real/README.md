# Lingdong Tron2 右臂 pi05 真机推理 (`pi05_lingdong_right_arm`)

Synchronous-chunk deployment of the `pi05_lingdong_right_arm` checkpoint on the Tron2 (right arm + right
gripper), in the style of the Franka client (`packages/openpi-client/.../franka_collect`). **Default is DRY RUN**:
observe → infer → log. Nothing is sent to the robot unless `--execute` is given *and* you press
Enter at the summary prompt (or pass `--yes`). **Joint targets are sent verbatim; gripper values are converted to integer percentages**
(see "Execution / safety" below).

## ⚠️ Must-know: Orin XLA bug
With JAX/XLA's default Triton GEMM autotuning on this Orin, the pi05 forward pass is numerically wrong (log shows
`gemm_fusion_autotuner ... Results do not match the reference`) and the policy outputs the **same chunk for any
state / image / prompt** (≈ dataset mean). `run.sh`, `serve.sh` and `LocalPolicy` therefore set
`XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"`. Never run this model without it
(this also applies to `/mnt/hdc/.infer-test` benchmarks and any `serve_policy.py` launch).

## Files
| file | role |
|---|---|
| `main.py` | CLI + control loop (modes `run`, `camera-check`, `state-check`) |
| `common.py` | constants, 18-D raw layout by joint name, 8-D selection, policy obs, prompts, camera serials |
| `safety.py` | raw passthrough: NaN/inf/shape check only, state range warnings, RAW banner |
| `tron2_link.py` | Tron2 WebSocket (`ws://10.192.1.2:5000`) client, read-only unless armed; `FakeLink` |
| `executor.py` | raw servoj streaming: one servoj per model step at 30 Hz, no interpolation; holds left arm/head/left gripper; hold on stop |
| `cameras.py` | RealSense capture (640x480 bgr8@30 + depth, → RGB), latest-frame with age check |
| `backends.py` | `Tron2Backend` (live) and `ReplayBackend` (LeRobot episode, PyAV decode) |
| `policy_backends.py` | `LocalPolicy` (in-process, default) / `WebsocketPolicy` (serve_policy.py) |
| `run.sh`, `serve.sh` | launchers (env: no prealloc, Triton off, JAX compile cache on /mnt/hdc, NO_PROXY) |
| `test_lingdong_real.py` | hardware-free unit tests |
| `live_view.py`, `live_view.html` | timestamped video, action/state curves and recording |
| `viewer_server.py` | persistent HTTP viewer, including startup-error reporting |
| `test_live_view.py` | telemetry, video encoding and persistent-viewer tests |

## Observation (as in training)
* state: robot `response_get_joint_state.data.q` (16 = L7, R7, head2) + `response_get_limx_2fclaw_state`
  `left/right_opening/100` → raw 18-D in dataset name order → right 8 = `abad/hip/yaw/knee/wrist_yaw/wrist_pitch/wrist_roll_R_Joint, right_gripper`.
* images: `observation.images.cam_high` (head D455) and `observation.images.cam_right_wrist`, RGB uint8 480x640;
  resize/pad to 224 happens inside the policy transform exactly as in training.
* **Wrist correction**: in the raw collection data the stream named `cam_left_wrist` shows the active right arm and
  `cam_right_wrist` the idle arm; the final dataset exchanged the two wrist MP4s (no flip/rotation). So the policy's
  `cam_right_wrist` = the camera the collection called `cam_left_wrist`. Default serial `427622271953` (the old
  deploy's "left_wrist" serial) — **verify with `--mode camera-check`** (the POLICY_cam_right_wrist image must show
  the right gripper); override with `--right-wrist-serial`. Replay on raw data uses `--replay-wrist-source cam_left_wrist` (default).
* prompt: `--task pick_up` uses the user-confirmed training string `pick up the object`.
  The `put_down` fallback is unverified and prints a warning; use `--prompt "<exact string>"`
  from the final dataset's `meta/tasks.jsonl` when needed.

## Run

### Live video and action/state curves

Visualization is enabled by default, including for the existing real-robot command:

```bash
bash examples/lingdong_real/run.sh --policy websocket --port 8000 --task pick_up --execute --execute-steps 30 --max-cycles 100
```

Open **http://10.192.1.155:8001** from the laptop on the robot LAN (or http://127.0.0.1:8001 on Orin).
The page shows head and corrected right-wrist RGB plus all eight right-arm/gripper action/state curves.
Orange is the raw model target for each successfully sent executor step; green is independently sampled
live robot feedback at 20 Hz. The gripper plot uses the model's 0..1 scale; the transport still multiplies
by 100, rounds and clamps to 0..100. During inference, the target line holds the last sent target.
In `--dry-run-execution`, targets go to FakeLink and are labelled simulated; normal dry runs have no sent-action line.

Camera previews and a combined camera/curve recording run at 8 FPS in a background worker; the viewer reads
the existing camera/robot caches and does not open another camera or send robot commands. Control callbacks
only enqueue telemetry. `--viz-fps 4` lowers recording load, `--viz-port 8002` selects another viewer port,
and `--no-visualize` disables telemetry/recording for the new run (the independent HTTP process remains available).
Port 8000 remains the policy server.

Each run's existing log directory also contains:

* `dashboard.mp4`: both cameras and all eight timestamp-aligned action/state curves (last 20 seconds).
* `visualization.html`: interactive recorded curves and synchronized video playback; keep it beside the MP4.
* `telemetry.jsonl` / `telemetry.csv`: timestamped action and feedback events (separate rows, no synthetic feedback).
* `visualization_meta.json`: recording status, video time offset and any dropped viewer events.

The HTTP viewer is an independent process, started automatically BEFORE robot/camera initialization.
It stays available after inference exits and displays initialization errors (such as a missing head camera).
It follows new runs automatically; the existing inference command reuses port 8001 instead of rebinding it.
The viewer reads log files and cached JPGs only; it does not connect to the robot or cameras.
If needed, start just the viewer with `.venv/bin/python examples/lingdong_real/viewer_server.py`.
For recorded playback, open `visualization.html` from the saved log directory alongside `dashboard.mp4`.

```bash
cd /mnt/hdc/RynnValue/pi-rl
examples/lingdong_real/run.sh --mode state-check                    # read-only robot state + range warnings
examples/lingdong_real/run.sh --mode camera-check                   # read-only, saves labelled jpgs
examples/lingdong_real/run.sh --backend replay --replay-episode 131 # offline replay, MAE vs recorded actions
examples/lingdong_real/run.sh --prompt "..."                        # live DRY RUN (cameras+state, no commands)
examples/lingdong_real/run.sh --prompt "..." --dry-run-execution    # executor against FakeLink
examples/lingdong_real/run.sh --prompt "..." --execute              # REAL MOTION after pressing Enter (--yes skips)
# Option B
examples/lingdong_real/serve.sh &   # port 8000
examples/lingdong_real/run.sh --policy websocket --port 8000 --prompt "..."
```
Useful flags: `--execute-steps 15` (1..30), `--control-hz 30`, `--num-steps 10`, `--warmup 2`,
`--max-cycles`, `--servo-mode {0,1}` (not sent by default, like the old deploy), `--yes`.
Logs: `/mnt/hdc/lingdong_real_logs/<time>_.../` → `run_meta.json`, `cycles.jsonl` (latency breakdown, safety report,
state, first action, MAE), `cycles.npz` (state8, raw_chunk, safe_targets (= raw_chunk[:execute_steps]), recorded),
`sent_targets.npy` (every right8 target sent), with `--dry-run-execution` also `fake_link_servoj_q16.npy` /
`fake_link_gripper.npy` (what the fake robot received, incl. 3 final hold re-sends), periodic input jpgs.

## Execution / safety (RAW passthrough — the only execution mode)
**Joint targets are passed through unchanged.** Each of the first `--execute-steps` actions of a chunk is sent
as one `request_servoj` (right 7 joints = model output, left arm/head = values held at start) at `--control-hz`, plus
the gripper command `right_opening = action[7] * 100` (the transport rounds and clamps it to an int 0..100; an unchanged
percent is not re-sent). There is **no** interpolation, first-action / inter-chunk ramp, per-step delta limit,
joint clipping, tracking-lag stop, or stale-state / stale-image stop. Gripper conversion at the transport remains as described above. A large first action is executed as a
direct jump at full servo speed. A warning banner is printed at start.

What remains:
* dry-run default; `Tron2Link` raises before sending any non-whitelisted title unless armed; only
  `request_get_joint_state` / `request_get_limx_2fclaw_state` while unarmed. `movej`, `moveh`, `emgy_stop` are never sent.
  `--dry-run-execution` streams to an in-memory `FakeLink` (robot untouched).
* never moves to an init pose / no `reset()`.
* `--execute`: warmup ≥1, prints a summary and waits for **Enter** (`n` / Ctrl+C aborts); `--yes` skips it (also
  required when stdin is not a TTY).
* start state outside the training / recorded range → warning only.
* stop on non-finite (NaN/inf) or wrong-shape policy output.
* Ctrl+C (once: finish the current chunk and stop; twice: immediate) / any exception → stop streaming and re-send the
  last target 3× (hold). **Keep the hardware e-stop in hand.**

Recommended first real run:
```bash
examples/lingdong_real/run.sh --policy websocket --port 8000 --task pick_up --execute --execute-steps 10 --max-cycles 5
```
