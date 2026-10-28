"""Lingdong Tron2 right-arm pi05 real-robot inference (synchronous chunking).

Default = DRY RUN: observe -> infer -> log; nothing is sent to the robot.
Real motion only with --execute (press Enter at the summary prompt, or pass --yes).
Execution is RAW: model actions are sent verbatim (no interpolation / ramp / clipping / rate limit / lag or
staleness stops); only NaN/inf output stops.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
from safety import RAW_BANNER, ActionSafety, SafetyViolation  # noqa: E402

log = logging.getLogger("lingdong_real")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["run", "camera-check", "state-check"], default="run")
    p.add_argument("--backend", choices=["tron2", "replay"], default="tron2")
    p.add_argument("--policy", choices=["local", "websocket"], default="local")
    p.add_argument("--config", default=C.CONFIG_NAME)
    p.add_argument("--checkpoint", default=C.DEFAULT_CHECKPOINT)
    p.add_argument("--num-steps", type=int, default=10, help="flow-matching denoise steps (training default 10)")
    p.add_argument("--compile-cache", default="/mnt/hdc/.jax_compile_cache_notriton")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--task", choices=sorted(C.TASK_PROMPTS), default="pick_up")
    p.add_argument("--prompt", default=None, help="exact training task string (overrides --task)")
    # robot / cameras
    p.add_argument("--robot-ip", default="10.192.1.2")
    p.add_argument("--robot-port", type=int, default=5000)
    p.add_argument("--head-serial", default=C.POLICY_CAMERA_SERIALS["cam_high"])
    p.add_argument("--right-wrist-serial", default=C.POLICY_CAMERA_SERIALS["cam_right_wrist"],
                   help="physical RIGHT wrist camera (see README: wrist correction)")
    p.add_argument("--no-depth", action="store_true", help="do not enable the depth stream")
    # replay
    p.add_argument("--replay-root", default=C.DEFAULT_REPLAY_ROOT)
    p.add_argument("--replay-episode", type=int, default=131)
    p.add_argument("--replay-wrist-source", default="cam_left_wrist",
                   help="recorded stream fed as cam_right_wrist (raw data: cam_left_wrist; final dataset: cam_right_wrist)")
    p.add_argument("--replay-start", default="auto")
    # loop
    p.add_argument("--execute-steps", type=int, default=15, help="actions executed per chunk (1..30)")
    p.add_argument("--control-hz", type=float, default=C.CONTROL_HZ)
    p.add_argument("--max-cycles", type=int, default=0, help="0 = until episode end / Ctrl+C")
    p.add_argument("--warmup", type=int, default=2, help="warmup inferences before the loop (mandatory >=1 with --execute)")
    p.add_argument("--execute", action="store_true", help="REALLY MOVE THE ROBOT (press Enter to confirm)")
    p.add_argument("--yes", "-y", action="store_true", help="with --execute: skip the Enter confirmation")
    p.add_argument("--dry-run-execution", action="store_true",
                   help="run the executor against an in-memory FakeLink (tests streaming/timing, robot untouched)")
    p.add_argument("--servo-mode", type=int, choices=[0, 1], default=None,
                   help="send request_set_servo_mode before streaming (old deploy never sent it explicitly)")
    p.add_argument("--gripper-speed", type=float, default=100)
    p.add_argument("--gripper-force", type=float, default=50)
    # logging
    p.add_argument("--log-root", default=C.DEFAULT_LOG_ROOT)
    p.add_argument("--save-images-every", type=int, default=10, help="save input jpgs every N cycles (0=off)")
    p.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True,
                   help="live camera/action/state page and synchronized video recording")
    p.add_argument("--viz-host", default="0.0.0.0")
    p.add_argument("--viz-port", type=int, default=8001, help="live viewer HTTP port (policy uses --port)")
    p.add_argument("--viz-fps", type=float, default=8, help="camera preview and MP4 frame rate")
    a = p.parse_args(argv)
    if not 1 <= a.execute_steps <= C.ACTION_HORIZON:
        p.error("--execute-steps must be in 1..30")
    if not 0 < a.control_hz <= 50:
        p.error("--control-hz must be in (0, 50]")
    if a.execute and a.backend != "tron2":
        p.error("--execute requires --backend tron2")
    if a.execute and a.dry_run_execution:
        p.error("--execute and --dry-run-execution are exclusive")
    if a.execute and a.warmup < 1:
        p.error("--execute requires --warmup >= 1 (first call compiles for ~40 s)")
    if not 1 <= a.viz_fps <= 15:
        p.error("--viz-fps must be in 1..15")
    if not 0 <= a.viz_port <= 65535:
        p.error("--viz-port must be in 0..65535")
    return a


NO_AGE_LIMIT = float("inf")  # raw mode: never stop on stale state / images


def _save_jpg(path, rgb):
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path, quality=90)
    except Exception as e:  # noqa: BLE001
        log.warning("could not save %s: %s", path, e)


def camera_check(args, out_dir):
    """Read-only: save one labelled frame per connected RealSense + print robot state."""
    import pyrealsense2 as rs

    from cameras import RealSenseCameras

    serials = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
               for d in rs.context().query_devices()}
    cams = RealSenseCameras({s: s for s in serials}, enable_depth=not args.no_depth)
    try:
        time.sleep(1.0)
        for s, name in serials.items():
            role = {args.head_serial: "POLICY_cam_high", args.right_wrist_serial: "POLICY_cam_right_wrist"}.get(s, "unused")
            img, _ = cams.get(s, 1.0)
            path = out_dir / f"{role}_{name.replace(' ', '_')}_{s}.jpg"
            _save_jpg(path, img)
            print(f"{s} {name:16s} -> {role:24s} {path}")
    finally:
        cams.close()
    print("Check: POLICY_cam_right_wrist must show the RIGHT arm's gripper, upright, like training wrist frames.")


def state_check(args):
    from tron2_link import Tron2Link

    link = Tron2Link(args.robot_ip, args.robot_port)
    try:
        link.wait_ready()
        st = link.get_state()
        s18 = C.build_raw_state(st["q16"], st["left_opening"], st["right_opening"])
        s8 = C.select_right(s18)
        print("accid", link.accid, "right8", np.round(s8, 4).tolist(), "left_open", st["left_opening"])
        for m in ActionSafety().check_state(s8):
            print("WARNING:", m)
        print("sent (read-only):", link.sent_counts)
    finally:
        link.close()


def _confirm(summary: str, assume_yes: bool = False) -> bool:
    print("\n" + "=" * 70 + "\n" + summary + "\n" + "=" * 70)
    if assume_yes:
        print("--yes given: starting without confirmation.")
        return True
    if not sys.stdin.isatty():
        print("stdin is not a TTY: cannot ask for Enter -> pass --yes to start without a prompt.")
        return False
    try:
        ans = input("Press Enter to move the robot (type n / Ctrl+C to abort): ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() not in ("n", "no", "q", "quit", "abort")


def run(args, out_dir: pathlib.Path):
    prompt, prompt_verified = C.resolve_prompt(args.task, args.prompt)
    safety = ActionSafety()
    link = cams = backend = policy = executor = viz = None
    stop = {"flag": False}

    def _sigint(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        log.warning("Ctrl+C: stopping after the current step (press again to abort immediately)")

    meta = {"args": vars(args), "prompt": prompt, "prompt_verified": prompt_verified,
            "execution": "raw_passthrough", "start_local": datetime.datetime.now().isoformat()}
    jsonl = open(out_dir / "cycles.jsonl", "w")
    rec = {k: [] for k in ("state8", "raw_chunk", "safe_targets", "recorded", "frame_index")}
    mae_rows = []
    try:
        if args.visualize and args.viz_port:
            from viewer_server import ensure_server

            ensure_server(args.log_root, args.viz_host, args.viz_port)
            log.info("viewer ready: http://10.192.1.155:%d (also shows initialization errors)", args.viz_port)
        # ---------------- backend
        if args.backend == "replay":
            from backends import ReplayBackend

            start = args.replay_start if args.replay_start == "auto" else int(args.replay_start)
            backend = ReplayBackend(args.replay_root, args.replay_episode, wrist_source=args.replay_wrist_source,
                                    start=start)
            meta["replay"] = {"episode": args.replay_episode, "task_index": backend.task_index,
                              "start": backend.start, "n": backend.n}
        else:
            from backends import Tron2Backend
            from cameras import RealSenseCameras
            from tron2_link import Tron2Link

            link = Tron2Link(args.robot_ip, args.robot_port)
            link.wait_ready()
            cams = RealSenseCameras({"cam_high": args.head_serial, "cam_right_wrist": args.right_wrist_serial},
                                    enable_depth=not args.no_depth)
            backend = Tron2Backend(link, cams, state_max_age=NO_AGE_LIMIT, image_max_age=NO_AGE_LIMIT)
        if args.visualize:
            from live_view import LiveView

            viz = LiveView(out_dir, link=link, cameras=cams, host=args.viz_host, port=args.viz_port,
                           fps=args.viz_fps,
                           execution="real" if args.execute else "fake" if args.dry_run_execution else "dry").start()
            meta["visualization_url"] = f"http://10.192.1.155:{viz.port}"
            viz.record_observation(backend.observe())
            viz.set_status("loading policy")
        # ---------------- policy
        if args.policy == "local":
            from policy_backends import LocalPolicy

            policy = LocalPolicy(args.config, args.checkpoint, num_steps=args.num_steps,
                                 compile_cache_dir=args.compile_cache)
        else:
            from policy_backends import WebsocketPolicy

            policy = WebsocketPolicy(args.host, args.port)
        meta["policy_metadata"] = {k: str(v) for k, v in (policy.metadata or {}).items()}
        meta["policy_load_s"] = policy.load_s
        obs = backend.observe()
        for m in safety.check_state(obs.state8, strict=False):
            log.warning(m)
        for i in range(args.warmup):
            if viz is not None:
                viz.set_status("warmup", warmup=i)
            t = time.monotonic()
            a, _ = policy.infer(C.build_policy_obs(obs.state8, obs.images["cam_high"], obs.images["cam_right_wrist"], prompt))
            log.info("warmup %d: %.0f ms, actions %s finite=%s", i, (time.monotonic() - t) * 1e3, a.shape,
                     bool(np.isfinite(a).all()))
        # ---------------- executor
        if args.execute or args.dry_run_execution:
            from executor import RobotExecutor

            print(RAW_BANNER, flush=True)
            log.warning("RAW execution: model actions are sent verbatim (only NaN/inf stops)")

            exec_link = link
            if args.dry_run_execution:
                from tron2_link import FakeLink

                q16 = np.zeros(16)
                q16[7:14] = obs.state8[:7]
                exec_link = FakeLink(q16, right_opening=obs.state8[7] * 100)
            obs = backend.observe()
            for m in safety.check_state(obs.state8):  # warnings only, never blocks
                log.warning("start state: %s", m)
            a, _ = policy.infer(C.build_policy_obs(obs.state8, obs.images["cam_high"], obs.images["cam_right_wrist"], prompt))
            _, rep = safety.process(a[: args.execute_steps], obs.state8)
            if args.execute:
                if viz is not None:
                    viz.set_status("waiting for confirmation")
                summary = (f"robot {args.robot_ip} accid {link.accid}\nprompt {prompt!r} (verified={prompt_verified})\n"
                           f"right-wrist camera serial {args.right_wrist_serial}\n"
                           f"current right8 {np.round(obs.state8, 3).tolist()}\nfirst action {np.round(a[0], 3).tolist()}\n"
                           f"first action is {rep.first_jump:.3f} rad from the current pose and is sent DIRECTLY "
                           f"(no ramp); max step inside the chunk {rep.max_step:.3f} rad\n"
                           f"execute {args.execute_steps}/30 steps per chunk, one servoj per step @ {args.control_hz} Hz\n"
                           "RAW MODE: no clipping / rate limit / ramp / lag or staleness stops. Only NaN/inf stops.\n"
                           "Keep the e-stop in hand. Ctrl+C = stop & hold.")
                if not _confirm(summary, args.yes):
                    log.warning("not confirmed -> exiting without sending anything")
                    return meta
                link.arm_commands("I_UNDERSTAND_THE_ROBOT_WILL_MOVE")
            else:
                exec_link.arm_commands("I_UNDERSTAND_THE_ROBOT_WILL_MOVE")
            executor = RobotExecutor(exec_link, control_hz=args.control_hz, gripper_speed=args.gripper_speed,
                                     gripper_force=args.gripper_force, servo_mode=args.servo_mode,
                                     on_target=viz.record_target if viz is not None else None)
            executor.start()
            meta["executor_link"] = "real" if args.execute else "fake"
        signal.signal(signal.SIGINT, _sigint)
        # ---------------- loop
        cycle = 0
        period_target = args.execute_steps / args.control_hz
        while not stop["flag"]:
            if args.max_cycles and cycle >= args.max_cycles:
                break
            if args.backend == "replay" and backend.done:
                break
            t0 = time.monotonic()
            obs = backend.observe()
            if viz is not None:
                viz.record_observation(obs)
                viz.set_status("inferring", cycle=cycle + 1)
            t1 = time.monotonic()
            inp = C.build_policy_obs(obs.state8, obs.images["cam_high"], obs.images["cam_right_wrist"], prompt)
            t2 = time.monotonic()
            actions, ptiming = policy.infer(inp)
            t3 = time.monotonic()
            if actions.shape != (C.ACTION_HORIZON, 8) or not np.isfinite(actions).all():
                raise SafetyViolation(f"bad policy output {actions.shape} finite={np.isfinite(actions).all()}")
            reference = executor.last_target8 if executor is not None else obs.state8
            chunk = actions[: args.execute_steps]
            safe, rep = safety.process(chunk, reference)  # verbatim copy; raises only on NaN/inf/shape
            t4 = time.monotonic()
            exec_stats = {}
            if executor is not None:
                if viz is not None:
                    viz.set_status("executing", cycle=cycle + 1, infer_ms=1e3 * (t3 - t2))
                exec_stats = executor.execute(safe)
            elif args.backend == "tron2":  # live dry run: keep the real cadence
                time.sleep(max(0.0, period_target - (time.monotonic() - t0)))
            t5 = time.monotonic()
            row = {"cycle": cycle, "t": t0, "frame_index": obs.frame_index,
                   "lat_ms": {"observe": 1e3 * (t1 - t0), "prep": 1e3 * (t2 - t1), "infer": 1e3 * (t3 - t2),
                              "safety": 1e3 * (t4 - t3), "execute": 1e3 * (t5 - t4), "cycle": 1e3 * (t5 - t0)},
                   "policy_timing": ptiming, "ages_s": obs.ages_s, "chunk": rep.to_dict(), "exec": exec_stats,
                   "state8": obs.state8.round(5).tolist(), "first_action": actions[0].round(5).tolist()}
            rec_actions = None
            if args.backend == "replay":
                rec_actions = backend.recorded_actions8(obs.frame_index, C.ACTION_HORIZON)
                k = len(rec_actions)
                err = np.abs(actions[:k] - rec_actions)
                row["mae_chunk"] = err.mean(0).round(5).tolist()
                row["mae_exec_steps"] = err[: args.execute_steps].mean(0).round(5).tolist()
                mae_rows.append(err[: args.execute_steps].mean(0))
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            rec["state8"].append(obs.state8)
            rec["raw_chunk"].append(actions)
            rec["safe_targets"].append(np.pad(safe, ((0, args.execute_steps - len(safe)), (0, 0))))
            rec["frame_index"].append(-1 if obs.frame_index is None else obs.frame_index)
            if rec_actions is not None:
                rec["recorded"].append(np.pad(rec_actions, ((0, C.ACTION_HORIZON - len(rec_actions)), (0, 0)),
                                              constant_values=np.nan))
            if args.save_images_every and cycle % args.save_images_every == 0:
                _save_jpg(out_dir / f"c{cycle:04d}_cam_high.jpg", obs.images["cam_high"])
                _save_jpg(out_dir / f"c{cycle:04d}_cam_right_wrist.jpg", obs.images["cam_right_wrist"])
            log.info("cycle %d%s: obs %.0f infer %.0f exec %.0f total %.0f ms | jump %.3f maxstep %.3f%s", cycle,
                     f" f{obs.frame_index}" if obs.frame_index is not None else "", row["lat_ms"]["observe"],
                     row["lat_ms"]["infer"], row["lat_ms"]["execute"], row["lat_ms"]["cycle"], rep.first_jump,
                     rep.max_step,
                     f" | mae {np.mean(row['mae_exec_steps'][:7]):.4f} rad grip {row['mae_exec_steps'][7]:.3f}"
                     if "mae_exec_steps" in row else "")
            backend.advance(args.execute_steps)
            cycle += 1
    except (SafetyViolation, KeyboardInterrupt) as e:
        log.error("STOP: %s: %s", type(e).__name__, e)
        meta["stop_reason"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        log.exception("STOP on exception")
        meta["stop_reason"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if executor is not None:
            executor.hold()
        if viz is not None:
            try:
                viz.close()
            except Exception:
                log.exception("could not finalize visualization (control has stopped)")
        if executor is not None and executor.sent_targets:
            np.save(out_dir / "sent_targets.npy", np.asarray(executor.sent_targets))
        if executor is not None and args.dry_run_execution:
            meta["fake_link_sent"] = executor.link.sent_counts
            fl = executor.link.commands
            sj = [c[2] for c in fl if c[1] == "request_servoj"]
            if sj:  # every servoj q16 the fake robot received (incl. the final hold re-sends)
                np.save(out_dir / "fake_link_servoj_q16.npy", np.asarray(sj))
            gr = [c[2] for c in fl if c[1] == "request_set_limx_2fclaw_cmd"]
            if gr:
                np.save(out_dir / "fake_link_gripper.npy", np.asarray(gr, dtype=np.float64))
        if link is not None:
            meta["robot_link_sent"] = dict(link.sent_counts)
            link.close()
        if cams is not None:
            cams.close()
        if policy is not None:
            policy.close()
        jsonl.close()
        arrays = {k: np.asarray(v) for k, v in rec.items() if v}
        if arrays:
            np.savez_compressed(out_dir / "cycles.npz", **arrays)
        if mae_rows:
            m = np.mean(mae_rows, 0)
            meta["mae_exec_steps_per_channel"] = m.round(5).tolist()
            meta["mae_exec_steps_joint_mean_rad"] = float(m[:7].mean())
        meta["end_local"] = datetime.datetime.now().isoformat()
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        log.info("logs: %s", out_dir)
    return meta


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    args = parse_args(argv)
    tag = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{args.backend}_{'EXEC' if args.execute else 'dry'}"
    out_dir = C.ensure_dir(pathlib.Path(args.log_root) / tag)
    if args.mode == "camera-check":
        camera_check(args, out_dir)
    elif args.mode == "state-check":
        state_check(args)
    else:
        run(args, out_dir)


if __name__ == "__main__":
    main()
