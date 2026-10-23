"""Repeat operator-judged attempts, with deadlines and explicit fixed-start resets."""

import contextlib
import json
import pathlib
import threading
import time
from math import isfinite

import numpy as np

from openpi_client.franka_collect.client import PROMPT, prepare_observation
from openpi_client.franka_collect.reset import START_POSE


class _Request:
    """Only policy I/O runs in the worker; ROS and all control stay on the main thread."""

    def __init__(self, factory, observation, expected, client=None):
        self.client = client
        self.done = threading.Event()
        self.canceled = threading.Event()
        self.lock = threading.Lock()
        self.value = self.error = None

        def work():
            try:
                policy = client if client is not None else factory()
                with self.lock:
                    self.client = policy
                if self.canceled.is_set():
                    policy.close()
                    return
                if policy.get_server_metadata().get("config") != expected:
                    raise ValueError(f"Expected policy config {expected}")
                self.value = policy.infer(observation)
            except Exception as exc:
                self.error = exc
            finally:
                self.done.set()

        self.thread = threading.Thread(target=work, name="pi05-evaluation-request", daemon=True)
        self.thread.start()

    def abort(self):
        self.canceled.set()
        with self.lock:
            client = self.client
        if client is not None:
            client.close()
        self.thread.join(timeout=0.5)

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class Evaluator:
    def __init__(
        self,
        reader,
        executor,
        control,
        policy_factory,
        *,
        attempts=10,
        timeout=300,
        report_path=None,
        action_path=None,
        prompt=PROMPT,
        representation="absolute",
        observation_timeout=10,
        clock=time.monotonic,
    ):
        if attempts < 1 or not isfinite(timeout) or timeout <= 0:
            raise ValueError("Attempts and attempt timeout must be positive and finite")
        self.reader, self.executor, self.control = reader, executor, control
        self.policy_factory, self.clock = policy_factory, clock
        self.attempts, self.timeout = attempts, timeout
        self.report_path = pathlib.Path(report_path) if report_path is not None else None
        self.action_path = pathlib.Path(action_path) if action_path is not None else None
        self.prompt, self.observation_timeout = prompt, observation_timeout
        self.expected = "pi05_pku_franka_collect_objects" + (
            "_relative" if representation == "relative" else ""
        )
        self.report = {
            "mode": "evaluation",
            "attempts_requested": attempts,
            "attempt_timeout_s": timeout,
            "attempts": [],
        }
        self._save()

    def _save(self):
        successes = sum(entry["result"] == "success" for entry in self.report["attempts"])
        failures = sum(entry["result"] == "failure" for entry in self.report["attempts"])
        completed = successes + failures
        self.report["summary"] = {
            "completed": completed,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / completed if completed else 0.0,
        }
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.report_path.with_name(self.report_path.name + ".tmp")
            temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(self.report_path)

    def _attempt(self, entry):
        started, paused, pause_held = self.clock(), False, False
        request = policy = None

        def stop_requested():
            nonlocal paused, pause_held
            if self.clock() - started >= self.timeout:
                entry.update(result="failure", reason="timeout")
                return True
            event = self.control.next_event("paused" if paused else "running")
            if event == "quit":
                raise KeyboardInterrupt("User exited evaluation")
            if event in ("success", "failure"):
                entry.update(result=event, reason="manual")
            elif event == "pause":
                paused = True
            elif event == "resume":
                paused, pause_held = False, False
                print("继续本次尝试：重新读取观测。", flush=True)
            return paused or entry["result"] is not None

        try:
            while entry["result"] is None:
                stop_requested()
                if entry["result"] is not None:
                    break
                if paused:
                    if not pause_held:
                        self.executor.hold()
                        pause_held = True
                        print("已暂停：s 成功；f 失败；c 继续；q 退出。", flush=True)
                    self.reader.spin_for(0.03)
                    continue
                remaining = self.timeout - (self.clock() - started)
                raw = self.reader.get_obs(
                    timeout=min(self.observation_timeout, remaining), should_stop=stop_requested
                )
                if raw is None or stop_requested():
                    continue
                request = _Request(
                    self.policy_factory,
                    prepare_observation(raw, self.prompt),
                    self.expected,
                    policy,
                )
                entry["requests"] += 1
                while not request.done.is_set():
                    if stop_requested():
                        break
                    self.reader.spin_for(0.02)
                if paused or entry["result"] is not None:
                    self.executor.hold()
                    pause_held = paused
                    request.abort()
                    request = policy = None
                    continue
                result = request.result()
                policy = request.client
                self.report["server"] = policy.get_server_metadata()
                request = None
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.shape != (30, 7) or not np.isfinite(actions).all():
                    raise ValueError("Expected finite model actions [30,7]")
                if self.action_path is not None:
                    self.action_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(self.action_path, actions)
                self.executor.execute(actions, should_pause=stop_requested)
        except KeyboardInterrupt:
            if entry["result"] is None:
                entry.update(result="failure", reason="interrupted")
            raise
        except Exception as exc:
            if entry["result"] is None:
                reason = "timeout" if self.clock() - started >= self.timeout else "error"
                entry.update(result="failure", reason=reason, error=str(exc))
            else:
                entry["cleanup_error"] = str(exc)
        finally:
            entry["duration_s"] = round(self.clock() - started, 3)
            try:
                self.executor.hold()
            finally:
                try:
                    if request is not None:
                        request.abort()
                    elif policy is not None:
                        policy.close()
                finally:
                    self._save()

    def _reset(self, entry):
        self.control.discard_pending_input()
        print("请按 r 复位/重试；q 退出。", flush=True)
        while True:
            event = self.control.next_event("reset")
            if event == "quit":
                return False
            if event == "reset":
                try:
                    entry["reset"] = self.executor.reset_to_pose(START_POSE)
                    entry["reset_status"] = "completed"
                    self._save()
                    print("复位完成。", flush=True)
                    return True
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    entry["reset_error"] = str(exc)
                    self._save()
                    self.executor.hold()
                    print(f"复位失败：{exc}；可按 r 重试，或 q 退出。", flush=True)
            self.reader.spin_for(0.03)

    def _wait_start(self, index):
        self.control.discard_pending_input()
        print(f"第 {index + 1}/{self.attempts} 轮已就绪，请按空格开始；q 退出。", flush=True)
        while True:
            event = self.control.next_event("start")
            if event == "start":
                return True
            if event == "quit":
                return False
            self.reader.spin_for(0.03)

    def run(self):
        try:
            for index in range(self.attempts):
                if not self._wait_start(index):
                    break
                entry = {
                    "attempt": index + 1,
                    "result": None,
                    "requests": 0,
                    "reset_status": "waiting",
                }
                self.report["attempts"].append(entry)
                self._save()
                print(
                    f"开始尝试 {index + 1}/{self.attempts}，最长 {self.timeout:g} 秒；"
                    "s 成功；f 失败。",
                    flush=True,
                )
                self._attempt(entry)
                label = "成功" if entry["result"] == "success" else "失败"
                extra = "（运行超时）" if entry["reason"] == "timeout" else ""
                print(
                    f"尝试 {index + 1}：{label}{extra}，用时 {entry['duration_s']:.1f} 秒。",
                    flush=True,
                )
                if not self._reset(entry):
                    break
        except (KeyboardInterrupt, EOFError):
            print("已退出评测，已保存当前结果。", flush=True)
        finally:
            self._save()
        print(json.dumps(self.report["summary"], ensure_ascii=False), flush=True)
        return self.report


def run_evaluation(args):
    from openpi_client.franka_collect.action_execution import ActionExecutor
    from openpi_client.franka_collect.client import create_client
    from openpi_client.franka_collect.evaluation_control import EvaluationControl
    from openpi_client.franka_collect.live_observation import LiveObservationReader

    with contextlib.ExitStack() as stack:
        reader = stack.enter_context(LiveObservationReader(args.namespace, args.env_config))
        executor = stack.enter_context(
            ActionExecutor(reader, args.action_representation, args.execute_steps, args.control_hz)
        )
        control = stack.enter_context(EvaluationControl(reader))
        runner = Evaluator(
            reader,
            executor,
            control,
            lambda: create_client(args.host, args.port, retry=False),
            attempts=args.attempts,
            timeout=args.attempt_timeout,
            report_path=args.report,
            action_path=args.output,
            prompt=args.prompt,
            representation=args.action_representation,
            observation_timeout=args.observation_timeout,
        )
        runner.report.update(
            endpoint={"host": args.host, "port": args.port},
            action_representation=args.action_representation,
            prompt=args.prompt,
            topics=reader.topics,
        )
        return runner.run()
