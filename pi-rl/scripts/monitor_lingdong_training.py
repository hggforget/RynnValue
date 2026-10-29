"""Monitor one tmux training pane and resume known failures from complete checkpoints.

The monitor is a persistent local process, not a promise of a future chat turn.
It does not interrupt a healthy run, overwrite checkpoints, skip bad samples, or
change batch size/learning rate. Unknown failures retain their logs for diagnosis.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

CONFIG_NAME = "pi05_lingdong_right_arm"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RECOVERABLE = (
    "VIDEO_READ_FAILED",
    "InvalidDataError",
    "Error parsing OBU",
    "Input/output error",
    "Connection reset by peer",
    "Transport endpoint is not connected",
    "ResourceExhaustedError",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "out of memory",
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017 - training uses Python 3.10


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def option(argv, name, default=None):
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


def replace_option(argv, name, value):
    result = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == name:
            skip = True
        elif not arg.startswith(name + "="):
            result.append(arg)
    return [*result, name, str(value)]


def resume_argv(argv, experiment, python):
    if CONFIG_NAME not in argv or "--overwrite" in argv or any(a.startswith("--overwrite=") for a in argv):
        raise ValueError("Refusing an incompatible or overwrite-enabled training command")
    result = replace_option(list(argv), "--exp-name", experiment)
    result = [a for a in result if a != "--no-resume"]
    result[0] = python
    if "--resume" not in result:
        result.append("--resume")
    return result


def complete_checkpoints(directory):
    """Ignore temporary/incomplete saves; keep both parameters and optimizer state."""
    result = []
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if not path.name.isdigit() or not path.is_dir():
            continue
        try:
            metadata = read_json(path / "_CHECKPOINT_METADATA")
            required = [
                path / "params/_METADATA",
                path / "train_state/_METADATA",
                path / "assets/lingdong_right_arm/norm_stats.json",
            ]
            if metadata.get("commit_timestamp_nsecs") and all(p.is_file() for p in required):
                result.append((int(path.name), path))
        except (OSError, ValueError):
            continue
    return sorted(result)


def classify_exit(log):
    log = ANSI.sub("", log).replace("\r", "\n")
    traceback = log[log.rfind("Traceback") :] if "Traceback" in log else ""
    if "KeyboardInterrupt" in traceback:
        return "manual_stop"
    # A failed final asynchronous save must not be mistaken for completion.
    if "Waiting for checkpoint manager to finish" in log and not any(
        x in log.split("Waiting for checkpoint manager to finish")[-1] for x in ("Traceback", "Error")
    ):
        return "complete"
    if traceback and any(token.lower() in traceback.lower() for token in RECOVERABLE):
        return "recoverable_error"
    return "needs_attention"


def progress_step(log):
    log = ANSI.sub("", log)
    steps = [int(x) for x in re.findall(r"(?:Step\s+|\|\s*)(\d+)(?::|/\d+\s*\[)", log)]
    return max(steps, default=None)


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        argv = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        return {"pid": int(pid), "parent": int(fields[1]), "started": fields[19], "argv": argv}
    except (OSError, ValueError):
        return None


def training_in_pane(shell_pid):
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        item = process(int(path.name))
        if (
            item
            and item["parent"] == shell_pid
            and CONFIG_NAME in item["argv"]
            and any(a.endswith("scripts/train.py") for a in item["argv"])
        ):
            return item
    return None


def tail(path, offset):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(offset, path.stat().st_size - 256 * 1024))
        return stream.read().decode(errors="replace")


def run_plan(path):
    """Run in the training pane so it inherits the operator's existing environment."""
    plan = read_json(path)
    environment = {**os.environ, **plan["environment"]}
    if plan["dataset_fingerprint"] != hashlib.sha256(Path(plan["inventory"]).read_bytes()).hexdigest():
        raise RuntimeError("Dataset changed since monitor attachment; refusing to resume silently")
    checkpoint = Path(plan["checkpoint"])
    if (int(checkpoint.name), checkpoint) not in complete_checkpoints(checkpoint.parent):
        raise RuntimeError("Selected checkpoint is no longer complete")
    cpu_environment = {**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    subprocess.run(
        [plan["argv"][0], "scripts/compute_lingdong_norm_stats.py", "--dataset-root", plan["dataset_root"], "--check"],
        cwd=plan["cwd"],
        env=cpu_environment,
        check=True,
    )
    os.chdir(plan["cwd"])
    print("MONITOR_RESUME " + json.dumps({"checkpoint": str(checkpoint), "argv": plan["argv"]}), flush=True)
    os.execvpe(plan["argv"][0], plan["argv"], environment)


class Supervisor:
    def __init__(self, config, state_dir):
        self.config = config
        self.directory = state_dir
        self.log_path = state_dir / "live.log"
        self.offset = 0
        self.pid = config["initial_pid"]
        self.experiment = option(config["argv"], "--exp-name")
        self.cwd = Path(config["cwd"])
        base = Path(option(config["argv"], "--checkpoint-base-dir", "../output/checkpoints"))
        self.checkpoint_root = (self.cwd / base).resolve() / CONFIG_NAME
        self.dataset_root = option(config["argv"], "--data.dataset-root")
        self.inventory = Path(self.dataset_root) / "lerobot_2026-10-26_12-09-32_final/meta/file_checksums.json"
        fingerprint = hashlib.sha256(self.inventory.read_bytes()).hexdigest()
        self.fingerprint = config.get("dataset_fingerprint", fingerprint)
        if self.fingerprint != fingerprint:
            raise ValueError("Dataset changed since monitor attachment")
        self.restarts = 0
        self.failures_without_progress = 0
        self.saved_step = 0
        self.last_resume_step = 0
        self.last_seen_step = None
        self.last_progress = time.time()
        self.pending_restart_since = None
        self.handled_exit = False
        self.status = "watching"
        self.last_checkpoint_scan = 0.0
        self.checkpoint = None
        self.last_heartbeat = 0.0
        previous_state = state_dir / "status.json"
        if previous_state.is_file():
            previous = read_json(previous_state)
            self.restarts = previous.get("automatic_restarts", 0)
            self.last_resume_step = previous.get("last_resume_step", 0)
            self.failures_without_progress = previous.get("failures_without_progress", 0)
            self.offset = previous.get("log_offset", 0)
            self.pending_restart_since = previous.get("pending_restart_since")

    def event(self, event, **values):
        row = {"time": now(), "event": event, **values}
        with (self.directory / "events.jsonl").open("a") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def save_status(self):
        write_json(
            self.directory / "status.json",
            {
                "updated_at": now(),
                "status": self.status,
                "pane": self.config["pane"],
                "pid": self.pid,
                "experiment": self.experiment,
                "step": self.last_seen_step,
                "latest_saved_step": self.saved_step,
                "latest_checkpoint": str(self.checkpoint) if self.checkpoint else None,
                "automatic_restarts": self.restarts,
                "last_progress_at": self.last_progress,
                "recovery_patch_ready": (self.directory / "recovery_ready").exists(),
                "log_offset": self.offset,
                "last_resume_step": self.last_resume_step,
                "failures_without_progress": self.failures_without_progress,
                "pending_restart_since": self.pending_restart_since,
            },
        )

    def restart(self, shell_pid):
        if not (self.directory / "recovery_ready").exists():
            self.status = "waiting_for_recovery_patch"
            return
        candidates = complete_checkpoints(self.checkpoint_root / self.experiment)
        experiment = self.experiment
        if not candidates:
            fallback = self.config.get("fallback_experiment")
            candidates = (
                [
                    c
                    for c in complete_checkpoints(self.checkpoint_root / fallback)
                    if c[0] == self.config.get("fallback_step")
                ]
                if fallback
                else []
            )
            experiment = fallback
        if not candidates:
            self.status = "needs_attention"
            self.event("no_complete_checkpoint_available")
            self.handled_exit = True
            return
        step, checkpoint = candidates[-1]
        failures_without_progress = self.failures_without_progress + 1 if step <= self.last_resume_step else 0
        if failures_without_progress >= 3:
            self.status = "needs_attention"
            self.event("repeated_failure_without_checkpoint_progress", checkpoint=str(checkpoint))
            self.handled_exit = True
            return
        if hashlib.sha256(self.inventory.read_bytes()).hexdigest() != self.fingerprint:
            self.status = "needs_attention"
            self.event("dataset_changed_refusing_automatic_resume")
            self.handled_exit = True
            return
        # Never inject a command into a pane now occupied by another process.
        pane_command = subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", self.config["pane"], "#{pane_current_command}"],
            text=True,
        ).strip()
        if pane_command not in ("bash", "zsh", "sh") or training_in_pane(shell_pid):
            self.status = "waiting_for_idle_pane"
            return
        screen = subprocess.check_output(
            ["tmux", "capture-pane", "-p", "-t", self.config["pane"], "-S", "-5"],
            text=True,
        )
        last_line = next((line for line in reversed(screen.splitlines()) if line.strip()), "")
        if not re.search(r"[$#]\s*$", ANSI.sub("", last_line)):
            self.status = "waiting_for_idle_pane"
            return
        plan = {
            **self.config,
            "argv": resume_argv(self.config["argv"], experiment, self.config["python"]),
            "checkpoint": str(checkpoint),
            "dataset_fingerprint": self.fingerprint,
            "inventory": str(self.inventory),
            "dataset_root": self.dataset_root,
        }
        plan_path = self.directory / f"resume_{self.restarts + 1}.json"
        write_json(plan_path, plan)
        self.offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        command = shlex.join([self.config["python"], str(Path(__file__).resolve()), "--run-plan", str(plan_path)])
        subprocess.run(["tmux", "send-keys", "-t", self.config["pane"], "-l", command], check=True)
        subprocess.run(["tmux", "send-keys", "-t", self.config["pane"], "Enter"], check=True)
        self.failures_without_progress = failures_without_progress
        self.restarts += 1
        self.last_resume_step = step
        self.saved_step, self.checkpoint = step, checkpoint
        self.experiment = experiment
        self.last_seen_step = None
        self.pending_restart_since = time.time()
        self.last_progress = time.time()
        self.status = "restarting"
        self.event("automatic_resume_requested", experiment=experiment, checkpoint=str(checkpoint))

    def poll(self):
        shell_pid = int(
            subprocess.check_output(
                ["tmux", "display-message", "-p", "-t", self.config["pane"], "#{pane_pid}"],
                text=True,
            ).strip()
        )
        active = training_in_pane(shell_pid)
        log = tail(self.log_path, self.offset)
        if active:
            experiment = option(active["argv"], "--exp-name")
            if active["pid"] != self.pid:
                self.event("training_process_attached", pid=active["pid"], experiment=experiment)
                if self.pending_restart_since is None:
                    self.offset = self.log_path.stat().st_size if self.log_path.exists() else 0
                    log = ""
                if experiment != self.experiment:
                    self.saved_step, self.checkpoint = 0, None
                    self.last_checkpoint_scan = 0
                self.pid = active["pid"]
                self.config["argv"] = active["argv"]
                self.config["initial_pid"] = self.pid
                write_json(self.directory / "config.json", self.config)
                self.experiment = experiment
                self.handled_exit = False
                self.pending_restart_since = None
                self.last_progress = time.time()
            step = progress_step(log)
            if step is not None and step != self.last_seen_step:
                self.last_seen_step = step
                self.last_progress = time.time()
            self.status = "running" if step is not None else "initializing"
            if time.time() - self.last_progress > 1200:
                self.status = "stalled_needs_attention"
            if time.time() - self.last_checkpoint_scan > 60:
                candidates = complete_checkpoints(self.checkpoint_root / self.experiment)
                if candidates:
                    self.saved_step, self.checkpoint = candidates[-1]
                self.last_checkpoint_scan = time.time()
            if time.time() - self.last_heartbeat > 60:
                self.event("heartbeat", status=self.status, step=self.last_seen_step, saved_step=self.saved_step)
                self.last_heartbeat = time.time()
        elif self.pending_restart_since is not None and time.time() - self.pending_restart_since < 180:
            self.status = "restarting"
        elif not self.handled_exit:
            reason = classify_exit(log)
            if reason == "complete":
                candidates = complete_checkpoints(self.checkpoint_root / self.experiment)
                final_step = int(option(self.config["argv"], "--num-train-steps", "20000")) - 1
                if not candidates or candidates[-1][0] < final_step:
                    reason = "needs_attention"
                else:
                    self.saved_step, self.checkpoint = candidates[-1]
            failure_path = self.directory / f"exit_{time.time_ns()}.log"
            failure_path.write_text(log)
            self.event("training_process_exited", reason=reason, pid=self.pid, last_step=self.last_seen_step)
            if reason == "recoverable_error":
                self.restart(shell_pid)
            else:
                self.status = reason
                self.handled_exit = True
        self.save_status()
        return self.status != "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/tmp/tron2-monitor/config.json"))
    parser.add_argument("--state-dir", type=Path, default=Path("/tmp/tron2-monitor"))
    parser.add_argument("--run-plan", type=Path)
    args = parser.parse_args()
    if args.run_plan:
        run_plan(args.run_plan)
        return
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (args.state_dir / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.state_dir / "monitor.pid").write_text(str(os.getpid()) + "\n")
    config = read_json(args.config)
    supervisor = Supervisor(config, args.state_dir)
    config["dataset_fingerprint"] = supervisor.fingerprint
    write_json(args.config, config)
    supervisor.event("monitor_started", interval_seconds=config.get("interval_seconds", 15), pid=os.getpid())
    while True:
        try:
            if not supervisor.poll():
                supervisor.event("training_complete")
                break
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            supervisor.event("monitor_poll_error", error=f"{type(exc).__name__}: {exc}")
        time.sleep(config.get("interval_seconds", 15))


if __name__ == "__main__":
    main()
