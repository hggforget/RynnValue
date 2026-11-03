"""Nebula worker entry point for the local Lingdong and PKU Franka launchers."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

LAUNCHERS = (
    "scripts/train_pi05_lingdong_right_arm.sh",
    "scripts/train_sft_pku_franka_pickplace.sh",
    "scripts/train_iql_pku_franka_pickplace.sh",
    "scripts/train_sft_pku_franka_pickplace_collect_objects.sh",
    "scripts/train_sft_pku_franka_pickplace_collect_drawer.sh",
    "scripts/train_sft_pku_franka_super.sh",
)


def training_command(root, launch):
    # Existing Lingdong manifests have no launcher field.
    launcher = launch.get("launcher", LAUNCHERS[0])
    if launcher not in LAUNCHERS:
        raise ValueError(f"Unsupported Nebula training launcher: {launcher}")
    return ["bash", str(root / launcher), *launch["train_args"]]


def direct_environment():
    # Nebula and OSS endpoints are reachable directly from the internal network.
    return {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}


def run_setup(root, env):
    command = ["bash", str(root / "scripts/setup_nebula.sh")]
    with tempfile.TemporaryDirectory(prefix="nebula-setup-") as directory:
        local_log = Path(directory) / "setup.log"
        with (
            local_log.open("w") as log,
            subprocess.Popen(
                command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as process,
        ):
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        # Keep an accessible copy even when the platform's log bucket is private.
        try:
            log_dir = Path(env.get("NEBULA_SETUP_LOG_DIR", "/oss/embodied-ai-data/lingdong/nebula_setup_logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            target = log_dir / (Path(env["EXP_NAME"]).name + ".log")
            shutil.copyfile(local_log, target)
            print(f"Setup log saved: {target}", flush=True)
        except OSError as error:
            print(f"Could not copy setup log to OSS: {error}", file=sys.stderr, flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)


def setup_oss_mounts():
    for target, link in (
        (Path("/data/oss_bucket_0"), Path("/oss/embodied-ai-data")),
        (Path("/data/oss_bucket_1"), Path("/public/damo_rynnvla_data_zjk")),
    ):
        if not target.is_dir():
            raise FileNotFoundError(f"Nebula OSS mount is missing: {target}")
        if not link.exists():
            subprocess.run(["sudo", "mkdir", "-p", str(link.parent)], check=True)
            subprocess.run(["sudo", "ln", "-sfn", str(target), str(link)], check=True)
        elif link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected existing OSS path: {link}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    launch = json.loads(args.launch_config.read_text())
    env = direct_environment()
    env.update(launch["env"])
    env["PYTHON"] = sys.executable
    # Discover devices on the allocated worker, using its scheduler visibility.
    for name in ("GPUS", "FSDP_DEVICES", "BATCH_SIZE"):
        env.pop(name, None)
    env.setdefault("WANDB_MODE", "offline")
    command = training_command(root, launch)
    print(json.dumps({"env": launch["env"], "command": command}, indent=2), flush=True)
    if args.dry_run:
        return

    setup_oss_mounts()
    subprocess.run(["nvidia-smi"], check=True, env=env)
    if not launch.get("skip_setup", False):
        run_setup(root, env)
    # Check the actual container's JAX backend before opening a training run.
    subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"],
        env=env,
        check=True,
    )
    os.chdir(root)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
