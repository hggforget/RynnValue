"""Nebula worker entry point; run the same Lingdong launcher used locally."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def direct_environment():
    # Nebula and OSS endpoints are reachable directly from the internal network.
    return {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}


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
    command = ["bash", str(root / "scripts/train_pi05_lingdong_right_arm.sh"), *launch["train_args"]]
    print(json.dumps({"env": launch["env"], "command": command}, indent=2), flush=True)
    if args.dry_run:
        return

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

    subprocess.run(["nvidia-smi"], check=True, env=env)
    if not launch.get("skip_setup", False):
        subprocess.run(["bash", str(root / "scripts/setup_nebula.sh")], cwd=root, env=env, check=True)
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
