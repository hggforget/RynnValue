"""Stage a small source snapshot and submit one Lingdong SFT/IQL job to Nebula."""

import argparse
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def copy_sources(destination):
    # Explicit allowlist: the workspace contains checkpoints and a 41 GB core dump.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".git")
    for name in ("src", "scripts", "packages", "third_party", "assets", "configs"):
        shutil.copytree(ROOT / name, destination / name, ignore=ignore)
    for name in ("run_nebula.py", "README.md", "LICENSE", "LICENSE_GEMMA.txt", "pyproject.toml"):
        shutil.copy2(ROOT / name, destination / name)


def reference_settings(reference_root):
    """Read existing settings as data, without executing the reference submit script."""
    source = reference_root / "submit.sh"
    text = source.read_text() if source.exists() else ""
    auth = dict(
        re.findall(r"^#\s*(user_id|user_name|access_id|access_key|nebula_project)=([^\s]+)", text, re.MULTILINE)
    )
    oss = {}
    for name in ("oss_access_id", "oss_access_key", "oss_bucket", "oss_endpoint"):
        match = re.search(r"--" + name + r"=([^\s\\]+)", text)
        value = os.environ.get("NEBULA_" + name.upper()) or (match[1].strip("\"'") if match else None)
        if value:
            oss[name] = value
    return auth, oss


def local_preflight(launch, python, requested_gpus):
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    env.pop("INIT_CHECKPOINT", None)
    env.update(launch["env"])
    # Only the local preview models the requested allocation; the uploaded
    # manifest leaves GPU discovery and global batch sizing to the worker.
    env.update(GPUS=",".join(map(str, range(requested_gpus))))
    env.pop("FSDP_DEVICES", None)
    env.pop("BATCH_SIZE", None)
    env.update(PYTHON=python, DRY_RUN="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    result = subprocess.run(
        ["bash", "scripts/train_pi05_lingdong_right_arm.sh", *launch["train_args"]],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="", flush=True)
    env["PYTHONPATH"] = os.pathsep.join(
        str(ROOT / path) for path in ("src", "packages/openpi-client/src", "third_party/jaxrl2")
    )
    trainer_args = shlex.split(result.stdout.splitlines()[-1])[2:]
    subprocess.run(
        [
            python,
            "-c",
            "import sys, tyro; from openpi.training import config; "
            "name=sys.argv[1]; "
            "tyro.extras.overridable_config_cli({name: (name, config.get_config(name))}, args=sys.argv[1:]); "
            "print('Trainer arguments validated')",
            *trainer_args,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    norm = [
        python,
        "scripts/compute_lingdong_norm_stats.py",
        "--dataset-root",
        env["DATA_ROOT"],
        "--action-representation",
        env["ACTION_REPRESENTATION"],
        "--action-horizon",
        env["ACTION_HORIZON"],
        "--check",
    ]
    subprocess.run(norm, cwd=ROOT, env=env, check=True)
    if env["METHOD"] == "iql":
        command = [
            python,
            "scripts/check_lingdong_iql_data.py",
            "--dataset-root",
            env["DATA_ROOT"],
            "--task",
            env["TASK"],
            "--reward-source",
            env["REWARD_SOURCE"],
        ]
        if env.get("REWARD_LABELS_PATH"):
            command += ["--reward-labels-path", env["REWARD_LABELS_PATH"]]
        if env.get("ASSUME_SUCCESS") == "1":
            command.append("--assume-success")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sft", "iql"), default="sft")
    parser.add_argument("--task", choices=("all", "pick_up", "put_down"), default="all")
    parser.add_argument("--action-representation", choices=("absolute", "relative"), default="absolute")
    parser.add_argument(
        "--gpus",
        type=int,
        choices=(1, 2, 4, 8),
        default=8,
        help="Scheduler GPU request; the worker detects its visible devices.",
    )
    parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    parser.add_argument("--num-train-steps", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exp-name")
    parser.add_argument("--job-name")
    parser.add_argument("--queue", default="damo_eai_rfm_l20z")
    parser.add_argument("--algo-name", default="pytorch271")
    parser.add_argument("--data-root", default="/oss/embodied-ai-data/lingdong")
    parser.add_argument("--weight-params-path", default="/oss/embodied-ai-data/pku/pi05_base/params")
    parser.add_argument("--ckpt-base-dir", default="/oss/embodied-ai-data/lingdong/nebula_checkpoints")
    parser.add_argument("--reward-labels-path")
    parser.add_argument("--reward-source", choices=("progress", "terminal"), default="progress")
    parser.add_argument("--assume-success", action="store_true")
    parser.add_argument("--skip-setup", action="store_true", help="Use only with a prepared worker image.")
    parser.add_argument(
        "--pip-index-url", help="Package index reachable from the worker; defaults to the internal index."
    )
    parser.add_argument(
        "--wheelhouse",
        help="Offline wheel bundle on the worker. Auto-detects the shared bundle; an empty value disables auto-detection.",
    )
    parser.add_argument("--preflight-python", default="/opt/conda/envs/pi-rl/bin/python")
    parser.add_argument("--reference-root", type=Path, default=Path("/mnt/rynnbot_hangzhou/hdc/pi-rl"))
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Stage and validate; do not contact Nebula.")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Extra trainer flags after --.")
    args = parser.parse_args()
    if args.batch_size_per_gpu < 1 or args.num_train_steps < 1 or args.num_workers < 0:
        parser.error("Batch size and steps must be positive; workers must be nonnegative.")
    if args.task != "all" and args.action_representation != "absolute":
        parser.error("Single-task presets require absolute actions.")
    if (
        args.method == "iql"
        and not args.reward_labels_path
        and not (args.reward_source == "terminal" and args.assume_success)
    ):
        parser.error(
            "Lingdong IQL requires real reward labels, or an explicit --reward-source terminal --assume-success baseline."
        )
    # Changes to semantic settings belong in the explicit arguments above so that
    # local validation checks exactly what will run on the worker.
    extra = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    for flag in extra:
        if flag.replace("_", "-").startswith(
            (
                "--data.",
                "--model.action-horizon",
                "--weight-loader",
                "--fsdp-devices",
                "--batch-size",
                "--checkpoint-base-dir",
                "--exp-name",
                "--overwrite",
            )
        ):
            parser.error(f"Use launcher options for data/resources; unsupported override: {flag}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment = args.exp_name or f"lingdong_{args.task}_{args.method}_{args.action_representation}_nebula_{stamp}"
    launch = {
        "env": {
            "METHOD": args.method,
            "TASK": args.task,
            "ACTION_REPRESENTATION": args.action_representation,
            "ACTION_HORIZON": "30",
            "BATCH_SIZE_PER_GPU": str(args.batch_size_per_gpu),
            "DATA_ROOT": args.data_root,
            "WEIGHT_PARAMS_PATH": args.weight_params_path,
            "CKPT_BASE_DIR": args.ckpt_base_dir,
            "EXP_NAME": experiment,
            "OPENPI_DATA_HOME": "/oss/embodied-ai-data/pku/openpi_cache",
            "SKIP_NORM": "1",
            "REWARD_SOURCE": args.reward_source,
            "ASSUME_SUCCESS": str(int(args.assume_success)),
            "WANDB_MODE": "offline",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "train_args": ["--num-train-steps", str(args.num_train_steps), "--num-workers", str(args.num_workers), *extra],
        "skip_setup": args.skip_setup,
    }
    if args.reward_labels_path:
        launch["env"]["REWARD_LABELS_PATH"] = args.reward_labels_path
    if args.pip_index_url:
        launch["env"]["PIP_INDEX_URL"] = args.pip_index_url
    if args.wheelhouse is not None:
        launch["env"]["NEBULA_WHEELHOUSE"] = args.wheelhouse
    local_preflight(launch, args.preflight_python, args.gpus)
    stage = args.stage_dir or Path(tempfile.mkdtemp(prefix="lingdong-nebula-"))
    if args.stage_dir:
        stage.mkdir(parents=True, exist_ok=False)
    stage = stage.resolve()
    copy_sources(stage)
    (stage / "nebula_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    (stage / "cluster.json").write_text(
        json.dumps(
            {
                "worker": {
                    "cpu": max(800, args.gpus * 400),
                    "gpu": args.gpus * 100,
                    "memory": 300000,
                    "disk": 300000,
                }
            },
            indent=2,
        )
        + "\n"
    )
    size = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
    summary = {
        "stage": str(stage),
        "source_mb": round(size / 1024**2, 2),
        "queue": args.queue,
        "requested_gpus": args.gpus,
        "training_gpus": "auto",
        "job_name": args.job_name or experiment,
        "launch": launch,
        "proxy": "disabled",
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    auth, oss = reference_settings(args.reference_root)
    config = Path(os.environ.get("NEBULA_USER_CONFIG", str(Path.home() / ".nebulactl/config.ini")))
    if not config.exists() and not all(
        auth.get(key) for key in ("user_id", "access_id", "access_key", "nebula_project")
    ):
        parser.error("Nebula credentials missing: set NEBULA_USER_CONFIG to your existing config.ini.")
    if len(oss) != 4:
        parser.error("OSS mount credentials missing: set NEBULA_OSS_ACCESS_ID/KEY/BUCKET/ENDPOINT.")
    if oss["oss_bucket"] != "embodied-ai-data,damo-rynnvla-data-zjk":
        parser.error("Mount order must be embodied-ai-data,damo-rynnvla-data-zjk.")
    nebulactl = shutil.which("nebulactl") or str(Path(sys.executable).parent / "nebulactl")
    if not Path(nebulactl).is_file():
        parser.error("nebulactl not found; set SUBMIT_PYTHON to an environment containing it.")
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    # Credentials stay outside the staged/uploaded source and out of process args.
    with tempfile.TemporaryDirectory(prefix="lingdong-nebula-auth-") as auth_path:
        auth_dir = Path(auth_path)
        if not config.exists():
            config = auth_dir / "config.ini"
            config.write_text("\n".join(f"{key}={value}" for key, value in auth.items()) + "\n")
            config.chmod(0o600)
        env["NEBULA_USER_CONFIG"] = str(config)
        properties = auth_dir / "mounts.properties"
        properties.write_text("\n".join(f"{key}={value}" for key, value in oss.items()) + "\n")
        properties.chmod(0o600)
        command = [
            nebulactl,
            "run",
            "mdl",
            "--property_file",
            str(properties),
            "--queue=" + args.queue,
            "--job_name=" + summary["job_name"],
            "--user_params=--launch-config=nebula_launch.json",
            "--entry=run_nebula.py",
            "--algo_name=" + args.algo_name,
            "--worker_count=1",
            "--file.cluster_file=cluster.json",
            "--public_pool_job_type=queuing",
            "--enable_oss_cache=true",
            "--nebula_work_dir=" + str(stage),
            "--nebula_workspace=" + experiment,
        ]
        secrets = [auth.get("access_id", ""), auth.get("access_key", "")]
        secrets += oss["oss_access_id"].split(",") + oss["oss_access_key"].split(",")
        # nebulactl versions may log the generated mount config; redact it here.
        with subprocess.Popen(
            command, cwd=stage, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process:
            for raw_line in process.stdout:
                line = raw_line
                for secret in secrets:
                    if secret:
                        line = line.replace(secret, "<redacted>")
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == "__main__":
    main()
