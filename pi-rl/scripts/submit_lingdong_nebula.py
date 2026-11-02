"""Stage a small source snapshot and submit one Lingdong SFT/IQL job to Nebula."""

import argparse
from datetime import datetime
from datetime import timezone
import os
from pathlib import Path
import shlex
import subprocess

if __package__:
    from .nebula_common import stage_launch
    from .nebula_common import submit_staged_job
else:
    from nebula_common import stage_launch
    from nebula_common import submit_staged_job

ROOT = Path(__file__).resolve().parents[1]


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017 - Python 3.10 worker/submit envs.
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
    stage, summary = stage_launch(args, launch, prefix="lingdong-nebula-", root=ROOT)
    if args.dry_run:
        return

    submit_staged_job(args, stage, summary, parser)


if __name__ == "__main__":
    main()
