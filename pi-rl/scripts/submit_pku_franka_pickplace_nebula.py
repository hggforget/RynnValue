"""Submit PKU Franka pick-and-place SFT, IQL, or joint SFT to Nebula."""

import argparse
from datetime import datetime
from datetime import timezone
import json
import math
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
DATA_ROOT = "/oss/embodied-ai-data/pku/"
PICKPLACE = DATA_ROOT + "fr3_real_pickplace_20261027_20261030_merged"
PICKPLACE_REWARDS = PICKPLACE + "_progress_expert_sup_step025000_on_vlm4b90k"
COLLECT = DATA_ROOT + "fr3_collect_objects_20261017_progress_expert_sup_step005000"
DRAWER = DATA_ROOT + "fr3_drawer_20261102_03"
VARIANTS = {
    "sft": ("scripts/train_sft_pku_franka_pickplace.sh", "pi05_pku_franka_pickplace", "pku_franka_pickplace"),
    "iql": ("scripts/train_iql_pku_franka_pickplace.sh", "pi05_pku_franka_pickplace_iql", "pku_franka_pickplace"),
    "cotrain-sft": (
        "scripts/train_sft_pku_franka_pickplace_collect_objects.sh",
        "pi05_pku_franka_pickplace_collect_objects",
        "pku_franka_pickplace_collect_objects",
    ),
    "cotrain3-sft": (
        "scripts/train_sft_pku_franka_pickplace_collect_drawer.sh",
        "pi05_pku_franka_pickplace_collect_drawer",
        "pku_franka_pickplace_collect_drawer",
    ),
}


def validate_norm_stats(path):
    """SKIP_NORM uses the stats included in the source snapshot."""
    stats = json.loads(path.read_text())["norm_stats"]
    for key, dimension in (("state", 14), ("actions", 7)):
        for field in ("mean", "std", "q01", "q99"):
            values = stats[key].get(field)
            if values is None or len(values) != dimension or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Invalid {key}.{field} normalization in {path}; recompute without --skip-norm")


def local_preflight(launch, python, requested_gpus):
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    env.update(launch["env"])
    for name in ("INIT_CHECKPOINT", "FSDP_DEVICES", "BATCH_SIZE"):
        env.pop(name, None)
    env.update(
        GPUS=",".join(map(str, range(requested_gpus))),
        PYTHON=python,
        DRY_RUN="1",
        JAX_PLATFORMS="cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    result = subprocess.run(
        ["bash", launch["launcher"], *launch["train_args"]],
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
    # The launcher prints the exact trainer command on its last line. Parse that
    # command with the real config CLI and validate both datasets/reward metadata.
    trainer_args = shlex.split(result.stdout.splitlines()[-1])[2:]
    code = (
        "import sys, tyro; from openpi.training import config; "
        "name=sys.argv[1]; "
        "c=tyro.extras.overridable_config_cli({name:(name,config.get_config(name))}, args=sys.argv[1:]); "
        "d=c.data.create(c.assets_dirs,c.model); "
        "assert c.iql is None or abs(c.data.reward_discount-c.iql.discount)<1e-8, 'IQL reward discounts differ'; "
        "print('Trainer arguments and dataset metadata validated')"
    )
    subprocess.run([python, "-c", code, *trainer_args], cwd=ROOT, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(VARIANTS), default="sft")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    parser.add_argument("--num-train-steps", type=int, default=20_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exp-name")
    parser.add_argument("--job-name")
    parser.add_argument("--queue", default="damo_eai_rfm_l20z")
    parser.add_argument("--algo-name", default="pytorch271")
    parser.add_argument("--repo-id", help="Pick-and-place path; IQL defaults to the expert-labeled dataset.")
    parser.add_argument("--collect-repo-id", help="Collect-objects path for cotrain-sft / cotrain3-sft.")
    parser.add_argument("--drawer-root", help="Drawer collection root for --method cotrain3-sft.")
    parser.add_argument("--weight-params-path", default=DATA_ROOT + "pi05_base/params")
    parser.add_argument("--ckpt-base-dir", default=DATA_ROOT + "nebula_checkpoints")
    parser.add_argument("--openpi-data-home", default=DATA_ROOT + "openpi_cache")
    parser.add_argument(
        "--skip-norm", action="store_true", help="Use validated normalization included in the snapshot."
    )
    parser.add_argument("--resume", action="store_true", help="Resume --exp-name with its policy/optimizer/IQL states.")
    parser.add_argument("--skip-setup", action="store_true", help="Use only with a prepared worker image.")
    parser.add_argument("--pip-index-url")
    parser.add_argument("--wheelhouse", help="Worker wheel bundle; an empty string disables auto-detection.")
    parser.add_argument("--preflight-python", default="/opt/conda/envs/pi-rl/bin/python")
    parser.add_argument(
        "--skip-local-preflight",
        action="store_true",
        help="Defer data/config validation to the worker when local data or training dependencies are unavailable.",
    )
    parser.add_argument("--reference-root", type=Path, default=Path("/mnt/rynnbot_hangzhou/hdc/pi-rl"))
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Stage source and print settings without contacting Nebula."
    )
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Extra trainer flags after --.")
    args = parser.parse_args(argv)
    if args.batch_size_per_gpu < 1 or args.num_train_steps < 1 or args.num_workers < 0:
        parser.error("Batch size and steps must be positive; workers must be nonnegative.")
    if args.resume and not args.exp_name:
        parser.error("--resume requires the original --exp-name")
    if args.collect_repo_id is not None and args.method not in ("cotrain-sft", "cotrain3-sft"):
        parser.error("--collect-repo-id requires --method cotrain-sft or cotrain3-sft")
    if args.drawer_root is not None and args.method != "cotrain3-sft":
        parser.error("--drawer-root requires --method cotrain3-sft")
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
                "--assets-base-dir",
                "--exp-name",
                "--overwrite",
                "--resume",
            )
        ):
            parser.error(f"Use explicit submission options for data/resources/resume; unsupported override: {flag}")
    launcher, config, asset = VARIANTS[args.method]
    if args.skip_norm:
        validate_norm_stats(ROOT / "assets" / config / asset / "norm_stats.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017 - Python 3.10 worker/submit envs.
    experiment = args.exp_name or f"fr3_pickplace_{args.method.replace('-', '_')}_nebula_{stamp}"
    env = {
        "BATCH_SIZE_PER_GPU": str(args.batch_size_per_gpu),
        "WEIGHT_PARAMS_PATH": args.weight_params_path,
        "CKPT_BASE_DIR": args.ckpt_base_dir,
        "EXP_NAME": experiment,
        "OPENPI_DATA_HOME": args.openpi_data_home,
        "SKIP_NORM": str(int(args.skip_norm)),
        "NEBULA_SETUP_LOG_DIR": DATA_ROOT + "nebula_setup_logs",
        "WANDB_MODE": "offline",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    repo = args.repo_id or (PICKPLACE_REWARDS if args.method == "iql" else PICKPLACE)
    if args.method in ("cotrain-sft", "cotrain3-sft"):
        env.update(PICKPLACE_REPO_ID=repo, COLLECT_REPO_ID=args.collect_repo_id or COLLECT)
    else:
        env["REPO_ID"] = repo
    if args.method == "cotrain3-sft":
        env["DRAWER_ROOT"] = args.drawer_root or DRAWER
    if args.pip_index_url:
        env["PIP_INDEX_URL"] = args.pip_index_url
    if args.wheelhouse is not None:
        env["NEBULA_WHEELHOUSE"] = args.wheelhouse
    launch = {
        "launcher": launcher,
        "env": env,
        "train_args": ["--num-train-steps", str(args.num_train_steps), "--num-workers", str(args.num_workers), *extra],
        "skip_setup": args.skip_setup,
        "local_preflight": not args.skip_local_preflight,
    }
    if args.resume:
        launch["train_args"].append("--resume")
    if args.skip_local_preflight:
        print("Local data/config preflight deferred to the worker.", flush=True)
    else:
        local_preflight(launch, args.preflight_python, args.gpus)
    stage, summary = stage_launch(args, launch, prefix="pku-pickplace-nebula-", root=ROOT)
    if not args.dry_run:
        submit_staged_job(args, stage, summary, parser)


if __name__ == "__main__":
    main()
