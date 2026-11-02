"""Source staging and credential handling shared by Nebula training submitters."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def copy_sources(destination, root=ROOT):
    # Explicit allowlist: the workspace contains checkpoints and a 41 GB core dump.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".git")
    for name in ("src", "scripts", "packages", "third_party", "assets", "configs"):
        shutil.copytree(root / name, destination / name, ignore=ignore)
    for name in ("run_nebula.py", "README.md", "LICENSE", "LICENSE_GEMMA.txt", "pyproject.toml"):
        shutil.copy2(root / name, destination / name)


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


def stage_launch(args, launch, *, prefix, root=ROOT):
    stage = args.stage_dir or Path(tempfile.mkdtemp(prefix=prefix))
    if args.stage_dir:
        stage.mkdir(parents=True, exist_ok=False)
    stage = stage.resolve()
    copy_sources(stage, root=root)
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
        "job_name": args.job_name or launch["env"]["EXP_NAME"],
        "launch": launch,
        "proxy": "disabled",
    }
    print(json.dumps(summary, indent=2), flush=True)
    return stage, summary


def submit_staged_job(args, stage, summary, parser):
    experiment = summary["launch"]["env"]["EXP_NAME"]
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
    with tempfile.TemporaryDirectory(prefix="nebula-auth-") as auth_path:
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
