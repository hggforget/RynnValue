"""Offline Nebula staging and worker routing; never allocate cluster resources."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

import run_nebula as worker

from . import nebula_common as common
from . import submit_lingdong_nebula as lingdong
from . import submit_pku_franka_pickplace_nebula as submit


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for name in ("src", "scripts", "packages", "third_party", "assets", "configs"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(name)
    for name in ("run_nebula.py", "README.md", "LICENSE", "LICENSE_GEMMA.txt", "pyproject.toml"):
        (root / name).write_text(name)
    for name in ("checkpoints", ".git", ".aws"):
        (root / name).mkdir()
        (root / name / "excluded.txt").write_text("must not be staged")
    (root / "core-python-123").write_text("must not be staged")
    (root / "src/__pycache__").mkdir()
    (root / "src/__pycache__/module.pyc").write_bytes(b"cache")
    monkeypatch.setattr(submit, "ROOT", root)
    monkeypatch.setattr(lingdong, "ROOT", root)
    return root


@pytest.mark.parametrize("method", submit.VARIANTS)
def test_each_variant_stages_correct_data_launcher_and_worker_resources(source_tree, tmp_path, monkeypatch, method):
    monkeypatch.setattr(common, "submit_staged_job", lambda *args: pytest.fail("Dry-run contacted Nebula"))
    monkeypatch.setattr(submit, "submit_staged_job", lambda *args: pytest.fail("Dry-run contacted Nebula"))
    stage = tmp_path / "staged"
    submit.main(
        [
            "--method",
            method,
            "--dry-run",
            "--skip-local-preflight",
            "--stage-dir",
            str(stage),
            "--gpus",
            "2",
            "--batch-size-per-gpu",
            "8",
            "--num-train-steps",
            "12",
            "--num-workers",
            "0",
            "--wheelhouse",
            "/oss/wheels",
            "--",
            "--save-interval",
            "10",
        ]
    )
    launch = json.loads((stage / "nebula_launch.json").read_text())
    assert launch["launcher"] == submit.VARIANTS[method][0]
    assert launch["local_preflight"] is False
    assert launch["env"]["SKIP_NORM"] == "0"
    assert launch["env"]["BATCH_SIZE_PER_GPU"] == "8"
    assert launch["env"]["NEBULA_WHEELHOUSE"] == "/oss/wheels"
    assert launch["env"]["CKPT_BASE_DIR"] == "/oss/embodied-ai-data/pku/nebula_checkpoints"
    assert not {"GPUS", "FSDP_DEVICES", "BATCH_SIZE", "CUDA_VISIBLE_DEVICES"} & launch["env"].keys()
    assert launch["train_args"] == ["--num-train-steps", "12", "--num-workers", "0", "--save-interval", "10"]
    if method == "cotrain-sft":
        assert launch["env"]["PICKPLACE_REPO_ID"] == submit.PICKPLACE
        assert launch["env"]["COLLECT_REPO_ID"] == submit.COLLECT
        assert "REPO_ID" not in launch["env"]
    else:
        assert launch["env"]["REPO_ID"] == (submit.PICKPLACE_REWARDS if method == "iql" else submit.PICKPLACE)
    cluster = json.loads((stage / "cluster.json").read_text())
    assert cluster["worker"]["gpu"] == 200
    assert worker.training_command(stage, launch)[1] == str(stage / launch["launcher"])
    assert (stage / "src/source.txt").is_file()
    assert not (stage / "checkpoints").exists()
    assert not (stage / ".git").exists()
    assert not (stage / ".aws").exists()
    assert not (stage / "core-python-123").exists()
    assert not (stage / "src/__pycache__").exists()


def test_resume_overrides_and_skip_norm_use_the_matching_staged_assets(source_tree, tmp_path):
    config, asset = submit.VARIANTS["cotrain-sft"][1:]
    stats_path = source_tree / "assets" / config / asset / "norm_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats = {
        key: {field: [0.0] * dim for field in ("mean", "std", "q01", "q99")}
        for key, dim in (("state", 14), ("actions", 7))
    }
    stats_path.write_text(json.dumps({"norm_stats": stats}))
    stage = tmp_path / "staged"
    submit.main(
        [
            "--method",
            "cotrain-sft",
            "--dry-run",
            "--skip-local-preflight",
            "--stage-dir",
            str(stage),
            "--skip-norm",
            "--resume",
            "--exp-name",
            "existing_run",
            "--repo-id",
            "/oss/custom/pickplace",
            "--collect-repo-id",
            "/oss/custom/collect",
            "--weight-params-path",
            "/oss/custom/weights/params",
            "--ckpt-base-dir",
            "/oss/custom/checkpoints",
        ]
    )
    launch = json.loads((stage / "nebula_launch.json").read_text())
    assert launch["env"]["SKIP_NORM"] == "1"
    assert launch["env"]["EXP_NAME"] == "existing_run"
    assert launch["env"]["PICKPLACE_REPO_ID"] == "/oss/custom/pickplace"
    assert launch["env"]["COLLECT_REPO_ID"] == "/oss/custom/collect"
    assert launch["env"]["CKPT_BASE_DIR"] == "/oss/custom/checkpoints"
    assert launch["env"]["WEIGHT_PARAMS_PATH"] == "/oss/custom/weights/params"
    assert launch["train_args"][-1] == "--resume"
    assert (stage / "assets" / config / asset / "norm_stats.json").read_bytes() == stats_path.read_bytes()
    stats["state"]["mean"] = [0.0] * 20
    stats_path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError, match="state.mean"):
        submit.validate_norm_stats(stats_path)


@pytest.mark.parametrize(
    "args",
    [
        ["--resume"],
        ["--gpus", "3"],
        ["--batch-size-per-gpu", "0"],
        ["--num-train-steps", "0"],
        ["--num-workers", "-1"],
        ["--collect-repo-id", "/unused/collect"],
        ["--", "--data.repo-id", "/different/data"],
        ["--", "--overwrite"],
        ["--", "--resume"],
        ["--", "--fsdp-devices=1"],
        ["--", "--batch_size", "1"],
        ["--", "--assets-base-dir", "/tmp/elsewhere"],
    ],
)
def test_invalid_or_conflicting_options_fail_before_staging(source_tree, tmp_path, args):
    stage = tmp_path / "staged"
    with pytest.raises(SystemExit) as result:
        submit.main(["--dry-run", "--skip-local-preflight", "--stage-dir", str(stage), *args])
    assert result.value.code == 2
    assert not stage.exists()


def test_local_preflight_models_requested_gpus_without_uploading_device_ids(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=("/training/python scripts/train_iql.py pi05_pku_franka_pickplace_iql --exp-name preview\n"),
        )

    monkeypatch.setattr(submit.subprocess, "run", run)
    for key in ("HTTP_PROXY", "https_proxy", "GPUS", "BATCH_SIZE", "FSDP_DEVICES", "INIT_CHECKPOINT"):
        monkeypatch.setenv(key, "inherited-value")
    launch = {
        "launcher": submit.VARIANTS["iql"][0],
        "env": {"BATCH_SIZE_PER_GPU": "16", "REPO_ID": submit.PICKPLACE_REWARDS},
        "train_args": ["--num-train-steps", "10"],
    }
    submit.local_preflight(launch, "/training/python", 4)
    assert calls[0][0][:2] == ["bash", launch["launcher"]]
    assert calls[0][1]["env"]["GPUS"] == "0,1,2,3"
    assert calls[0][1]["env"]["PYTHON"] == "/training/python"
    assert calls[1][0][3] == "pi05_pku_franka_pickplace_iql"
    assert calls[1][1]["env"]["JAX_PLATFORMS"] == "cpu"
    assert not any(key.lower().endswith("_proxy") for key in calls[0][1]["env"])
    assert not {"GPUS", "BATCH_SIZE", "FSDP_DEVICES"} & launch["env"].keys()


@pytest.mark.parametrize("launcher", [*worker.LAUNCHERS, None])
def test_worker_keeps_scheduler_visibility_and_uses_its_own_python(tmp_path, monkeypatch, launcher):
    launch = {"env": {"EXP_NAME": "test", "PYTHON": "/wrong/python"}, "train_args": [], "skip_setup": True}
    if launcher is not None:
        launch["launcher"] = launcher
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch))
    monkeypatch.setattr(sys, "argv", ["run_nebula.py", "--launch-config", str(path)])
    for name in ("GPUS", "FSDP_DEVICES", "BATCH_SIZE", "HTTPS_PROXY", "http_proxy"):
        monkeypatch.setenv(name, "inherited-value")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(worker, "setup_oss_mounts", lambda: None)
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker.os, "chdir", lambda path: None)

    def execute(executable, command, env):
        assert executable == "bash"
        assert command[1].endswith(launcher or worker.LAUNCHERS[0])
        assert env["PYTHON"] == sys.executable
        assert env["CUDA_VISIBLE_DEVICES"] == "2,5"
        assert not {"GPUS", "FSDP_DEVICES", "BATCH_SIZE"} & env.keys()
        assert not any(key.lower().endswith("_proxy") for key in env)
        raise StopIteration

    monkeypatch.setattr(worker.os, "execvpe", execute)
    with pytest.raises(StopIteration):
        worker.main()


def test_worker_rejects_unsupported_paths():
    with pytest.raises(ValueError, match="Unsupported"):
        worker.training_command(Path("/source"), {"launcher": "../../arbitrary.sh", "train_args": []})


def test_shared_transport_keeps_credentials_outside_snapshot_and_redacts_output(
    source_tree, tmp_path, monkeypatch, capsys
):
    stage = tmp_path / "stage"
    stage.mkdir()
    executable = tmp_path / "nebulactl"
    executable.touch()
    auth = {"user_id": "test", "access_id": "auth-id", "access_key": "auth-secret", "nebula_project": "test"}
    oss = {
        "oss_access_id": "oss-id",
        "oss_access_key": "oss-secret",
        "oss_bucket": "embodied-ai-data,damo-rynnvla-data-zjk",
        "oss_endpoint": "internal",
    }
    monkeypatch.setattr(common, "reference_settings", lambda root: (auth, oss))
    monkeypatch.setattr(common.shutil, "which", lambda command: str(executable))
    monkeypatch.setenv("NEBULA_USER_CONFIG", str(tmp_path / "missing-config"))
    monkeypatch.setenv("HTTP_PROXY", "inherited-proxy")

    class Process:
        returncode = 0
        stdout = ("auth-secret oss-secret oss-id auth-id\n",)

        def __init__(self, command, **kwargs):
            properties = Path(command[command.index("--property_file") + 1])
            assert properties.parent != stage
            assert properties.stat().st_mode & 0o777 == 0o600
            assert "oss-secret" in properties.read_text()
            config_path = Path(kwargs["env"]["NEBULA_USER_CONFIG"])
            assert config_path.stat().st_mode & 0o777 == 0o600
            assert not any(key.lower().endswith("_proxy") for key in kwargs["env"])
            assert not any("secret" in arg for arg in command)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr(common.subprocess, "Popen", Process)
    args = argparse.Namespace(reference_root=source_tree, queue="test", algo_name="pytorch271")
    summary = {"job_name": "test", "launch": {"env": {"EXP_NAME": "test"}}}
    common.submit_staged_job(args, stage, summary, argparse.ArgumentParser())
    output = capsys.readouterr().out
    assert "secret" not in output
    assert "<redacted>" in output
    assert not list(stage.iterdir())


def test_lingdong_submission_still_uses_legacy_worker_default(source_tree, tmp_path, monkeypatch):
    monkeypatch.setattr(lingdong, "local_preflight", lambda *args: None)
    stage = tmp_path / "lingdong-stage"
    monkeypatch.setattr(sys, "argv", ["submit_lingdong_nebula.py", "--dry-run", "--stage-dir", str(stage)])
    lingdong.main()
    launch = json.loads((stage / "nebula_launch.json").read_text())
    assert launch["env"]["METHOD"] == "sft"
    assert worker.training_command(stage, launch)[1].endswith("scripts/train_pi05_lingdong_right_arm.sh")
