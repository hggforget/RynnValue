"""Check that Lingdong launchers use matching training and normalization representations."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import tyro

from openpi.policies import lingdong_policy as policy
from openpi.training import config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_env(tmp_path):
    repo = tmp_path / "data" / policy.DATASET_NAMES[0]
    (repo / "meta").mkdir(parents=True)
    (repo / "meta/info.json").write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCH_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    for key in ("INIT_CHECKPOINT", "ACTION_REPRESENTATION", "ACTION_HORIZON", "BATCH_SIZE", "SKIP_NORM", "DRY_RUN"):
        env.pop(key, None)
    env.update(
        PYTHON=str(fake_python),
        DATA_ROOT=str(repo.parent),
        WEIGHT_PARAMS_PATH=str(weights),
        GPUS="0,1",
        FSDP_DEVICES="2",
        BATCH_SIZE_PER_GPU="16",
        EXP_NAME="test_lingdong",
        LAUNCH_LOG=str(tmp_path / "calls.jsonl"),
    )
    return env


@pytest.mark.parametrize(
    ("script", "extra_env", "cli", "representation", "horizon"),
    [
        ("train_pi05_lingdong_right_arm.sh", {}, [], "absolute", 30),
        ("train_pi05_lingdong_right_arm.sh", {"ACTION_REPRESENTATION": "relative"}, [], "relative", 30),
        ("train_pi05_lingdong_right_arm_relative.sh", {}, [], "relative", 30),
        (
            "train_pi05_lingdong_right_arm.sh",
            {},
            ["--data.action-representation", "relative", "--model.action-horizon=5", "--batch-size", "64"],
            "relative",
            5,
        ),
    ],
)
def test_training_and_norm_preflight_agree(launch_env, script, extra_env, cli, representation, horizon):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *cli],
        env={**launch_env, **extra_env},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(launch_env["LAUNCH_LOG"]).read_text().splitlines()]
    assert len(calls) == 3
    for args in calls[:2]:
        assert args[0] == "scripts/compute_lingdong_norm_stats.py"
        assert args[args.index("--action-representation") + 1] == representation
        assert int(args[args.index("--action-horizon") + 1]) == horizon
        assert args[args.index("--dataset-root") + 1] == launch_env["DATA_ROOT"]
    assert calls[1][-1] == "--check"
    name = policy.RELATIVE_CONFIG_NAME if representation == "relative" else policy.CONFIG_NAME
    assert calls[2][:2] == ["scripts/train.py", name]
    # Parse the actual generated command, including forwarded CLI overrides.
    parsed = tyro.extras.overridable_config_cli({name: (name, config.get_config(name))}, args=calls[2][1:])
    assert parsed.data.action_representation == representation
    assert parsed.model.action_horizon == horizon
    assert parsed.batch_size == (64 if "--batch-size" in cli else 32)


def test_skip_norm_still_checks_relative_assets(launch_env):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_pi05_lingdong_right_arm_relative.sh")],
        env={**launch_env, "SKIP_NORM": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(launch_env["LAUNCH_LOG"]).read_text().splitlines()]
    assert len(calls) == 2
    assert calls[0][-1] == "--check"
    assert calls[0][calls[0].index("--action-representation") + 1] == "relative"
    assert calls[1][1] == policy.RELATIVE_CONFIG_NAME


def test_relative_dry_run_does_not_launch_python(launch_env):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_pi05_lingdong_right_arm_relative.sh")],
        env={**launch_env, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[1:3] == ["scripts/train.py", policy.RELATIVE_CONFIG_NAME]
    assert not Path(launch_env["LAUNCH_LOG"]).exists()


@pytest.mark.parametrize("extra_env", [{"ACTION_REPRESENTATION": "bad"}, {"ACTION_HORIZON": "0"}])
def test_invalid_action_settings_fail_before_python(launch_env, extra_env):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_pi05_lingdong_right_arm.sh")],
        env={**launch_env, **extra_env},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0
    assert not Path(launch_env["LAUNCH_LOG"]).exists()
