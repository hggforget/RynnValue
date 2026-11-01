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
    for key in (
        "INIT_CHECKPOINT", "ACTION_REPRESENTATION", "ACTION_HORIZON", "TASK", "BATCH_SIZE", "SKIP_NORM", "DRY_RUN",
        "METHOD", "REWARD_SOURCE", "REWARD_LABELS_PATH", "ASSUME_SUCCESS",
    ):
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
    name = policy.RELATIVE_CROP_CONFIG_NAME if representation == "relative" else policy.CROP_CONFIG_NAME
    assert calls[2][:2] == ["scripts/train.py", name]
    # Parse the actual generated command, including forwarded CLI overrides.
    parsed = tyro.extras.overridable_config_cli({name: (name, config.get_config(name))}, args=calls[2][1:])
    assert parsed.data.action_representation == representation
    assert parsed.data.crop_head
    assert parsed.data.task == "all"
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
    assert calls[1][1] == policy.RELATIVE_CROP_CONFIG_NAME


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
    assert command[1:3] == ["scripts/train.py", policy.RELATIVE_CROP_CONFIG_NAME]
    assert not Path(launch_env["LAUNCH_LOG"]).exists()


@pytest.mark.parametrize(
    ("script", "extra_env", "cli", "task"),
    [
        ("train_pi05_lingdong_right_arm_pick_up.sh", {}, [], "pick_up"),
        ("train_pi05_lingdong_right_arm_put_down.sh", {}, [], "put_down"),
        ("train_pi05_lingdong_right_arm.sh", {"TASK": "pick_up"}, [], "pick_up"),
        ("train_pi05_lingdong_right_arm.sh", {"TASK": "put_down"}, [], "put_down"),
        ("train_pi05_lingdong_right_arm.sh", {}, ["--data.task", "pick_up"], "pick_up"),
        ("train_pi05_lingdong_right_arm.sh", {}, ["--data.task=put_down"], "put_down"),
    ],
)
@pytest.mark.parametrize("representation", ["absolute", "relative"])
def test_single_task_launchers_select_matching_config_and_keep_shared_norms(
    launch_env, script, extra_env, cli, task, representation
):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *cli],
        env={**launch_env, **extra_env, "ACTION_REPRESENTATION": representation},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(launch_env["LAUNCH_LOG"]).read_text().splitlines()]
    assert len(calls) == 3
    # Both tasks reuse the full-dataset norms for their action representation.
    for args in calls[:2]:
        assert args[0] == "scripts/compute_lingdong_norm_stats.py"
        assert args[args.index("--action-representation") + 1] == representation
        assert "--task" not in args
    names = policy.SINGLE_TASK_RELATIVE_CROP_CONFIGS if representation == "relative" else policy.SINGLE_TASK_CROP_CONFIGS
    name = names[task]
    assert calls[2][:2] == ["scripts/train.py", name]
    parsed = tyro.extras.overridable_config_cli({name: (name, config.get_config(name))}, args=calls[2][1:])
    assert parsed.data.task == task
    assert parsed.data.crop_head
    assert parsed.data.action_representation == representation
    assert parsed.model.action_horizon == 30


@pytest.mark.parametrize("task", ["all", "pick_up", "put_down"])
@pytest.mark.parametrize("reward_source", ["progress", "terminal"])
@pytest.mark.parametrize("representation", ["absolute", "relative"])
def test_iql_launch_checks_rewards_before_norms_and_passes_matching_flags(launch_env, task, reward_source, representation):
    labels = str(Path(launch_env["DATA_ROOT"]).parent / "reward labels.json")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_iql_lingdong_right_arm.sh")],
        env={**launch_env, "TASK": task, "ACTION_REPRESENTATION": representation,
             "REWARD_SOURCE": reward_source, "REWARD_LABELS_PATH": labels,
             "ASSUME_SUCCESS": "1" if reward_source == "terminal" else "0"},
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(launch_env["LAUNCH_LOG"]).read_text().splitlines()]
    assert len(calls) == 4
    assert calls[0][0] == "scripts/check_lingdong_iql_data.py"
    assert calls[0][calls[0].index("--task") + 1] == task
    assert calls[0][calls[0].index("--reward-source") + 1] == reward_source
    assert calls[0][calls[0].index("--reward-labels-path") + 1] == labels
    assert ("--assume-success" in calls[0]) == (reward_source == "terminal")
    assert all(args[0] == "scripts/compute_lingdong_norm_stats.py" for args in calls[1:3])
    names = policy.SINGLE_TASK_RELATIVE_CROP_CONFIGS if representation == "relative" else policy.SINGLE_TASK_CROP_CONFIGS
    joint = policy.RELATIVE_CROP_CONFIG_NAME if representation == "relative" else policy.CROP_CONFIG_NAME
    base = joint if task == "all" else names[task]
    name = f"{base}_iql"
    assert calls[3][:2] == ["scripts/train_iql.py", name]
    parsed = tyro.extras.overridable_config_cli({name: (name, config.get_config(name))}, args=calls[3][1:])
    assert parsed.data.task == task
    assert parsed.data.action_representation == representation
    assert parsed.data.reward_labels_path == labels
    assert parsed.data.reward_source == reward_source
    assert parsed.data.assume_success == (reward_source == "terminal")
    assert parsed.iql.proprio_key == "iql_state"


def test_iql_reward_cli_overrides_match_the_preflight_and_dry_run(launch_env):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_iql_lingdong_right_arm.sh"),
         "--data.reward-source=terminal", "--data.assume-success"],
        env={**launch_env, "DRY_RUN": "1"},
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout.splitlines()[-1])
    assert args[1:3] == ["scripts/train_iql.py", f"{policy.CROP_CONFIG_NAME}_iql"]
    assert "--data.assume-success" in args
    assert not Path(launch_env["LAUNCH_LOG"]).exists()


@pytest.mark.parametrize(
    "extra_env",
    [
        {"ACTION_REPRESENTATION": "bad"},
        {"ACTION_HORIZON": "0"},
        {"TASK": "unknown"},
        {"METHOD": "unknown"},
        {"METHOD": "iql", "REWARD_SOURCE": "fake"},
        {"METHOD": "iql", "ASSUME_SUCCESS": "yes"},
    ],
)
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


@pytest.mark.parametrize(
    ("script", "settings"),
    [
        ("train_pi05_lingdong_right_arm.sh", {}),
        ("train_pi05_lingdong_right_arm_relative.sh", {}),
        ("train_pi05_lingdong_right_arm_pick_up.sh", {}),
        ("train_pi05_lingdong_right_arm_put_down.sh", {}),
        ("train_iql_lingdong_right_arm.sh", {}),
        ("train_iql_lingdong_right_arm.sh", {"TASK": "pick_up"}),
        ("train_iql_lingdong_right_arm.sh", {"TASK": "put_down"}),
        ("train_iql_lingdong_right_arm.sh", {"ACTION_REPRESENTATION": "relative"}),
    ],
)
@pytest.mark.parametrize(("visible", "expected_gpus"), [(None, 4), ("3,5", 2), ("GPU-test-uuid", 1)])
def test_all_task_launchers_detect_visible_gpus(launch_env, tmp_path, script, settings, visible, expected_gpus):
    env = {**launch_env, **settings, "DRY_RUN": "1"}
    for name in ("GPUS", "FSDP_DEVICES", "BATCH_SIZE", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    # A visibility mask must win over the four physical GPUs reported by NVML.
    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\nprintf 'GPU 0: fake\\nGPU 1: fake\\nGPU 2: fake\\nGPU 3: fake\\n'\n")
    nvidia.chmod(0o755)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script)],
        env=env, capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[command.index("--fsdp-devices") + 1] == str(expected_gpus)
    assert command[command.index("--batch-size") + 1] == str(16 * expected_gpus)
    assert f"GPUs={visible or '0,1,2,3'}," in result.stdout
    assert not Path(launch_env["LAUNCH_LOG"]).exists()


def test_lingdong_rejects_mismatched_fsdp(launch_env):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_pi05_lingdong_right_arm.sh"), "--fsdp-devices", "1"],
        env={**launch_env, "DRY_RUN": "1"}, capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode != 0
    assert "FSDP must match the detected GPU count: 2" in result.stderr


@pytest.mark.parametrize("visible", ["", "-1"])
def test_no_visible_gpus_fails_before_training(launch_env, visible):
    env = {**launch_env, "CUDA_VISIBLE_DEVICES": visible, "DRY_RUN": "1"}
    env.pop("GPUS", None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_pi05_lingdong_right_arm.sh")],
        env=env, capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode != 0
    assert "No usable GPUs" in result.stderr
    assert not Path(launch_env["LAUNCH_LOG"]).exists()
