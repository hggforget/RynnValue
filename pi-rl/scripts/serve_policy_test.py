from pathlib import Path
import subprocess
import sys

import pytest
import tyro

from openpi.training import config

from . import serve_policy


@pytest.mark.parametrize("mode", ["standard", "rtc"])
@pytest.mark.parametrize("config_name", ["pi05_lingdong_right_arm", "pi05_lingdong_right_arm_crop"])
def test_modes_are_explicit_and_share_only_checkpoint_loading(monkeypatch, mode, config_name):
    captured = {}
    sentinel = object()

    def load(train_config, checkpoint, **kwargs):
        captured.update(config=train_config, checkpoint=checkpoint, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", load)  # noqa: SLF001
    original = config.get_config(config_name)
    args = serve_policy.Args(
        inference_mode=mode,
        dataset_root="/oss/example/lingdong",
        policy=serve_policy.Checkpoint(config=config_name, dir="/oss/example/checkpoint"),
    )
    assert serve_policy.create_policy(args) is sentinel
    assert captured["checkpoint"] == args.policy.dir
    assert captured["config"].data.dataset_root == args.dataset_root
    assert config.get_config(config_name) is original
    assert captured["config"].data.crop_head == config_name.endswith("_crop")
    if mode == "standard":
        assert captured["kwargs"] == {"default_prompt": None}
    else:
        from openpi.policies.rtc_policy import RTCPolicy

        assert captured["kwargs"]["policy_factory"].func is RTCPolicy


def test_rtc_rejects_unsupported_combinations_before_loading(monkeypatch):
    monkeypatch.setattr(
        serve_policy._policy_config,  # noqa: SLF001
        "create_trained_policy",
        lambda *a, **k: pytest.fail("Unsupported request loaded weights"),
    )
    for args in (
        serve_policy.Args(inference_mode="rtc"),
        serve_policy.Args(inference_mode="rtc", policy=serve_policy.Checkpoint("pi05_droid", "unused")),
        serve_policy.Args(
            inference_mode="rtc",
            dsrl_ckpt_dir="unused",
            policy=serve_policy.Checkpoint("pi05_lingdong_right_arm", "unused"),
        ),
    ):
        with pytest.raises(ValueError, match="RTC"):
            serve_policy.create_policy(args)


@pytest.mark.parametrize("mode", ["standard", "rtc"])
@pytest.mark.parametrize("config_name", ["pi05_lingdong_right_arm", "pi05_lingdong_right_arm_crop"])
def test_shell_launchers_produce_parseable_independent_modes(mode, config_name):
    import os
    import shlex

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "DRY_RUN": "1", "INFERENCE_MODE": mode, "POLICY_CONFIG": config_name}
    result = subprocess.run(
        ["bash", "scripts/serve_pi05_lingdong_right_arm.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stdout)
    parsed = tyro.cli(serve_policy.Args, args=command[2:])
    assert parsed.inference_mode == mode
    assert isinstance(parsed.policy, serve_policy.Checkpoint)
    assert parsed.policy.config == config_name
    assert parsed.dataset_root == "/oss/embodied-ai-data/lingdong"
    assert "/mnt/" not in result.stdout


def test_default_cli_mode_is_original_inference():
    assert tyro.cli(serve_policy.Args, args=[]).inference_mode == "standard"


def test_launcher_selects_checkout_client_even_with_an_older_installed_package(tmp_path):
    import os

    python = tmp_path / "inspect_python"
    python.write_text(
        f"#!{sys.executable}\n"
        "from openpi_client.rtc_action_chunk_broker import RTCActionChunkBroker\n"
        "print(RTCActionChunkBroker.__name__)\n"
    )
    python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/serve_pi05_lingdong_right_arm.sh"],
        cwd=root,
        env={**os.environ, "PYTHON": str(python), "DRY_RUN": "0", "INFERENCE_MODE": "standard"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RTCActionChunkBroker"
