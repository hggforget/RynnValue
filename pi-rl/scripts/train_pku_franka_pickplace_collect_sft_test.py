"""Offline integration checks for joint pick-and-place / collect-objects SFT."""

import dataclasses
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tyro

from openpi import transforms
from openpi.policies import pku_franka_collect_policy as policy
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader

from . import compute_norm_stats_fast as norms

CONFIG = "pi05_pku_franka_pickplace_collect_objects"
ASSET = "pku_franka_pickplace_collect_objects"
TRIPLE_CONFIG = "pi05_pku_franka_pickplace_collect_drawer"
TRIPLE_ASSET = "pku_franka_pickplace_collect_drawer"
SUPER_CONFIG = "pi05_pku_franka_super"
ROOT = Path(__file__).resolve().parents[1]


class ByteTokenizer:
    """Small local tokenizer so the tests do not download model assets."""

    def tokenize(self, prompt, state=None):
        del state
        values = np.frombuffer(prompt.encode(), dtype=np.uint8).astype(np.int32)
        return np.pad(values, (0, 200 - len(values))), np.arange(200) < len(values)


@pytest.fixture
def local_repos(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config.ModelTransformFactory,
        "__call__",
        lambda self, model: transforms.Group(
            inputs=[
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(ByteTokenizer(), discrete_state_input=True),
                transforms.PadStatesAndActions(model.action_dim),
            ]
        ),
    )
    video = tmp_path / "source.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), 64, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    repos = tuple(str(tmp_path / name) for name in ("pickplace data", "collect data"))
    expected = []
    for dataset_index, repo in enumerate(repos):
        root = Path(repo)
        (root / "meta").mkdir(parents=True)
        features = {
            **{
                f"observation.images.{camera}": {"dtype": "video", "shape": [16, 16, 3]}
                for camera in ("primary", "wrist")
            },
            **{
                key: {"dtype": "float32", "shape": [dim]}
                for key, dim in (
                    ("observation.state.cartesian", 6),
                    ("observation.state.gripper", 1),
                    ("observation.state.joints", 7),
                    ("observation.state", 20),
                    ("action", 7),
                )
            },
            **{
                key: {"dtype": "int64", "shape": [1]} for key in ("index", "frame_index", "episode_index", "task_index")
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        }
        if dataset_index == 1:
            features["progress"] = {"dtype": "float32", "shape": [1]}
        episodes, statistics = [], []
        prompts = ["Put the red block on the blue plate.", "Put the red block on the green plate."]
        if dataset_index == 1:
            prompts = ["lace each object on the plate that matches its color."]
        for episode in range(2 if dataset_index == 0 else 4):
            value = dataset_index * 10 + episode
            cartesian = np.full((4, 6), value, dtype=np.float32)
            gripper = np.full((4, 1), 0.5, dtype=np.float32)
            joints = np.full((4, 7), value + 0.25, dtype=np.float32)
            actions = np.column_stack((cartesian + np.arange(4, dtype=np.float32)[:, None] / 10, gripper))
            columns = {
                "observation.state.cartesian": cartesian.tolist(),
                "observation.state.gripper": gripper.tolist(),
                "observation.state.joints": joints.tolist(),
                "observation.state": np.full((4, 20), np.nan).tolist(),
                "action": actions.tolist(),
                "timestamp": np.arange(4, dtype=np.float32) / 30,
                "index": np.arange(4) + episode * 4,
                "frame_index": np.arange(4),
                "episode_index": [episode] * 4,
                "task_index": [episode if dataset_index == 0 else 0] * 4,
            }
            if dataset_index == 1:
                columns["progress"] = [float("nan")] * 4  # SFT must never consume reward annotations.
            path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(columns), path)
            for camera in ("primary", "wrist"):
                target = root / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(video, target)
            prompt = prompts[episode if dataset_index == 0 else 0]
            episodes.append({"episode_index": episode, "length": 4, "tasks": [prompt]})
            statistics.append({"episode_index": episode, "stats": {}})
            if dataset_index == 0 or episode not in (0, 2):
                expected.append((np.concatenate((cartesian, gripper, joints), axis=-1), actions, prompt, dataset_index))
        info = {
            "codebase_version": "v2.1",
            "robot_type": "franka",
            "fps": 30,
            "features": features,
            "total_episodes": len(episodes),
            "total_frames": len(episodes) * 4,
            "total_tasks": len(prompts),
            "chunks_size": 1000,
            "total_chunks": 1,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
        for filename, rows in (
            ("episodes.jsonl", episodes),
            ("episodes_stats.jsonl", statistics),
            ("tasks.jsonl", [{"task_index": i, "task": text} for i, text in enumerate(prompts)]),
        ):
            (root / "meta" / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (root / "meta/info.json").write_text(json.dumps(info))
    preset = config.get_config(CONFIG)
    preset = dataclasses.replace(
        preset, assets_base_dir=str(tmp_path / "assets"), data=dataclasses.replace(preset.data, repo_ids=repos)
    )
    return preset, repos, expected


def test_joint_sft_keeps_task_namespaces_and_filters_only_collect_episodes(local_repos):
    preset, repos, expected = local_repos
    data = preset.data.create(preset.assets_dirs, preset.model)
    assert preset.iql is None
    assert data.episode_ids == {repos[1]: (1, 3)}
    assert data.action_sequence_keys == ("action",)
    dataset = data_loader.make_torch_dataset(data, preset.model.action_horizon, preset.model)
    transformed = data_loader.transform_dataset(dataset, data, skip_norm_stats=True)
    assert len(dataset) == len(transformed) == 16
    for index, (states, actions, task, source) in enumerate(expected):
        for frame in (0, 3):
            sample = transformed[index * 4 + frame]
            np.testing.assert_array_equal(sample["state"][:14], states[frame])
            action_indices = np.minimum(frame + np.arange(30), 3)
            np.testing.assert_array_equal(sample["actions"][:, :7], actions[action_indices])
            assert not sample["actions"][:, 7:].any()
            text = "".join(chr(int(value)) for value in sample["tokenized_prompt"][sample["tokenized_prompt_mask"]])
            assert text == (task if source == 0 else policy.EXPERT_COLLECT_OBJECTS_PROMPT)
            assert set(sample["image"]) == {"base_0_rgb", "left_wrist_0_rgb"}
            assert all(image.shape == (224, 224, 3) for image in sample["image"].values())
            assert "progress" not in sample


def test_normalization_scans_both_sources_and_matches_training_chunks(local_repos, monkeypatch):
    preset, repos, expected = local_repos
    # Exercise the public --repo-ids path, rather than pre-overriding the config only.
    monkeypatch.setattr(
        config,
        "get_config",
        lambda name: dataclasses.replace(
            preset,
            data=config.LeRobotPkuFrankaPickplaceCollectDataConfig(
                repo_id=ASSET,
                assets=config.AssetsConfig(asset_id=ASSET),
            ),
        ),
    )
    norms.main(CONFIG, repo_ids=list(repos), batch_size=1, num_workers=0)
    stats = normalize.load(preset.assets_dirs / ASSET)
    state_rows = np.concatenate([entry[0] for entry in expected])
    action_chunks = np.concatenate(
        [actions[np.minimum(frame + np.arange(30), 3)] for _, actions, _, _ in expected for frame in range(4)]
    )
    for key, values in (("state", state_rows), ("actions", action_chunks)):
        np.testing.assert_allclose(stats[key].mean, values.mean(axis=0, dtype=np.float64), atol=1e-5)
        np.testing.assert_allclose(stats[key].std, values.std(axis=0, dtype=np.float64), atol=1e-5)
        assert stats[key].q01.shape == stats[key].mean.shape
        assert np.isfinite(stats[key].q99).all()
    data = preset.data.create(preset.assets_dirs, preset.model)
    loader = data_loader.create_torch_data_loader(data, preset.model, 30, 4, num_batches=1, shuffle=True)
    observation, actions = next(iter(loader))
    assert observation.state.shape == (4, 32)
    assert actions.shape == (4, 30, 32)
    assert np.isfinite(observation.state).all()
    assert np.isfinite(actions).all()


@pytest.mark.parametrize("change", ["fps", "action", "count", "duplicate"])
def test_joint_config_rejects_incompatible_sources(local_repos, change):
    preset, repos, _ = local_repos
    if change in ("count", "duplicate"):
        ids = repos[:1] if change == "count" else (repos[0], repos[0])
        factory = dataclasses.replace(preset.data, repo_ids=ids)
    else:
        path = Path(repos[1]) / "meta/info.json"
        info = json.loads(path.read_text())
        if change == "fps":
            info["fps"] = 5
        else:
            info["features"]["action"]["representation"] = "relative_delta_pose"
        path.write_text(json.dumps(info))
        factory = preset.data
    with pytest.raises(ValueError, match="30 Hz|absolute_target_pose|two distinct"):
        factory.create(preset.assets_dirs, preset.model)


def test_serving_keeps_caller_instruction_without_training_mounts(local_repos, monkeypatch):
    preset, _, _ = local_repos
    monkeypatch.setattr(policy, "validate_action_metadata", lambda *args: pytest.fail("Serving read dataset metadata"))
    factory = dataclasses.replace(preset.data, repo_ids=("/unmounted/pickplace", "/unmounted/collect"))
    data = factory.create_for_inference(preset.assets_dirs, preset.model)
    raw = {
        "observation.state.cartesian": np.zeros(6),
        "observation.state.gripper": np.zeros(1),
        "observation.state.joints": np.zeros(7),
        "observation.images.primary": np.zeros((16, 16, 3), np.uint8),
        "observation.images.wrist": np.zeros((16, 16, 3), np.uint8),
        "prompt": "Put the yellow block on the blue plate.",
    }
    result = transforms.compose(data.data_transforms.inputs)(raw)
    assert result["prompt"] == raw["prompt"]
    assert "actions" not in result


def test_launcher_and_norm_cli_use_identical_ordered_repos(local_repos, tmp_path, monkeypatch):
    preset, repos, _ = local_repos
    weights = tmp_path / "weights/params"
    weights.mkdir(parents=True)
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "PICKPLACE_REPO_ID": repos[0],
        "COLLECT_REPO_ID": repos[1],
        "WEIGHT_PARAMS_PATH": str(weights),
        "GPUS": "0,1",
        "DRY_RUN": "1",
        "SKIP_NORM": "0",
    }
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/train_sft_pku_franka_pickplace_collect_objects.sh"),
            "--num-train-steps",
            "123",
            "--resume",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()[-2:]]
    for args, flag in zip(commands, ("--repo-ids", "--data.repo-ids"), strict=True):
        start = args.index(flag) + 1
        assert args[start : start + 2] == list(repos)
    train_args = commands[1][commands[1].index(CONFIG) :]
    parsed = tyro.extras.overridable_config_cli({CONFIG: (CONFIG, config.get_config(CONFIG))}, args=train_args)
    assert parsed.data.repo_ids == repos
    assert parsed.batch_size == 32
    assert parsed.fsdp_devices == 2
    assert parsed.num_train_steps == 123
    assert parsed.resume
    assert parsed.iql is None
    # Parse and execute the actual generated normalization arguments using a tiny local dataset.
    norm_args = commands[0][commands[0].index("scripts/compute_norm_stats_fast.py") + 1 :]
    assert norm_args[norm_args.index("--config-name") + 1] == CONFIG
    monkeypatch.setattr(config, "get_config", lambda name: preset)
    tyro.cli(norms.main, args=norm_args)
    assert (preset.assets_dirs / ASSET / "norm_stats.json").is_file()


@pytest.fixture
def triple_repos(local_repos, tmp_path):
    base, repos, expected = local_repos
    expected = list(expected)
    drawer_root = tmp_path / "drawer collection"
    drawer_repos = []
    for offset, name in enumerate(("close", "open")):
        root = drawer_root / f"drawer-{name}" / "2026-11-03"
        shutil.copytree(repos[0], root)
        drawer_repos.append(str(root))
        prompt = f"{name.capitalize()} the white drawer."
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["total_tasks"] = 1
        info["features"]["observation.capture_timestamp_ns"] = {"dtype": "int64", "shape": [1]}
        info["features"]["action.relative"] = {"dtype": "float32", "shape": [7]}
        info_path.write_text(json.dumps(info))
        (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
        episodes = []
        for episode in range(2):
            path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
            table = pq.read_table(path)
            for key in ("observation.state.cartesian", "observation.state.joints", "action"):
                values = np.asarray(table[key].to_pylist(), dtype=np.float32) + 20 * (offset + 1)
                if key == "action":
                    values[:, -1] = 0.5  # Preserve the normalized gripper command.
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(values.tolist()))
            table = table.set_column(table.schema.get_field_index("task_index"), "task_index", pa.array([0] * 4))
            table = table.append_column("observation.capture_timestamp_ns", pa.array([1000] * 4))
            table = table.append_column("action.relative", pa.array([[float("nan")] * 7] * 4))
            pq.write_table(table, path)
            states = np.concatenate(
                [np.asarray(table[key].to_pylist()) for key in (
                    "observation.state.cartesian", "observation.state.gripper", "observation.state.joints",
                )],
                axis=-1,
            )
            expected.append((states, np.asarray(table["action"].to_pylist()), prompt, offset + 2))
            episodes.append({"episode_index": episode, "length": 4, "tasks": [prompt]})
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    preset = config.get_config(TRIPLE_CONFIG)
    preset = dataclasses.replace(
        preset,
        assets_base_dir=base.assets_base_dir,
        data=dataclasses.replace(preset.data, repo_ids=repos, dataset_root=str(drawer_root)),
    )
    return preset, repos, tuple(drawer_repos), expected


@pytest.fixture
def super_repos(triple_repos, tmp_path):
    base, repos, collection_repos, expected = triple_repos
    expected = list(expected)
    additional_roots = (tmp_path / "new drawer", tmp_path / "new plate")
    extra_repos = []
    prompts = ("Open the wooden drawer.", "Put the blue block on the yellow plate.")
    for offset, (directory, prompt) in enumerate(zip(additional_roots, prompts, strict=True)):
        root = directory if offset == 0 else directory / "blue-to-yellow/2026-11-03"
        shutil.copytree(collection_repos[0], root)
        extra_repos.append(str(root))
        (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
        episodes = []
        for episode in range(2):
            path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
            table = pq.read_table(path)
            for key in ("observation.state.cartesian", "observation.state.joints", "action"):
                values = np.asarray(table[key].to_pylist(), dtype=np.float32) + 100 * (offset + 1)
                if key == "action":
                    values[:, -1] = 0.5
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(values.tolist()))
            pq.write_table(table, path)
            states = np.concatenate(
                [np.asarray(table[key].to_pylist()) for key in (
                    "observation.state.cartesian", "observation.state.gripper", "observation.state.joints",
                )], axis=-1,
            )
            expected.append((states, np.asarray(table["action"].to_pylist()), prompt, offset + 4))
            episodes.append({"episode_index": episode, "length": 4, "tasks": [prompt]})
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    preset = config.get_config(SUPER_CONFIG)
    preset = dataclasses.replace(
        preset, assets_base_dir=base.assets_base_dir,
        data=dataclasses.replace(
            preset.data, repo_ids=repos, dataset_root=base.data.dataset_root,
            additional_dataset_roots=tuple(map(str, additional_roots)),
        ),
    )
    return preset, repos, (*collection_repos, *extra_repos), expected


@pytest.mark.parametrize("fixture_name", ["triple_repos", "super_repos"])
def test_collection_sft_preserves_task_prompts_and_only_filters_collect(request, fixture_name):
    preset, repos, drawer_repos, expected = request.getfixturevalue(fixture_name)
    data = preset.data.create(preset.assets_dirs, preset.model)
    assert data.repo_ids == (*repos, *drawer_repos)
    assert data.episode_ids == {repos[1]: (1, 3)}
    assert preset.iql is None
    dataset = data_loader.make_torch_dataset(data, 30, preset.model)
    transformed = data_loader.transform_dataset(dataset, data, skip_norm_stats=True)
    assert len(dataset) == len(expected) * 4
    for episode, (states, actions, task, source) in enumerate(expected):
        for frame in (0, 3):
            sample = transformed[episode * 4 + frame]
            np.testing.assert_array_equal(sample["state"][:14], states[frame])
            np.testing.assert_array_equal(sample["actions"][:, :7], actions[np.minimum(frame + np.arange(30), 3)])
            assert not sample["actions"][:, 7:].any()
            text = "".join(chr(int(value)) for value in sample["tokenized_prompt"][sample["tokenized_prompt_mask"]])
            assert text == (policy.EXPERT_COLLECT_OBJECTS_PROMPT if source == 1 else task)
            assert set(sample["image"]) == {"base_0_rgb", "left_wrist_0_rgb"}
            assert "progress" not in sample
            assert "action.relative" not in sample
            assert "observation.capture_timestamp_ns" not in sample


@pytest.mark.parametrize("fixture_name", ["triple_repos", "super_repos"])
def test_shared_norms_include_every_collection(request, fixture_name, monkeypatch):
    preset, repos, _, expected = request.getfixturevalue(fixture_name)
    monkeypatch.setattr(config, "get_config", lambda name: preset)
    norms.main(
        preset.name, repo_ids=list(repos), dataset_root=preset.data.dataset_root, batch_size=1, num_workers=0,
        additional_dataset_roots=getattr(preset.data, "additional_dataset_roots", None),
    )
    stats = normalize.load(preset.assets_dirs / preset.data.assets.asset_id)
    states = np.concatenate([entry[0] for entry in expected])
    actions = np.concatenate(
        [values[np.minimum(frame + np.arange(30), 3)] for _, values, _, _ in expected for frame in range(4)]
    )
    for key, values in (("state", states), ("actions", actions)):
        # Online batch moments use float32; allow its rounding at large fixture values.
        np.testing.assert_allclose(stats[key].mean, values.mean(axis=0, dtype=np.float64), rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(stats[key].std, values.std(axis=0, dtype=np.float64), rtol=1e-6, atol=1e-5)
    data = preset.data.create(preset.assets_dirs, preset.model)
    loader = data_loader.create_torch_data_loader(data, preset.model, 30, 4, num_batches=1)
    observation, actions = next(iter(loader))
    assert observation.state.shape == (4, 32)
    assert actions.shape == (4, 30, 32)
    assert np.isfinite(observation.state).all()
    assert np.isfinite(actions).all()


@pytest.mark.parametrize("change", ["empty", "duplicate", "fps", "camera", "action"])
def test_triple_config_rejects_invalid_drawer_data(triple_repos, tmp_path, change):
    preset, repos, drawer_repos, _ = triple_repos
    factory = preset.data
    if change == "empty":
        empty = tmp_path / "empty drawer"
        empty.mkdir()
        factory = dataclasses.replace(factory, dataset_root=str(empty))
    elif change == "duplicate":
        factory = dataclasses.replace(factory, dataset_root=repos[0])
    else:
        path = Path(drawer_repos[0]) / "meta/info.json"
        info = json.loads(path.read_text())
        if change == "fps":
            info["fps"] = 5
        elif change == "camera":
            del info["features"]["observation.images.wrist"]
        else:
            info["features"]["action"]["representation"] = "relative_delta_pose"
        path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="No LeRobot|distinct datasets|30 Hz|camera|absolute_target_pose"):
        factory.create(preset.assets_dirs, preset.model)


@pytest.mark.parametrize("fixture_name", ["triple_repos", "super_repos"])
def test_collection_serving_does_not_need_training_mounts(request, fixture_name, monkeypatch):
    preset, _, _, _ = request.getfixturevalue(fixture_name)
    monkeypatch.setattr(policy, "validate_action_metadata", lambda *args: pytest.fail("Serving read dataset metadata"))
    factory = dataclasses.replace(
        preset.data, repo_ids=("/unmounted/pickplace", "/unmounted/collect"), dataset_root="/unmounted/drawer",
    )
    data = factory.create_for_inference(preset.assets_dirs, preset.model)
    assert data.action_sequence_keys == ("action",)


@pytest.mark.parametrize("fixture_name", ["triple_repos", "super_repos"])
def test_collection_launcher_routes_identical_data_to_statistics_and_training(request, fixture_name, tmp_path, monkeypatch):
    preset, repos, _, _ = request.getfixturevalue(fixture_name)
    super_mode = fixture_name == "super_repos"
    launcher = "train_sft_pku_franka_super.sh" if super_mode else "train_sft_pku_franka_pickplace_collect_drawer.sh"
    weights = tmp_path / "weights/params"
    weights.mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / launcher), "--num-train-steps", "123"],
        env={
            **os.environ, "PYTHON": sys.executable, "PICKPLACE_REPO_ID": repos[0], "COLLECT_REPO_ID": repos[1],
            "DRAWER_ROOT": preset.data.dataset_root, "WEIGHT_PARAMS_PATH": str(weights),
            "GPUS": "0,1", "DRY_RUN": "1", "SKIP_NORM": "0", "FR3_INCLUDE_DRAWER": "0",
            "FR3_SUPER_COTRAIN": "0",
            **({
                "EXTRA_DRAWER_ROOT": preset.data.additional_dataset_roots[0],
                "PLATE_ROOT": preset.data.additional_dataset_roots[1],
            } if super_mode else {}),
        },
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    norm_command, train_command = [shlex.split(line) for line in result.stdout.splitlines()[-2:]]
    assert norm_command[norm_command.index("--dataset-root") + 1] == preset.data.dataset_root
    assert train_command[train_command.index("--data.dataset-root") + 1] == preset.data.dataset_root
    parsed = tyro.extras.overridable_config_cli(
        {preset.name: (preset.name, config.get_config(preset.name))},
        args=train_command[train_command.index(preset.name):],
    )
    assert parsed.data.repo_ids == repos
    assert parsed.data.dataset_root == preset.data.dataset_root
    assert parsed.batch_size == 32
    assert parsed.fsdp_devices == 2
    assert parsed.num_train_steps == 123
    if super_mode:
        start = norm_command.index("--additional-dataset-roots") + 1
        assert tuple(norm_command[start:start + 2]) == preset.data.additional_dataset_roots
        assert parsed.data.additional_dataset_roots == preset.data.additional_dataset_roots
    monkeypatch.setattr(config, "get_config", lambda name: preset)
    tyro.cli(norms.main, args=norm_command[norm_command.index("scripts/compute_norm_stats_fast.py") + 1:])
    assert (preset.assets_dirs / preset.data.assets.asset_id / "norm_stats.json").is_file()


@pytest.mark.parametrize("change", ["missing", "overlap", "fps"])
def test_super_validates_additional_collections(super_repos, tmp_path, change):
    preset, _, _, _ = super_repos
    factory = preset.data
    if change == "missing":
        factory = dataclasses.replace(factory, additional_dataset_roots=(str(tmp_path / "missing"),))
    elif change == "overlap":
        factory = dataclasses.replace(factory, additional_dataset_roots=(factory.dataset_root,))
    else:
        path = Path(factory.additional_dataset_roots[0]) / "meta/info.json"
        info = json.loads(path.read_text())
        info["fps"] = 5
        path.write_text(json.dumps(info))
    with pytest.raises((ValueError, FileNotFoundError), match="not mounted|distinct datasets|30 Hz"):
        factory.create(preset.assets_dirs, preset.model)
