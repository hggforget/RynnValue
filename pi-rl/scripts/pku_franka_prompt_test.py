"""FR3 prompt semantics, episode leakage, sampling and real LeRobot integration."""

import dataclasses
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from openpi.policies import pku_franka_language as language
from openpi.training import config
from openpi.training import data_loader
from openpi.training import pku_franka_prompt_data as metadata

from . import compute_norm_stats_fast as norms
from .train_pku_franka_pickplace_collect_sft_test import local_repos as _local_repos_fixture

local_repos = _local_repos_fixture

CONFIG = "pi05_pku_franka_pickplace_prompt_aug"
BASELINE = "pi05_pku_franka_pickplace_prompt_baseline"


def instruction(**kwargs):
    return dataclasses.replace(
        language.EpisodeInstruction(
            repo_id="repo",
            episode_index=0,
            length=4,
            source_dataset="source",
            source_episode_index=0,
            date="2026-11-03",
            note="",
            split="train",
            fine_task={
                "kind": "pick_place",
                "object_color": "blue",
                "object_shape": "cylinder",
                "target_kind": "plate",
                "target_color": "green",
                "scene": "with_distractors",
            },
            original_prompt="Put the blue cylindrical block on the green plate.",
        ),
        **kwargs,
    )


def test_all_eight_plate_masks():
    record = instruction()
    prompts = {language.prompt_for_mask(record, mask) for mask in itertools.product((False, True), repeat=3)}
    assert prompts == {
        "Put a block on a plate.",
        "Put a blue block on a plate.",
        "Put a cylindrical block on a plate.",
        "Put a block on a green plate.",
        "Put a blue cylindrical block on a plate.",
        "Put a blue block on a green plate.",
        "Put a cylindrical block on a green plate.",
        record.original_prompt,
    }


def test_drawer_masks_never_become_plate():
    record = instruction(
        fine_task={"target_kind": "white_drawer", "object_color": "blue", "object_shape": "cylinder"},
        original_prompt="Put the blue cylindrical block in the white drawer.",
    )
    prompts = {language.prompt_for_mask(record, mask) for mask in itertools.product((False, True), repeat=2)}
    assert len(prompts) == 4
    assert all("drawer" in prompt and "plate" not in prompt for prompt in prompts)
    assert "Put a block in a drawer." in prompts
    assert record.original_prompt in prompts


@pytest.mark.parametrize("shape", language.SHAPES)
def test_shapes_and_colors_come_from_labels(shape):
    record = instruction(
        original_prompt="",
        fine_task={
            "target_kind": "plate",
            "object_color": "yellow",
            "object_shape": shape,
            "target_color": "red",
        },
    )
    text = language.prompt_for_mask(record, (True, True, True))
    assert text == f"Put the yellow {language.SHAPES[shape]} block on the red plate."
    assert language.SHAPES["rectangular"] != language.SHAPES["flat_rectangular"]


@pytest.mark.parametrize("drawer", [False, True])
def test_sampling_distribution_and_preserved_actions(drawer):
    record = instruction()
    if drawer:
        record = instruction(
            fine_task={**record.fine_task, "target_kind": "white_drawer"},
            original_prompt="Put the blue cylindrical block in the white drawer.",
        )
    transform = language.PkuFrankaPromptAugmentation(({0: record},))
    n = record.num_attributes
    probabilities = transform.drawer_probabilities if drawer else transform.plate_probabilities
    masks = list(itertools.product((False, True), repeat=n))
    expected = {
        language.prompt_for_mask(record, mask): probabilities[sum(mask)]
        / len(list(itertools.combinations(range(n), sum(mask))))
        for mask in masks
    }
    actions = np.arange(14).reshape(2, 7)
    sample = {"dataset_index": np.int64(0), "episode_index": np.int64(0), "actions": actions}
    state = np.random.get_state()
    try:
        np.random.seed(15)
        prompts = [transform(sample)["prompt"] for _ in range(5000)]
    finally:
        np.random.set_state(state)
    assert set(prompts) == set(expected)
    for prompt, probability in expected.items():
        assert prompts.count(prompt) / len(prompts) == pytest.approx(probability, abs=0.025)
    result = transform(sample)
    assert result["actions"] is actions
    assert "episode_index" not in result
    assert "dataset_index" not in result
    assert sample["episode_index"] == 0
    assert transform({"prompt": "custom inference instruction"}) == {"prompt": "custom inference instruction"}


@pytest.mark.parametrize(
    "probabilities", [(1, 0), (-0.1, 0.4, 0.3, 0.4), (0.5, 0.5, 0.5, 0.5), (float("nan"), 0, 0, 1)]
)
def test_invalid_sampling_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        language.PkuFrankaPromptAugmentation((), plate_probabilities=probabilities)


def write_metadata(root, count=20):
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    rows = []
    for i in range(count):
        record = instruction()
        rows.append(
            {
                "episode_index": i,
                "length": 4 + i,
                "prompt": record.original_prompt,
                "fine_task": record.fine_task,
                "source_dataset": "2026-11-03/pick-blue-cylinder-to-green-plate",
                "source_episode_index": i,
                "note": "review" if i >= count - 2 else "",
            }
        )
    rows.append({"episode_index": count, "length": 4, "fine_task": {"kind": "drawer_open"}})
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    return root


def test_split_provenance_quality_and_duplicates(tmp_path):
    root = write_metadata(tmp_path / "one")
    settings = {"validation_fraction": 0.3, "split_seed": 42, "include_noted_episodes": False}
    records = metadata.episode_records((str(root),), **settings)
    assert len(records) == 20  # drawer_open omitted; each original appears only once.
    assert sum(r.split == "excluded_note" for r in records) == 2
    train = {r.source_episode_index for r in records if r.split == "train"}
    validation = {r.source_episode_index for r in records if r.split == "validation"}
    assert train
    assert validation
    assert train.isdisjoint(validation)
    assert metadata.episode_records((str(root),), **settings) == records
    assert all(r.date == "2026-11-03" and r.source_dataset and r.fine_task["scene"] for r in records)
    path = tmp_path / "manifest.json"
    metadata.save_manifest(path, records, **settings)
    saved = json.loads(path.read_text())
    assert len(saved["episodes"]) == 20
    assert saved["episodes"][-1]["note"] == "review"
    included = metadata.episode_records((str(root),), **{**settings, "include_noted_episodes": True})
    assert all(r.split in {"train", "validation"} for r in included)
    other = write_metadata(tmp_path / "two")
    with pytest.raises(ValueError, match="Duplicate source"):
        metadata.episode_records((str(root), str(other)), **settings)
    with pytest.raises(ValueError, match="overlap"):
        metadata.discover_repos((str(root), str(root)))


def test_conflicting_full_prompt_is_rejected(tmp_path):
    root = write_metadata(tmp_path / "conflict")
    path = root / "meta/episodes.jsonl"
    rows = list(map(json.loads, path.read_text().splitlines()))
    rows[0]["prompt"] = rows[0]["prompt"].replace("blue", "red")
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="disagrees with fine_task"):
        metadata.episode_records((str(root),), validation_fraction=0.3, split_seed=42, include_noted_episodes=False)


def test_episode_balancing_keeps_repo_local_identities():
    class Raw:
        def __init__(self, ids):
            self.hf_dataset = {"episode_index": ids}

        def __len__(self):
            return len(self.hf_dataset["episode_index"])

    class Multi:
        def __init__(self):
            self._datasets = [Raw([0, 1, 1, 1]), Raw([0, 0])]

        def __len__(self):
            return sum(map(len, self._datasets))

    weights = data_loader.episode_balanced_weights(data_loader.TransformedDataset(Multi(), []))
    np.testing.assert_allclose(weights, [1, 1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    assert weights[:1].sum() == pytest.approx(weights[1:4].sum())
    assert weights[:1].sum() == pytest.approx(weights[4:].sum())


@pytest.fixture
def fine_repos(local_repos):
    _, repos, _ = local_repos
    for index, repo in enumerate(repos):
        root = Path(repo)
        rows = list(map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()))
        record = instruction()
        fine = record.fine_task if index == 0 else {**record.fine_task, "target_kind": "white_drawer"}
        prompt = record.original_prompt if index == 0 else "Put the blue cylindrical block in the white drawer."
        for row in rows:
            row.update(fine_task=fine, prompt=prompt, tasks=[prompt], source_dataset=f"source-{index}", note="")
        if index == 1:
            rows[0]["fine_task"] = {"kind": "drawer_open"}
            rows[2]["note"] = "needs review"
        (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        tasks = list(map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines()))
        for task in tasks:
            task["task"] = prompt
        (root / "meta/tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)) + "\n")
    return repos


def test_config_and_lerobot_pipeline(fine_repos, tmp_path):
    preset = config.get_config(CONFIG)
    model = dataclasses.replace(preset.model, action_horizon=2)
    factory = dataclasses.replace(
        preset.data,
        repo_ids=fine_repos,
        validation_fraction=0.5,
        plate_probabilities=(1, 0, 0, 0),
        drawer_probabilities=(1, 0, 0),
    )
    train = factory.create(tmp_path / "assets", model)
    val = dataclasses.replace(factory, split="validation").create(tmp_path / "assets", model)
    train_keys = {(repo, e) for repo, eps in train.episode_ids.items() for e in eps}
    val_keys = {(repo, e) for repo, eps in val.episode_ids.items() for e in eps}
    assert train_keys
    assert val_keys
    assert train_keys.isdisjoint(val_keys)
    assert train_keys | val_keys == {(fine_repos[0], 0), (fine_repos[0], 1), (fine_repos[1], 1), (fine_repos[1], 3)}
    assert train.episode_balanced_sampling
    assert not val.episode_balanced_sampling
    raw = data_loader.make_torch_dataset(train, 2, model)
    assert len(raw) == 4 * len(train_keys)
    weights = data_loader.episode_balanced_weights(raw)
    np.testing.assert_allclose(weights, 0.25)
    sample = raw[0]
    augmented = data_loader.transform_dataset(raw, train, skip_norm_stats=True)[0]
    np.testing.assert_array_equal(augmented["actions"][:, :7], sample["action"])
    assert augmented["state"].shape == (model.action_dim,)
    # The fixture's byte tokenizer returns int32 tokens, so decode their byte values.
    text = bytes(augmented["tokenized_prompt"][augmented["tokenized_prompt_mask"]].tolist()).decode()
    assert text in {"Put a block on a plate.", "Put a block in a drawer."}
    loader = data_loader.create_torch_data_loader(
        train,
        model,
        action_horizon=2,
        batch_size=2,
        num_workers=0,
        num_batches=1,
        seed=42,
        skip_norm_stats=True,
        framework="pytorch",
    )
    observations, actions = next(iter(loader))
    assert observations.state.shape == (2, model.action_dim)
    assert actions.shape == (2, 2, model.action_dim)
    stats_loader, batches = norms.create_torch_dataloader(train, 2, 1, model, 0, skip_video=True)
    assert batches == len(raw)
    assert len(list(stats_loader)) == len(raw)
    # Validation is deterministic full-prompt evaluation even when train augmentation is enabled.
    val_raw = data_loader.make_torch_dataset(val, 2, model)
    item = val_raw[0]
    mapped = val.data_transforms.inputs[0](item)
    assert val.training_transforms.inputs[0](mapped)["prompt"] == item["prompt"]
    inference = dataclasses.replace(factory, repo_ids=("/unmounted",)).create_for_inference(tmp_path, model)
    assert not inference.training_transforms.inputs
    assert not inference.repo_ids


def test_baseline_has_same_budget_and_full_prompts():
    augmented, baseline = (config.get_config(name) for name in (CONFIG, BASELINE))
    assert augmented.model == baseline.model
    assert augmented.lr_schedule == baseline.lr_schedule
    assert augmented.num_train_steps == baseline.num_train_steps == 20_000
    assert dataclasses.replace(augmented.data, augment_prompts=False) == baseline.data
    record = instruction()
    transform = language.PkuFrankaPromptAugmentation(({0: record},), augment=False)
    assert (
        transform({"actions": np.zeros((2, 7)), "dataset_index": 0, "episode_index": 0})["prompt"]
        == record.original_prompt
    )


def test_launcher_uses_same_sources_and_split_for_stats_and_training(fine_repos, tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "GPUS": "0",
        "DRY_RUN": "1",
        "PICKPLACE_REPO_ID": fine_repos[0],
        "PLATE_ROOT": fine_repos[1],
        "DRAWER_ROOT": fine_repos[1],
        "WEIGHT_PARAMS_PATH": str(weights),
        "VALIDATION_FRACTION": "0.2",
        "SPLIT_SEED": "7",
        "INCLUDE_NOTED_EPISODES": "1",
    }
    script = Path(__file__).with_name("train_sft_pku_franka_pickplace_prompt_aug.sh")
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines() if "scripts/" in line]
    norm, train = commands
    assert (
        norm[norm.index("--validation-fraction") + 1] == train[train.index("--data.validation-fraction") + 1] == "0.2"
    )
    assert norm[norm.index("--split-seed") + 1] == train[train.index("--data.split-seed") + 1] == "7"
    assert (
        norm[norm.index("--repo-ids") + 1 : norm.index("--repo-ids") + 4]
        == train[train.index("--data.repo-ids") + 1 : train.index("--data.repo-ids") + 4]
    )
    assert "--include-noted-episodes" in norm
    assert "--data.include-noted-episodes" in train

    def parsed_norm(factory, *args):
        del args
        assert factory.validation_fraction == 0.2
        assert factory.split_seed == 7
        assert factory.include_noted_episodes
        assert factory.repo_ids == (fine_repos[0], fine_repos[1], fine_repos[1])
        raise RuntimeError("norm CLI parsed")

    monkeypatch.setattr(config.LeRobotPkuFrankaPromptDataConfig, "create", parsed_norm)
    with pytest.raises(RuntimeError, match="norm CLI parsed"):
        norms.tyro.cli(norms.main, args=norm[norm.index("scripts/compute_norm_stats_fast.py") + 1 :])
    parsed = config.tyro.extras.overridable_config_cli(
        {CONFIG: (CONFIG, config.get_config(CONFIG))}, args=train[train.index("scripts/train.py") + 1 :]
    )
    assert parsed.data.validation_fraction == 0.2
    assert parsed.data.split_seed == 7
    assert parsed.data.include_noted_episodes
