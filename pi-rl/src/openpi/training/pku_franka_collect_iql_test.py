"""Integration checks for the mounted expert-labeled FR3 dataset."""

import dataclasses
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import rl_data_loader


@pytest.fixture(scope="module", params=["absolute", "relative"])
def expert_dataset(request):
    suffix = "_relative" if request.param == "relative" else ""
    config = training_config.get_config(f"pi05_pku_franka_collect_objects{suffix}_iql")
    root = pathlib.Path(config.data.repo_id)
    if not (root / "meta/info.json").is_file():
        pytest.skip("The expert-labeled FR3 dataset is not mounted")
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    data = config.data.create(config.assets_dirs, config.model)
    dataset = rl_data_loader.create_rl_dataset(data, config.model.action_horizon, config.model, skip_norm_stats=True)
    return config, data, dataset, info, episodes


@pytest.mark.manual
def test_sft_policy_tokens_use_the_same_new_instruction(expert_dataset):
    config, _, _, info, _ = expert_dataset
    sft = training_config.get_config(config.name.removesuffix("_iql"))
    factory = dataclasses.replace(sft.data, repo_id=config.data.repo_id)
    data = factory.create(sft.assets_dirs, sft.model)
    dataset = data_loader.make_torch_dataset(data, sft.model.action_horizon, sft.model)
    dataset = data_loader.transform_dataset(dataset, data, skip_norm_stats=True)
    sample = dataset[0]
    tokenizer = next(t.tokenizer for t in data.model_transforms.inputs if isinstance(t, transforms.TokenizePrompt))
    text = tokenizer._tokenizer.decode(sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist())  # noqa: SLF001
    assert info["progress_annotation"]["instruction_override"] in text
    assert "Action:" in text


@pytest.mark.manual
def test_iql_policy_tokens_match_the_expert_annotation_instruction(expert_dataset):
    config, data, dataset, info, episodes = expert_dataset
    assert len(dataset) == sum(ep["length"] for ep in episodes) == info["total_frames"]
    assert data.episode_ids is None  # Failed recordings remain available to IQL.
    sample = dataset[0]
    tokenizer = next(t.tokenizer for t in data.model_transforms.inputs if isinstance(t, transforms.TokenizePrompt))
    text = tokenizer._tokenizer.decode(sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist())  # noqa: SLF001
    assert info["progress_annotation"]["instruction_override"] in text
    assert "Leave all wooden-textured blocks on the table." in text
    assert "Action:" in text  # The longer instruction and state were not truncated.
    assert sample["state"].shape == (config.model.action_dim,)
    assert sample["actions"].shape == (config.model.action_horizon, config.model.action_dim)
    assert set(sample["image"]) == {"base_0_rgb", "left_wrist_0_rgb"}
    for key in ("next_image_base", "next_image_wrist"):
        assert sample[key].shape == (224, 224, 3)


@pytest.mark.manual
def test_iql_uses_recorded_actions_and_negative_seconds_potential(expert_dataset):
    config, data, dataset, info, _ = expert_dataset
    horizon = config.model.action_horizon
    path = pathlib.Path(data.repo_id) / info["data_path"].format(episode_chunk=0, episode_index=0)
    table = pq.read_table(path)
    sample = dataset[0]
    expected_actions = np.asarray(table[data.action_sequence_keys[0]].slice(0, horizon).to_pylist(), dtype=np.float32)
    np.testing.assert_array_equal(sample["actions"][:, :7], expected_actions)
    assert not sample["actions"][:, 7:].any()
    progress = table["progress"].to_numpy()
    assert progress[0] < 0
    expected_reward = config.iql.discount**horizon * progress[horizon] - progress[0]
    assert sample["mask"] == 1.0
    np.testing.assert_allclose(sample["chunk_reward"], expected_reward, rtol=1e-6)


@pytest.mark.manual
def test_iql_terminal_penalties_include_new_failures_and_preserve_recoveries(expert_dataset):
    config, data, dataset, info, episodes = expert_dataset
    offset = 0
    for episode in episodes:
        eid = episode["episode_index"]
        if eid in (71, 78, 83):
            sample = dataset[offset + episode["length"] - 1]
            path = pathlib.Path(data.repo_id) / info["data_path"].format(
                episode_chunk=eid // info["chunks_size"], episode_index=eid,
            )
            progress = pq.read_table(path, columns=["progress"])["progress"].to_numpy()
            # 71 records recovered grasp failures; 78 and 83 explicitly end in failure.
            penalty = config.data.failure_penalty if eid in (78, 83) else 0.0
            assert sample["mask"] == 0.0
            np.testing.assert_allclose(sample["chunk_reward"], -progress[-1] - penalty, rtol=1e-6)
        offset += episode["length"]
