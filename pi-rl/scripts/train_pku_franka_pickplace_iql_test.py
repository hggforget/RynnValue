"""Integration checks using the mounted, expert-labeled pick-and-place dataset."""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.training import config
from openpi.training import rl_data_loader

from . import train_iql


@pytest.fixture(scope="module")
def pickplace():
    preset = config.get_config("pi05_pku_franka_pickplace_iql")
    root = Path(preset.data.repo_id)
    if not (root / "meta/info.json").is_file():
        pytest.skip("Expert-labeled FR3 pick-and-place is not mounted")
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    data = preset.data.create(preset.assets_dirs, preset.model)
    dataset = rl_data_loader.create_rl_dataset(data, preset.model.action_horizon, preset.model, skip_norm_stats=True)
    return preset, data, dataset, info, episodes


@pytest.mark.manual
def test_real_video_samples_match_progress_actions_tasks_and_terminal_boundaries(pickplace):
    preset, data, dataset, info, episodes = pickplace
    assert len(dataset) == sum(row["length"] for row in episodes) == info["total_frames"]
    assert data.episode_ids is None
    tokenizer = next(t.tokenizer for t in data.model_transforms.inputs if isinstance(t, transforms.TokenizePrompt))
    horizon = preset.model.action_horizon
    offset = 0
    for episode in episodes:
        eid, length = episode["episode_index"], episode["length"]
        if eid in (0, len(episodes) // 2, len(episodes) - 1):
            table = pq.read_table(
                Path(data.repo_id)
                / info["data_path"].format(
                    episode_chunk=eid // info["chunks_size"],
                    episode_index=eid,
                )
            )
            progress = table["progress"].to_numpy()
            for frame in (0, length - horizon - 1, length - 2, length - 1):
                sample = dataset[offset + frame]
                indices = np.minimum(frame + np.arange(horizon), length - 1)
                actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[indices]
                np.testing.assert_array_equal(sample["actions"][:, :7], actions)
                assert not sample["actions"][:, 7:].any()
                steps = min(horizon, length - 1 - frame)
                expected = sum(
                    preset.iql.discount**k * (progress[frame + k + 1] - progress[frame + k]) for k in range(steps)
                )
                np.testing.assert_allclose(sample["chunk_reward"], expected, rtol=1e-5, atol=1e-6)
                assert sample["mask"] == float(frame + horizon < length - 1)
                assert sample["iql_state"].shape == (14 + info["total_tasks"],)
                task = int(table["task_index"][frame].as_py())
                np.testing.assert_array_equal(sample["iql_state"][14:], np.eye(info["total_tasks"])[task])
                np.testing.assert_array_equal(sample["next_iql_state"][14:], sample["iql_state"][14:])
                for key in ("next_image_base", "next_image_wrist"):
                    assert sample[key].shape == (224, 224, 3)
                    assert sample[key].dtype == np.uint8
                text = tokenizer._tokenizer.decode(  # noqa: SLF001
                    sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist()
                )
                assert episode["tasks"][0] in text
                assert "Action:" in text
        offset += length


@pytest.mark.manual
def test_real_batch_updates_task_conditioned_critic_and_restores_state(pickplace):
    if jax.default_backend() != "cpu":
        pytest.skip("Run this small-encoder smoke test with JAX_PLATFORMS=cpu")
    preset, _, dataset, _, _ = pickplace
    samples = [dataset[0], dataset[len(dataset) - 1]]
    batch = jax.tree.map(lambda *values: np.stack(values), *samples)
    # Keep the 30x7 action chunk and 58-D conditioning; a small image/encoder
    # makes optimizer, advantage and restore checks practical on CPU.
    batch["image"] = {key: value[:, ::7, ::7] for key, value in batch["image"].items()}
    for key in ("next_image_base", "next_image_wrist"):
        batch[key] = batch[key][:, ::7, ::7]
    preset = dataclasses.replace(
        preset,
        iql=dataclasses.replace(
            preset.iql,
            encoder_type="small",
            hidden_dims=(16, 16),
            cnn_features=(8, 8),
            cnn_strides=(2, 2),
            latent_dim=8,
        ),
    )
    keys = {
        "image_keys": preset.iql.image_keys,
        "next_image_keys": preset.iql.next_image_keys,
        "proprio_key": preset.iql.proprio_key,
        "next_proprio_key": preset.iql.next_proprio_key,
        "action_dim": preset.iql.action_dim,
    }
    observations, actions = train_iql.host_sample_for_iql_init(
        batch,
        image_keys=keys["image_keys"],
        proprio_key=keys["proprio_key"],
        action_dim=keys["action_dim"],
    )
    agent = train_iql.init_pi_iql(preset, observations, actions)
    obs, actions, view = train_iql.split_batch(batch, **keys)
    assert obs.state.shape == (2, 32)
    assert actions.shape == (2, 30, 32)
    assert view["actions"].shape == (2, 30, 7)
    assert view["observations"]["proprio"].shape == (2, 58)
    assert view["observations"]["pixels"].dtype == np.uint8
    before = agent.iql_save_pytree()
    metrics = agent.update(view)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(metrics))
    assert agent.bellman_discount == pytest.approx(preset.iql.discount**30)
    _, weights, _, qs, values, advantages = agent.compute_advs(view)
    assert weights.shape == values.shape == advantages.shape == (2,)
    assert qs.shape == (2, 2)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert (weights <= 100).all()
    after = agent.iql_save_pytree()
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True))
    agent.iql_load_pytree(before)
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(agent.iql_save_pytree()), strict=True):
        np.testing.assert_array_equal(expected, actual)
