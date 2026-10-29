import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.policies import lingdong_policy as policy
from openpi.shared import normalize

from . import compute_lingdong_norm_stats as norms


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "datasets"
    repo = root / policy.DATASET_NAMES[0]
    meta = repo / "meta"
    meta.mkdir(parents=True)
    names = [f"unused_{i}" for i in range(18)]
    for index, name in zip(policy.RIGHT_INDICES, policy.RIGHT_NAMES, strict=True):
        names[index] = name
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 21,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {"shape": [18], "names": names},
            "action": {"shape": [18], "names": names},
            **{key: {"dtype": "video"} for key in policy.VIDEO_KEYS},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 21}) + "\n")
    state = np.arange(21 * 18, dtype=np.float32).reshape(21, 18) / 100
    state[:, 17] = np.repeat([0.4, 0.43, 0.63, 0.94, 0.91, 0.71, 0.4], 3)
    action = state + np.float32(0.1)
    action[:, 17] = state[:, 17]
    path = repo / "data/chunk-000/episode_000000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"observation.state": state.tolist(), "action": action.tolist()}), path)
    return root, state, action


def test_norms_use_binary_actions_and_continuous_states(dataset, tmp_path):
    root, state, action = dataset
    output = tmp_path / "assets"
    norms.compute(root, output, workers=1)
    norms.compute(root, output, check=True)
    saved = normalize.load(output)
    expected_action = action[:, list(policy.RIGHT_INDICES)].copy()
    expected_action[:, -1] = np.r_[np.zeros(2), np.ones(9), np.zeros(10)]
    for key, expected in (("state", state[:, list(policy.RIGHT_INDICES)]), ("actions", expected_action)):
        np.testing.assert_allclose(saved[key].mean, np.mean(expected, axis=0, dtype=np.float64))
        np.testing.assert_allclose(saved[key].std, np.std(expected, axis=0, dtype=np.float64))
        np.testing.assert_allclose(saved[key].q01, np.quantile(expected, 0.01, axis=0))
        np.testing.assert_allclose(saved[key].q99, np.quantile(expected, 0.99, axis=0))
    assert saved["actions"].q01[-1] == 0
    assert saved["actions"].q99[-1] == 1
    assert saved["state"].q99[-1] < 1


def test_training_uses_same_episode_labels_for_overlapping_and_terminal_chunks(dataset):
    root, _, raw = dataset
    repo = root / policy.DATASET_NAMES[0]
    expected = np.r_[np.zeros(2), np.ones(9), np.zeros(10)]
    training = policy.LingdongTrainingActions(action_horizon=5, repo_ids=(str(repo),))
    selected = raw[:, list(policy.RIGHT_INDICES)]
    for start in range(len(raw)):
        indices = np.minimum(start + np.arange(5), len(raw) - 1)
        chunk = selected[indices].copy()
        result = training({"actions": chunk, "dataset_index": 0, "episode_index": 0, "frame_index": start})
        np.testing.assert_array_equal(result["actions"][:, -1], expected[indices])
        np.testing.assert_array_equal(result["actions"][:, :7], chunk[:, :7])
        np.testing.assert_array_equal(chunk, selected[indices])
        assert "frame_index" not in result
    assert training({"actions": selected[12:17], "episode_index": 0, "frame_index": 12})["actions"][0, -1] == 0


def test_pre_binary_norm_manifest_is_rejected(dataset, tmp_path):
    root, _, _ = dataset
    output = tmp_path / "assets"
    norms.compute(root, output, workers=1)
    manifest, _ = norms.dataset_manifest(root)
    old = {key: value for key, value in manifest.items() if key not in ("action_gripper", "signature")}
    old["version"] = 1
    old["signature"] = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    (output / "norm_stats_manifest.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="preprocessing changed"):
        norms.compute(root, output, check=True)
