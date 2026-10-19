# ruff: noqa: PT009, PT027
import ast
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest

import numpy as np

from openpi.policies import pku_franka_collect_policy as collect
from openpi.policies import pku_franka_collect_rl_policy as rl


def _transition(frame=0):
    return {
        "observation.images.primary": np.zeros((2, 3, 8, 12), dtype=np.float32),
        "observation.images.wrist": np.ones((2, 8, 12, 3), dtype=np.uint8),
        "observation.state.cartesian": np.array([[0.4, 0, 0.2, 3.1, 0, 0], [0.5, 0, 0.2, 3.1, 0, 0]]),
        "observation.state.gripper": np.array([1.0, 0.0]),
        "observation.state.joints": np.zeros((2, 7)),
        "observation.state.target": np.full((2, 6), np.nan),
        "action": np.zeros((3, 7)),
        "progress": np.array([0.1, 0.2, 0.3, 0.4]),
        "episode_index": np.int64(0),
        "frame_index": np.int64(frame),
        "prompt": collect.COLLECT_OBJECTS_PROMPT,
    }


class CollectRLTest(unittest.TestCase):
    def test_nonterminal_transition_and_discounted_potential(self):
        transformed = rl.PkuFrankaCollectRLInputs({0: 10}, discount=0.9)(_transition())
        self.assertEqual(transformed["state"].shape, (14,))
        self.assertEqual(transformed["next_state"].shape, (14,))
        self.assertEqual(transformed["image"]["base_0_rgb"].shape, (8, 12, 3))
        self.assertEqual(transformed["next_image_wrist"].shape, (224, 224, 3))
        self.assertEqual(transformed["mask"], 1)
        self.assertAlmostEqual(float(transformed["chunk_reward"]), 0.9**3 * 0.4 - 0.1, places=6)
        self.assertTrue(np.isfinite(transformed["state"]).all())

    def test_exact_terminal_and_padded_terminal_absorb_potential(self):
        transform = rl.PkuFrankaCollectRLInputs({0: 10}, discount=0.9)
        for frame in [6, 7, 9]:
            with self.subTest(frame=frame):
                transformed = transform(_transition(frame))
                self.assertEqual(transformed["mask"], 0)
                self.assertAlmostEqual(float(transformed["chunk_reward"]), -0.1, places=6)

    def test_failed_episode_retained_with_terminal_penalty(self):
        transform = rl.PkuFrankaCollectRLInputs({0: 10}, failed_episode_ids=(0,), discount=0.9)
        self.assertAlmostEqual(float(transform(_transition(7))["chunk_reward"]), -0.1 - 0.9, places=6)
        self.assertEqual(transform(_transition())["mask"], 1)

    def test_progress_is_required_and_finite(self):
        item = _transition()
        item["progress"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite progress"):
            rl.PkuFrankaCollectRLInputs({0: 10})(item)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({"fps": 30, "features": {}}))
            with self.assertRaisesRegex(ValueError, "progress"):
                rl.collection_metadata(directory)

    def test_rl_checkpoint_inference_does_not_require_training_labels(self):
        item = _transition()
        for key in list(item):
            if key.startswith("observation.") and key != "observation.state.target":
                item[key] = np.asarray(item[key])[0]
        del item["action"]
        del item["progress"]
        output = rl.PkuFrankaCollectRLInputs()(item)
        self.assertEqual(output["state"].shape, (14,))
        self.assertNotIn("actions", output)
        self.assertEqual(rl.collection_metadata(None), ({"fps": 30}, {}))

    def test_window_scores_match_brute_force_and_episode_tail(self):
        progress = np.array([0, 0.2, 0.1, 0.5, 0.4])
        for discount in [0.98, 1.0]:
            expected = [
                sum(discount**k * (progress[t + k + 1] - progress[t + k]) for k in range(min(3, len(progress) - 1 - t)))
                for t in range(len(progress))
            ]
            np.testing.assert_allclose(rl.window_advantages(progress, 3, discount), expected, atol=1e-10)

    def test_trainer_uses_chunk_reward_once_and_preserves_legacy_path(self):
        path = pathlib.Path(__file__).resolve().parents[3] / "scripts/train_iql.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_chunk_reward")
        namespace = {"np": np, "jax": types.SimpleNamespace(device_get=lambda value: value)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        fn = namespace["_chunk_reward"]
        np.testing.assert_array_equal(fn({"chunk_reward": [-2.0], "progress": [[10, 20]]}, 0.9), [-2.0])
        np.testing.assert_allclose(fn({"progress": [[0.1, 0.2, 0.4]]}, 0.9), [0.9**2 * 0.4 - 0.1], atol=1e-7)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "PyArrow is available on the dataset host")
    def test_filter_maps_global_indices_and_excludes_failed_episodes(self):
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "meta/info.json").write_text(json.dumps({"fps": 30, "features": {"progress": {}}}))
            (root / "meta/episodes.jsonl").write_text(
                "\n".join(json.dumps({"episode_index": i, "length": 5}) for i in [0, 1])
            )
            for episode in [0, 1]:
                pq.write_table(
                    pa.table(
                        {
                            "index": np.arange(5) + 100 * episode,
                            "episode_index": [episode] * 5,
                            "progress": [0, 0.2, 0.1, 0.4, 0.3],
                        }
                    ),
                    root / f"data/chunk-000/episode_{episode:06d}.parquet",
                )
            kept, threshold = rl.filtered_frame_indices(directory, (0,), 1, 1.0, "zero", 0.7)
            self.assertEqual(kept, (100, 102))
            self.assertEqual(threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
