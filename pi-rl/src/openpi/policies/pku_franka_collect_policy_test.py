# Keep these runnable with NumPy and the standard library on the collection host.
# ruff: noqa: PT009, PT027

import json
import pathlib
import tempfile
import unittest

import numpy as np

from openpi.policies import pku_franka_collect_policy as policy


def _sample():
    return {
        "observation.images.primary": np.zeros((8, 12, 3), dtype=np.uint8),
        "observation.images.wrist": np.ones((3, 8, 12), dtype=np.float32),
        "observation.state.cartesian": np.array([0.4, -0.1, 0.2, 3.1, 0, -0.2], dtype=np.float32),
        "observation.state.gripper": np.float32(0.75),
        "observation.state.joints": np.arange(7, dtype=np.float32),
        "observation.state": np.full(20, np.nan),
        "observation.state.target": np.full(6, np.nan),
        "action": np.tile([0.5, -0.2, 0.3, -3.1, 0.1, 0.2, 1.0], (30, 1)).astype(np.float32),
        "action.relative": np.full((30, 7), np.nan),
        "prompt": b"lace each object on the plate that matches its color.",
    }


class CollectPolicyTest(unittest.TestCase):
    def test_measured_state_excludes_target_and_combined_state(self):
        sample = _sample()
        inputs = policy.PkuFrankaCollectInputs()(sample)
        expected = np.concatenate([sample["observation.state.cartesian"], [0.75], sample["observation.state.joints"]])
        np.testing.assert_array_equal(inputs["state"], expected)
        self.assertEqual(inputs["state"].shape, (policy.STATE_DIM,))
        self.assertTrue(np.isfinite(inputs["state"]).all())

    def test_absolute_actions_and_gripper_are_preserved(self):
        sample = _sample()
        sample["action"][0, -1] = 0
        inputs = policy.PkuFrankaCollectInputs()(sample)
        np.testing.assert_array_equal(inputs["actions"], sample["action"])
        padded = np.pad(inputs["actions"], ((0, 0), (0, 32 - policy.ACTION_DIM)))
        outputs = policy.PkuFrankaCollectOutputs()({"actions": padded})
        np.testing.assert_array_equal(outputs["actions"], sample["action"])

    def test_two_cameras_and_prompt_typo(self):
        inputs = policy.PkuFrankaCollectInputs()(_sample())
        self.assertEqual(set(inputs["image"]), {"base_0_rgb", "left_wrist_0_rgb"})
        self.assertTrue(all(inputs["image_mask"].values()))
        for image in inputs["image"].values():
            self.assertEqual(image.shape, (8, 12, 3))
            self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(inputs["image"]["left_wrist_0_rgb"], np.full((8, 12, 3), 255))
        self.assertEqual(inputs["prompt"], policy.COLLECT_OBJECTS_PROMPT)

    def test_inference_needs_no_action_or_recorded_target(self):
        sample = _sample()
        for key in ["action", "action.relative", "observation.state", "observation.state.target"]:
            del sample[key]
        sample["observation.state.gripper"] = np.array([0.75])
        sample["prompt"] = "a different instruction"
        inputs = policy.PkuFrankaCollectInputs()(sample)
        self.assertNotIn("actions", inputs)
        self.assertEqual(inputs["prompt"], "a different instruction")
        sample["prompt"] = ""
        self.assertNotIn("prompt", policy.PkuFrankaCollectInputs()(sample))

    def test_invalid_state_action_and_image_shapes(self):
        for key, value, message in [
            ("observation.state.joints", np.zeros(14), "measured Cartesian"),
            ("action", np.zeros((30, 8)), "action chunk"),
            ("observation.images.primary", np.zeros((8, 12)), "RGB image"),
        ]:
            with self.subTest(key=key):
                sample = _sample()
                sample[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    policy.PkuFrankaCollectInputs()(sample)

    def test_episode_exclusion_keeps_later_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            meta = pathlib.Path(directory) / "meta"
            meta.mkdir()
            episodes_path = meta / "episodes.jsonl"
            episodes_path.write_text("\n".join(json.dumps({"episode_index": i}) for i in [0, 1, 2, 64]))
            self.assertEqual(policy.training_episode_ids(directory, (0, 2)), (1, 64))
            self.assertIsNone(policy.training_episode_ids(directory, ()))
            with self.assertRaisesRegex(ValueError, "No FR3 training episodes"):
                policy.training_episode_ids(directory, (0, 1, 2, 64))

    def test_checkpoint_serving_does_not_require_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(policy.training_episode_ids(directory, (0, 2)))


if __name__ == "__main__":
    unittest.main()
