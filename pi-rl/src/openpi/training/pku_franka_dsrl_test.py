# ruff: noqa: PT009, PT027
import dataclasses
import unittest

import numpy as np

from openpi.training import pku_franka_dsrl as dsrl


class _Policy:
    def __init__(self):
        self.calls = 0

    def infer(self, observation, *, noise):
        assert noise.shape == (30, 32)
        assert "observation.state.joints" in observation
        self.calls += 1
        return {"actions": np.tile([0.4, 0, 0.2, 3.1, 0, 0, 1], (30, 1))}


class _Agent:
    def __init__(self):
        self.updates = []

    def sample_actions(self, observation):
        assert observation["pixels"].shape == (1, 64, 64, 6, 1)
        assert observation["state"].shape == (1, 14, 1)
        return np.zeros((1, 32), dtype=np.float32)

    def update(self, batch):
        self.updates.append(batch)


class DsrlTest(unittest.TestCase):
    def test_noise_to_absolute_action_and_replay_update_loop(self):
        config = dsrl.DsrlConfig(
            batch_size=2, start_online_updates=2, random_queries=0, utd_ratio=1, max_updates=3, max_episodes=2
        )
        env, policy, agent = dsrl.FakeCollectEnv(12), _Policy(), _Agent()
        result = dsrl.run_training(env, policy, agent, config)
        self.assertEqual(result["updates"], 3)
        self.assertEqual(len(agent.updates), 3)
        self.assertEqual(policy.calls, result["queries"])
        batch = agent.updates[0]
        self.assertEqual(batch["actions"].shape, (2, 1, 32))
        self.assertEqual(batch["observations"]["state"].shape, (2, 14, 1))
        self.assertEqual(batch["next_observations"]["pixels"].shape, (2, 64, 64, 6, 1))
        np.testing.assert_allclose(batch["discount"], config.discount**6)

    def test_termination_absorbs_potential_but_truncation_bootstraps(self):
        config = dsrl.DsrlConfig(discount=0.9)
        result = {"rewards": [0, 1], "terminated": True, "truncated": False}
        reward, mask, count = dsrl.transition_reward({"progress": 0.2}, {"progress": 0.9}, result, config)
        self.assertAlmostEqual(reward, 0.9 - 0.2)
        self.assertEqual((mask, count), (0, 2))
        result.update(terminated=False, truncated=True)
        reward, mask, _ = dsrl.transition_reward({"progress": 0.2}, {"progress": 0.9}, result, config)
        self.assertAlmostEqual(reward, 0.9 + 0.9**2 * 0.9 - 0.2)
        self.assertEqual(mask, 1)

    def test_explicit_next_observation_survives_episode_reset(self):
        replay = dsrl.TransitionReplay(4, seed=0)
        replay.items.append(
            {
                "observations": {"state": np.array([1])},
                "next_observations": {"state": np.array([9])},
                "masks": np.float32(0),
            }
        )
        batch = replay.sample(3)
        np.testing.assert_array_equal(batch["next_observations"]["state"], np.full((3, 1), 9))

    def test_sparse_online_mode_and_invalid_config(self):
        result = {"rewards": [0, 1], "terminated": True, "truncated": False}
        config = dataclasses.replace(dsrl.DsrlConfig(), shaping_weight=0)
        self.assertEqual(dsrl.transition_reward({}, {}, result, config)[1], 0)
        with self.assertRaises(ValueError):
            dsrl.DsrlConfig(execute_steps=31)


if __name__ == "__main__":
    unittest.main()
