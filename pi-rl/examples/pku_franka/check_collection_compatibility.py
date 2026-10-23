"""Compare planning with the deployed collection env.step(), without constructing ROS clients."""

import copy
from types import SimpleNamespace

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import make_env_config
from crisp_py.robot import Pose
import numpy as np
from openpi_client.franka_collect.action_execution import ExecutionLimits
from openpi_client.franka_collect.action_execution import plan_targets
from scipy.spatial.transform import Rotation


def reference_steps(config, seed, actions, relative):
    # Bypass __init__: all hardware interfaces below are in-memory stand-ins.
    env = object.__new__(ManipulatorCartesianEnv)
    env.config = copy.deepcopy(config)
    env.config.use_relative_actions = relative
    env.action_space = SimpleNamespace(shape=(7,))
    env.timestep = 0
    env.robot = SimpleNamespace(target_pose=Pose(seed[:3].copy(), Rotation.from_euler("xyz", seed[3:])))
    env.robot.set_target = lambda *, pose: setattr(env.robot, "target_pose", pose.copy())
    gripper = [None]
    env.gripper = SimpleNamespace(set_target=lambda value: gripper.__setitem__(0, float(value)))
    env._get_obs = dict  # noqa: SLF001 - replace only observation I/O, retain the deployed step().
    outputs = []
    for action in actions:
        env.step(np.asarray(action), block=False)
        outputs.append(np.r_[env.robot.target_pose.to_pos_euler_array(), gripper[0]])
    return np.asarray(outputs)


def assert_same(first, second):
    np.testing.assert_allclose(first[:, :3], second[:, :3], atol=1e-9)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", a[3:6]).as_matrix(),
            Rotation.from_euler("xyz", b[3:6]).as_matrix(),
            atol=1e-9,
        )
    np.testing.assert_allclose(first[:, 6], second[:, 6])


def main():
    config = make_env_config("fr3_dual_cam_640x480")
    seed = np.array([0.3, 0.1, 0.2, 0.25, -0.15, 0.4])
    limits = ExecutionLimits(max_step_m=1, max_step_rad=1, max_goal_m=1, max_goal_rad=2)
    for representation in ("absolute", "relative"):
        actions = np.array([[0.002, -0.001, 0.003, 0.01, -0.02, 0.015, 1.02], [0, 0, 0, -0.01, 0.02, -0.015, -0.1]])
        if representation == "absolute":
            actions[:, :6] += seed
        expected = reference_steps(config, seed, actions, representation == "relative")
        actual = plan_targets(actions, seed, representation, 2, limits, **config.safety_box)
        assert_same(actual, expected)
        print(f"PASS: {representation} Cartesian/Euler/gripper semantics match installed env.step()", flush=True)
    seed = np.array([0.3, 0.1, 0.2, 0, 0, 0])
    requested = np.array([[0.02, 0, 0, 0, 0, 0.1, 1], [0, 0, 0, 0, 0, 0, 1]])
    bounded = np.array([[0.005, 0, 0, 0, 0, 0.03, 1], [0, 0, 0, 0, 0, 0, 1]])
    expected = reference_steps(config, seed, bounded, relative=True)
    actual = plan_targets(requested, seed, "relative", 2, **config.safety_box)
    assert_same(actual, expected)
    print("PASS: bounded relative actions followed by zero hold the committed target", flush=True)
    seed[2] = 0.001
    actions = np.array([[0.3, 0.1, -0.001, 0, 0, 0, 1]])
    expected = reference_steps(config, seed, actions, relative=False)
    actual = plan_targets(actions, seed, "absolute", 1, **config.safety_box)
    assert_same(actual, expected)
    print("PASS: effective collection workspace matches, including legitimate negative Z", flush=True)
    print("No robot objects, ROS nodes or command publishers were constructed.", flush=True)


if __name__ == "__main__":
    main()
