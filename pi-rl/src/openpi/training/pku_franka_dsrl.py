"""FR3 DSRL: two RGB cameras + measured state -> SAC noise -> frozen VLA.

The environment factory must implement the absolute Cartesian contract below.
The legacy single-camera delta-action RobotClient is not compatible.
"""

import argparse
from collections import deque
import dataclasses
import importlib
import json
import pathlib
import sys

import numpy as np

from openpi.policies import pku_franka_collect_policy as collect


@dataclasses.dataclass(frozen=True)
class DsrlConfig:
    name: str = "pi05_pku_franka_collect_objects_dsrl"
    policy_config: str = "pi05_pku_franka_collect_objects"
    seed: int = 42
    horizon: int = 30
    execute_steps: int = 6
    control_hz: float = 30.0
    image_size: int = 64
    discount: float = 0.99
    shaping_weight: float = 1.0
    batch_size: int = 256
    start_online_updates: int = 500
    random_queries: int = 50
    utd_ratio: int = 20
    max_updates: int = 500_000
    max_episodes: int = 500
    max_episode_steps: int = 1_800
    buffer_capacity: int = 5_000

    def __post_init__(self):
        if not 1 <= self.execute_steps <= self.horizon or not 0 < self.discount <= 1:
            raise ValueError("Invalid DSRL action horizon, execute_steps, or discount")
        if min(self.batch_size, self.utd_ratio, self.max_episodes, self.max_updates, self.image_size) < 1:
            raise ValueError("DSRL sizes and budgets must be positive")
        if self.buffer_capacity < max(self.batch_size, self.start_online_updates):
            raise ValueError("Replay capacity must cover batch_size and start_online_updates")


def sac_observation(observation: dict, config: DsrlConfig) -> dict:
    transformed = collect.PkuFrankaCollectInputs()(observation)
    images = []
    for key in ["base_0_rgb", "left_wrist_0_rgb"]:
        image = transformed["image"][key]
        rows = np.linspace(0, image.shape[0] - 1, config.image_size).astype(int)
        cols = np.linspace(0, image.shape[1] - 1, config.image_size).astype(int)
        images.append(image[rows[:, None], cols])
    return {
        "pixels": np.concatenate(images, axis=-1)[..., None],
        "state": transformed["state"][:, None],
    }


class TransitionReplay:
    """Store actual next observations, including terminal/truncated transitions."""

    def __init__(self, capacity: int, seed: int):
        self.items = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def sample(self, batch_size: int) -> dict:
        rows = [self.items[int(i)] for i in self.rng.integers(len(self), size=batch_size)]
        batch = {}
        for key in rows[0]:
            if isinstance(rows[0][key], dict):
                batch[key] = {k: np.stack([row[key][k] for row in rows]) for k in rows[0][key]}
            else:
                batch[key] = np.stack([row[key] for row in rows]).astype(np.float32)
        return batch


def transition_reward(current: dict, following: dict, result: dict, config: DsrlConfig) -> tuple[float, float, int]:
    rewards = np.asarray(result["rewards"], dtype=np.float32).reshape(-1)
    if not len(rewards) or not np.isfinite(rewards).all():
        raise ValueError("Environment must return finite per-executed-step rewards")
    mask = float(not result["terminated"])
    reward = float(np.dot(config.discount ** np.arange(len(rewards)), rewards))
    if config.shaping_weight:
        potentials = np.asarray([current["progress"], following["progress"]], dtype=float)
        if not np.isfinite(potentials).all():
            raise ValueError("Online progress must be finite")
        reward += config.shaping_weight * (config.discount ** len(rewards) * mask * potentials[1] - potentials[0])
    return reward, mask, len(rewards)


def run_training(env, policy, agent, config: DsrlConfig, *, on_episode=None) -> dict:
    """Train the SAC noise actor; the supplied VLA only performs inference.

    env.reset() -> raw dataset-schema observation, optionally including progress.
    env.step_chunk([N,7], publish_hz=...) -> {observation, rewards[N], terminated,
    truncated}. Observations contain both RGB streams and all measured fields.
    Truncations bootstrap from the actual final observation; terminations do not.
    """
    replay = TransitionReplay(config.buffer_capacity, config.seed)
    rng = np.random.default_rng(config.seed)
    updates = queries = 0
    for episode in range(config.max_episodes):
        current = env.reset()
        episode_steps = 0
        episode_reward = 0.0
        while episode_steps < config.max_episode_steps:
            observation = sac_observation(current, config)
            if config.shaping_weight and "progress" not in current:
                raise ValueError("Supply online progress in the environment, or set shaping_weight=0 for sparse RL")
            if queries < config.random_queries:
                noise = rng.normal(0, 0.1, (1, 32)).astype(np.float32)
            else:
                batched = {k: v[None] for k, v in observation.items()}
                noise = np.asarray(agent.sample_actions(batched), dtype=np.float32).reshape(1, 32)
            model_input = dict(current)
            model_input.setdefault("prompt", collect.COLLECT_OBJECTS_PROMPT)
            full_noise = np.repeat(noise, config.horizon, axis=0)
            actions = np.asarray(policy.infer(model_input, noise=full_noise)["actions"], dtype=np.float32).copy()
            if actions.shape != (config.horizon, 7) or not np.isfinite(actions).all():
                raise ValueError("VLA must output a finite absolute Cartesian action chunk [H,7]")
            actions[:, 6] = np.clip(actions[:, 6], 0, 1)
            count = min(config.execute_steps, config.max_episode_steps - episode_steps)
            result = env.step_chunk(actions[:count], publish_hz=config.control_hz)
            following = result["observation"]
            reward, mask, executed = transition_reward(current, following, result, config)
            if executed > count:
                raise ValueError("Environment executed more actions than requested")
            replay.items.append(
                {
                    "observations": observation,
                    "next_observations": sac_observation(following, config),
                    "actions": noise.copy(),
                    "rewards": np.float32(reward),
                    "masks": np.float32(mask),
                    "discount": np.float32(config.discount**executed),
                }
            )
            current = following
            queries += 1
            episode_steps += executed
            episode_reward += reward
            if len(replay) >= max(config.batch_size, config.start_online_updates):
                for _ in range(min(config.utd_ratio, config.max_updates - updates)):
                    agent.update(replay.sample(config.batch_size))
                    updates += 1
            if result["terminated"] or result["truncated"] or updates >= config.max_updates:
                break
        summary = {"episode": episode + 1, "updates": updates, "queries": queries, "reward": episode_reward}
        if on_episode is not None:
            on_episode(summary)
        if updates >= config.max_updates:
            break
    return summary


class FakeCollectEnv:
    """Protocol test environment; its synthetic progress is not a task evaluation."""

    def __init__(self, episode_steps: int = 24):
        self.episode_steps = episode_steps
        self.steps = 0
        self.pose = np.array([0.4, 0, 0.2, 3.1, 0, 0], dtype=np.float32)

    def _observation(self):
        return {
            "observation.images.primary": np.zeros((32, 48, 3), dtype=np.uint8),
            "observation.images.wrist": np.full((32, 48, 3), 64, dtype=np.uint8),
            "observation.state.cartesian": self.pose.copy(),
            "observation.state.gripper": np.float32(1),
            "observation.state.joints": np.zeros(7, dtype=np.float32),
            "progress": self.steps / self.episode_steps,
        }

    def reset(self):
        self.steps = 0
        return self._observation()

    def step_chunk(self, actions, *, publish_hz):
        del publish_hz
        count = min(len(actions), self.episode_steps - self.steps)
        self.pose = np.asarray(actions[count - 1, :6]).copy()
        self.steps += count
        terminated = self.steps >= self.episode_steps
        rewards = np.zeros(count, dtype=np.float32)
        rewards[-1] = float(terminated)
        return {"observation": self._observation(), "rewards": rewards, "terminated": terminated, "truncated": False}

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--policy-config", default=DsrlConfig.policy_config)
    parser.add_argument("--env-factory", help="module:function returning the FR3 environment")
    parser.add_argument("--env-kwargs", default="{}", help="JSON keyword arguments for the environment factory")
    parser.add_argument("--fake-env", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--output-dir", default="./checkpoints/pi05_pku_franka_collect_objects_dsrl")
    parser.add_argument("--resume-agent")
    for name in [
        "seed",
        "execute_steps",
        "batch_size",
        "start_online_updates",
        "random_queries",
        "utd_ratio",
        "max_updates",
        "max_episodes",
        "max_episode_steps",
        "buffer_capacity",
    ]:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(DsrlConfig, name))
    parser.add_argument("--shaping-weight", type=float, default=DsrlConfig.shaping_weight)
    args = parser.parse_args()
    fields = {f.name: getattr(args, f.name) for f in dataclasses.fields(DsrlConfig) if hasattr(args, f.name)}
    config = DsrlConfig(**fields)
    if args.print_config:
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return
    if not args.base_checkpoint or (not args.fake_env and not args.env_factory):
        parser.error("Provide --base-checkpoint and either --env-factory or --fake-env")
    # Runtime-only imports keep config inspection and protocol tests independent of JAX.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "third_party/jaxrl2"))
    from flax.core import freeze  # noqa: PLC0415
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner  # noqa: PLC0415

    from openpi.policies.policy_config import create_trained_policy  # noqa: PLC0415
    from openpi.training.config import get_config  # noqa: PLC0415

    base = get_config(config.policy_config)
    if base.model.action_horizon != config.horizon or base.model.action_dim != 32:
        raise ValueError("DSRL base policy must use horizon=30 and internal action_dim=32")
    # Serving uses checkpoint-bundled norm stats and never needs training labels.
    # In particular, an IQL/Filter-BC checkpoint must load on the unlabeled robot host.
    base = dataclasses.replace(base, data=dataclasses.replace(base.data, repo_id=None))
    sample = sac_observation(FakeCollectEnv().reset(), config)
    learner = PixelSACLearner(
        config.seed,
        {k: v[None] for k, v in sample.items()},
        np.zeros((1, 1, 32), dtype=np.float32),
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        discount=config.discount**config.execute_steps,
        encoder_type="small",
        num_cameras=2,
        num_qs=10,
        color_jitter=False,
        softmax_temperature=-1,
    )
    if args.resume_agent:
        learner.restore_checkpoint(args.resume_agent)
    policy = create_trained_policy(base, args.base_checkpoint)

    class Agent:
        def sample_actions(self, observation):
            return learner.sample_actions(freeze(observation))

        def update(self, batch):
            return learner.update(freeze(batch))

    if args.fake_env:
        env = FakeCollectEnv()
    else:
        module, function = args.env_factory.split(":", 1)
        env = getattr(importlib.import_module(module), function)(**json.loads(args.env_kwargs))
    output = pathlib.Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))

    def checkpoint(summary):
        print(json.dumps(summary), flush=True)
        if summary["updates"]:
            learner.save_checkpoint(str(output), summary["updates"], keep_every_n_steps=None, keep=3)

    try:
        run_training(env, policy, Agent(), config, on_episode=checkpoint)
    finally:
        env.close()


if __name__ == "__main__":
    main()
