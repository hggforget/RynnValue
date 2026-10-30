# adapted from openpi
import dataclasses
import enum
import functools
import logging
import socket
from typing import Literal

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Network interface to bind for incoming WebSocket connections.
    host: str = "0.0.0.0"
    # Record the policy's behavior for debugging.
    record: bool = False

    # Select a separate policy implementation; ordinary inference remains the default.
    inference_mode: Literal["standard", "rtc"] = "standard"
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_schedule: Literal["exp", "linear", "hard"] = "exp"
    # Optional metadata source override, useful for mounted OSS datasets.
    dataset_root: str | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # ── DSRL options (optional, only used when dsrl_ckpt_dir is set) ──
    # Directory containing the trained DSRL (SAC) checkpoint.
    # When set, the policy server wraps the base pi05 policy with the
    # DSRL noise predictor for guided inference.
    dsrl_ckpt_dir: str | None = None
    # Arm mode for DSRL observation processing.
    dsrl_arm_mode: str = "single"
    # SAC encoder input image size.
    dsrl_resize_image: int = 64


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    optional = {}
    if args.inference_mode == "rtc":
        if args.dsrl_ckpt_dir or not isinstance(args.policy, Checkpoint):
            raise ValueError("RTC requires an explicit Lingdong checkpoint and cannot be combined with DSRL")
        if args.policy.config not in ("pi05_lingdong_right_arm", "pi05_lingdong_right_arm_crop"):
            raise ValueError("RTC currently supports absolute Lingdong full-frame and crop configs")
        from openpi.models.rtc import RTCConfig
        from openpi.policies.rtc_policy import RTCPolicy

        optional["policy_factory"] = functools.partial(
            RTCPolicy,
            rtc_config=RTCConfig(args.rtc_max_guidance_weight, args.rtc_prefix_schedule),
        )
    match args.policy:
        case Checkpoint():
            config = _config.get_config(args.policy.config)
            if args.dataset_root is not None:
                if not hasattr(config.data, "dataset_root"):
                    raise ValueError("This policy config does not accept a dataset_root override")
                config = dataclasses.replace(
                    config, data=dataclasses.replace(config.data, dataset_root=args.dataset_root)
                )
            base_policy = _policy_config.create_trained_policy(
                config, args.policy.dir, default_prompt=args.default_prompt, **optional
            )
        case Default():
            base_policy = create_default_policy(args.env, default_prompt=args.default_prompt)

    # Optionally wrap with DSRL noise predictor
    if args.dsrl_ckpt_dir:
        from openpi.policies.dsrl_wrapper import DsrlPolicyWrapper

        logging.info("Wrapping policy with DSRL (ckpt=%s, arm_mode=%s)", args.dsrl_ckpt_dir, args.dsrl_arm_mode)
        base_policy = DsrlPolicyWrapper(
            base_policy=base_policy,
            dsrl_ckpt_dir=args.dsrl_ckpt_dir,
            arm_mode=args.dsrl_arm_mode,
            resize_image=args.dsrl_resize_image,
        )

    return base_policy


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)
    policy_metadata["inference_mode"] = args.inference_mode
    if isinstance(args.policy, Checkpoint):
        policy_metadata.update(config=args.policy.config, checkpoint=args.policy.dir)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
