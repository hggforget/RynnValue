"""Policy backends: in-process JAX policy (default) or websocket client to scripts/serve_policy.py."""

from __future__ import annotations

import logging
import os
import time

import numpy as np

log = logging.getLogger("policy")
TRITON_OFF = "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"


class LocalPolicy:
    def __init__(self, config_name: str, checkpoint: str, *, num_steps: int | None = None,
                 compile_cache_dir: str | None = None):
        t0 = time.monotonic()
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_gpu_enable_triton_gemm=false" not in flags:
            # Orin: Triton GEMM autotuning yields wrong numerics -> constant actions. Must be set before jax init.
            os.environ["XLA_FLAGS"] = (flags + " " + TRITON_OFF).strip()
            log.warning("XLA_FLAGS += %s", TRITON_OFF)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax

        if compile_cache_dir:
            os.makedirs(compile_cache_dir, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", compile_cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        from openpi.policies import policy_config
        from openpi.training import config as _config

        cfg = _config.get_config(config_name)
        sample_kwargs = {"num_steps": int(num_steps)} if num_steps else None
        self._policy = policy_config.create_trained_policy(cfg, checkpoint, sample_kwargs=sample_kwargs)
        self.metadata = dict(self._policy.metadata)
        self.action_horizon = cfg.model.action_horizon
        self.load_s = time.monotonic() - t0
        log.info("local policy %s loaded in %.1fs (num_steps=%s, devices=%s)", config_name, self.load_s,
                 num_steps or "default(10)", jax.devices())

    def infer(self, obs: dict) -> tuple[np.ndarray, dict]:
        t = time.monotonic()
        out = self._policy.infer(obs)
        actions = np.asarray(out["actions"])  # blocks until the device result is ready
        timing = {"policy_wall_ms": (time.monotonic() - t) * 1e3}
        timing.update({f"server_{k}": v for k, v in (out.get("policy_timing") or {}).items()})
        return actions, timing

    def close(self):
        pass


class WebsocketPolicy:
    def __init__(self, host: str, port: int, *, retry_s: float = 60.0):
        # The policy server is local / on the LAN; never route it through the SOCKS proxy.
        for k in ("NO_PROXY", "no_proxy"):
            v = os.environ.get(k, "")
            if host not in v.split(","):
                os.environ[k] = ",".join(x for x in (v, host) if x)
        from openpi_client import websocket_client_policy

        t0 = time.monotonic()
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self._client.get_server_metadata()
        self.action_horizon = None
        self.load_s = time.monotonic() - t0
        log.info("connected to policy server %s:%d metadata=%s", host, port, self.metadata)

    def infer(self, obs: dict) -> tuple[np.ndarray, dict]:
        t = time.monotonic()
        out = self._client.infer(obs)
        timing = {"policy_wall_ms": (time.monotonic() - t) * 1e3}
        for k, v in (out.get("policy_timing") or {}).items():
            timing[f"policy_{k}"] = v
        for k, v in (out.get("server_timing") or {}).items():
            timing[f"server_{k}"] = v
        return np.asarray(out["actions"]), timing

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
