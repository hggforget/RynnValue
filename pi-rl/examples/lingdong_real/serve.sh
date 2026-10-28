#!/bin/bash
# Option B: serve the checkpoint over websocket, then run main.py --policy websocket --port 8000
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# REQUIRED on this Orin: with XLA's Triton GEMM autotuning the pi05 forward pass is numerically wrong
# ("Results do not match the reference" in the log) and actions ignore all inputs. Verified 2026-10-28.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0${XLA_FLAGS:+ $XLA_FLAGS}"
CKPT="${CKPT:-/mnt/hdc/checkpoints/pi05-lingdong-right-arm/pi05_lingdong_right_arm/lingdong_right_arm_1027_163358/10000}"
cd "$REPO" && exec .venv/bin/python scripts/serve_policy.py --port "${PORT:-8000}" policy:checkpoint \
  --policy.config pi05_lingdong_right_arm --policy.dir "$CKPT"
