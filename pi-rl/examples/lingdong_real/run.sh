#!/bin/bash
# Launch Lingdong right-arm pi05 inference on the Orin. Default is DRY RUN.
#   examples/lingdong_real/run.sh --backend replay --replay-episode 131 --task pick_up
#   examples/lingdong_real/run.sh --mode camera-check
#   examples/lingdong_real/run.sh --prompt "<exact task>" --execute      # moves the robot after confirmation
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# REQUIRED on this Orin: with XLA's Triton GEMM autotuning the pi05 forward pass is numerically wrong
# ("Results do not match the reference" in the log) and actions ignore all inputs. Verified 2026-10-28.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0${XLA_FLAGS:+ $XLA_FLAGS}"
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_ALLOCATOR || true
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/mnt/hdc/.jax_compile_cache_notriton}"
export NO_PROXY="localhost,127.0.0.1,::1,10.192.1.0/24${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export UV_CACHE_DIR=/mnt/hdc/.uv-cache UV_LINK_MODE=copy
exec "$REPO/.venv/bin/python" "$HERE/main.py" "$@"
