#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"${PYTHON:-python3}" -m openpi.training.pku_franka_dsrl "$@"
