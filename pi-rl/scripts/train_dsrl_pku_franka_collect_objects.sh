#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
# The environment may have an editable openpi install from another checkout; prefer this repo.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"${PYTHON:-python3}" -m openpi.training.pku_franka_dsrl "$@"
