#!/usr/bin/env bash
# Run directly: defaults select the dataset, GPUs, experiment name, and normalization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'

DEFAULT_PYTHON=/opt/conda/envs/pi-rl/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
CONFIG=pi05_pku_franka_unified_relative
DEFAULT_DATA_ROOT=/oss/embodied-ai-data/pku/fr3_unified_5hz_relative
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
WEIGHT_PARAMS_PATH="${WEIGHT_PARAMS_PATH:-/oss/embodied-ai-data/pku/pi05_base/params}"
EXP_NAME="${EXP_NAME:-fr3_unified_relative_$(date +%m%d_%H%M%S)}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$PROJECT_ROOT/../output/checkpoints}"
if [[ -z "${GPUS:-}" ]]; then
    GPUS=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || true)
        if [[ "$GPU_COUNT" -gt 1 ]]; then
            GPUS=$(seq -s, 0 $((GPU_COUNT - 1)))
        fi
    fi
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
FSDP_DEVICES="${FSDP_DEVICES:-${#GPU_LIST[@]}}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
SHARED_CACHE=/mnt/rynnbot_hangzhou/hdc/pku/openpi_cache
if [[ -z "${OPENPI_DATA_HOME:-}" && -f "$SHARED_CACHE/big_vision/paligemma_tokenizer.model" ]]; then
    export OPENPI_DATA_HOME="$SHARED_CACHE"
fi

if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
    echo "Unified collection is not mounted: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -d "$WEIGHT_PARAMS_PATH" ]]; then
    echo "Base checkpoint is not mounted: $WEIGHT_PARAMS_PATH" >&2
    exit 1
fi
NORM_STATS_FILE="./assets/$CONFIG/pku_franka_unified_relative/norm_stats.json"
validate_norm_stats() {
    "$PYTHON" - "$NORM_STATS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    stats = json.load(stream)["norm_stats"]
for key in ("state", "actions"):
    for field in ("mean", "std", "q01", "q99"):
        if len(stats.get(key, {}).get(field, [])) != 7:
            raise SystemExit(f"Invalid {key}.{field} norms; run with SKIP_NORM=0.")
PY
}

# Reuse bundled stats for the default collection. Compute them automatically if
# absent or invalid; changing the collection defaults to recomputing its stats.
if [[ "$DATA_ROOT" == "$DEFAULT_DATA_ROOT" ]]; then
    NORM_MODE="${SKIP_NORM:-auto}"
else
    NORM_MODE="${SKIP_NORM:-0}"
fi
case "$NORM_MODE" in
    auto)
        if validate_norm_stats >/dev/null 2>&1; then
            NORM_MODE=1
        else
            NORM_MODE=0
        fi
        ;;
    0|1) ;;
    *) echo "SKIP_NORM must be auto, 0, or 1" >&2; exit 1 ;;
esac
if [[ "$NORM_MODE" == "0" ]]; then
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" scripts/compute_norm_stats_fast.py \
        --config-name "$CONFIG" --dataset-root "$DATA_ROOT" --num-workers 0 --batch-size 1
fi
validate_norm_stats

echo "Starting $CONFIG: experiment=$EXP_NAME, GPUs=$GPUS, data=$DATA_ROOT"

exec "$PYTHON" scripts/train.py "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --data.dataset-root "$DATA_ROOT" \
    --weight-loader.params-path "$WEIGHT_PARAMS_PATH" \
    --checkpoint-base-dir "$CKPT_BASE_DIR" \
    --fsdp-devices "$FSDP_DEVICES" \
    "$@"
