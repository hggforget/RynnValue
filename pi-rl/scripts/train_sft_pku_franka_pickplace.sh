#!/usr/bin/env bash
# SFT for the merged Oct 27-30 FR3 pick-and-place dataset (30 Hz, absolute actions).
# Run from any directory:
#   GPUS=0 bash /path/to/pi-rl/scripts/train_sft_pku_franka_pickplace.sh
#   GPUS=0,1 BATCH_SIZE_PER_GPU=16 DRY_RUN=1 bash scripts/train_sft_pku_franka_pickplace.sh
#   EXP_NAME=existing_experiment SKIP_NORM=1 bash scripts/train_sft_pku_franka_pickplace.sh --resume
# Extra arguments are forwarded to train.py; use REPO_ID to override the dataset for both passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/jaxrl2${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${PYTHON:-}" ]]; then
    PYTHON=python3
    for candidate in "$PROJECT_ROOT/.venv/bin/python" /opt/conda/envs/pi-rl/bin/python /opt/conda/envs/pku-pi-rl/bin/python; do
        if [[ -x "$candidate" ]]; then
            PYTHON="$candidate"
            break
        fi
    done
fi
CONFIG=pi05_pku_franka_pickplace
ASSET_ID=pku_franka_pickplace
REPO_ID="${REPO_ID:-/oss/embodied-ai-data/pku/fr3_real_pickplace_20261027_20261030_merged}"
EXP_NAME="${EXP_NAME:-fr3_pickplace_$(date +%Y%m%d_%H%M%S)}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-./checkpoints}"
WEIGHT_PARAMS_PATH="${WEIGHT_PARAMS_PATH:-/oss/embodied-ai-data/pku/pi05_base/params}"

if [[ -z "${GPUS:-}" ]]; then
    GPUS="${CUDA_VISIBLE_DEVICES:-}"
    if [[ -z "$GPUS" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)
    fi
    GPUS="${GPUS:-0}"
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
FSDP_DEVICES="${FSDP_DEVICES:-${#GPU_LIST[@]}}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
if [[ ! "$BATCH_SIZE_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE_PER_GPU must be a positive integer." >&2
    exit 1
fi
BATCH_SIZE="${BATCH_SIZE:-$((BATCH_SIZE_PER_GPU * ${#GPU_LIST[@]}))}"
CLI_ARGS=("$@")
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
    case "${CLI_ARGS[i]}" in
        --batch-size) BATCH_SIZE="${CLI_ARGS[i+1]:-}" ;;
        --batch-size=*) BATCH_SIZE="${CLI_ARGS[i]#*=}" ;;
        --fsdp-devices) FSDP_DEVICES="${CLI_ARGS[i+1]:-}" ;;
        --fsdp-devices=*) FSDP_DEVICES="${CLI_ARGS[i]#*=}" ;;
        --data.*|--assets-base-dir|--assets-base-dir=*)
            echo "Use REPO_ID to select the dataset; this launcher fixes the adapter and norm assets for both passes." >&2
            exit 1
            ;;
    esac
done
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || ((BATCH_SIZE % ${#GPU_LIST[@]} != 0)); then
    echo "Global BATCH_SIZE must be positive and divisible by the selected GPU count." >&2
    exit 1
fi
if [[ ! "$FSDP_DEVICES" =~ ^[1-9][0-9]*$ ]] || ((${#GPU_LIST[@]} % FSDP_DEVICES != 0)); then
    echo "FSDP_DEVICES must be a positive divisor of the selected GPU count." >&2
    exit 1
fi
if [[ ! -f "$REPO_ID/meta/info.json" ]]; then
    echo "Dataset is not mounted at $REPO_ID; set REPO_ID to its local path." >&2
    exit 1
fi
if [[ ! -d "$WEIGHT_PARAMS_PATH" ]]; then
    echo "Base weights are not mounted at $WEIGHT_PARAMS_PATH; set WEIGHT_PARAMS_PATH to pi05_base/params." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# Reuse the mounted tokenizer cache together with the local pi05_base weights.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/oss/embodied-ai-data/pku/openpi_cache}"
NORM_ARGS=(scripts/compute_norm_stats_fast.py --config-name "$CONFIG" --repo-id "$REPO_ID" --num-workers 0 --batch-size 1)
TRAIN_ARGS=(
    scripts/train.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --data.repo-id "$REPO_ID"
    --checkpoint-base-dir "$CKPT_BASE_DIR"
    --weight-loader.params-path "$WEIGHT_PARAMS_PATH"
    --fsdp-devices "$FSDP_DEVICES"
    --batch-size "$BATCH_SIZE"
    "$@"
)
echo "Starting $CONFIG: GPUs=$GPUS, FSDP=$FSDP_DEVICES, global_batch=$BATCH_SIZE, experiment=$EXP_NAME"
echo "Dataset: $REPO_ID (per-task prompts, all episodes, absolute actions)"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if [[ "${SKIP_NORM:-0}" != "1" ]]; then
        printf '%q ' env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false "$PYTHON" "${NORM_ARGS[@]}"
        printf '\n'
    fi
    printf '%q ' "$PYTHON" "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

# Recompute with this adapter and every frame; batch 1 includes the final frame.
if [[ "${SKIP_NORM:-0}" != "1" ]]; then
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false "$PYTHON" "${NORM_ARGS[@]}"
fi
NORM_STATS_FILE="./assets/$CONFIG/$ASSET_ID/norm_stats.json"
"$PYTHON" - "$NORM_STATS_FILE" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as stream:
    stats = json.load(stream)["norm_stats"]
for key, dimension in [("state", 14), ("actions", 7)]:
    for field in ["mean", "std", "q01", "q99"]:
        values = stats[key].get(field)
        if values is None or len(values) != dimension or not all(math.isfinite(value) for value in values):
            raise SystemExit(f"Invalid {key}.{field} normalization; recompute without SKIP_NORM=1.")
PY
exec "$PYTHON" "${TRAIN_ARGS[@]}"
