#!/usr/bin/env bash
# Run from a training environment with pi-rl installed and the dataset mounted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
# Prefer this checkout's OpenPI and vendored IQL learner over editable installs elsewhere.
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/jaxrl2${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_PYTHON=/opt/conda/envs/pku-pi-rl/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
METHOD="${METHOD:-sft}"
CONFIG_PREFIX=pi05_pku_franka_collect_objects
case "${ACTION_REPRESENTATION:-absolute}" in
    absolute) ;;
    relative) CONFIG_PREFIX+=_relative ;;
    *) echo "ACTION_REPRESENTATION must be absolute or relative" >&2; exit 1 ;;
esac
case "$METHOD" in
    sft) DEFAULT_CONFIG="$CONFIG_PREFIX"; TRAIN_SCRIPT=scripts/train.py ;;
    iql) DEFAULT_CONFIG="${CONFIG_PREFIX}_iql"; TRAIN_SCRIPT=scripts/train_iql.py ;;
    filter_bc) DEFAULT_CONFIG="${CONFIG_PREFIX}_filter_bc"; TRAIN_SCRIPT=scripts/train.py ;;
    *) echo "METHOD must be sft, iql, or filter_bc" >&2; exit 1 ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
DEFAULT_REPO_ID=/oss/embodied-ai-data/pku/fr3_collect_objects_20261017
if [[ "$METHOD" == "iql" ]]; then
    DEFAULT_REPO_ID+=_progress_expert_sup_step005000
fi
REPO_ID="${REPO_ID:-$DEFAULT_REPO_ID}"
EXP_NAME="${EXP_NAME:-$(basename "$REPO_ID")_$(date +%m%d_%H%M%S)}"
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
if [[ "$METHOD" == "sft" ]]; then
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
        esac
    done
    if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || ((BATCH_SIZE % ${#GPU_LIST[@]} != 0)); then
        echo "Global BATCH_SIZE must be positive and divisible by the selected GPU count." >&2
        exit 1
    fi
fi
CKPT_BASE_DIR="${CKPT_BASE_DIR:-./checkpoints}"
WEIGHT_PARAMS_PATH="${WEIGHT_PARAMS_PATH:-/oss/embodied-ai-data/pku/pi05_base/params}"

case "$METHOD:$CONFIG" in
    sft:pi05_pku_franka_collect_objects) ;;
    sft:pi05_pku_franka_collect_objects_relative) ;;
    iql:pi05_pku_franka_collect_objects_iql) ;;
    iql:pi05_pku_franka_collect_objects_relative_iql) ;;
    filter_bc:pi05_pku_franka_collect_objects_filter_bc) ;;
    filter_bc:pi05_pku_franka_collect_objects_relative_filter_bc) ;;
    *) echo "Unsupported CONFIG: $CONFIG" >&2; exit 1 ;;
esac
ASSET_ID=pku_franka_collect_objects
case "$CONFIG" in
    pi05_pku_franka_collect_objects_relative*) CONFIG_REPRESENTATION=relative; ASSET_ID+=_relative ;;
    *) CONFIG_REPRESENTATION=absolute ;;
esac
if [[ -n "${ACTION_REPRESENTATION:-}" && "$ACTION_REPRESENTATION" != "$CONFIG_REPRESENTATION" ]]; then
    echo "CONFIG=$CONFIG conflicts with ACTION_REPRESENTATION=$ACTION_REPRESENTATION" >&2
    exit 1
fi
if [[ ! -f "$REPO_ID/meta/info.json" ]]; then
    echo "Dataset is not mounted at $REPO_ID; set REPO_ID to its local path." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# Always recompute: stats must use this adapter and the same episode whitelist.
# Batch 1 includes the last frame instead of dropping an incomplete batch.
if [[ "${SKIP_NORM:-0}" != "1" ]]; then
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" scripts/compute_norm_stats_fast.py \
        --config-name "$CONFIG" --repo-id "$REPO_ID" --num-workers 0 --batch-size 1
fi

# Each preset has its own stats, and the action representations have separate
# asset IDs. Reject old 20-D state stats, including when SKIP_NORM is requested.
NORM_STATS_FILE="./assets/$CONFIG/$ASSET_ID/norm_stats.json"
"$PYTHON" - "$NORM_STATS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    stats = json.load(stream)["norm_stats"]
for key, dimension in [("state", 14), ("actions", 7)]:
    for field in ["mean", "std", "q01", "q99"]:
        values = stats[key].get(field)
        if values is None or len(values) != dimension:
            raise SystemExit(f"Invalid {key}.{field} normalization; recompute without SKIP_NORM=1.")
PY

# No --overwrite: existing experiments require an explicit --resume or a new name.
TRAIN_ARGS=(
    "$TRAIN_SCRIPT" "$CONFIG"
    --exp-name "$EXP_NAME"
    --data.repo-id "$REPO_ID"
    --checkpoint-base-dir "$CKPT_BASE_DIR"
    --fsdp-devices "$FSDP_DEVICES"
)
if [[ "$METHOD" == "sft" ]]; then
    TRAIN_ARGS+=(--batch-size "$BATCH_SIZE")
    echo "Starting $CONFIG: GPUs=$GPUS, FSDP=$FSDP_DEVICES, global_batch=$BATCH_SIZE, batch_per_gpu=$((BATCH_SIZE / ${#GPU_LIST[@]})), experiment=$EXP_NAME"
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=(--weight-loader.params-path "${INIT_CHECKPOINT%/}/params")
elif [[ -d "$WEIGHT_PARAMS_PATH" ]]; then
    TRAIN_ARGS+=(--weight-loader.params-path "$WEIGHT_PARAMS_PATH")
else
    echo "Warning: base weights not found at $WEIGHT_PARAMS_PATH; using the config's default loader." >&2
fi
"$PYTHON" "${TRAIN_ARGS[@]}" "$@"
