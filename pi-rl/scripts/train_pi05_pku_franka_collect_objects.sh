#!/usr/bin/env bash
# Run from a training environment with pi-rl installed and the dataset mounted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
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
REPO_ID="${REPO_ID:-/home/hdc/workspace/DATA/local/fr3_collect_objects_20261017}"
EXP_NAME="${EXP_NAME:-fr3_collect_objects_20261017}"
GPUS="${GPUS:-0}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-./checkpoints}"

case "$METHOD:$CONFIG" in
    sft:pi05_pku_franka_collect_objects|sft:pi05_pku_franka_collect_objects_lora) ;;
    sft:pi05_pku_franka_collect_objects_relative|sft:pi05_pku_franka_collect_objects_relative_lora) ;;
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
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=(--weight-loader.params-path "${INIT_CHECKPOINT%/}/params")
fi
"$PYTHON" "${TRAIN_ARGS[@]}" "$@"
