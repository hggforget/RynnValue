#!/usr/bin/env bash
# Run from a training environment with pi-rl installed and the dataset mounted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
METHOD="${METHOD:-sft}"
case "$METHOD" in
    sft) DEFAULT_CONFIG=pi05_pku_franka_collect_objects; TRAIN_SCRIPT=scripts/train.py ;;
    iql) DEFAULT_CONFIG=pi05_pku_franka_collect_objects_iql; TRAIN_SCRIPT=scripts/train_iql.py ;;
    filter_bc) DEFAULT_CONFIG=pi05_pku_franka_collect_objects_filter_bc; TRAIN_SCRIPT=scripts/train.py ;;
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
    iql:pi05_pku_franka_collect_objects_iql) ;;
    filter_bc:pi05_pku_franka_collect_objects_filter_bc) ;;
    *) echo "Unsupported CONFIG: $CONFIG" >&2; exit 1 ;;
esac
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

# Both presets use the same 14-D measured state and 7-D absolute action stats.
# Reject old 20-D state stats. SKIP_NORM also requires the same action contract.
NORM_STATS_FILE="./assets/$CONFIG/pku_franka_collect_objects/norm_stats.json"
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
INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    INIT_ARGS+=(--weight-loader.params-path "${INIT_CHECKPOINT%/}/params")
fi
"$PYTHON" "$TRAIN_SCRIPT" "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --data.repo-id "$REPO_ID" \
    --checkpoint-base-dir "$CKPT_BASE_DIR" \
    --fsdp-devices "$FSDP_DEVICES" \
    "${INIT_ARGS[@]}" \
    "$@"
