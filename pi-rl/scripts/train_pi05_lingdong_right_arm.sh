#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_PYTHON=/opt/conda/envs/pku-pi-rl/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
CONFIG=pi05_lingdong_right_arm

# Explicit paths take priority. Otherwise reuse the byte-verified SSD copies
# prepared for this final dataset and base checkpoint when they are available.
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    WEIGHT_PARAMS_PATH="${INIT_CHECKPOINT%/}/params"
fi
TRAINING_PATHS=$("$PYTHON" - "${DATA_ROOT:-}" "${WEIGHT_PARAMS_PATH:-}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

default_data = Path('/mnt/rynnbot_hangzhou/hdc/pku/datasets/lingdong')
default_weights = Path('/mnt/rynnbot_hangzhou/hdc/pku/openpi_cache/openpi-assets/checkpoints/pi05_base/params')
cache = Path('/tmp/lingdong_training_cache')
name = 'lerobot_2026-10-26_12-09-32_final'
data, weights = sys.argv[1:]
if not data:
    data = str(default_data)
    try:
        record = json.loads((cache / name / '.training_cache.json').read_text())
        signature = hashlib.sha256((default_data / name / 'meta/file_checksums.json').read_bytes()).hexdigest()
        if (record['status'] == 'complete' and record['source'] == str(default_data / name)
                and record['source_inventory_sha256'] == signature
                and (cache / name / 'meta/info.json').is_file()):
            data = str(cache)
    except (OSError, ValueError, KeyError):
        pass
if not weights:
    weights = str(default_weights)
    try:
        record = json.loads((cache / 'pi05_base/source_manifest.json').read_text())
        expected = record['files']['_METADATA']['sha256']
        if (record['status'] == 'complete' and record['source'] == str(default_weights)
                and hashlib.sha256((default_weights / '_METADATA').read_bytes()).hexdigest() == expected
                and hashlib.sha256((cache / 'pi05_base/params/_METADATA').read_bytes()).hexdigest() == expected):
            weights = str(cache / 'pi05_base/params')
    except (OSError, ValueError, KeyError):
        pass
print(data)
print(weights)
PY
)
mapfile -t SELECTED_PATHS <<< "$TRAINING_PATHS"
DATA_ROOT="${SELECTED_PATHS[0]}"
WEIGHT_PARAMS_PATH="${SELECTED_PATHS[1]}"
EXP_NAME="${EXP_NAME:-lingdong_right_arm_$(date +%m%d_%H%M%S)}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$PROJECT_ROOT/../output/checkpoints}"
# Match train_sft_pku_franka_collect_objects.sh: use every visible physical GPU
# unless the caller supplies GPUS; shard the model across the selected devices.
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
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
if [[ ! "$BATCH_SIZE_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE_PER_GPU must be a positive integer." >&2
    exit 1
fi
BATCH_SIZE="${BATCH_SIZE:-$((BATCH_SIZE_PER_GPU * ${#GPU_LIST[@]}))}"
# Keep the CLI's global-batch override visible in the launch summary as well.
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
export CUDA_VISIBLE_DEVICES="$GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
if [[ -z "${OPENPI_DATA_HOME:-}" ]]; then
    export OPENPI_DATA_HOME=/mnt/rynnbot_hangzhou/hdc/pku/openpi_cache
fi
if [[ ! -d "$WEIGHT_PARAMS_PATH" ]]; then
    echo "Pi0.5 base parameters are unavailable: $WEIGHT_PARAMS_PATH; set WEIGHT_PARAMS_PATH to a mounted params directory." >&2
    exit 1
fi
if [[ ! -f "$DATA_ROOT/lerobot_2026-10-26_12-09-32_final/meta/info.json" ]]; then
    echo "Final Lingdong dataset is unavailable under $DATA_ROOT; set DATA_ROOT to its parent directory." >&2
    exit 1
fi

TRAIN_ARGS=(
    scripts/train.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --data.dataset-root "$DATA_ROOT"
    --weight-loader.params-path "$WEIGHT_PARAMS_PATH"
    --checkpoint-base-dir "$CKPT_BASE_DIR"
    --fsdp-devices "$FSDP_DEVICES"
    --batch-size "$BATCH_SIZE"
)
echo "Starting $CONFIG: data=$DATA_ROOT, weights=$WEIGHT_PARAMS_PATH, GPUs=$GPUS, FSDP=$FSDP_DEVICES, global_batch=$BATCH_SIZE, batch_per_gpu=$((BATCH_SIZE / ${#GPU_LIST[@]})), experiment=$EXP_NAME"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$PYTHON" "${TRAIN_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

# Match reference SFT preflight: compute norms on CPU unless SKIP_NORM=1.
# Every retained frame contributes once; no videos are decoded. Even when
# skipping, require finite 8-D stats with the exact current source fingerprint.
if [[ "${SKIP_NORM:-0}" != "1" ]]; then
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" scripts/compute_lingdong_norm_stats.py --dataset-root "$DATA_ROOT"
fi
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PYTHON" scripts/compute_lingdong_norm_stats.py --dataset-root "$DATA_ROOT" --check
exec "$PYTHON" "${TRAIN_ARGS[@]}" "$@"
