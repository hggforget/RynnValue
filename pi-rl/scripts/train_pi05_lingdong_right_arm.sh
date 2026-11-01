#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/jaxrl2:$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_PYTHON=/opt/conda/envs/pku-pi-rl/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=/opt/conda/envs/pi-rl/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-absolute}"
ACTION_HORIZON="${ACTION_HORIZON:-30}"
TASK="${TASK:-all}"
METHOD="${METHOD:-sft}"
REWARD_SOURCE="${REWARD_SOURCE:-progress}"
REWARD_LABELS_PATH="${REWARD_LABELS_PATH:-}"
ASSUME_SUCCESS="${ASSUME_SUCCESS:-0}"

# Read data and base weights directly from OSS unless explicitly overridden.
DATA_ROOT="${DATA_ROOT:-/oss/embodied-ai-data/lingdong}"
WEIGHT_PARAMS_PATH="${WEIGHT_PARAMS_PATH:-/oss/embodied-ai-data/pku/pi05_base/params}"
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    WEIGHT_PARAMS_PATH="${INIT_CHECKPOINT%/}/params"
fi
CKPT_BASE_DIR="${CKPT_BASE_DIR:-../output/checkpoints}"
# All SFT/IQL/task wrappers share this runtime GPU discovery. Respect a CUDA
# visibility mask supplied by the scheduler before counting physical GPUs.
if [[ -z "${GPUS:-}" || "${GPUS:-}" == auto ]]; then
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        GPUS="$CUDA_VISIBLE_DEVICES"
    else
        GPU_COUNT=0
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || true)
        fi
        if [[ "$GPU_COUNT" -eq 0 ]]; then
            echo "No visible GPUs detected; check the worker allocation or set GPUS explicitly." >&2
            exit 1
        fi
        GPUS=$(seq -s, 0 $((GPU_COUNT - 1)))
    fi
fi
if [[ -z "$GPUS" || "$GPUS" == -1 || "$GPUS" == none || "$GPUS" == ,* || "$GPUS" == *, || "$GPUS" == *,,* ]]; then
    echo "No usable GPUs in GPUS/CUDA_VISIBLE_DEVICES: $GPUS" >&2
    exit 1
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
for GPU_ID in "${GPU_LIST[@]}"; do
    [[ "$GPU_ID" =~ ^[A-Za-z0-9_./-]+$ ]] || { echo "Invalid GPU identifier: $GPU_ID" >&2; exit 1; }
done
# Shard the model across all devices discovered on this worker.
FSDP_DEVICES="${#GPU_LIST[@]}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
if [[ ! "$BATCH_SIZE_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE_PER_GPU must be a positive integer." >&2
    exit 1
fi
BATCH_SIZE="${BATCH_SIZE:-$((BATCH_SIZE_PER_GPU * ${#GPU_LIST[@]}))}"
# Keep CLI overrides in sync with normalization and the launch summary.
CLI_ARGS=("$@")
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
    case "${CLI_ARGS[i]}" in
        --batch-size) BATCH_SIZE="${CLI_ARGS[i+1]:-}" ;;
        --batch-size=*) BATCH_SIZE="${CLI_ARGS[i]#*=}" ;;
        --fsdp-devices|--fsdp_devices)
            [[ "${CLI_ARGS[i+1]:-}" == "$FSDP_DEVICES" ]] || { echo "FSDP must match the detected GPU count: $FSDP_DEVICES" >&2; exit 1; } ;;
        --fsdp-devices=*|--fsdp_devices=*)
            [[ "${CLI_ARGS[i]#*=}" == "$FSDP_DEVICES" ]] || { echo "FSDP must match the detected GPU count: $FSDP_DEVICES" >&2; exit 1; } ;;
        --data.action-representation) ACTION_REPRESENTATION="${CLI_ARGS[i+1]:-}" ;;
        --data.action-representation=*) ACTION_REPRESENTATION="${CLI_ARGS[i]#*=}" ;;
        --model.action-horizon) ACTION_HORIZON="${CLI_ARGS[i+1]:-}" ;;
        --model.action-horizon=*) ACTION_HORIZON="${CLI_ARGS[i]#*=}" ;;
        --data.dataset-root) DATA_ROOT="${CLI_ARGS[i+1]:-}" ;;
        --data.dataset-root=*) DATA_ROOT="${CLI_ARGS[i]#*=}" ;;
        --data.task) TASK="${CLI_ARGS[i+1]:-}" ;;
        --data.task=*) TASK="${CLI_ARGS[i]#*=}" ;;
        --data.reward-source) REWARD_SOURCE="${CLI_ARGS[i+1]:-}" ;;
        --data.reward-source=*) REWARD_SOURCE="${CLI_ARGS[i]#*=}" ;;
        --data.reward-labels-path) REWARD_LABELS_PATH="${CLI_ARGS[i+1]:-}" ;;
        --data.reward-labels-path=*) REWARD_LABELS_PATH="${CLI_ARGS[i]#*=}" ;;
        --data.assume-success) ASSUME_SUCCESS=1 ;;
        --data.no-assume-success) ASSUME_SUCCESS=0 ;;
    esac
done
case "$ACTION_REPRESENTATION" in
    absolute) CONFIG=pi05_lingdong_right_arm_crop; EXP_PREFIX=lingdong_right_arm_crop ;;
    relative) CONFIG=pi05_lingdong_right_arm_relative_crop; EXP_PREFIX=lingdong_right_arm_relative_crop ;;
    *) echo "ACTION_REPRESENTATION must be absolute or relative." >&2; exit 1 ;;
esac
case "$TASK" in
    all) ;;
    pick_up|put_down)
        CONFIG="pi05_lingdong_right_arm_${TASK}_crop"
        EXP_PREFIX="lingdong_right_arm_${TASK}_crop"
        if [[ "$ACTION_REPRESENTATION" == relative ]]; then
            CONFIG="pi05_lingdong_right_arm_${TASK}_relative_crop"
            EXP_PREFIX="lingdong_right_arm_${TASK}_relative_crop"
        fi
        ;;
    *) echo "TASK must be all, pick_up or put_down." >&2; exit 1 ;;
esac
case "$METHOD" in
    sft) TRAIN_SCRIPT=scripts/train.py ;;
    iql)
        CONFIG+=_iql
        EXP_PREFIX+=_iql
        TRAIN_SCRIPT=scripts/train_iql.py
        if [[ "$REWARD_SOURCE" != progress && "$REWARD_SOURCE" != terminal ]]; then
            echo "REWARD_SOURCE must be progress or terminal." >&2
            exit 1
        fi
        if [[ "$ASSUME_SUCCESS" != 0 && "$ASSUME_SUCCESS" != 1 ]]; then
            echo "ASSUME_SUCCESS must be 0 or 1." >&2
            exit 1
        fi
        ;;
    *) echo "METHOD must be sft or iql." >&2; exit 1 ;;
esac
if [[ ! "$ACTION_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACTION_HORIZON must be a positive integer." >&2
    exit 1
fi
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}_$(date +%m%d_%H%M%S)}"
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || ((BATCH_SIZE % ${#GPU_LIST[@]} != 0)); then
    echo "Global BATCH_SIZE must be positive and divisible by the selected GPU count." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
if [[ -z "${OPENPI_DATA_HOME:-}" ]]; then
    export OPENPI_DATA_HOME=/oss/embodied-ai-data/pku/openpi_cache
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
    "$TRAIN_SCRIPT" "$CONFIG"
    --exp-name "$EXP_NAME"
    --data.dataset-root "$DATA_ROOT"
    --data.task "$TASK"
    --weight-loader.params-path "$WEIGHT_PARAMS_PATH"
    --checkpoint-base-dir "$CKPT_BASE_DIR"
    --fsdp-devices "$FSDP_DEVICES"
    --batch-size "$BATCH_SIZE"
    --model.action-horizon "$ACTION_HORIZON"
)
if [[ "$METHOD" == iql ]]; then
    TRAIN_ARGS+=(--data.reward-source "$REWARD_SOURCE")
    if [[ -n "$REWARD_LABELS_PATH" ]]; then
        TRAIN_ARGS+=(--data.reward-labels-path "$REWARD_LABELS_PATH")
    fi
    if [[ "$ASSUME_SUCCESS" == 1 ]]; then
        TRAIN_ARGS+=(--data.assume-success)
    else
        TRAIN_ARGS+=(--data.no-assume-success)
    fi
fi
echo "Starting $CONFIG: method=$METHOD, data=$DATA_ROOT, task=$TASK, weights=$WEIGHT_PARAMS_PATH, checkpoints=$CKPT_BASE_DIR, openpi_cache=$OPENPI_DATA_HOME, GPUs=$GPUS, FSDP=$FSDP_DEVICES, global_batch=$BATCH_SIZE, batch_per_gpu=$((BATCH_SIZE / ${#GPU_LIST[@]})), action_representation=$ACTION_REPRESENTATION, action_horizon=$ACTION_HORIZON, experiment=$EXP_NAME"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$PYTHON" "${TRAIN_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

if [[ "$METHOD" == iql ]]; then
    REWARD_ARGS=(--dataset-root "$DATA_ROOT" --task "$TASK" --reward-source "$REWARD_SOURCE")
    if [[ -n "$REWARD_LABELS_PATH" ]]; then
        REWARD_ARGS+=(--reward-labels-path "$REWARD_LABELS_PATH")
    fi
    if [[ "$ASSUME_SUCCESS" == 1 ]]; then
        REWARD_ARGS+=(--assume-success)
    fi
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" scripts/check_lingdong_iql_data.py "${REWARD_ARGS[@]}"
fi

# Match reference SFT preflight: compute norms on CPU unless SKIP_NORM=1.
# Single-task presets share full-dataset norms with the two-task baseline.
# Whole-episode gripper events are shared across modes; relative statistics cover
# every chunk position with terminal padding. No videos are decoded. Even when
# skipping, require finite 8-D stats with the current source/preprocessing fingerprint.
NORM_ARGS=(
    --dataset-root "$DATA_ROOT"
    --action-representation "$ACTION_REPRESENTATION"
    --action-horizon "$ACTION_HORIZON"
)
if [[ "${SKIP_NORM:-0}" != "1" ]]; then
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" scripts/compute_lingdong_norm_stats.py "${NORM_ARGS[@]}"
fi
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PYTHON" scripts/compute_lingdong_norm_stats.py "${NORM_ARGS[@]}" --check
exec "$PYTHON" "${TRAIN_ARGS[@]}" "$@"
