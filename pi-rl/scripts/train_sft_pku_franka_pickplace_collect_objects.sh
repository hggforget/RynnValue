#!/usr/bin/env bash
# Joint SFT for pick-and-place and collect-objects (30 Hz, absolute actions).
# Run from any directory:
#   GPUS=0 bash /path/to/pi-rl/scripts/train_sft_pku_franka_pickplace_collect_objects.sh
#   GPUS=0,1 BATCH_SIZE_PER_GPU=16 DRY_RUN=1 bash scripts/train_sft_pku_franka_pickplace_collect_objects.sh
#   EXP_NAME=existing_experiment SKIP_NORM=1 bash scripts/train_sft_pku_franka_pickplace_collect_objects.sh --resume
# Extra arguments are forwarded to train.py; use PICKPLACE_REPO_ID / COLLECT_REPO_ID to override both passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/jaxrl2${PYTHONPATH:+:$PYTHONPATH}"

check_python() {
    "$1" - <<'PY'
import importlib.util

missing = [name for name in ("numpy", "jax", "flax", "torch", "lerobot", "tyro", "av") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing dependencies: {', '.join(missing)}")
try:
    compatible_lerobot = importlib.util.find_spec("lerobot.common.datasets.lerobot_dataset") is not None
except (ImportError, AttributeError, ValueError):
    compatible_lerobot = False
if not compatible_lerobot:
    raise SystemExit("requires LeRobot with lerobot.common.datasets (see pi-rl/pyproject.toml)")
PY
}

if [[ -n "${PYTHON:-}" ]]; then
    # An explicit override must not silently launch a different environment.
    if ! problem=$(check_python "$PYTHON" 2>&1); then
        echo "Requested Python is not a usable training environment: $PYTHON ($problem)" >&2
        exit 1
    fi
else
    PYTHON_CANDIDATES=()
    add_python() {
        local candidate="${1:-}" previous
        [[ -n "$candidate" ]] || return 0
        if [[ ! -x "$candidate" ]]; then
            candidate=$(command -v "$candidate" || true)
        fi
        [[ -n "$candidate" && -x "$candidate" ]] || return 0
        for previous in "${PYTHON_CANDIDATES[@]}"; do
            [[ "$previous" != "$candidate" ]] || return 0
        done
        PYTHON_CANDIDATES+=("$candidate")
    }
    [[ -z "${VIRTUAL_ENV:-}" ]] || add_python "$VIRTUAL_ENV/bin/python"
    [[ -z "${CONDA_PREFIX:-}" ]] || add_python "$CONDA_PREFIX/bin/python"
    add_python "$PROJECT_ROOT/.venv/bin/python"
    add_python python
    add_python python3

    # Discover named environments even when conda has not been activated.
    CONDA_ROOTS=()
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        CONDA_ROOTS+=("${CONDA_PREFIX%%/envs/*}")
    fi
    conda_executable="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ "$conda_executable" == */bin/conda ]]; then
        CONDA_ROOTS+=("${conda_executable%/bin/conda}")
    fi
    for name in python python3; do
        path_python=$(command -v "$name" || true)
        if [[ "$path_python" == */bin/python* ]]; then
            CONDA_ROOTS+=("${path_python%/bin/python*}")
        fi
    done
    CONDA_ROOTS+=(/opt/conda /opt/miniconda3 /opt/anaconda3)
    if [[ -n "${HOME:-}" ]]; then
        CONDA_ROOTS+=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge")
    fi
    for conda_root in "${CONDA_ROOTS[@]}"; do
        add_python "$conda_root/envs/pi-rl/bin/python"
        add_python "$conda_root/envs/pku-pi-rl/bin/python"
    done
    if [[ -n "${HOME:-}" && -f "$HOME/.conda/environments.txt" ]]; then
        while IFS= read -r conda_environment || [[ -n "$conda_environment" ]]; do
            [[ -z "$conda_environment" ]] || add_python "$conda_environment/bin/python"
        done < "$HOME/.conda/environments.txt"
    fi

    PYTHON=""
    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if problem=$(check_python "$candidate" 2>&1); then
            PYTHON="$candidate"
            break
        fi
        echo "Skipping Python: $candidate ($problem)" >&2
    done
    if [[ -z "$PYTHON" ]]; then
        echo "No compatible training Python found. Install the pi-rl dependencies in a virtualenv/Conda environment, or set PYTHON=/path/to/training/env/bin/python." >&2
        exit 1
    fi
fi
CONFIG=pi05_pku_franka_pickplace_collect_objects
ASSET_ID=pku_franka_pickplace_collect_objects
PICKPLACE_REPO_ID="${PICKPLACE_REPO_ID:-/oss/embodied-ai-data/pku/fr3_real_pickplace_20261027_20261030_merged}"
COLLECT_REPO_ID="${COLLECT_REPO_ID:-/oss/embodied-ai-data/pku/fr3_collect_objects_20261017_progress_expert_sup_step005000}"
EXP_NAME="${EXP_NAME:-fr3_pickplace_collect_sft_$(date +%Y%m%d_%H%M%S)}"
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
            echo "Use PICKPLACE_REPO_ID and COLLECT_REPO_ID to select datasets; this launcher fixes the adapter and norm assets for both passes." >&2
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
for dataset_path in "$PICKPLACE_REPO_ID" "$COLLECT_REPO_ID"; do
    if [[ ! -f "$dataset_path/meta/info.json" ]]; then
        echo "Dataset is not mounted at $dataset_path; set PICKPLACE_REPO_ID / COLLECT_REPO_ID to local paths." >&2
        exit 1
    fi
done
if [[ ! -d "$WEIGHT_PARAMS_PATH" ]]; then
    echo "Base weights are not mounted at $WEIGHT_PARAMS_PATH; set WEIGHT_PARAMS_PATH to pi05_base/params." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# Reuse the mounted tokenizer cache together with the local pi05_base weights.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/oss/embodied-ai-data/pku/openpi_cache}"
NORM_ARGS=(scripts/compute_norm_stats_fast.py --config-name "$CONFIG" --repo-ids "$PICKPLACE_REPO_ID" "$COLLECT_REPO_ID" --num-workers 0 --batch-size 1)
TRAIN_ARGS=(
    scripts/train.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --data.repo-ids "$PICKPLACE_REPO_ID" "$COLLECT_REPO_ID"
    --checkpoint-base-dir "$CKPT_BASE_DIR"
    --weight-loader.params-path "$WEIGHT_PARAMS_PATH"
    --fsdp-devices "$FSDP_DEVICES"
    --batch-size "$BATCH_SIZE"
    "$@"
)
echo "Starting $CONFIG: GPUs=$GPUS, FSDP=$FSDP_DEVICES, global_batch=$BATCH_SIZE, experiment=$EXP_NAME"
echo "Python: $PYTHON"
echo "Pick-and-place: $PICKPLACE_REPO_ID (all episodes, per-task prompts)"
echo "Collect-objects: $COLLECT_REPO_ID (existing SFT instruction and episode exclusions)"
echo "Joint SFT: sample uniformly over eligible frames; shared state/action normalization."
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if [[ "${SKIP_NORM:-0}" != "1" ]]; then
        printf '%q ' env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false "$PYTHON" "${NORM_ARGS[@]}"
        printf '\n'
    fi
    printf '%q ' "$PYTHON" "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

# Recompute shared stats with both datasets and the same episode filters; batch 1 includes the final frame.
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
