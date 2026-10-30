#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/oss/embodied-ai-data/pku/openpi_cache}"

DEFAULT_PYTHON=/opt/conda/envs/pi-rl/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
CHECKPOINT="${CHECKPOINT:-/oss/embodied-ai-data/lingdong/pi05_lingdong_right_arm/lingdong_right_arm_1027_163358/10000}"
# Keep the legacy default paired with the legacy default checkpoint. Crop-trained
# checkpoints must use their matching *_crop config.
POLICY_CONFIG="${POLICY_CONFIG:-pi05_lingdong_right_arm}"
INFERENCE_MODE="${INFERENCE_MODE:-standard}"
case "$INFERENCE_MODE" in
    standard|rtc) ;;
    *) echo 'INFERENCE_MODE must be standard or rtc' >&2; exit 1 ;;
esac

ARGS=(
    scripts/serve_policy.py
    --inference-mode "$INFERENCE_MODE"
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8000}"
    --dataset-root "${DATA_ROOT:-/oss/embodied-ai-data/lingdong}"
)
if [[ "$INFERENCE_MODE" == rtc ]]; then
    ARGS+=(
        --rtc-max-guidance-weight "${RTC_MAX_GUIDANCE_WEIGHT:-10}"
        --rtc-prefix-schedule "${RTC_PREFIX_SCHEDULE:-exp}"
    )
fi
ARGS+=(policy:checkpoint --policy.config "$POLICY_CONFIG" --policy.dir "$CHECKPOINT")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "$PYTHON" "${ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "$PYTHON" "${ARGS[@]}" "$@"
