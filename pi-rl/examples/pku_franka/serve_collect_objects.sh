#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
MODEL_ROOT="${MODEL_ROOT:-$HOME/MODELS/pi05-pku-franka-collect-objects}"
MODE="${ACTION_REPRESENTATION:-absolute}"
case "$MODE" in
    absolute)
        CONFIG=pi05_pku_franka_collect_objects
        EXPERIMENT=fr3_collect_objects_20261017_1020_190034
        DEFAULT_PORT=8000
        ;;
    relative)
        CONFIG=pi05_pku_franka_collect_objects_relative
        EXPERIMENT=fr3_collect_objects_20261017_1021_010646
        DEFAULT_PORT=8001
        ;;
    *) echo 'ACTION_REPRESENTATION must be absolute or relative' >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/$CONFIG/$EXPERIMENT/19999}"
cd "$PROJECT_ROOT"
exec "${PYTHON:-$PROJECT_ROOT/.venv/bin/python}" scripts/serve_policy.py \
    --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-$DEFAULT_PORT}" \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CHECKPOINT" "$@"
