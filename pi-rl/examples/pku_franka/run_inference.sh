#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PI05_RUNTIME_DIR="${PI05_RUNTIME_DIR:-$PROJECT_ROOT/.runtime/pku_franka_collect_objects}"
VENDOR_DIR="${PI05_VENDOR_DIR:-$PI05_RUNTIME_DIR/vendor}"
export PYTHONPATH="$PROJECT_ROOT/packages/openpi-client/src:$VENDOR_DIR${PYTHONPATH:+:$PYTHONPATH}"
ROBOT_RUNTIME="${ROBOT_RUNTIME:-$HOME/workspace/data_collection_web/scripts/run.sh}"
if [[ -n "${PYTHON:-}" ]]; then
    exec "$PYTHON" -m openpi_client.franka_collect.client "$@"
elif [[ -f "$ROBOT_RUNTIME" ]]; then
    exec bash "$ROBOT_RUNTIME" python -m openpi_client.franka_collect.client "$@"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    exec "$PROJECT_ROOT/.venv/bin/python" -m openpi_client.franka_collect.client "$@"
else
    exec python3 -m openpi_client.franka_collect.client "$@"
fi
