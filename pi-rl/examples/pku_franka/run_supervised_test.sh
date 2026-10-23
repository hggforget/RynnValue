#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PI05_RUNTIME_DIR="${PI05_RUNTIME_DIR:-$PROJECT_ROOT/.runtime/pku_franka_collect_objects}"
export PYTHONPATH="$PROJECT_ROOT/packages/openpi-client/src:$PI05_RUNTIME_DIR/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec bash "${ROBOT_RUNTIME:-$HOME/workspace/data_collection_web/scripts/run.sh}" \
    python -m openpi_client.franka_collect.test_supervisor -- "$@"
