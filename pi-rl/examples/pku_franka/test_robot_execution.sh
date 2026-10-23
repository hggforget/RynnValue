#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-$PROJECT_ROOT/.venv/bin/python}" -m openpi_client.franka_collect.action_test "$@"
