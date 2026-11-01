#!/usr/bin/env bash
# Install the training dependencies validated in the local pi-rl environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
"$PYTHON" -m pip install -r scripts/requirements-nebula.txt
LEROBOT_ROOT="${LEROBOT_ROOT:-/public/damo_rynnvla_data_zjk/RynnValue/third_party/lerobot}"
[[ -f "$LEROBOT_ROOT/pyproject.toml" ]] || { echo "LeRobot source missing: $LEROBOT_ROOT" >&2; exit 1; }
"$PYTHON" -m pip install --no-deps "$LEROBOT_ROOT"
"$PYTHON" -m pip install --no-deps -e packages/openpi-client -e third_party/jaxrl2 -e .
PYTHONPATH="src:third_party/jaxrl2:packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c 'from openpi.training import config, data_loader, rl_data_loader; import jaxrl2; print("Training imports passed")'
