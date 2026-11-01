#!/usr/bin/env bash
# Install the training dependencies validated in the local pi-rl environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
read -r REQUIREMENTS_SHA _ < <(sha256sum scripts/requirements-nebula.txt)
SHARED_WHEELHOUSE="/oss/embodied-ai-data/lingdong/nebula_wheels/pi_rl_py310_${REQUIREMENTS_SHA:0:12}"
if [[ ! -v NEBULA_WHEELHOUSE && -f "$SHARED_WHEELHOUSE/manifest.json" ]]; then
    export NEBULA_WHEELHOUSE="$SHARED_WHEELHOUSE"
fi
if [[ -n "${NEBULA_WHEELHOUSE:-}" ]]; then
    [[ -d "$NEBULA_WHEELHOUSE" ]] || { echo "Wheelhouse missing: $NEBULA_WHEELHOUSE" >&2; exit 1; }
    export PIP_NO_INDEX=1
    export PIP_FIND_LINKS="$NEBULA_WHEELHOUSE"
    echo "Installing from offline wheelhouse: $NEBULA_WHEELHOUSE"
else
    # Public mirrors may be unreachable from Nebula's worker network.
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.antfin-inc.com/simple/}"
    "$PYTHON" scripts/check_nebula_pip_index.py
fi
"$PYTHON" -m pip install -r scripts/requirements-nebula.txt
LEROBOT_ROOT="${LEROBOT_ROOT:-/public/damo_rynnvla_data_zjk/RynnValue/third_party/lerobot}"
if [[ -n "${NEBULA_WHEELHOUSE:-}" && -f "$NEBULA_WHEELHOUSE/lerobot-0.1.0-py3-none-any.whl" ]]; then
    "$PYTHON" -m pip install --no-deps --force-reinstall "$NEBULA_WHEELHOUSE/lerobot-0.1.0-py3-none-any.whl"
elif [[ -f "$LEROBOT_ROOT/pyproject.toml" ]]; then
    "$PYTHON" -m pip install --no-deps "$LEROBOT_ROOT"
else
    echo "LeRobot unavailable: include its wheel in NEBULA_WHEELHOUSE or set LEROBOT_ROOT to its source." >&2
    exit 1
fi
"$PYTHON" -m pip install --no-deps -e packages/openpi-client -e third_party/jaxrl2 -e .
PYTHONPATH="src:third_party/jaxrl2:packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c 'from openpi.training import config, data_loader, rl_data_loader; import jaxrl2; print("Training imports passed")'
