#!/usr/bin/env bash
# --method sft | iql | cotrain-sft; extra trainer flags follow --.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_PYTHON="${SUBMIT_PYTHON:-/opt/conda/envs/rynnvalue_r1/bin/python}"
[[ -x "$SUBMIT_PYTHON" ]] || SUBMIT_PYTHON=python3
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
exec "$SUBMIT_PYTHON" "$ROOT/scripts/submit_pku_franka_pickplace_nebula.py" "$@"
