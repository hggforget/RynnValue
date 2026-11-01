#!/usr/bin/env bash
# Run on a connected Python 3.10 Linux host, then publish the directory to OSS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
WHEELHOUSE="${1:?Usage: prepare_nebula_wheelhouse.sh /path/to/wheels}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Nebula requires Python 3.10 wheels"'
mkdir -p "$WHEELHOUSE"
"$PYTHON" -m pip wheel --disable-pip-version-check --progress-bar off --prefer-binary \
    --wheel-dir "$WHEELHOUSE" -r scripts/requirements-nebula.txt \
    hatchling setuptools wheel editables hatch-vcs setuptools-scm poetry-core pip==24.0
