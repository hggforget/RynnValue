#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=pick_up
exec bash "$SCRIPT_DIR/train_pi05_lingdong_right_arm.sh" "$@"
