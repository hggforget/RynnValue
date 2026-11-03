#!/usr/bin/env bash
# Three-source SFT: pick-and-place + collect-objects + all drawer tasks.
# Optional data overrides: PICKPLACE_REPO_ID, COLLECT_REPO_ID, DRAWER_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FR3_INCLUDE_DRAWER=1
export EXP_NAME="${EXP_NAME:-fr3_pickplace_collect_drawer_sft_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/train_sft_pku_franka_pickplace_collect_objects.sh" "$@"
