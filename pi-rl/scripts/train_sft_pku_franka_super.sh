#!/usr/bin/env bash
# Five-source SFT: pickplace + collect-objects + original/new drawer + new plate.
# Data overrides: PICKPLACE_REPO_ID, COLLECT_REPO_ID, DRAWER_ROOT,
# EXTRA_DRAWER_ROOT, PLATE_ROOT. GPU/Python discovery is automatic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FR3_INCLUDE_DRAWER=1
export FR3_SUPER_COTRAIN=1
export EXP_NAME="${EXP_NAME:-fr3_super_sft_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/train_sft_pku_franka_pickplace_collect_objects.sh" "$@"
