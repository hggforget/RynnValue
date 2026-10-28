#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize from OSS weights; let the normal launcher name and save the new run.
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-/oss/embodied-ai-data/lingdong/pi05_lingdong_right_arm/lingdong_right_arm_1027_163358/10000}"

exec bash "$SCRIPT_DIR/train_pi05_lingdong_right_arm.sh" "$@"
