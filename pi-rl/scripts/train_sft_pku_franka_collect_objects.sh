#!/usr/bin/env bash
# SFT with the expert dataset and the revised colored-block instruction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD=sft
export REPO_ID="${REPO_ID:-/oss/embodied-ai-data/pku/fr3_collect_objects_20261017_progress_expert_sup_step005000}"

exec bash "$SCRIPT_DIR/train_pi05_pku_franka_collect_objects.sh" "$@"
