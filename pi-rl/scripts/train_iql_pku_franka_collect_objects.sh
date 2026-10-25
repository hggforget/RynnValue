#!/usr/bin/env bash
# Independent IQL: pi05_base policy initialization and random critic/value.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD=iql
export REPO_ID="${REPO_ID:-/oss/embodied-ai-data/pku/fr3_collect_objects_20261017_progress_expert_sup_step005000}"
# Ignore any SFT initialization inherited from an earlier shell command.
export INIT_CHECKPOINT=""

exec bash "$SCRIPT_DIR/train_pi05_pku_franka_collect_objects.sh" "$@"
