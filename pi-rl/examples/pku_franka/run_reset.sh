#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PI05_RUNTIME_DIR="${PI05_RUNTIME_DIR:-$PROJECT_ROOT/.runtime/pku_franka_collect_objects}"
export PYTHONPATH="$PROJECT_ROOT/packages/openpi-client/src:$PI05_RUNTIME_DIR/vendor${PYTHONPATH:+:$PYTHONPATH}"
ROBOT_RUNTIME="${ROBOT_RUNTIME:-$HOME/workspace/data_collection_web/scripts/run.sh}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == 1 ]]; then EXECUTE="${EXECUTE:-0}"; else EXECUTE="${EXECUTE:-1}"; fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]] || [[ "$EXECUTE" != 0 && "$EXECUTE" != 1 ]]; then
    echo 'EXECUTE and DRY_RUN must be 0 or 1' >&2; exit 2
fi
if [[ "$EXECUTE" == 1 && "$DRY_RUN" == 1 ]]; then
    echo 'Choose either EXECUTE=1 or DRY_RUN=1' >&2; exit 2
fi
RUN_DIR="${RUN_DIR:-$PI05_RUNTIME_DIR/results/$(date +%Y%m%d_%H%M%S)_reset_$$}"
mkdir -p "$RUN_DIR"
ARGS=(--namespace "${TARGET_NAMESPACE:-left}" --env-config "${ENV_CONFIG:-fr3_dual_cam_640x480}"
      --timeout "${RESET_TIMEOUT:-30}" --observation-timeout "${OBSERVATION_TIMEOUT:-10}"
      --report "$RUN_DIR/report.json")
if [[ "$EXECUTE" == 1 ]]; then
    COMMAND=(python -m openpi_client.franka_collect.test_supervisor
             --client-module openpi_client.franka_collect.reset --)
    ARGS+=(--execute)
else
    COMMAND=(python -m openpi_client.franka_collect.reset)
    ARGS+=(--dry-run-execution)
fi
printf 'Reset: XYZ [0.3550, 0.0277, 0.2356] m | Euler [-pi, 0, 0] | gripper open\n'
printf 'Execute: %s | Report: %s\nLog: %s\n' "$EXECUTE" "$RUN_DIR/report.json" "$RUN_DIR/client.log"
bash "$ROBOT_RUNTIME" "${COMMAND[@]}" "${ARGS[@]}" 2>&1 | (
    trap '' INT
    exec tee "$RUN_DIR/client.log"
)
