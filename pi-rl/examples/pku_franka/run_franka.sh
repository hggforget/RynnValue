#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Franka 端常用参数：直接修改这里，也可以用同名环境变量覆盖。
MODE="${MODE:-inference}"                       # inference / replay
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-absolute}"  # absolute / relative
INFERENCE_HOST="${INFERENCE_HOST:-219.223.195.77}"  # 4090 服务器 IP
REPEAT="${REPEAT:-100}"                         # 推理请求次数
TARGET_NAMESPACE="${TARGET_NAMESPACE:-left}"
ENV_CONFIG="${ENV_CONFIG:-fr3_dual_cam_640x480}"
OBSERVATION_TIMEOUT="${OBSERVATION_TIMEOUT:-10}"
EXECUTE="${EXECUTE:-0}"                         # 1=执行预测动作，0=仅预测
DRY_RUN="${DRY_RUN:-0}"                         # 1=规划执行目标，但不发送控制命令
EXECUTE_STEPS="${EXECUTE_STEPS:-10}"             # 每次预测执行前多少步，再读取新观测
CONTROL_HZ="${CONTROL_HZ:-30}"
PROMPT="${PROMPT:-place each object on the plate that matches its color.}"
DATASET="${DATASET:-$HOME/workspace/DATA/local/fr3_collect_objects_20261017}"
EPISODE="${EPISODE:-1}"
FRAME="${FRAME:-0}"
FRAME_STEP="${FRAME_STEP:-1}"
export PI05_RUNTIME_DIR="${PI05_RUNTIME_DIR:-$PROJECT_ROOT/.runtime/pku_franka_collect_objects}"

case "${1:-}" in
    inference|replay) MODE="$1"; shift ;;
    -h|--help)
        cat <<'HELP'
用法：bash examples/pku_franka/run_franka.sh [inference|replay]

不带参数：实时观测推理，默认绝对动作模型、100 次请求。
replay：回放数据集观测，默认轨迹 1、从第 0 帧开始、100 帧。
常用配置集中在脚本顶部；ACTION_REPRESENTATION=relative 切换相对模型。
默认直接连接 219.223.195.77:8000，相对模型使用 8001。
每次运行的动作和报告保存到独立目录，启动时会打印路径。
实时模式需要机器人和相机驱动已经启动。默认只预测动作。
EXECUTE=1 启用机械臂和夹爪控制；DRY_RUN=1 仅规划执行目标。
EXECUTE_STEPS 控制每次执行的步数，CONTROL_HZ 控制动作发布频率。
replay 始终只回放观测，不能启用动作执行。
HELP
        exit 0
        ;;
esac
if (( $# > 0 )); then
    echo 'Configure run_franka.sh through its settings or environment variables; use run_inference.sh for full CLI options' >&2
    exit 2
fi

case "$ACTION_REPRESENTATION" in
    absolute) DEFAULT_PORT=8000 ;;
    relative) DEFAULT_PORT=8001 ;;
    *) echo 'ACTION_REPRESENTATION must be absolute or relative' >&2; exit 2 ;;
esac
PORT="${PORT:-$DEFAULT_PORT}"
ARGS=(--mode "$MODE" --host "$INFERENCE_HOST" --port "$PORT"
      --action-representation "$ACTION_REPRESENTATION" --repeat "$REPEAT" --prompt "$PROMPT")
case "$MODE" in
    inference)
        ARGS+=(--namespace "$TARGET_NAMESPACE" --env-config "$ENV_CONFIG"
               --observation-timeout "$OBSERVATION_TIMEOUT"
               --execute-steps "$EXECUTE_STEPS" --control-hz "$CONTROL_HZ")
        ;;
    replay)
        ARGS+=(--dataset "$DATASET" --episode "$EPISODE" --frame "$FRAME" --frame-step "$FRAME_STEP")
        ;;
    *) echo 'MODE must be inference or replay' >&2; exit 2 ;;
esac
if [[ "$EXECUTE" != 0 && "$EXECUTE" != 1 ]] || [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo 'EXECUTE and DRY_RUN must be 0 or 1' >&2; exit 2
fi
if [[ "$EXECUTE" == 1 && "$DRY_RUN" == 1 ]]; then
    echo 'Choose either EXECUTE=1 or DRY_RUN=1' >&2; exit 2
fi
if [[ "$MODE" == replay && ( "$EXECUTE" == 1 || "$DRY_RUN" == 1 ) ]]; then
    echo 'replay only reads recorded observations; action execution requires inference' >&2; exit 2
fi
if [[ "$EXECUTE" == 1 ]]; then ARGS+=(--execute); fi
if [[ "$DRY_RUN" == 1 ]]; then ARGS+=(--dry-run-execution); fi

RUN_DIR="${RUN_DIR:-$PI05_RUNTIME_DIR/results/$(date +%Y%m%d_%H%M%S)_${MODE}_${ACTION_REPRESENTATION}_$$}"
mkdir -p "$RUN_DIR"
printf 'Mode: %s | Model: %s | Server: ws://%s:%s\n' "$MODE" "$ACTION_REPRESENTATION" "$INFERENCE_HOST" "$PORT"
printf 'Execute: %s | Dry run: %s | Steps/query: %s | Control: %s Hz\n' "$EXECUTE" "$DRY_RUN" "$EXECUTE_STEPS" "$CONTROL_HZ"
printf 'Actions: %s\nReport: %s\n' "$RUN_DIR/actions.npy" "$RUN_DIR/report.json"
printf 'Log: %s\n' "$RUN_DIR/client.log"
bash "$PROJECT_ROOT/examples/pku_franka/run_inference.sh" "${ARGS[@]}" \
    --output "$RUN_DIR/actions.npy" --report "$RUN_DIR/report.json" 2>&1 | tee "$RUN_DIR/client.log"
