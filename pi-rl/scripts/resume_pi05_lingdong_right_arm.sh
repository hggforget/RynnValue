#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export CKPT_BASE_DIR="${CKPT_BASE_DIR:-../output/checkpoints}"
export EXP_NAME="${EXP_NAME:-lingdong_right_arm_1027_163358}"
# Keep the staging location consistent with forwarded training CLI overrides.
CLI_ARGS=("$@")
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
    case "${CLI_ARGS[i]}" in
        --checkpoint-base-dir) CKPT_BASE_DIR="${CLI_ARGS[i+1]:?Missing --checkpoint-base-dir value}" ;;
        --checkpoint-base-dir=*) CKPT_BASE_DIR="${CLI_ARGS[i]#*=}" ;;
        --exp-name) EXP_NAME="${CLI_ARGS[i+1]:?Missing --exp-name value}" ;;
        --exp-name=*) EXP_NAME="${CLI_ARGS[i]#*=}" ;;
    esac
done
: "${CKPT_BASE_DIR:?CKPT_BASE_DIR must not be empty}"
: "${EXP_NAME:?EXP_NAME must not be empty}"
CHECKPOINT_DIR="$CKPT_BASE_DIR/pi05_lingdong_right_arm/$EXP_NAME/10000"
OSS_CKPT_BASE_DIR="${OSS_CKPT_BASE_DIR:-/oss/embodied-ai-data/lingdong}"
SOURCE_CHECKPOINT_DIR="$OSS_CKPT_BASE_DIR/pi05_lingdong_right_arm/$EXP_NAME/10000"

case "$CKPT_BASE_DIR" in
    /oss|/oss/*|*://*)
        echo "CKPT_BASE_DIR must be a local filesystem path: $CKPT_BASE_DIR" >&2
        exit 1
        ;;
esac

checkpoint_ready() {
    [[ -f "$1/train_state/_METADATA" && -f "$1/params/_METADATA" &&
       -f "$1/assets/lingdong_right_arm/norm_stats.json" && -f "$1/_CHECKPOINT_METADATA" ]]
}

if ! checkpoint_ready "$CHECKPOINT_DIR"; then
    if [[ -e "$CHECKPOINT_DIR" ]]; then
        echo "Local checkpoint is incomplete: $CHECKPOINT_DIR; choose another CKPT_BASE_DIR." >&2
        exit 1
    fi
    if ! checkpoint_ready "$SOURCE_CHECKPOINT_DIR"; then
        echo "Complete step-10000 checkpoint not found at $SOURCE_CHECKPOINT_DIR" >&2
        exit 1
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "Would copy OSS checkpoint: $SOURCE_CHECKPOINT_DIR -> $CHECKPOINT_DIR"
    else
        # Publish the local step directory only after all checkpoint files arrive.
        STAGING_DIR="${CHECKPOINT_DIR}.download"
        mkdir -p "$STAGING_DIR"
        echo "Copying OSS checkpoint: $SOURCE_CHECKPOINT_DIR -> $CHECKPOINT_DIR"
        cp -R "$SOURCE_CHECKPOINT_DIR/params" "$SOURCE_CHECKPOINT_DIR/train_state" \
            "$SOURCE_CHECKPOINT_DIR/assets" "$STAGING_DIR/"
        cp "$SOURCE_CHECKPOINT_DIR/_CHECKPOINT_METADATA" "$STAGING_DIR/"
        mv -T "$STAGING_DIR" "$CHECKPOINT_DIR"
    fi
fi

# Resume locally to 20,000 total steps; future checkpoints stay under CKPT_BASE_DIR.
exec bash "$SCRIPT_DIR/train_pi05_lingdong_right_arm.sh" \
    --resume --num-train-steps 20000 "$@"
