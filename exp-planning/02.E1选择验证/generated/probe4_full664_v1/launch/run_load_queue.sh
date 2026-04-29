#!/usr/bin/env bash
set -euo pipefail

BATCH_NAME="${1:-probe4_full664_load_queue_$(date +%Y%m%d-%H%M%S)}"
if [ "$#" -gt 0 ]; then
  shift
fi

REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"
LOG_DIR="$REMOTE_ROOT/experiments/${BATCH_NAME}"
mkdir -p "$LOG_DIR"

cd "$REMOTE_ROOT"
echo "BATCH_NAME=${BATCH_NAME}"
echo "LOG=${LOG_DIR}/queue_scheduler.log"

PYTHONPATH=. conda run -n sim_base python check/run_probe4_full664_load_queue.py \
  --batch-name "$BATCH_NAME" \
  "$@" | tee "$LOG_DIR/queue_scheduler.log"
