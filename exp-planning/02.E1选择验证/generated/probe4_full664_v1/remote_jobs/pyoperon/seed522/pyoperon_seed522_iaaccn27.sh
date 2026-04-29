#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <BATCH_NAME> <WORKERS> [retry]" >&2
  exit 2
fi

BATCH_NAME="$1"
WORKERS="$2"
RETRY_MODE="${3:-}"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"
EXTRA_ARGS=()
if [ "$RETRY_MODE" = "retry" ]; then
  EXTRA_ARGS+=(--retry-failed)
fi

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n sim_base python check/launch_e1_benchmark.py run \
  --tool pyoperon \
  --slice-csv "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/probe4_full664_v1/slices/pyoperon/seed522/iaaccn27.csv" \
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params/pyoperon.json" \
  --output-root "$REMOTE_ROOT/experiments/${BATCH_NAME}/pyoperon/seed522/iaaccn27" \
  --seed 522 \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"
