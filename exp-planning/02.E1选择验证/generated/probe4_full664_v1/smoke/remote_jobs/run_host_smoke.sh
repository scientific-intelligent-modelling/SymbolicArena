#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <host-label> <batch-name>" >&2
  exit 2
fi

HOST_LABEL="$1"
BATCH_NAME="$2"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"
ASSET_ROOT="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/probe4_full664_v1/smoke"
SEED="990"
LOG_ROOT="$REMOTE_ROOT/experiments/${BATCH_NAME}/__smoke_logs/${HOST_LABEL}"
mkdir -p "$LOG_ROOT"
cd "$REMOTE_ROOT"
export PYTHONPATH=.

run_tool() {
  local tool="$1"
  local tool_arg="$2"
  local env_name="$3"
  local slice_csv="$ASSET_ROOT/slices/${HOST_LABEL}.csv"
  local params_json="$ASSET_ROOT/params/${tool}.json"
  local out_root="$REMOTE_ROOT/experiments/${BATCH_NAME}/${tool}/${HOST_LABEL}"
  echo "[start] ${HOST_LABEL} ${tool}"
  conda run -n "$env_name" python check/launch_e1_benchmark.py run \
    --tool "$tool_arg" \
    --slice-csv "$slice_csv" \
    --params-json "$params_json" \
    --output-root "$out_root" \
    --seed "$SEED" \
    --workers 1 \
    --retry-failed > "$LOG_ROOT/${tool}.log" 2>&1 &
}

run_tool "udsr" "udsr" "sim_dso"
run_tool "dso" "dso" "sim_dso"
run_tool "imcts" "iMCTS" "sim_iMCTS"
run_tool "pyoperon" "pyoperon" "sim_base"
wait
echo "[done] ${HOST_LABEL} smoke"
