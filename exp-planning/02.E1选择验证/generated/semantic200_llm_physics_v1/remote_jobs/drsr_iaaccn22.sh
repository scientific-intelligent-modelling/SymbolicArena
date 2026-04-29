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
if [ "${CONFIRM_SEMANTIC200_LLM_PHYSICS:-}" != "semantic200_llm_physics_v1" ]; then
  echo "Refusing to launch drsr/iaaccn22: export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v1 after explicit user confirmation." >&2
  exit 3
fi

if [ "$RETRY_MODE" = "retry" ]; then
  EXTRA_ARGS+=(--retry-failed)
fi

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n sim_llm python check/launch_e1_benchmark.py run \
  --tool drsr \
  --slice-csv "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v1/slices/iaaccn22.csv" \
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v1/params/drsr_semantic.json" \
  --output-root "$REMOTE_ROOT/experiments/${BATCH_NAME}/drsr/iaaccn22" \
  --seed 1314 \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"
