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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

conda run -n sim_base python check/launch_e1_benchmark.py run \
  --tool pyoperon \
  --slice-csv "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/probe4_full664_v1/slices/pyoperon/seed521/iaaccn24.csv" \
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params/pyoperon.json" \
  --output-root "$REMOTE_ROOT/experiments/${BATCH_NAME}/pyoperon/seed521/iaaccn24" \
  --seed 521 \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"
