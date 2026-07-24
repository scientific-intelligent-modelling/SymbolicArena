#!/usr/bin/env bash
set -uo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
cd "$REMOTE_ROOT"

RSYNC_FILTERS=(
  "--exclude=.git/"
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=*.pyo"
  "--exclude=$BATCH_DIR/audit/"
  "--exclude=$BATCH_DIR/collect/"
  "--exclude=$BATCH_DIR/harvest/"
  "--exclude=$BATCH_DIR/runs/"
  "--exclude=$BATCH_DIR/remote-experiments/"
  "--exclude=$BATCH_DIR/queues/load_queue_*/"
  "--exclude=$BATCH_DIR/smoke/audit/"
  "--exclude=$BATCH_DIR/smoke/collect/"
  "--exclude=$BATCH_DIR/smoke/harvest/"
  "--exclude=$BATCH_DIR/smoke/runs/"
  "--exclude=$BATCH_DIR/smoke/remote-experiments/"
  "--exclude=$BATCH_DIR/smoke/queues/load_queue_*/"
  "--exclude=$BATCH_DIR/deploy/*.log"
  "--exclude=$BATCH_DIR/deploy/preflight_*.json"
)

SYNC_ITEMS=(
  "check/run_e1_candidate200_12alg_load_queue.py"
  "check/launch_e1_benchmark.py"
  "scientific_intelligent_modelling/"
  "benchmark-control/compliance/"
  "A_Neurips_experiments/README.md"
  "A_Neurips_experiments/rebuttal/"
)

failures=0
for target in \
  10.10.100.23 10.10.100.24 10.10.100.25 10.10.100.26 \
  10.10.100.27 10.10.100.28 10.10.100.29
do
  echo "[fanout] iaaccn22 -> $target"
  if ! timeout 900 rsync -aR "${RSYNC_FILTERS[@]}" "${SYNC_ITEMS[@]}" \
    "$target:$REMOTE_ROOT/"; then
    echo "SYNC_FAIL $target"
    failures=$((failures + 1))
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "[fanout] failed targets: $failures"
  exit 1
fi

echo "[fanout] all internal targets synchronized"
