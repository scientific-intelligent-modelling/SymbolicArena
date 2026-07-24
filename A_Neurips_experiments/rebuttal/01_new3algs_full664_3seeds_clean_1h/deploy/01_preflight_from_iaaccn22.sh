#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
cd "$REMOTE_ROOT"

BATCH_ID="$(tr -d '\r\n' < "$BATCH_DIR/BATCH_NAME.txt")"
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1

python check/run_e1_candidate200_12alg_load_queue.py \
  --batch-name "${BATCH_ID}_preflight" \
  --source-csv "$BATCH_DIR/queues/full664_source.csv" \
  --expected-rows 664 \
  --queue-root "$BATCH_DIR/queues/load_queue_full" \
  --params-root "$BATCH_DIR/params" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --tools fepysr jaxsr symbolfit \
  --seeds 520 521 522 \
  --noise-sigmas 0 \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --preflight-only \
  --preflight-host-timeout 900 \
  --preflight-report "$BATCH_DIR/deploy/preflight_8hosts_full664.json" \
  2>&1 | tee "$BATCH_DIR/deploy/preflight_8hosts_full664.log"

python benchmark-control/compliance/launchers/check_preflight_report.py \
  --batch-dir "$BATCH_DIR" \
  --report "$BATCH_DIR/deploy/preflight_8hosts_full664.json" \
  --expected-hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29
