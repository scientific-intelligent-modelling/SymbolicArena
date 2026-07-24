#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
cd "$REMOTE_ROOT"

BATCH_ID="$(tr -d '\r\n' < "$BATCH_DIR/BATCH_NAME.txt")"
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1

jq -e '.audit_passed == true and .expected_total_tasks == 18' \
  "$BATCH_DIR/smoke/audit/audit_gate_summary.json" >/dev/null

echo "[full] launching $BATCH_ID: 5976 tasks on iaaccn22~29"

python check/run_e1_candidate200_12alg_load_queue.py \
  --batch-name "$BATCH_ID" \
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
  --max-jobs-per-host 70 \
  --max-new-jobs-per-host-per-poll 5 \
  --session-prefix neurips_rebuttal_new3_1h_ \
  --host-session-count-prefix neurips_rebuttal_new3_1h_ \
  --poll-seconds 60 \
  2>&1 | tee "$BATCH_DIR/deploy/full_dispatch_from_iaaccn22.log"
