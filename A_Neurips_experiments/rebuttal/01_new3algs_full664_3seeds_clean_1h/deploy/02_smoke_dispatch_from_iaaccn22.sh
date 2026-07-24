#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
cd "$REMOTE_ROOT"

BATCH_ID="$(tr -d '\r\n' < "$BATCH_DIR/BATCH_NAME.txt")"
SMOKE_ID="${BATCH_ID}_smoke"
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1

jq -e '.ready_for_smoke == true' \
  "$BATCH_DIR/preflight/preflight_gate_summary.json" >/dev/null

python check/run_e1_candidate200_12alg_load_queue.py \
  --batch-name "$SMOKE_ID" \
  --source-csv "$BATCH_DIR/smoke/queues/smoke_2datasets_source.csv" \
  --expected-rows 2 \
  --queue-root "$BATCH_DIR/smoke/queues/load_queue_full" \
  --params-root "$BATCH_DIR/smoke/params" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --tools fepysr jaxsr symbolfit \
  --seeds 520 521 522 \
  --noise-sigmas 0 \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --max-jobs-per-host 3 \
  --max-new-jobs-per-host-per-poll 3 \
  --session-prefix neurips_rebuttal_new3_smoke_ \
  --host-session-count-prefix neurips_rebuttal_new3_smoke_ \
  --poll-seconds 30 \
  2>&1 | tee "$BATCH_DIR/deploy/smoke_dispatch_from_iaaccn22.log"

python benchmark-control/compliance/launchers/collect_remote_batch.py \
  --batch-dir "$BATCH_DIR/smoke" \
  --batch-id "$SMOKE_ID" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --controller-host iaaccn22 \
  --use-internal-ips

HARVEST_ARGS=()
for host in iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29; do
  HARVEST_ARGS+=(
    --experiment-root
    "$BATCH_DIR/smoke/remote-experiments/$host"
  )
done

python benchmark-control/compliance/launchers/harvest_batch.py \
  --batch-dir "$BATCH_DIR/smoke" \
  "${HARVEST_ARGS[@]}"

python benchmark-control/compliance/launchers/audit_batch.py \
  --batch-dir "$BATCH_DIR/smoke" \
  --write-heartbeat \
  --write-rerun \
  --round-id 1

python benchmark-control/compliance/launchers/check_audit_success.py \
  --batch-dir "$BATCH_DIR/smoke" \
  --expected-total-tasks 18
