#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
cd "$REMOTE_ROOT"

BATCH_ID="$(tr -d '\r\n' < "$BATCH_DIR/BATCH_NAME.txt")"
STATE_SUMMARY="$BATCH_DIR/queues/load_queue_full/state/$BATCH_ID.latest.json"
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1

jq -e '
  ((.task_states.done // 0) == 5976)
  and ((.task_states.pending // 0) == 0)
  and ((.task_states.running // 0) == 0)
  and ([.task_states[]] | add == 5976)
' "$STATE_SUMMARY" >/dev/null

python benchmark-control/compliance/launchers/collect_remote_batch.py \
  --batch-dir "$BATCH_DIR" \
  --batch-id "$BATCH_ID" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --controller-host iaaccn22 \
  --use-internal-ips

HARVEST_ARGS=()
for host in iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29; do
  HARVEST_ARGS+=(
    --experiment-root
    "$BATCH_DIR/remote-experiments/$host"
  )
done

python benchmark-control/compliance/launchers/harvest_batch.py \
  --batch-dir "$BATCH_DIR" \
  "${HARVEST_ARGS[@]}"

python benchmark-control/compliance/launchers/audit_batch.py \
  --batch-dir "$BATCH_DIR" \
  --write-heartbeat \
  --write-rerun \
  --round-id 1

python benchmark-control/compliance/launchers/check_audit_success.py \
  --batch-dir "$BATCH_DIR" \
  --expected-total-tasks 5976

python check/analyze_neurips_rebuttal_full664.py \
  --batch-dir "$BATCH_DIR" \
  --stage3-run-level \
  "A_Neurips_experiments/stage3_664dats_4probes_3seeds_1h/probe4_postprocess_run_level.csv" \
  --output-dir "$BATCH_DIR/analysis"

bash "$BATCH_DIR/deploy/05_generate_symf_from_iaaccn22.sh"
