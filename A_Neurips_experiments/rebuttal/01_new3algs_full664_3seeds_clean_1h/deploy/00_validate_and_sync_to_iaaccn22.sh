#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
BATCH_ID="$(tr -d '\r\n' < "$BATCH_DIR/BATCH_NAME.txt")"
QUEUE_ROOT="$BATCH_DIR/queues/load_queue_full"

pytest -q tests/test_prepare_neurips_rebuttal_full664.py

test "$(awk 'END {print NR - 1}' "$BATCH_DIR/manifest/datasets.csv")" -eq 664
test "$(awk 'END {print NR - 1}' "$BATCH_DIR/manifest/tasks.csv")" -eq 5976
test "$(awk 'END {print NR - 1}' "$BATCH_DIR/smoke/manifest/tasks.csv")" -eq 18

python check/run_e1_candidate200_12alg_load_queue.py \
  --batch-name "$BATCH_ID" \
  --source-csv "$BATCH_DIR/queues/full664_source.csv" \
  --expected-rows 664 \
  --queue-root "$QUEUE_ROOT" \
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
  --dry-run \
  2>&1 | tee "$BATCH_DIR/deploy/dry_run.log"

jq -e '
  (.tasks | length) == 5976
  and ([.tasks[].state] | unique) == ["pending"]
' "$QUEUE_ROOT/state/$BATCH_ID.state.json" >/dev/null

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

CONTROLLER_ONLY_SYNC_ITEMS=(
  "check/analyze_neurips_rebuttal_full664.py"
  "check/audit_neurips_rebuttal_completed.py"
  "check/prepare_symf_formal_judge_params.py"
  "check/generate_symf_formal_metrics.py"
  "check/merge_symf_formal_shards.py"
  "check/merge_neurips_rebuttal_metrics.py"
  "A_Neurips_experiments/stage3_664dats_4probes_3seeds_1h/probe4_postprocess_run_level.csv"
  "A_Neurips_experiments/stage3_664dats_4probes_3seeds_1h/probe4_current_run_level_raw_digest_7968.csv"
)

echo "[sync] local -> iaaccn22"
timeout 900 rsync -aR "${RSYNC_FILTERS[@]}" \
  "${SYNC_ITEMS[@]}" "${CONTROLLER_ONLY_SYNC_ITEMS[@]}" \
  iaaccn22:/home/zhangziwen/workplace/scientific-intelligent-modelling/

echo "[sync] iaaccn22 -> iaaccn23~29"
timeout 1800 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
  "cd /home/zhangziwen/workplace/scientific-intelligent-modelling && bash '$BATCH_DIR/deploy/00_fanout_from_iaaccn22.sh'"

echo "[sync] validated and synchronized: $BATCH_ID"
