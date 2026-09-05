#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
ASSET_ROOT="$REPO_ROOT/AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1"
SESSION=stage5_all_15alg_fullcpu_v1
BATCH=all_15alg_fullcpu_v1_formal
STATE="$ASSET_ROOT/queues/formal/state/${BATCH}.state.json"
EVENTS="$ASSET_ROOT/queues/formal/state/${BATCH}.events.jsonl"
WRAPPER_REL=scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py
WRAPPER="$REPO_ROOT/$WRAPPER_REL"
WRAPPER_SHA256=738746dc26e7d721efee0ac169843353f209baba2f602d5f9e35de37f72f9d4e
FIX_COMMIT=5894ad03
AUDIT="$ASSET_ROOT/runtime/qlattice_global_best_requeue_20260905.audit.json"
SYNC_AUDIT="$ASSET_ROOT/runtime/qlattice_global_best_sync_20260905.audit"
CONTROLLER_LOG="$ASSET_ROOT/logs/controller_formal.log"
HOSTS=(iaaccn22 iaaccn23 iaaccn24 iaaccn26 iaaccn27 iaaccn28 iaaccn29)

cd "$REPO_ROOT"

restart_controller() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "$SESSION" /bin/bash -lc "
    cd '$REPO_ROOT'
    export PYTHONPATH=.
    export SIM_QUEUE_CONTROLLER_IS_LOCAL=1
    python '$REPO_ROOT/check/run_e1_candidate200_12alg_load_queue.py' \\
      --batch-name '$BATCH' \\
      --source-csv '$ASSET_ROOT/frozen_inputs/sources/formal3h_13alg_ssr50_source.csv' \\
      --expected-rows 50 \\
      --queue-root '$ASSET_ROOT/queues/formal' \\
      --params-root '$ASSET_ROOT/params' \\
      --task-id-allowlist-csv '$ASSET_ROOT/manifests/new_queue_6607.csv' \\
      --hosts ${HOSTS[*]} \\
      --tools gplearn llmsr pyoperon drsr pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit \\
      --seeds 520 521 522 \\
      --noise-sigmas 0 0.01 0.05 \\
      --condition-dispatch-mode sequential-non-llm-backfill \\
      --controller-host iaaccn22 \\
      --use-internal-ips \\
      --remote-root '$REPO_ROOT' \\
      --remote-data-root /home/zhangziwen/sim-datasets-data \\
      --max-jobs-per-host 256 \\
      --max-new-jobs-per-host-per-poll 128 \\
      --load-tier-new-jobs '0.50:128,0.75:64,0.90:32,0.98:8,1.00:2' \\
      --max-load-ratio 1.00 \\
      --max-memory-used-ratio 0.90 \\
      --min-free-mem-gb 32 \\
      --session-prefix all_conditions_cpu_v2_ \\
      --host-session-count-prefix all_conditions_cpu_v2_ \\
      --poll-seconds 30 \\
      --retry-limit 1 \\
      --prioritize-llm \\
      --llm-model-assignment from-params \\
      --llm-default-bucket turbo \\
      --llm-model-bucket-limits base:0,turbo:30 \\
      --skip-support-sync \\
      --force-rerun-existing >>'$CONTROLLER_LOG' 2>&1
  "
}

trap restart_controller EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux send-keys -t "$SESSION" C-c
fi
for _ in $(seq 1 30); do
  if ! pgrep -f "run_e1_candidate200_12alg_load_queue.py.*--batch-name $BATCH" >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f "run_e1_candidate200_12alg_load_queue.py.*--batch-name $BATCH" >/dev/null; then
  pgrep -f "run_e1_candidate200_12alg_load_queue.py.*--batch-name $BATCH" | xargs -r kill -TERM
  sleep 3
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "required_sha256=$WRAPPER_SHA256"
  echo "iaaccn22=$(sha256sum "$WRAPPER" | awk '{print $1}')"
} >"$SYNC_AUDIT"

for host in iaaccn23 iaaccn24 iaaccn26 iaaccn27 iaaccn28 iaaccn29; do
  ip="10.10.100.${host#iaaccn}"
  if timeout 90 rsync -a --relative "$WRAPPER_REL" "$ip:$REPO_ROOT/"; then
    actual=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" "sha256sum '$WRAPPER'" | awk '{print $1}')
    echo "$host=$actual" >>"$SYNC_AUDIT"
  else
    echo "$host=sync_failed" >>"$SYNC_AUDIT"
    exit 1
  fi
done
echo "iaaccn25=quarantined_ssh_unavailable" >>"$SYNC_AUDIT"

if awk -F= -v expected="$WRAPPER_SHA256" '$1 ~ /^iaaccn2[2346789]$/ && $2 != expected {exit 1}' "$SYNC_AUDIT"; then
  echo "reachable_host_hashes_ok=true" >>"$SYNC_AUDIT"
else
  echo "reachable_host_hashes_ok=false" >>"$SYNC_AUDIT"
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH=. python "$ASSET_ROOT/runtime/requeue_qlattice_global_best_20260905.py" \
  --state "$STATE" \
  --events "$EVENTS" \
  --wrapper "$WRAPPER" \
  --expected-wrapper-sha256 "$WRAPPER_SHA256" \
  --fix-commit "$FIX_COMMIT" \
  --expected-task-count 450 \
  --audit-output "$AUDIT"

restart_controller
sleep 15
echo "finished_at=$(date --iso-8601=seconds)" >>"$SYNC_AUDIT"
echo "controller=$(pgrep -af "run_e1_candidate200_12alg_load_queue.py.*--batch-name $BATCH" || true)" >>"$SYNC_AUDIT"
echo "tmux=$(tmux list-sessions -F '#S|#{session_dead}' | grep -F "$SESSION" || true)" >>"$SYNC_AUDIT"
trap - EXIT
cat "$SYNC_AUDIT"
