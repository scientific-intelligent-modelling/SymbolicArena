#!/usr/bin/env bash
set -u

REPO_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
RUNTIME="$REPO_ROOT/AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1/runtime"
AUDIT_DIR="$RUNTIME/abort_fullcpu_20260905-1118"
HOST_SCRIPT="$RUNTIME/stop_fullcpu_host_20260905.py"
TARGETS="$AUDIT_DIR/targets_all_6607.json"
EXPERIMENT_ROOT="$REPO_ROOT/experiments/all_15alg_fullcpu_v1_formal"
AUDIT="$AUDIT_DIR/final_stop_verification.audit"

failed=0
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "batch=all_15alg_fullcpu_v1_formal"
} >"$AUDIT"

run_local() {
  local host=$1
  local output
  output=$(python3 "$HOST_SCRIPT" \
    --targets "$TARGETS" \
    --marker "$EXPERIMENT_ROOT/ABORTED_DO_NOT_USE.json" 2>&1)
  local rc=$?
  printf 'host=%s rc=%s output=%s\n' "$host" "$rc" "$output" >>"$AUDIT"
  return "$rc"
}

run_remote() {
  local suffix=$1
  local host="iaaccn$suffix"
  local ip="10.10.100.$suffix"
  local attempt output rc
  for attempt in 1 2 3; do
    if ! timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$ip" \
      "mkdir -p '$AUDIT_DIR'"; then
      printf 'host=%s attempt=%s stage=mkdir rc=failed\n' "$host" "$attempt" >>"$AUDIT"
      sleep 5
      continue
    fi
    if ! timeout 180 scp -o BatchMode=yes -o ConnectTimeout=15 \
      "$HOST_SCRIPT" "$TARGETS" "$ip:$AUDIT_DIR/"; then
      printf 'host=%s attempt=%s stage=copy rc=failed\n' "$host" "$attempt" >>"$AUDIT"
      sleep 5
      continue
    fi
    output=$(timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=15 "$ip" \
      "python3 '$AUDIT_DIR/$(basename "$HOST_SCRIPT")' --targets '$AUDIT_DIR/$(basename "$TARGETS")' --marker '$EXPERIMENT_ROOT/ABORTED_DO_NOT_USE.json'" 2>&1)
    rc=$?
    printf 'host=%s attempt=%s stage=reap rc=%s output=%s\n' \
      "$host" "$attempt" "$rc" "$output" >>"$AUDIT"
    if [[ "$rc" == "0" ]]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_local iaaccn22 || failed=1
for suffix in 23 24 25 26 27 28 29; do
  run_remote "$suffix" || failed=1
done

controller_processes=$(pgrep -af 'run_e1_candidate200_12alg_load_queue.py.*all_15alg_fullcpu_v1_formal' || true)
controller_session=absent
if tmux has-session -t stage5_all_15alg_fullcpu_v1 2>/dev/null; then
  controller_session=present
  failed=1
fi
if [[ -n "$controller_processes" ]]; then
  failed=1
fi
{
  echo "controller_session=$controller_session"
  echo "controller_processes=${controller_processes:-none}"
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "ok=$([[ "$failed" == "0" ]] && echo true || echo false)"
} >>"$AUDIT"
cat "$AUDIT"
exit "$failed"
