#!/usr/bin/env bash
set -u -o pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
AUDITOR="check/audit_neurips_rebuttal_completed.py"
cd "$REMOTE_ROOT"

BATCH_ID="$(tr -d '\r\n' < "$BATCH_DIR/BATCH_NAME.txt")"
STATE="$BATCH_DIR/queues/load_queue_full/state/$BATCH_ID.state.json"
EXPERIMENT_ROOT="$REMOTE_ROOT/experiments/$BATCH_ID"
STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="$BATCH_DIR/monitoring/completed_audit/$STAMP"
mkdir -p "$REPORT_DIR"

STATE_SNAPSHOT="$REPORT_DIR/state.snapshot.json"
snapshot_ok=0
for _ in 1 2 3 4 5; do
  if cp "$STATE" "$STATE_SNAPSHOT" \
    && jq -e '.tasks | type == "object"' "$STATE_SNAPSHOT" >/dev/null; then
    snapshot_ok=1
    break
  fi
  sleep 1
done
if [[ "$snapshot_ok" -ne 1 ]]; then
  echo "无法冻结有效 state 快照: $STATE" >&2
  exit 1
fi

state_basename="$(basename "$STATE_SNAPSHOT")"
auditor_basename="$(basename "$AUDITOR")"
failed=0

python "$AUDITOR" \
  --state "$STATE_SNAPSHOT" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --host iaaccn22 \
  --min-runtime 3300 \
  --output "$REPORT_DIR/iaaccn22.json" >/dev/null || failed=1

for suffix in 23 24 25 26 27 28 29; do
  host="iaaccn${suffix}"
  ip="10.10.100.${suffix}"
  report="$REPORT_DIR/$host.json"

  timeout 20 scp \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    "$AUDITOR" \
    "$STATE_SNAPSHOT" \
    "$ip:/tmp/" || {
      jq -n \
        --arg host "$host" \
        '{
          host: $host,
          done_tasks: 0,
          validated_results: 0,
          passed: false,
          issues: [{issue: "sync_failed"}]
        }' > "$report"
      failed=1
      continue
    }

  timeout 60 ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    "$ip" \
    "python /tmp/$auditor_basename \
      --state /tmp/$state_basename \
      --experiment-root $EXPERIMENT_ROOT \
      --host $host \
      --min-runtime 3300" > "$report" || failed=1
done

expected_done="$(
  jq '[.tasks[] | select(.state == "done")] | length' "$STATE_SNAPSHOT"
)"
state_sha256="$(sha256sum "$STATE_SNAPSHOT" | awk '{print $1}')"

jq -s \
  --arg created_at "$(date --iso-8601=seconds)" \
  --arg state_snapshot "$STATE_SNAPSHOT" \
  --arg state_sha256 "$state_sha256" \
  --argjson expected_done "$expected_done" \
  '{
    created_at: $created_at,
    state_snapshot: $state_snapshot,
    state_sha256: $state_sha256,
    expected_done_tasks: $expected_done,
    audited_done_tasks: (map(.done_tasks) | add),
    validated_results: (map(.validated_results) | add),
    all_hosts_passed: all(.passed == true),
    issue_count: (map(.issues | length) | add),
    passed: (
      all(.passed == true)
      and ((map(.done_tasks) | add) == $expected_done)
      and ((map(.validated_results) | add) == $expected_done)
    ),
    host_reports: .
  }' "$REPORT_DIR"/iaaccn*.json \
  > "$REPORT_DIR/summary.json"

cat "$REPORT_DIR/summary.json"
jq -e '.passed == true' "$REPORT_DIR/summary.json" >/dev/null || failed=1

exit "$failed"
