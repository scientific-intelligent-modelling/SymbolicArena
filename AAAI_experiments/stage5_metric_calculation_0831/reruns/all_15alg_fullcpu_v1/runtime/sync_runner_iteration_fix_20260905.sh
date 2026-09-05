#!/usr/bin/env bash
set -u

REPO_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
RUNNER_REL=scientific_intelligent_modelling/benchmarks/runner.py
RUNNER="$REPO_ROOT/$RUNNER_REL"
EXPECTED_SHA256=33d7ed02f1615a8b8758b5aace8c2f03a404a26631dbefe26f84d48d0b8ad37b
AUDIT="$REPO_ROOT/AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1/runtime/runner_iteration_sync_20260905.audit"

cd "$REPO_ROOT" || exit 2
failed=0
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "expected_sha256=$EXPECTED_SHA256"
  echo "iaaccn22=$(sha256sum "$RUNNER" | awk '{print $1}')"
} >"$AUDIT"

for suffix in 23 24 26 27 28 29; do
  ip="10.10.100.$suffix"
  if timeout 120 rsync -a --relative "$RUNNER_REL" "$ip:$REPO_ROOT/"; then
    actual=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" \
      "sha256sum '$RUNNER'" | awk '{print $1}')
    echo "iaaccn$suffix=$actual" >>"$AUDIT"
    [[ "$actual" == "$EXPECTED_SHA256" ]] || failed=1
  else
    echo "iaaccn$suffix=sync_failed" >>"$AUDIT"
    failed=1
  fi
done

echo "iaaccn25=quarantined_ssh_unavailable" >>"$AUDIT"
echo "finished_at=$(date --iso-8601=seconds)" >>"$AUDIT"
echo "ok=$([[ "$failed" == "0" ]] && echo true || echo false)" >>"$AUDIT"
cat "$AUDIT"
exit "$failed"
