#!/usr/bin/env bash
set -u

REMOTE_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
RUNNER_REL=scientific_intelligent_modelling/benchmarks/runner.py
SOURCE="$REMOTE_ROOT/$RUNNER_REL"
EXPECTED_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
FAILED=0

for host_id in 23 24 26 27 28 29; do
  host="10.10.100.$host_id"
  if ! timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "mkdir -p '$REMOTE_ROOT/scientific_intelligent_modelling/benchmarks'"; then
    echo "PREP_FAIL iaaccn$host_id"
    FAILED=1
    continue
  fi
  if ! timeout 120 rsync -a --checksum \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
    "$SOURCE" "$host:$REMOTE_ROOT/$RUNNER_REL"; then
    echo "SYNC_FAIL iaaccn$host_id"
    FAILED=1
    continue
  fi
  actual_sha="$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "sha256sum '$REMOTE_ROOT/$RUNNER_REL'" | awk '{print $1}')"
  if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "VERIFY_FAIL iaaccn$host_id expected=$EXPECTED_SHA actual=$actual_sha"
    FAILED=1
    continue
  fi
  echo "SYNCED iaaccn$host_id sha256=$actual_sha"
done

exit "$FAILED"
