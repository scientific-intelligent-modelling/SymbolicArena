#!/usr/bin/env bash
set -u

REMOTE_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
RUNTIME_FILES=(
  scientific_intelligent_modelling/benchmarks/runner.py
  check/run_e1_candidate200_12alg_load_queue.py
)
FAILED=0

for host_id in 23 24 26 27 28 29; do
  host="10.10.100.$host_id"
  for relative_path in "${RUNTIME_FILES[@]}"; do
    source_path="$REMOTE_ROOT/$relative_path"
    expected_sha="$(sha256sum "$source_path" | awk '{print $1}')"
    target_dir="$(dirname "$source_path")"
    if ! timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
      "mkdir -p '$target_dir'"; then
      echo "PREP_FAIL iaaccn$host_id file=$relative_path"
      FAILED=1
      continue
    fi
    if ! timeout 120 rsync -a --checksum \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$source_path" "$host:$source_path"; then
      echo "SYNC_FAIL iaaccn$host_id file=$relative_path"
      FAILED=1
      continue
    fi
    actual_sha="$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
      "sha256sum '$source_path'" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      echo "VERIFY_FAIL iaaccn$host_id file=$relative_path expected=$expected_sha actual=$actual_sha"
      FAILED=1
      continue
    fi
    echo "SYNCED iaaccn$host_id file=$relative_path sha256=$actual_sha"
  done
done

exit "$FAILED"
