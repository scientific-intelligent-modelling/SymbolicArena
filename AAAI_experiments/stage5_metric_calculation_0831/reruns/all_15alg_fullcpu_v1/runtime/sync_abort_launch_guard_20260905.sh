#!/usr/bin/env bash
set -euo pipefail

REL_ROOT="AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1"
REMOTE_REPO="/home/zhangziwen/workplace/scientific-intelligent-modelling"
FILES=(
  "commands/run_all_15alg_preflight.sh"
  "commands/run_all_15alg_formal.sh"
)

if [[ "${1:-}" == "--fanout" ]]; then
  cd "$REMOTE_REPO"
  for number in {23..29}; do
    host="10.10.100.${number}"
    for relative in "${FILES[@]}"; do
      timeout 30 scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        "$REL_ROOT/$relative" "$host:$REMOTE_REPO/$REL_ROOT/$relative"
    done
  done
  exit 0
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
REMOTE_RUNTIME="$REMOTE_REPO/$REL_ROOT/runtime"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
  "mkdir -p '$REMOTE_RUNTIME'"
for relative in "${FILES[@]}"; do
  timeout 30 scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$REL_ROOT/$relative" "iaaccn22:$REMOTE_REPO/$REL_ROOT/$relative"
done
timeout 30 scp -q -o BatchMode=yes -o ConnectTimeout=10 \
  "$REL_ROOT/runtime/sync_abort_launch_guard_20260905.sh" \
  "iaaccn22:$REMOTE_RUNTIME/sync_abort_launch_guard_20260905.sh"
timeout 240 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
  "bash '$REMOTE_RUNTIME/sync_abort_launch_guard_20260905.sh' --fanout"

for relative in "${FILES[@]}"; do
  local_sha="$(sha256sum "$REL_ROOT/$relative" | awk '{print $1}')"
  printf '%s local=%s\n' "$relative" "$local_sha"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
    "sha256sum '$REMOTE_REPO/$REL_ROOT/$relative'"
  for number in {23..29}; do
    timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
      "timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 10.10.100.${number} sha256sum '$REMOTE_REPO/$REL_ROOT/$relative'"
  done
done
