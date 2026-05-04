#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/family/workplace/scientific-intelligent-modelling"
REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BATCH_NAME="${BATCH_NAME:-e1_candidate200_seed1314_w5_$STAMP}"
echo "BATCH_NAME=${BATCH_NAME}"

start_job() {
  local host="$1"
  local session="$2"
  local remote_job="$3"
  local rel_slice="$4"
  local rel_params="$5"
  local workers="$6"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p \"$REMOTE_ROOT/check\" \"$REMOTE_ROOT/$(dirname "$rel_slice")\" \"$REMOTE_ROOT/$(dirname "$rel_params")\" \"$REMOTE_ROOT/$(dirname "$remote_job")\""
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/check/launch_e1_benchmark.py" "$host:$REMOTE_ROOT/check/launch_e1_benchmark.py"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_slice" "$host:$REMOTE_ROOT/$rel_slice"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_params" "$host:$REMOTE_ROOT/$rel_params"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$remote_job" "$host:$REMOTE_ROOT/$remote_job"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "chmod +x \"$REMOTE_ROOT/$remote_job\" && tmux kill-session -t \"$session\" >/dev/null 2>&1 || true; tmux new-session -d -s \"$session\" /bin/bash \"$REMOTE_ROOT/$remote_job\" \"$BATCH_NAME\" \"$workers\""
  echo "STARTED ${host} ${session}"
}

start_job "iaaccn22" "e1_w5_tpsr_22" "exp-planning/02.E1选择验证/generated/remote_jobs/W5/tpsr_iaaccn22.sh" "exp-planning/02.E1选择验证/generated/slices/W5/tpsr/iaaccn22.csv" "exp-planning/02.E1选择验证/generated/params/tpsr.json" "25"
start_job "iaaccn23" "e1_w5_tpsr_23" "exp-planning/02.E1选择验证/generated/remote_jobs/W5/tpsr_iaaccn23.sh" "exp-planning/02.E1选择验证/generated/slices/W5/tpsr/iaaccn23.csv" "exp-planning/02.E1选择验证/generated/params/tpsr.json" "25"
start_job "iaaccn24" "e1_w5_tpsr_24" "exp-planning/02.E1选择验证/generated/remote_jobs/W5/tpsr_iaaccn24.sh" "exp-planning/02.E1选择验证/generated/slices/W5/tpsr/iaaccn24.csv" "exp-planning/02.E1选择验证/generated/params/tpsr.json" "25"
start_job "iaaccn25" "e1_w5_tpsr_25" "exp-planning/02.E1选择验证/generated/remote_jobs/W5/tpsr_iaaccn25.sh" "exp-planning/02.E1选择验证/generated/slices/W5/tpsr/iaaccn25.csv" "exp-planning/02.E1选择验证/generated/params/tpsr.json" "25"

echo "WAVE_DONE W5"
