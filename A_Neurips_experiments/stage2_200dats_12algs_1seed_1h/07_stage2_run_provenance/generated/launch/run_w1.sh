#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/family/workplace/scientific-intelligent-modelling"
REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BATCH_NAME="${BATCH_NAME:-e1_candidate200_seed1314_w1_$STAMP}"
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

start_job "iaaccn22" "e1_w1_gplearn_22" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/gplearn_iaaccn22.sh" "exp-planning/02.E1选择验证/generated/slices/W1/gplearn/iaaccn22.csv" "exp-planning/02.E1选择验证/generated/params/gplearn.json" "32"
start_job "iaaccn23" "e1_w1_gplearn_23" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/gplearn_iaaccn23.sh" "exp-planning/02.E1选择验证/generated/slices/W1/gplearn/iaaccn23.csv" "exp-planning/02.E1选择验证/generated/params/gplearn.json" "32"
start_job "iaaccn24" "e1_w1_gplearn_24" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/gplearn_iaaccn24.sh" "exp-planning/02.E1选择验证/generated/slices/W1/gplearn/iaaccn24.csv" "exp-planning/02.E1选择验证/generated/params/gplearn.json" "32"
start_job "iaaccn25" "e1_w1_gplearn_25" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/gplearn_iaaccn25.sh" "exp-planning/02.E1选择验证/generated/slices/W1/gplearn/iaaccn25.csv" "exp-planning/02.E1选择验证/generated/params/gplearn.json" "32"
start_job "iaaccn26" "e1_w1_llmsr_26" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/llmsr_iaaccn26.sh" "exp-planning/02.E1选择验证/generated/slices/W1/llmsr/iaaccn26.csv" "exp-planning/02.E1选择验证/generated/params/llmsr.json" "25"
start_job "iaaccn27" "e1_w1_llmsr_27" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/llmsr_iaaccn27.sh" "exp-planning/02.E1选择验证/generated/slices/W1/llmsr/iaaccn27.csv" "exp-planning/02.E1选择验证/generated/params/llmsr.json" "25"
start_job "iaaccn28" "e1_w1_llmsr_28" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/llmsr_iaaccn28.sh" "exp-planning/02.E1选择验证/generated/slices/W1/llmsr/iaaccn28.csv" "exp-planning/02.E1选择验证/generated/params/llmsr.json" "25"
start_job "iaaccn29" "e1_w1_llmsr_29" "exp-planning/02.E1选择验证/generated/remote_jobs/W1/llmsr_iaaccn29.sh" "exp-planning/02.E1选择验证/generated/slices/W1/llmsr/iaaccn29.csv" "exp-planning/02.E1选择验证/generated/params/llmsr.json" "25"

echo "WAVE_DONE W1"
