#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"
REMOTE_HOST_23="${REMOTE_HOST_23:-10.10.100.23}"
REMOTE_HOST_24="${REMOTE_HOST_24:-10.10.100.24}"
REMOTE_HOST_25="${REMOTE_HOST_25:-10.10.100.25}"
REMOTE_HOST_26="${REMOTE_HOST_26:-10.10.100.26}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BATCH_NAME="${BATCH_NAME:-semantic200_llm_physics_v2_4host_seed1314_${STAMP}}"
WORKERS="${WORKERS:-50}"

if [ "${CONFIRM_SEMANTIC200_LLM_PHYSICS:-}" != "semantic200_llm_physics_v2_4host" ]; then
  cat >&2 <<'EOF'
Refusing to launch semantic200 LLM physics rerun.

This batch reruns LLMSR and DRSR on Candidate-200 with physical/semantic
metadata injected into prompts. It must only be launched after explicit user
confirmation.

After confirmation, run:
  export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v2_4host
  bash exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/launch/run_semantic200_llm_queue.sh
EOF
  exit 3
fi

echo "BATCH_NAME=${BATCH_NAME}"
echo "WORKERS=${WORKERS}"

check_auth_local() {
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/remote_jobs/check_llm_auth.py"
  python "$script"
}

check_auth_remote() {
  local target="$1"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/remote_jobs/check_llm_auth.py"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "python '$script'"
}

start_local() {
  local tool="$1"
  local host="$2"
  local session="semantic200_${tool}_${host}"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/remote_jobs/${tool}_${host}.sh"
  chmod +x "$script"
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session" env CONFIRM_SEMANTIC200_LLM_PHYSICS="${CONFIRM_SEMANTIC200_LLM_PHYSICS}" /bin/bash "$script" "$BATCH_NAME" "$WORKERS"
  echo "STARTED $host $session"
}

start_remote() {
  local tool="$1"
  local host="$2"
  local target="$3"
  local session="semantic200_${tool}_${host}"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/remote_jobs/${tool}_${host}.sh"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" \
    "chmod +x '$script'; tmux kill-session -t '$session' >/dev/null 2>&1 || true; tmux new-session -d -s '$session' env CONFIRM_SEMANTIC200_LLM_PHYSICS='semantic200_llm_physics_v2_4host' /bin/bash '$script' '$BATCH_NAME' '$WORKERS'"
  echo "STARTED $host $session"
}

wait_local() {
  local session="$1"
  while tmux has-session -t "$session" >/dev/null 2>&1; do
    sleep 60
  done
  echo "FINISHED local $session"
}

wait_remote() {
  local session="$1"
  local target="$2"
  while timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" \
    "tmux has-session -t '$session' >/dev/null 2>&1"; do
    sleep 60
  done
  echo "FINISHED remote $session"
}

run_wave() {
  local tool="$1"
  start_local "$tool" "iaaccn23"
  start_remote "$tool" "iaaccn24" "$REMOTE_HOST_24"
  start_remote "$tool" "iaaccn25" "$REMOTE_HOST_25"
  start_remote "$tool" "iaaccn26" "$REMOTE_HOST_26"

  wait_local "semantic200_${tool}_iaaccn23" &
  local p1=$!
  wait_remote "semantic200_${tool}_iaaccn24" "$REMOTE_HOST_24" &
  local p2=$!
  wait_remote "semantic200_${tool}_iaaccn25" "$REMOTE_HOST_25" &
  local p3=$!
  wait_remote "semantic200_${tool}_iaaccn26" "$REMOTE_HOST_26" &
  local p4=$!
  wait "$p1" "$p2" "$p3" "$p4"
  echo "WAVE_DONE $tool"
}

cd "$REMOTE_ROOT"
check_auth_local
check_auth_remote "$REMOTE_HOST_24"
check_auth_remote "$REMOTE_HOST_25"
check_auth_remote "$REMOTE_HOST_26"
run_wave llmsr
run_wave drsr
echo "QUEUE_DONE $BATCH_NAME"
