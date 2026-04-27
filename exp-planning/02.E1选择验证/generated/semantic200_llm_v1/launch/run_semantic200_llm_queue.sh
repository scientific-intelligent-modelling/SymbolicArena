#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"
REMOTE_HOST_23="${REMOTE_HOST_23:-10.10.100.23}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BATCH_NAME="${BATCH_NAME:-semantic200_llm_v1_seed1314_${STAMP}}"
WORKERS="${WORKERS:-50}"

echo "BATCH_NAME=${BATCH_NAME}"
echo "WORKERS=${WORKERS}"

start_local() {
  local tool="$1"
  local session="semantic200_${tool}_22"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_v1/remote_jobs/${tool}_iaaccn22.sh"
  chmod +x "$script"
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session" /bin/bash "$script" "$BATCH_NAME" "$WORKERS"
  echo "STARTED iaaccn22 $session"
}

start_remote23() {
  local tool="$1"
  local session="semantic200_${tool}_23"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_v1/remote_jobs/${tool}_iaaccn23.sh"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST_23" \
    "chmod +x '$script'; tmux kill-session -t '$session' >/dev/null 2>&1 || true; tmux new-session -d -s '$session' /bin/bash '$script' '$BATCH_NAME' '$WORKERS'"
  echo "STARTED iaaccn23 $session"
}

wait_local() {
  local session="$1"
  while tmux has-session -t "$session" >/dev/null 2>&1; do
    sleep 60
  done
  echo "FINISHED iaaccn22 $session"
}

wait_remote23() {
  local session="$1"
  while timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST_23" \
    "tmux has-session -t '$session' >/dev/null 2>&1"; do
    sleep 60
  done
  echo "FINISHED iaaccn23 $session"
}

run_wave() {
  local tool="$1"
  start_local "$tool"
  start_remote23 "$tool"
  wait_local "semantic200_${tool}_22" &
  local p1=$!
  wait_remote23 "semantic200_${tool}_23" &
  local p2=$!
  wait "$p1" "$p2"
  echo "WAVE_DONE $tool"
}

cd "$REMOTE_ROOT"
run_wave llmsr
run_wave drsr
echo "QUEUE_DONE $BATCH_NAME"
