#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <tool: udsr|dso|imcts|pyoperon> <seed: 520|521|522> [batch_name] [retry]" >&2
  exit 2
fi

TOOL="$1"
SEED="$2"
BATCH_NAME="${3:-probe4_full664_v1_$(date +%Y%m%d-%H%M%S)}"
RETRY_MODE="${4:-retry}"
REPO_ROOT="/home/family/workplace/scientific-intelligent-modelling"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"

case " udsr dso imcts pyoperon " in *" $TOOL "*) ;; *) echo "invalid tool: $TOOL" >&2; exit 2;; esac
case " 520 521 522 " in *" $SEED "*) ;; *) echo "invalid seed: $SEED" >&2; exit 2;; esac

case "$TOOL" in
  udsr) DEFAULT_WORKERS="15" ;;
  dso) DEFAULT_WORKERS="15" ;;
  imcts) DEFAULT_WORKERS="24" ;;
  pyoperon) DEFAULT_WORKERS="24" ;;
esac
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"

echo "BATCH_NAME=${BATCH_NAME}"
echo "TOOL=${TOOL}"
echo "SEED=${SEED}"
echo "WORKERS=${WORKERS}"

start_job() {
  local host="$1"
  local session="probe4_full664_${TOOL}_s${SEED}_${host}"
  local rel_slice="exp-planning/02.E1选择验证/generated/probe4_full664_v1/slices/${TOOL}/seed${SEED}/${host}.csv"
  local rel_params="exp-planning/02.E1选择验证/generated/params/${TOOL}.json"
  local remote_job="exp-planning/02.E1选择验证/generated/probe4_full664_v1/remote_jobs/${TOOL}/seed${SEED}/${TOOL}_seed${SEED}_${host}.sh"
  if [ "$TOOL" = "imcts" ]; then rel_params="exp-planning/02.E1选择验证/generated/params/imcts.json"; fi

  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p \"$REMOTE_ROOT/check\" \"$REMOTE_ROOT/$(dirname "$rel_slice")\" \"$REMOTE_ROOT/$(dirname "$rel_params")\" \"$REMOTE_ROOT/$(dirname "$remote_job")\""
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/check/launch_e1_benchmark.py" "$host:$REMOTE_ROOT/check/launch_e1_benchmark.py"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_slice" "$host:$REMOTE_ROOT/$rel_slice"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_params" "$host:$REMOTE_ROOT/$rel_params"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$remote_job" "$host:$REMOTE_ROOT/$remote_job"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "chmod +x \"$REMOTE_ROOT/$remote_job\" && tmux kill-session -t \"$session\" >/dev/null 2>&1 || true; tmux new-session -d -s \"$session\" /bin/bash \"$REMOTE_ROOT/$remote_job\" \"$BATCH_NAME\" \"$WORKERS\" \"$RETRY_MODE\""
  echo "STARTED ${host} ${session}"
}

start_job "iaaccn23"
start_job "iaaccn24"
start_job "iaaccn25"
start_job "iaaccn26"
start_job "iaaccn27"
start_job "iaaccn28"
start_job "iaaccn29"
echo "WAVE_STARTED ${TOOL} seed${SEED}"
