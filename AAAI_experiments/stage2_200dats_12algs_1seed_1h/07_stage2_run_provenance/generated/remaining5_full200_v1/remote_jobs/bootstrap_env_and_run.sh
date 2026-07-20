#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <ENV_NAME> <JOB_SCRIPT> <BATCH_NAME> <WORKERS> [retry]" >&2
  exit 2
fi

ENV_NAME="$1"
JOB_SCRIPT="$2"
BATCH_NAME="$3"
WORKERS="$4"
RETRY_MODE="${5:-retry}"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"

cd "$REMOTE_ROOT"
export PYTHONPATH=.
BOOTSTRAP_LOG_DIR="$REMOTE_ROOT/experiments/$BATCH_NAME/_bootstrap_logs"
mkdir -p "$BOOTSTRAP_LOG_DIR"
exec > >(tee -a "$BOOTSTRAP_LOG_DIR/${ENV_NAME}_$(hostname).log") 2>&1

if ! conda run -n "$ENV_NAME" python -V; then
  echo "[bootstrap] missing env: $ENV_NAME"
  BOOTSTRAP_PY="/tmp/e1_create_env_${ENV_NAME}.py"
  cat > "$BOOTSTRAP_PY" <<'PY'
import sys
from scientific_intelligent_modelling.srkit.conda_env_manager import env_manager

env_name = sys.argv[1]
ok = env_manager.create_environment(env_name)
raise SystemExit(0 if ok else 1)
PY
  PYTHONPATH=. conda run -n sim_base python "$BOOTSTRAP_PY" "$ENV_NAME"
fi

conda run -n "$ENV_NAME" python -V
/bin/bash "$JOB_SCRIPT" "$BATCH_NAME" "$WORKERS" "$RETRY_MODE"
