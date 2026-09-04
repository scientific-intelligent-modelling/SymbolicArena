#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSET_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$ASSET_ROOT/../../../.." && pwd)"
LOG_DIR="$ASSET_ROOT/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/controller_formal.log") 2>&1
cd "$REPO_ROOT"
export PYTHONPATH=.
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1
QUEUE_SCRIPT="$REPO_ROOT/check/run_e1_candidate200_12alg_load_queue.py"
SOURCE_CSV="$ASSET_ROOT/frozen_inputs/ssr50_source.csv"
PARAMS_ROOT="$ASSET_ROOT/params"
ALLOWLIST="$ASSET_ROOT/manifests/all_conditions_noise_task_allowlist.csv"
QUEUE_ROOT="$ASSET_ROOT/queues/formal"
mkdir -p "$QUEUE_ROOT"

python "$QUEUE_SCRIPT" \
  --batch-name all_conditions_cpu_v2_noise_formal_formal \
  --source-csv "$SOURCE_CSV" \
  --expected-rows 50 \
  --queue-root "$QUEUE_ROOT" \
  --params-root "$PARAMS_ROOT" \
  --task-id-allowlist-csv "$ALLOWLIST" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --tools jaxsr imcts drsr \
  --seeds 520 521 522 \
  --noise-sigmas 0.01 0.05 \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \
  --remote-data-root /home/zhangziwen/sim-datasets-data \
  --max-jobs-per-host 18 \
  --max-new-jobs-per-host-per-poll 18 \
  --load-tier-new-jobs "0.50:18,0.70:18,0.85:18,0.95:18" \
  --max-load-ratio 0.95 \
  --max-memory-used-ratio 0.95 \
  --min-free-mem-gb 0 \
  --session-prefix all_conditions_cpu_v2_ \
  --host-session-count-prefix all_conditions_cpu_v2_ \
  --poll-seconds 60 \
  --retry-limit 1 \
  --llm-model-assignment from-params \
  --llm-default-bucket turbo \
  --llm-model-bucket-limits base:0,turbo:1 \
  --max-cpu-used-ratio 0.95 \
  --force-rerun-existing
