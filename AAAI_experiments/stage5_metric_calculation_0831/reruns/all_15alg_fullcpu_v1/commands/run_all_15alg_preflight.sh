#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSET_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${SIM_REPO_ROOT:-$(cd -- "$ASSET_ROOT/../../../.." && pwd)}"
mkdir -p "$ASSET_ROOT/logs" "$ASSET_ROOT/queues/preflight"
exec > >(tee -a "$ASSET_ROOT/logs/controller_preflight.log") 2>&1
cd "$REPO_ROOT"
export PYTHONPATH=.
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1
python "$REPO_ROOT/check/run_e1_candidate200_12alg_load_queue.py" \
  --batch-name all_15alg_fullcpu_v1_preflight \
  --source-csv "$ASSET_ROOT/frozen_inputs/sources/formal3h_13alg_ssr50_source.csv" \
  --expected-rows 50 \
  --queue-root "$ASSET_ROOT/queues/preflight" \
  --params-root "$ASSET_ROOT/params" \
  --task-id-allowlist-csv "$ASSET_ROOT/manifests/new_queue_6607.csv" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --tools gplearn llmsr pyoperon drsr pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit \
  --seeds 520 521 522 \
  --noise-sigmas 0 0.01 0.05 \
  --condition-dispatch-mode sequential \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \
  --remote-data-root /home/zhangziwen/sim-datasets-data \
  --max-jobs-per-host 256 \
  --max-new-jobs-per-host-per-poll 128 \
  --load-tier-new-jobs "0.50:128,0.75:64,0.90:32,0.98:8,1.00:2" \
  --max-load-ratio 1.00 \
  --max-memory-used-ratio 0.90 \
  --min-free-mem-gb 32 \
  --session-prefix all_conditions_cpu_v2_ \
  --host-session-count-prefix all_conditions_cpu_v2_ \
  --poll-seconds 30 \
  --retry-limit 1 \
  --prioritize-llm \
  --llm-model-assignment from-params \
  --llm-default-bucket turbo \
  --llm-model-bucket-limits base:0,turbo:30 \
  --preflight-only \
  --preflight-report "$ASSET_ROOT/reports/preflight_report.json"
