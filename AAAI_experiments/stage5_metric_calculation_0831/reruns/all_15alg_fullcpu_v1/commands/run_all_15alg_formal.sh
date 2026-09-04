#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSET_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${SIM_REPO_ROOT:-$(cd -- "$ASSET_ROOT/../../../.." && pwd)}"
mkdir -p "$ASSET_ROOT/logs" "$ASSET_ROOT/queues/formal"
exec > >(tee -a "$ASSET_ROOT/logs/controller_formal.log") 2>&1
cd "$REPO_ROOT"
export PYTHONPATH=.
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1
RESUME_ARGS=()
if [[ "${SIM_RESUME_EXISTING_QUEUE:-0}" == "1" ]]; then
  RESUME_ARGS+=(--skip-support-sync)
fi
python "$REPO_ROOT/check/run_e1_candidate200_12alg_load_queue.py" \
  --batch-name all_15alg_fullcpu_v1_formal \
  --source-csv "$ASSET_ROOT/frozen_inputs/sources/formal3h_13alg_ssr50_source.csv" \
  --expected-rows 50 \
  --queue-root "$ASSET_ROOT/queues/formal" \
  --params-root "$ASSET_ROOT/params" \
  --task-id-allowlist-csv "$ASSET_ROOT/manifests/new_queue_6607.csv" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --tools gplearn llmsr pyoperon drsr pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit \
  --seeds 520 521 522 \
  --noise-sigmas 0 0.01 0.05 \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \
  --remote-data-root /home/zhangziwen/sim-datasets-data \
  --max-jobs-per-host 230 \
  --max-cpu-used-ratio 0.8984375 \
  --max-new-jobs-per-host-per-poll 230 \
  --load-tier-new-jobs "0.50:230,0.70:230,0.85:230,0.90:230" \
  --max-load-ratio 0.90 \
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
  "${RESUME_ARGS[@]}" \
  --force-rerun-existing
