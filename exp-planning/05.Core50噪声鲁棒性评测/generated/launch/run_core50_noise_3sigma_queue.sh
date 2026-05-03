#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangziwen/workplace/scientific-intelligent-modelling
export PYTHONPATH=.
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1

BATCH_SUFFIX="${BATCH_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
RUN_LOG_ROOT="experiments/core50_noise_12alg_3seed_${BATCH_SUFFIX}"
mkdir -p "$RUN_LOG_ROOT"

COMMON_ARGS=(
  --source-csv "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
  --expected-rows 50
  --hosts iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29
  --seeds 0 1 2
  --controller-host iaaccn23
  --use-internal-ips
  --poll-seconds 60
  --retry-limit 1
  --max-jobs-per-host 100
  --load-tier-new-jobs "0.50:10,0.70:5,0.80:2"
  --max-load-ratio 0.80
  --max-memory-used-ratio 0.80
  --llm-model-assignment stable-half
  --llm-model-buckets base,turbo
  --llm-model-bucket-limits base:100,turbo:100
  --prioritize-llm
  --seed-dispatch-mode sequential
)

run_sigma() {
  local sigma_label="$1"
  local batch_name="core50_noise_${sigma_label}_12alg_3seed_${BATCH_SUFFIX}"
  local queue_root="exp-planning/05.Core50噪声鲁棒性评测/generated/load_queue/${sigma_label}"
  local params_root="exp-planning/05.Core50噪声鲁棒性评测/generated/params/${sigma_label}"
  local session_prefix="core50_noise_${sigma_label}_"
  local log_path="${RUN_LOG_ROOT}/controller_${sigma_label}.log"

  echo "[$(date '+%F %T')] START ${batch_name}" | tee -a "$log_path"
  python check/run_e1_candidate200_12alg_load_queue.py \
    --batch-name "$batch_name" \
    --queue-root "$queue_root" \
    --params-root "$params_root" \
    --session-prefix "$session_prefix" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_path"
  echo "[$(date '+%F %T')] DONE ${batch_name}" | tee -a "$log_path"
}

run_sigma sigma001
run_sigma sigma005
run_sigma sigma010

