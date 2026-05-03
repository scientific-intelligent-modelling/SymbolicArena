#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangziwen/workplace/scientific-intelligent-modelling
export PYTHONPATH=.
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1

BATCH_SUFFIX="${BATCH_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
RUN_LOG_ROOT="experiments/core50_noise_12alg_seed34_${BATCH_SUFFIX}"
mkdir -p "$RUN_LOG_ROOT"

HOSTS=(
  iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29
  iaaccn48 iaaccn49 iaaccn50 iaaccn51 iaaccn52 iaaccn53 iaaccn54 iaaccn55
)
HOST_REMOTE_ROOT_OVERRIDES="iaaccn48=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn49=/data3/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn50=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn51=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn52=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn53=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn54=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn55=/data1/zhangziwen/workplace/scientific-intelligent-modelling"
HOST_REMOTE_DATA_ROOT_OVERRIDES="iaaccn48=/data1/zhangziwen/sim-datasets-data,iaaccn49=/data3/zhangziwen/sim-datasets-data,iaaccn50=/data1/zhangziwen/sim-datasets-data,iaaccn51=/data1/zhangziwen/sim-datasets-data,iaaccn52=/data1/zhangziwen/sim-datasets-data,iaaccn53=/data1/zhangziwen/sim-datasets-data,iaaccn54=/data1/zhangziwen/sim-datasets-data,iaaccn55=/data1/zhangziwen/sim-datasets-data"

COMMON_ARGS=(
  --source-csv "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
  --expected-rows 50
  --hosts "${HOSTS[@]}"
  --seeds 3 4
  --controller-host iaaccn23
  --use-internal-ips
  --remote-root "/home/zhangziwen/workplace/scientific-intelligent-modelling"
  --remote-data-root "/home/zhangziwen/sim-datasets-data"
  --host-remote-root-overrides "$HOST_REMOTE_ROOT_OVERRIDES"
  --host-remote-data-root-overrides "$HOST_REMOTE_DATA_ROOT_OVERRIDES"
  --poll-seconds 60
  --retry-limit 1
  --max-jobs-per-host 100
  --load-tier-new-jobs "0.50:10,0.70:5,0.80:2"
  --max-load-ratio 0.80
  --max-memory-used-ratio 0.80
  --host-session-count-prefix core50_noise_
  --llm-model-assignment stable-half
  --llm-model-buckets base,turbo
  --llm-model-bucket-limits base:80,turbo:80
  --prioritize-llm
  --seed-dispatch-mode sequential
)

if [ "${SKIP_SUPPORT_SYNC:-0}" = "1" ]; then
  COMMON_ARGS+=(--skip-support-sync)
fi

run_sigma() {
  local sigma_label="$1"
  local batch_name="core50_noise_${sigma_label}_12alg_seed34_${BATCH_SUFFIX}"
  local queue_root="exp-planning/05.Core50噪声鲁棒性评测/generated/load_queue/${sigma_label}"
  local params_root="exp-planning/05.Core50噪声鲁棒性评测/generated/params/${sigma_label}"
  local session_prefix="core50_noise_${sigma_label}_seed34_"
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

SIGMAS=(${SIGMAS:-sigma001 sigma005 sigma010})
PARALLEL_SIGMAS="${PARALLEL_SIGMAS:-1}"

if [ "$PARALLEL_SIGMAS" = "1" ]; then
  pids=()
  for sigma in "${SIGMAS[@]}"; do
    run_sigma "$sigma" &
    pids+=("$!")
    # 轻微错峰，降低多个控制器首轮同时同步和探测导致的拥塞。
    sleep 20
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  for sigma in "${SIGMAS[@]}"; do
    run_sigma "$sigma"
  done
fi
