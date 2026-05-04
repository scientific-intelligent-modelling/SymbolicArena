#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=.
unset SIM_QUEUE_CONTROLLER_IS_LOCAL

BATCH_SUFFIX="${BATCH_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
BATCH_NAME="core50_drsr_llmbudget_seed567_${BATCH_SUFFIX}"
RUN_LOG_ROOT="experiments/${BATCH_NAME}"
ASSET_ROOT="exp-planning/04.Core50正式全量评测/generated/core50_drsr_llmbudget_seed567_20260504"
QUEUE_ROOT="${ASSET_ROOT}/load_queue"
PARAMS_ROOT="${ASSET_ROOT}/params"

mkdir -p "$RUN_LOG_ROOT"

HOSTS=(
  iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29
  iaaccn48 iaaccn49 iaaccn50 iaaccn51 iaaccn52 iaaccn53 iaaccn54 iaaccn55
)
if [ -n "${DISABLE_HOSTS:-}" ]; then
  FILTERED_HOSTS=()
  for host in "${HOSTS[@]}"; do
    case " ${DISABLE_HOSTS} " in
      *" ${host} "*) ;;
      *) FILTERED_HOSTS+=("$host") ;;
    esac
  done
  HOSTS=("${FILTERED_HOSTS[@]}")
fi
HOST_REMOTE_ROOT_OVERRIDES="iaaccn48=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn49=/data3/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn50=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn51=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn52=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn53=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn54=/data1/zhangziwen/workplace/scientific-intelligent-modelling,iaaccn55=/data1/zhangziwen/workplace/scientific-intelligent-modelling"
HOST_REMOTE_DATA_ROOT_OVERRIDES="iaaccn48=/data1/zhangziwen/sim-datasets-data,iaaccn49=/data3/zhangziwen/sim-datasets-data,iaaccn50=/data1/zhangziwen/sim-datasets-data,iaaccn51=/data1/zhangziwen/sim-datasets-data,iaaccn52=/data1/zhangziwen/sim-datasets-data,iaaccn53=/data1/zhangziwen/sim-datasets-data,iaaccn54=/data1/zhangziwen/sim-datasets-data,iaaccn55=/data1/zhangziwen/sim-datasets-data"

EXTRA_ARGS=()
if [ "${SKIP_SUPPORT_SYNC:-0}" = "1" ]; then
  EXTRA_ARGS+=(--skip-support-sync)
fi

python check/run_e1_candidate200_12alg_load_queue.py \
  --batch-name "$BATCH_NAME" \
  --source-csv "exp-planning/04.Core50正式全量评测/core50_datasets.csv" \
  --expected-rows 50 \
  --queue-root "$QUEUE_ROOT" \
  --params-root "$PARAMS_ROOT" \
  --session-prefix "core50_drsr_llmbudget_seed567_" \
  --hosts "${HOSTS[@]}" \
  --tools drsr \
  --seeds 5 6 7 \
  --controller-host iaaccn23 \
  --no-use-internal-ips \
  --remote-root "/home/zhangziwen/workplace/scientific-intelligent-modelling" \
  --remote-data-root "/home/zhangziwen/sim-datasets-data" \
  --host-remote-root-overrides "$HOST_REMOTE_ROOT_OVERRIDES" \
  --host-remote-data-root-overrides "$HOST_REMOTE_DATA_ROOT_OVERRIDES" \
  --poll-seconds 60 \
  --retry-limit 1 \
  --max-jobs-per-host 100 \
  --load-tier-new-jobs "0.50:10,0.70:5,0.80:2" \
  --max-load-ratio 0.80 \
  --max-memory-used-ratio 0.80 \
  --host-session-count-prefix core50_ \
  --llm-model-assignment stable-half \
  --llm-model-buckets base,turbo \
  --llm-model-bucket-limits base:80,turbo:80 \
  --prioritize-llm \
  --seed-dispatch-mode sequential \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${RUN_LOG_ROOT}/controller.log"
