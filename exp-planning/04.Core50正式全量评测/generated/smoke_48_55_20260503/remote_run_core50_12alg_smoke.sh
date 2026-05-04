#!/usr/bin/env bash
set -uo pipefail

ROOT="$1"
DATA_ROOT="$2"
BATCH_NAME="$3"
HOST_LABEL="$4"
SCRIPT_PATH="$5"

LOG_DIR="${ROOT}/experiments/${BATCH_NAME}/${HOST_LABEL}/__smoke_assets/host_logs"
mkdir -p "${LOG_DIR}"

cd "${ROOT}" || exit 2
python "${SCRIPT_PATH}" \
  --root "${ROOT}" \
  --data-root "${DATA_ROOT}" \
  --batch-name "${BATCH_NAME}" \
  --host-label "${HOST_LABEL}" \
  --dataset-index 9 \
  --timeout-seconds 300 \
  --outer-timeout-seconds 480 \
  --parallel 12 \
  > "${LOG_DIR}/remote_core50_12alg_smoke.log" 2>&1
