#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${REPO_ROOT}/A_Neurips_experiments/rebuttal/02_six_axis_uncertainty_15algs"
COLLECTOR="${OUT_DIR}/collect_aaai_new3_minute60.py"
SELECTED="${REPO_ROOT}/AAAI_experiments/stage4_ssr50_15algs_3seeds_3noise_3h/selected_runs_with_fepysr_rerun.csv"
PARTS_DIR="/tmp/symbolicarena_new3_minute60_parts"
REMOTE_TMP="/tmp/symbolicarena_new3_minute60"
HOSTS=(iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29)

mkdir -p "${PARTS_DIR}"

scp "${COLLECTOR}" "${SELECTED}" "iaaccn22:${REMOTE_TMP}/" || {
  ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 "mkdir -p '${REMOTE_TMP}'" || exit 1
  scp "${COLLECTOR}" "${SELECTED}" "iaaccn22:${REMOTE_TMP}/" || exit 1
}

ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
  "REMOTE_TMP='${REMOTE_TMP}' bash -s" <<'REMOTE'
set -u
hosts=(iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29)
for host in "${hosts[@]}"; do
  output="${REMOTE_TMP}/${host}.csv"
  if [[ "${host}" == "iaaccn22" ]]; then
    python "${REMOTE_TMP}/collect_aaai_new3_minute60.py" extract-host \
      --selected-runs-csv "${REMOTE_TMP}/selected_runs_with_fepysr_rerun.csv" \
      --host "${host}" \
      --output "${output}" || {
        echo "EXTRACT_FAIL ${host}" >&2
        continue
      }
  else
    ip="10.10.100.${host#iaaccn}"
    timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
      "mkdir -p '${REMOTE_TMP}'" || {
        echo "PREP_FAIL ${host}" >&2
        continue
      }
    scp "${REMOTE_TMP}/collect_aaai_new3_minute60.py" \
      "${REMOTE_TMP}/selected_runs_with_fepysr_rerun.csv" \
      "${ip}:${REMOTE_TMP}/" || {
        echo "SYNC_FAIL ${host}" >&2
        continue
      }
    timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
      "python '${REMOTE_TMP}/collect_aaai_new3_minute60.py' extract-host \
       --selected-runs-csv '${REMOTE_TMP}/selected_runs_with_fepysr_rerun.csv' \
       --host '${host}' --output '${output}'" || {
        echo "EXTRACT_FAIL ${host}" >&2
        continue
      }
    scp "${ip}:${output}" "${output}" || {
      echo "COLLECT_FAIL ${host}" >&2
      continue
    }
  fi
  echo "COLLECTED ${host}"
done
REMOTE

for host in "${HOSTS[@]}"; do
  scp "iaaccn22:${REMOTE_TMP}/${host}.csv" "${PARTS_DIR}/${host}.csv" || {
    echo "DOWNLOAD_FAIL ${host}" >&2
    continue
  }
done

python "${COLLECTOR}" merge \
  --parts-dir "${PARTS_DIR}" \
  --output "${OUT_DIR}/aaai_new3_minute60_run_level.csv"
