#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${REPO_ROOT}/A_Neurips_experiments/rebuttal/02_six_axis_uncertainty_15algs"
SNAPSHOT_DIR="$(
  find "${REPO_ROOT}/A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/symf/source_snapshots" \
    -mindepth 1 -maxdepth 1 -type d | head -n 1
)"
GENERATOR="${SNAPSHOT_DIR}/generate_symf_formal_metrics.py"
REMOTE_TMP="/tmp/symbolicarena_new3_minute60_symf"

ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
  "mkdir -p '${REMOTE_TMP}/output'"
scp \
  "${GENERATOR}" \
  "${OUT_DIR}/symf_params_ssr50.csv" \
  "${OUT_DIR}/symf_clean_minute60_run_level.csv" \
  "iaaccn22:${REMOTE_TMP}/"

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
  "cd /home/zhangziwen/workplace/scientific-intelligent-modelling && \
   PYTHONPATH=. conda run -n sim_base python \
   '${REMOTE_TMP}/generate_symf_formal_metrics.py' \
   --params-csv '${REMOTE_TMP}/symf_params_ssr50.csv' \
   --run-level-csv '${REMOTE_TMP}/symf_clean_minute60_run_level.csv' \
   --algorithms fepysr,jaxsr,symbolfit \
   --prefer-run-level-expression \
   --require-frozen-formula-source \
   --expected-runs 450 \
   --outdir '${REMOTE_TMP}/output'"; then
  # The remote sim_base environment lacks optional `tabulate`. The generator
  # writes and hashes every CSV/JSON result before rendering its README, so a
  # README-only failure is acceptable after all required outputs are present.
  ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 \
    "test -s '${REMOTE_TMP}/output/symbolic_metrics_formal.csv' && \
     test -s '${REMOTE_TMP}/output/symbolic_metrics_formal_dataset_summary.csv' && \
     test -s '${REMOTE_TMP}/output/symbolic_metrics_formal_algorithm_summary.csv' && \
     test -s '${REMOTE_TMP}/output/symbolic_metrics_formal_summary.json' && \
     test -s '${REMOTE_TMP}/output/symbolic_metrics_formal_provenance.json'" || exit 1
  echo "SYM-F metrics are complete; only optional remote README rendering failed."
fi

mkdir -p "${OUT_DIR}/symf_1h"
scp -r "iaaccn22:${REMOTE_TMP}/output/." "${OUT_DIR}/symf_1h/"
