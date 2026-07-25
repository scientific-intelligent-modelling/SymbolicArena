#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
PARAM_DIR="$BATCH_DIR/symf/params"
SYMF_DIR="$BATCH_DIR/symf/full664_7alg"
cd "$REMOTE_ROOT"

jq -e '
  (.final_ready == true)
  and (.new3.expected_tasks == 5976)
  and (.new3.present_results == 5976)
  and (.new3.identity_mismatches == 0)
  and (.audit_gate.valid == true)
  and (.stage3.rows == 7968)
  and (.stage3.raw_digest_rows == 7968)
' "$BATCH_DIR/analysis/analysis_summary.json" >/dev/null

python check/prepare_symf_formal_judge_params.py \
  --catalog-csv "$BATCH_DIR/manifest/datasets.csv" \
  --outdir "$PARAM_DIR" \
  --expected-datasets 664

jq -e '
  (.datasets == 664)
  and (.formula_parse_ok == 664)
  and (
    (.needs_review == 0)
    or (
      (.needs_review == 1)
      and (
        .unresolved_issue_counts[
          "formula_arg_mapped_by_remaining_position:F0->x"
        ] == 1
      )
    )
  )
' "$PARAM_DIR/symf_formal_judge_config.json" >/dev/null

python check/generate_symf_formal_metrics.py \
  --params-csv "$PARAM_DIR/symf_formal_judge_parameters.csv" \
  --run-level-csv "$BATCH_DIR/analysis/full664_7alg_run_level.csv" \
  --expected-runs 13944 \
  --outdir "$SYMF_DIR"

jq -e '
  (.runs == 13944)
  and (.datasets == 664)
  and (.algorithms == 7)
  and (.params_datasets == 664)
' "$SYMF_DIR/symbolic_metrics_formal_summary.json" >/dev/null

python check/merge_neurips_rebuttal_metrics.py \
  --performance-csv "$BATCH_DIR/analysis/full664_7alg_leaderboard.csv" \
  --symbolic-csv \
  "$SYMF_DIR/symbolic_metrics_formal_algorithm_summary.csv" \
  --output-csv \
  "$BATCH_DIR/analysis/full664_7alg_leaderboard_with_symf.csv" \
  --expected-algorithms 7

jq -e '
  (.algorithms == 7)
  and (
    (.algorithm_keys | sort)
    == ["dso", "fepysr", "imcts", "jaxsr", "pyoperon", "symbolfit", "udsr"]
  )
' "$BATCH_DIR/analysis/full664_7alg_leaderboard_with_symf.summary.json" \
  >/dev/null
