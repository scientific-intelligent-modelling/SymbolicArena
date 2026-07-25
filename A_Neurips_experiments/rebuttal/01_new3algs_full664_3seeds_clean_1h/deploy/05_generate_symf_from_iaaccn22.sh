#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
PARAM_DIR="$BATCH_DIR/symf/params"
SYMF_ROOT="$BATCH_DIR/symf/full664_7alg"
SHARD_ROOT="$SYMF_ROOT/shards"
SYMF_DIR="$SYMF_ROOT/final"
SHARD_ALGORITHMS=(dso fepysr imcts jaxsr pyoperon symbolfit udsr)
EXPECTED_ALGORITHM_KEYS="dso,fepysr,imcts,jaxsr,pyoperon,symbolfit,udsr"
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

mkdir -p "$SHARD_ROOT"
MERGE_ARGS=()
for algorithm in "${SHARD_ALGORITHMS[@]}"; do
  shard_dir="$SHARD_ROOT/$algorithm"
  shard_csv="$shard_dir/symbolic_metrics_formal.csv"
  if [[ -f "$shard_csv" ]] && python check/merge_symf_formal_shards.py \
    --params-csv "$PARAM_DIR/symf_formal_judge_parameters.csv" \
    --shard-csv "$shard_csv" \
    --expected-algorithm-keys "$algorithm" \
    --expected-runs 1992 \
    --expected-datasets 664 \
    --expected-seeds 520,521,522 \
    --expected-runs-per-algorithm 1992 \
    --validate-only >/dev/null; then
    echo "[symf] reuse validated shard: $algorithm"
  else
    if [[ -e "$shard_dir" ]]; then
      echo "[symf] invalid existing shard; refusing overwrite: $shard_dir" >&2
      exit 1
    fi
    shard_tmp="$SHARD_ROOT/.${algorithm}.tmp.$(date +%Y%m%d-%H%M%S).$$"
    python check/generate_symf_formal_metrics.py \
      --params-csv "$PARAM_DIR/symf_formal_judge_parameters.csv" \
      --run-level-csv "$BATCH_DIR/analysis/full664_7alg_run_level.csv" \
      --algorithms "$algorithm" \
      --expected-runs 1992 \
      --outdir "$shard_tmp"
    python check/merge_symf_formal_shards.py \
      --params-csv "$PARAM_DIR/symf_formal_judge_parameters.csv" \
      --shard-csv "$shard_tmp/symbolic_metrics_formal.csv" \
      --expected-algorithm-keys "$algorithm" \
      --expected-runs 1992 \
      --expected-datasets 664 \
      --expected-seeds 520,521,522 \
      --expected-runs-per-algorithm 1992 \
      --validate-only >/dev/null
    mv "$shard_tmp" "$shard_dir"
    echo "[symf] committed shard: $algorithm"
  fi
  MERGE_ARGS+=(--shard-csv "$shard_csv")
done

if [[ -d "$SYMF_DIR" ]] && python check/merge_symf_formal_shards.py \
  --params-csv "$PARAM_DIR/symf_formal_judge_parameters.csv" \
  --shard-csv "$SYMF_DIR/symbolic_metrics_formal.csv" \
  --expected-algorithm-keys "$EXPECTED_ALGORITHM_KEYS" \
  --expected-runs 13944 \
  --expected-datasets 664 \
  --expected-seeds 520,521,522 \
  --expected-runs-per-algorithm 1992 \
  --validate-only >/dev/null; then
  echo "[symf] reuse validated merged output"
else
  if [[ -e "$SYMF_DIR" ]]; then
    echo "[symf] invalid existing merged output; refusing overwrite: $SYMF_DIR" >&2
    exit 1
  fi
  merged_tmp="$SYMF_ROOT/.final.tmp.$(date +%Y%m%d-%H%M%S).$$"
  python check/merge_symf_formal_shards.py \
    --params-csv "$PARAM_DIR/symf_formal_judge_parameters.csv" \
    "${MERGE_ARGS[@]}" \
    --expected-algorithm-keys "$EXPECTED_ALGORITHM_KEYS" \
    --expected-runs 13944 \
    --expected-datasets 664 \
    --expected-seeds 520,521,522 \
    --expected-runs-per-algorithm 1992 \
    --outdir "$merged_tmp"
  mv "$merged_tmp" "$SYMF_DIR"
  echo "[symf] committed merged output"
fi

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
