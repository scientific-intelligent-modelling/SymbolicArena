#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
BATCH_DIR="A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
PARAM_DIR="$BATCH_DIR/symf/params"
SYMF_ROOT="$BATCH_DIR/symf/full664_7alg"
SHARD_ALGORITHMS=(dso fepysr imcts jaxsr pyoperon symbolfit udsr)
EXPECTED_ALGORITHM_KEYS="dso,fepysr,imcts,jaxsr,pyoperon,symbolfit,udsr"
SOURCE_PARAMS_CSV="$PARAM_DIR/symf_formal_judge_parameters.csv"
SOURCE_RUN_LEVEL_CSV="$BATCH_DIR/analysis/full664_7alg_run_level.csv"
SOURCE_PERFORMANCE_CSV="$BATCH_DIR/analysis/full664_7alg_leaderboard.csv"
SOURCE_SNAPSHOT_ROOT="$BATCH_DIR/symf/source_snapshots"
cd "$REMOTE_ROOT"

mkdir -p "$BATCH_DIR/symf"
exec 9>"$BATCH_DIR/symf/.generate_full664_7alg.lock"
if ! flock -n 9; then
  echo "[symf] another final SYM-F process holds the lock" >&2
  exit 1
fi

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

params_sha256="$(sha256sum "$SOURCE_PARAMS_CSV" | awk '{print $1}')"
run_level_sha256="$(sha256sum "$SOURCE_RUN_LEVEL_CSV" | awk '{print $1}')"
performance_sha256="$(
  sha256sum "$SOURCE_PERFORMANCE_CSV" | awk '{print $1}'
)"
generator_sha256="$(
  sha256sum check/generate_symf_formal_metrics.py | awk '{print $1}'
)"
merger_sha256="$(
  sha256sum check/merge_symf_formal_shards.py | awk '{print $1}'
)"
leaderboard_merger_sha256="$(
  sha256sum check/merge_neurips_rebuttal_metrics.py | awk '{print $1}'
)"
params_hash_prefix="${params_sha256:0:20}"
run_level_hash_prefix="${run_level_sha256:0:20}"
performance_hash_prefix="${performance_sha256:0:20}"
generator_hash_prefix="${generator_sha256:0:20}"
merger_hash_prefix="${merger_sha256:0:20}"
leaderboard_merger_hash_prefix="${leaderboard_merger_sha256:0:20}"
snapshot_id="$params_hash_prefix-$run_level_hash_prefix-$performance_hash_prefix-$generator_hash_prefix-$merger_hash_prefix-$leaderboard_merger_hash_prefix"
snapshot_dir="$SOURCE_SNAPSHOT_ROOT/$snapshot_id"
mkdir -p "$SOURCE_SNAPSHOT_ROOT"
if [[ ! -e "$snapshot_dir" ]]; then
  snapshot_tmp="$SOURCE_SNAPSHOT_ROOT/.${snapshot_id}.tmp.$$"
  mkdir "$snapshot_tmp"
  cp "$SOURCE_PARAMS_CSV" "$snapshot_tmp/params.csv"
  cp "$SOURCE_RUN_LEVEL_CSV" "$snapshot_tmp/run_level.csv"
  cp "$SOURCE_PERFORMANCE_CSV" "$snapshot_tmp/performance.csv"
  cp check/generate_symf_formal_metrics.py \
    "$snapshot_tmp/generate_symf_formal_metrics.py"
  cp check/merge_symf_formal_shards.py \
    "$snapshot_tmp/merge_symf_formal_shards.py"
  cp check/merge_neurips_rebuttal_metrics.py \
    "$snapshot_tmp/merge_neurips_rebuttal_metrics.py"
  test "$(
    sha256sum "$snapshot_tmp/params.csv" | awk '{print $1}'
  )" = "$params_sha256"
  test "$(
    sha256sum "$snapshot_tmp/run_level.csv" | awk '{print $1}'
  )" = "$run_level_sha256"
  test "$(
    sha256sum "$snapshot_tmp/performance.csv" | awk '{print $1}'
  )" = "$performance_sha256"
  test "$(
    sha256sum "$snapshot_tmp/generate_symf_formal_metrics.py" \
      | awk '{print $1}'
  )" = "$generator_sha256"
  test "$(
    sha256sum "$snapshot_tmp/merge_symf_formal_shards.py" \
      | awk '{print $1}'
  )" = "$merger_sha256"
  test "$(
    sha256sum "$snapshot_tmp/merge_neurips_rebuttal_metrics.py" \
      | awk '{print $1}'
  )" = "$leaderboard_merger_sha256"
  jq -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg params_sha256 "$params_sha256" \
    --arg run_level_sha256 "$run_level_sha256" \
    --arg performance_sha256 "$performance_sha256" \
    --arg generator_sha256 "$generator_sha256" \
    --arg merger_sha256 "$merger_sha256" \
    --arg leaderboard_merger_sha256 "$leaderboard_merger_sha256" \
    '{
      schema_version: 1,
      created_at: $created_at,
      params_sha256: $params_sha256,
      run_level_sha256: $run_level_sha256,
      performance_sha256: $performance_sha256,
      generator_sha256: $generator_sha256,
      merger_sha256: $merger_sha256,
      leaderboard_merger_sha256: $leaderboard_merger_sha256,
      expression_source: "run_level.expression_canonical"
    }' > "$snapshot_tmp/source_snapshot.json"
  mv "$snapshot_tmp" "$snapshot_dir"
fi

PARAMS_CSV="$snapshot_dir/params.csv"
RUN_LEVEL_CSV="$snapshot_dir/run_level.csv"
PERFORMANCE_CSV="$snapshot_dir/performance.csv"
GENERATOR_SCRIPT="$snapshot_dir/generate_symf_formal_metrics.py"
MERGER_SCRIPT="$snapshot_dir/merge_symf_formal_shards.py"
LEADERBOARD_MERGER_SCRIPT="$snapshot_dir/merge_neurips_rebuttal_metrics.py"
SOURCE_OUTPUT_ROOT="$SYMF_ROOT/by_source/$snapshot_id"
SHARD_ROOT="$SOURCE_OUTPUT_ROOT/shards"
SYMF_DIR="$SOURCE_OUTPUT_ROOT/final"
test "$(sha256sum "$PARAMS_CSV" | awk '{print $1}')" = "$params_sha256"
test "$(
  sha256sum "$RUN_LEVEL_CSV" | awk '{print $1}'
)" = "$run_level_sha256"
test "$(sha256sum "$PERFORMANCE_CSV" | awk '{print $1}')" \
  = "$performance_sha256"
test "$(sha256sum "$GENERATOR_SCRIPT" | awk '{print $1}')" \
  = "$generator_sha256"
test "$(sha256sum "$MERGER_SCRIPT" | awk '{print $1}')" \
  = "$merger_sha256"
test "$(sha256sum "$LEADERBOARD_MERGER_SCRIPT" | awk '{print $1}')" \
  = "$leaderboard_merger_sha256"
jq -e \
  --arg params_sha256 "$params_sha256" \
  --arg run_level_sha256 "$run_level_sha256" \
  --arg performance_sha256 "$performance_sha256" \
  --arg generator_sha256 "$generator_sha256" \
  --arg merger_sha256 "$merger_sha256" \
  --arg leaderboard_merger_sha256 "$leaderboard_merger_sha256" \
  '
    (.schema_version == 1)
    and (.params_sha256 == $params_sha256)
    and (.run_level_sha256 == $run_level_sha256)
    and (.performance_sha256 == $performance_sha256)
    and (.generator_sha256 == $generator_sha256)
    and (.merger_sha256 == $merger_sha256)
    and (
      .leaderboard_merger_sha256
      == $leaderboard_merger_sha256
    )
    and (.expression_source == "run_level.expression_canonical")
  ' "$snapshot_dir/source_snapshot.json" >/dev/null

PROVENANCE_ARGS=(
  --source-run-level-csv "$RUN_LEVEL_CSV"
  --generator-script "$GENERATOR_SCRIPT"
  --expected-expression-source run_level.expression_canonical
)

run_snapshot_python() {
  SIM_REPO_ROOT="$REMOTE_ROOT" python "$@"
}

mkdir -p "$SHARD_ROOT"
MERGE_ARGS=()
for algorithm in "${SHARD_ALGORITHMS[@]}"; do
  shard_dir="$SHARD_ROOT/$algorithm"
  shard_csv="$shard_dir/symbolic_metrics_formal.csv"
  if [[ -f "$shard_csv" ]] && run_snapshot_python "$MERGER_SCRIPT" \
    --params-csv "$PARAMS_CSV" \
    --shard-csv "$shard_csv" \
    --expected-algorithm-keys "$algorithm" \
    --expected-runs 1992 \
    --expected-datasets 664 \
    --expected-seeds 520,521,522 \
    --expected-runs-per-algorithm 1992 \
    "${PROVENANCE_ARGS[@]}" \
    --validate-only >/dev/null; then
    echo "[symf] reuse validated shard: $algorithm"
  else
    if [[ -e "$shard_dir" ]]; then
      echo "[symf] invalid existing shard; refusing overwrite: $shard_dir" >&2
      exit 1
    fi
    shard_tmp="$SHARD_ROOT/.${algorithm}.tmp.$(date +%Y%m%d-%H%M%S).$$"
    run_snapshot_python "$GENERATOR_SCRIPT" \
      --params-csv "$PARAMS_CSV" \
      --run-level-csv "$RUN_LEVEL_CSV" \
      --algorithms "$algorithm" \
      --expected-runs 1992 \
      --prefer-run-level-expression \
      --require-frozen-formula-source \
      --outdir "$shard_tmp"
    run_snapshot_python "$MERGER_SCRIPT" \
      --params-csv "$PARAMS_CSV" \
      --shard-csv "$shard_tmp/symbolic_metrics_formal.csv" \
      --expected-algorithm-keys "$algorithm" \
      --expected-runs 1992 \
      --expected-datasets 664 \
      --expected-seeds 520,521,522 \
      --expected-runs-per-algorithm 1992 \
      "${PROVENANCE_ARGS[@]}" \
      --validate-only >/dev/null
    mv "$shard_tmp" "$shard_dir"
    echo "[symf] committed shard: $algorithm"
  fi
  MERGE_ARGS+=(--shard-csv "$shard_csv")
done

if [[ -d "$SYMF_DIR" ]] && run_snapshot_python "$MERGER_SCRIPT" \
  --params-csv "$PARAMS_CSV" \
  "${MERGE_ARGS[@]}" \
  --expected-algorithm-keys "$EXPECTED_ALGORITHM_KEYS" \
  --expected-runs 13944 \
  --expected-datasets 664 \
  --expected-seeds 520,521,522 \
  --expected-runs-per-algorithm 1992 \
  "${PROVENANCE_ARGS[@]}" \
  --validate-only \
  --verify-output-dir "$SYMF_DIR" >/dev/null; then
  echo "[symf] reuse validated merged output"
else
  if [[ -e "$SYMF_DIR" ]]; then
    echo "[symf] invalid existing merged output; refusing overwrite: $SYMF_DIR" >&2
    exit 1
  fi
  merged_tmp="$SOURCE_OUTPUT_ROOT/.final.tmp.$(date +%Y%m%d-%H%M%S).$$"
  run_snapshot_python "$MERGER_SCRIPT" \
    --params-csv "$PARAMS_CSV" \
    "${MERGE_ARGS[@]}" \
    --expected-algorithm-keys "$EXPECTED_ALGORITHM_KEYS" \
    --expected-runs 13944 \
    --expected-datasets 664 \
    --expected-seeds 520,521,522 \
    --expected-runs-per-algorithm 1992 \
    "${PROVENANCE_ARGS[@]}" \
    --outdir "$merged_tmp"
  mv "$merged_tmp" "$SYMF_DIR"
  echo "[symf] committed merged output"
fi

run_snapshot_python "$MERGER_SCRIPT" \
  --params-csv "$PARAMS_CSV" \
  "${MERGE_ARGS[@]}" \
  --expected-algorithm-keys "$EXPECTED_ALGORITHM_KEYS" \
  --expected-runs 13944 \
  --expected-datasets 664 \
  --expected-seeds 520,521,522 \
  --expected-runs-per-algorithm 1992 \
  "${PROVENANCE_ARGS[@]}" \
  --validate-only \
  --verify-output-dir "$SYMF_DIR" >/dev/null

jq -e '
  (.runs == 13944)
  and (.datasets == 664)
  and (.algorithms == 7)
  and (.params_datasets == 664)
' "$SYMF_DIR/symbolic_metrics_formal_summary.json" >/dev/null

LEADERBOARD_DIR="$SOURCE_OUTPUT_ROOT/leaderboard"
COMBINED_CSV="$LEADERBOARD_DIR/full664_7alg_leaderboard_with_symf.csv"
COMBINED_SUMMARY="$LEADERBOARD_DIR/full664_7alg_leaderboard_with_symf.summary.json"
symbolic_csv="$SYMF_DIR/symbolic_metrics_formal_algorithm_summary.csv"
symbolic_sha256="$(sha256sum "$symbolic_csv" | awk '{print $1}')"
if [[ -d "$LEADERBOARD_DIR" ]] && jq -e \
  --arg performance_sha256 "$performance_sha256" \
  --arg symbolic_sha256 "$symbolic_sha256" \
  --arg output_sha256 "$(sha256sum "$COMBINED_CSV" | awk '{print $1}')" \
  '
    (.algorithms == 7)
    and (
      (.algorithm_keys | sort)
      == ["dso", "fepysr", "imcts", "jaxsr", "pyoperon", "symbolfit", "udsr"]
    )
    and (.performance_sha256 == $performance_sha256)
    and (.symbolic_sha256 == $symbolic_sha256)
    and (.output_sha256 == $output_sha256)
  ' "$COMBINED_SUMMARY" >/dev/null; then
  echo "[symf] reuse validated combined leaderboard"
else
  if [[ -e "$LEADERBOARD_DIR" ]]; then
    echo "[symf] invalid combined leaderboard; refusing overwrite" >&2
    exit 1
  fi
  leaderboard_tmp="$SOURCE_OUTPUT_ROOT/.leaderboard.tmp.$(date +%Y%m%d-%H%M%S).$$"
  mkdir "$leaderboard_tmp"
  combined_tmp="$leaderboard_tmp/full664_7alg_leaderboard_with_symf.csv"
  run_snapshot_python "$LEADERBOARD_MERGER_SCRIPT" \
    --performance-csv "$PERFORMANCE_CSV" \
    --symbolic-csv "$symbolic_csv" \
    --output-csv "$combined_tmp" \
    --expected-algorithms 7
  jq -e \
    --arg performance_sha256 "$performance_sha256" \
    --arg symbolic_sha256 "$symbolic_sha256" \
    --arg output_sha256 "$(sha256sum "$combined_tmp" | awk '{print $1}')" \
    '
      (.algorithms == 7)
      and (
        (.algorithm_keys | sort)
        == ["dso", "fepysr", "imcts", "jaxsr", "pyoperon", "symbolfit", "udsr"]
      )
      and (.performance_sha256 == $performance_sha256)
      and (.symbolic_sha256 == $symbolic_sha256)
      and (.output_sha256 == $output_sha256)
    ' "${combined_tmp%.csv}.summary.json" >/dev/null
  mv "$leaderboard_tmp" "$LEADERBOARD_DIR"
  echo "[symf] committed combined leaderboard"
fi

latest_csv="$BATCH_DIR/analysis/full664_7alg_leaderboard_with_symf.csv"
latest_summary="${latest_csv%.csv}.summary.json"
latest_csv_tmp="${latest_csv}.tmp.$$"
latest_summary_tmp="${latest_summary}.tmp.$$"
cp "$COMBINED_CSV" "$latest_csv_tmp"
cp "$COMBINED_SUMMARY" "$latest_summary_tmp"
mv "$latest_csv_tmp" "$latest_csv"
mv "$latest_summary_tmp" "$latest_summary"
