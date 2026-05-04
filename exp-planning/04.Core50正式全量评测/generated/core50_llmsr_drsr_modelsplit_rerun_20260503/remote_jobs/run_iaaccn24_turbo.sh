#!/usr/bin/env bash
set -euo pipefail
BATCH_NAME="${1:-core50_llmsr_drsr_modelsplit_rerun_20260503}"
MODEL_KEY="turbo"
MODEL_LABEL="deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
HOST_LABEL="iaaccn24"
REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
ASSET_REL="exp-planning/04.Core50正式全量评测/generated/core50_llmsr_drsr_modelsplit_rerun_20260503"
ASSET_ROOT="$REMOTE_ROOT/$ASSET_REL"
OUT_BASE="$REMOTE_ROOT/experiments/$BATCH_NAME/$MODEL_KEY"
LOG_DIR="$OUT_BASE/__host_launcher__"
mkdir -p "$LOG_DIR"
cd "$REMOTE_ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
echo "batch=$BATCH_NAME host=$HOST_LABEL model=$MODEL_LABEL start=$(date -Is)" | tee "$LOG_DIR/host_start.log"
pids=()
names=()
echo "START llmsr_seed0 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed0.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed0/$HOST_LABEL" --seed 0 --workers 10 > "$LOG_DIR/llmsr_seed0.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed0.exit") &
pids+=("$!")
names+=("llmsr_seed0")
echo "START llmsr_seed1 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed1.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed1/$HOST_LABEL" --seed 1 --workers 10 > "$LOG_DIR/llmsr_seed1.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed1.exit") &
pids+=("$!")
names+=("llmsr_seed1")
echo "START llmsr_seed2 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed2.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed2/$HOST_LABEL" --seed 2 --workers 10 > "$LOG_DIR/llmsr_seed2.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed2.exit") &
pids+=("$!")
names+=("llmsr_seed2")
echo "START llmsr_seed3 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed3.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed3/$HOST_LABEL" --seed 3 --workers 10 > "$LOG_DIR/llmsr_seed3.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed3.exit") &
pids+=("$!")
names+=("llmsr_seed3")
echo "START llmsr_seed4 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed4.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed4/$HOST_LABEL" --seed 4 --workers 10 > "$LOG_DIR/llmsr_seed4.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed4.exit") &
pids+=("$!")
names+=("llmsr_seed4")
echo "START drsr_seed0 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed0.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed0/$HOST_LABEL" --seed 0 --workers 10 > "$LOG_DIR/drsr_seed0.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed0.exit") &
pids+=("$!")
names+=("drsr_seed0")
echo "START drsr_seed1 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed1.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed1/$HOST_LABEL" --seed 1 --workers 10 > "$LOG_DIR/drsr_seed1.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed1.exit") &
pids+=("$!")
names+=("drsr_seed1")
echo "START drsr_seed2 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed2.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed2/$HOST_LABEL" --seed 2 --workers 10 > "$LOG_DIR/drsr_seed2.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed2.exit") &
pids+=("$!")
names+=("drsr_seed2")
echo "START drsr_seed3 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed3.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed3/$HOST_LABEL" --seed 3 --workers 10 > "$LOG_DIR/drsr_seed3.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed3.exit") &
pids+=("$!")
names+=("drsr_seed3")
echo "START drsr_seed4 $(date -Is)" | tee -a "$LOG_DIR/host_start.log"
(conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed4.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed4/$HOST_LABEL" --seed 4 --workers 10 > "$LOG_DIR/drsr_seed4.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed4.exit") &
pids+=("$!")
names+=("drsr_seed4")
status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "$pid"; then
    echo "DONE $name $(date -Is)" | tee -a "$LOG_DIR/host_done.log"
  else
    rc="$?"
    echo "FAIL $name rc=$rc $(date -Is)" | tee -a "$LOG_DIR/host_done.log"
    status=1
  fi
done
echo "host complete status=$status end=$(date -Is)" | tee -a "$LOG_DIR/host_done.log"
exit "$status"
