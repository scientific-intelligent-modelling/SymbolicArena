#!/usr/bin/env bash
set -euo pipefail
BATCH_NAME="${1:-core50_llmsr_drsr_modelsplit_rerun_20260503}"
MODEL_KEY="turbo"
HOST_LABEL="iaaccn24"
MAIN_SESSION="core50_llm_turbo_20260503"
REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
ASSET_REL="exp-planning/04.Core50正式全量评测/generated/core50_llmsr_drsr_modelsplit_rerun_20260503"
ASSET_ROOT="$REMOTE_ROOT/$ASSET_REL"
OUT_BASE="$REMOTE_ROOT/experiments/$BATCH_NAME/$MODEL_KEY"
LOG_DIR="$OUT_BASE/__retry_monitor__"
mkdir -p "$LOG_DIR"
cd "$REMOTE_ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "monitor start $(date -Is), waiting for $MAIN_SESSION" | tee -a "$LOG_DIR/monitor.log"
while tmux has-session -t "$MAIN_SESSION" >/dev/null 2>&1; do sleep 300; done
echo "main session ended $(date -Is)" | tee -a "$LOG_DIR/monitor.log"
for attempt in 1 2; do
  echo "retry attempt=$attempt start $(date -Is)" | tee -a "$LOG_DIR/monitor.log"
  pids=(); names=()
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed0.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed0/$HOST_LABEL" --seed 0 --workers 10 --retry-failed > "$LOG_DIR/llmsr_seed0.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed0.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("llmsr_seed0")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed1.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed1/$HOST_LABEL" --seed 1 --workers 10 --retry-failed > "$LOG_DIR/llmsr_seed1.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed1.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("llmsr_seed1")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed2.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed2/$HOST_LABEL" --seed 2 --workers 10 --retry-failed > "$LOG_DIR/llmsr_seed2.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed2.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("llmsr_seed2")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed3.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed3/$HOST_LABEL" --seed 3 --workers 10 --retry-failed > "$LOG_DIR/llmsr_seed3.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed3.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("llmsr_seed3")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool llmsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/llmsr/seed4.csv" --params-json "$ASSET_ROOT/params/llmsr_$MODEL_KEY.json" --output-root "$OUT_BASE/llmsr/seed4/$HOST_LABEL" --seed 4 --workers 10 --retry-failed > "$LOG_DIR/llmsr_seed4.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/llmsr_seed4.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("llmsr_seed4")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed0.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed0/$HOST_LABEL" --seed 0 --workers 10 --retry-failed > "$LOG_DIR/drsr_seed0.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed0.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("drsr_seed0")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed1.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed1/$HOST_LABEL" --seed 1 --workers 10 --retry-failed > "$LOG_DIR/drsr_seed1.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed1.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("drsr_seed1")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed2.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed2/$HOST_LABEL" --seed 2 --workers 10 --retry-failed > "$LOG_DIR/drsr_seed2.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed2.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("drsr_seed2")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed3.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed3/$HOST_LABEL" --seed 3 --workers 10 --retry-failed > "$LOG_DIR/drsr_seed3.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed3.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("drsr_seed3")
  (conda run -n sim_llm python check/launch_e1_benchmark.py run --tool drsr --slice-csv "$ASSET_ROOT/slices/$MODEL_KEY/drsr/seed4.csv" --params-json "$ASSET_ROOT/params/drsr_$MODEL_KEY.json" --output-root "$OUT_BASE/drsr/seed4/$HOST_LABEL" --seed 4 --workers 10 --retry-failed > "$LOG_DIR/drsr_seed4.attempt${attempt}.log" 2>&1; echo "$?" > "$LOG_DIR/drsr_seed4.attempt${attempt}.exit") &
  pids+=("$!")
  names+=("drsr_seed4")
  status=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then status=1; echo "retry fail ${names[$i]} attempt=$attempt" | tee -a "$LOG_DIR/monitor.log"; fi
  done
  python - "$OUT_BASE" <<'PY' | tee "$LOG_DIR/summary_attempt${attempt}.json" | tee -a "$LOG_DIR/monitor.log"
import json, sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1])
total=0; counts=Counter(); files=0
for p in root.glob("*/*/*/__launcher__/task_status.jsonl"):
    files += 1
    latest={}
    for line in p.read_text(errors="replace").splitlines():
        try: d=json.loads(line)
        except Exception: continue
        latest[d.get("task_key", str(len(latest)))] = d
    for d in latest.values():
        counts[str(d.get("status") or "unknown")] += 1; total += 1
print(json.dumps({"status_files":files,"latest_records":total,"status_counts":dict(counts)}, ensure_ascii=False))
PY
  if grep -q ""latest_records": 250" "$LOG_DIR/summary_attempt${attempt}.json" && ! grep -q ""error"" "$LOG_DIR/summary_attempt${attempt}.json"; then
    echo "retry monitor complete without error status $(date -Is)" | tee -a "$LOG_DIR/monitor.log"
    exit 0
  fi
  sleep 60
done
echo "retry monitor finished attempts; inspect $LOG_DIR" | tee -a "$LOG_DIR/monitor.log"
