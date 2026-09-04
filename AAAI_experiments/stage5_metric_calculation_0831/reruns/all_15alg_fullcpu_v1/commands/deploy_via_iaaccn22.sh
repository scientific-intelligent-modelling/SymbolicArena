#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
REMOTE_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
FILES=(
  'check/run_e1_candidate200_12alg_load_queue.py'
  'check/launch_e1_benchmark.py'
  'scientific_intelligent_modelling/benchmarks/runner.py'
  'scientific_intelligent_modelling/srkit/subprocess_runner.py'
  'scientific_intelligent_modelling/benchmarks/artifact_schema.py'
  'scientific_intelligent_modelling/benchmarks/normalizers.py'
  'scientific_intelligent_modelling/config/toolbox_config.json'
  'scientific_intelligent_modelling/algorithms/iMCTS_wrapper/MCTS-4-SR/iMCTS/regressor.py'
  'scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/pyoperon_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/fepysr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/jaxsr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/udsr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/ragsr_wrapper/wrapper.py'
  'scientific_intelligent_modelling/algorithms/symbolfit_wrapper/wrapper.py'
)
cd "$REPO_ROOT"
timeout 600 rsync -a --relative "${FILES[@]}" "iaaccn22:$REMOTE_ROOT/"
timeout 600 rsync -a --delete "AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1/" "iaaccn22:$REMOTE_ROOT/AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1/"
timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 "/bin/bash '$REMOTE_ROOT/AAAI_experiments/stage5_metric_calculation_0831/reruns/all_15alg_fullcpu_v1/commands/fanout_from_iaaccn22.sh'"
