set -eu
RUNTIME=/home/zhangziwen/sim-runtime/core50-opus-runtime
ROOT="$RUNTIME/core50_comparisons"
PYTHON=/home/zhangziwen/anaconda3/envs/sim_base/bin/python
export PYTHONPATH="$RUNTIME:$RUNTIME/vendor"
export TMPDIR="$RUNTIME/work"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$RUNTIME"
case "${1:?mode is required}" in
  prepare)
    exec "$PYTHON" -u "$ROOT/bin/run_core50_comparisons.py" prepare --prepare-workers 50 >> "$ROOT/prepare.log" 2>&1
    ;;
  run)
    exec "$PYTHON" -u "$ROOT/bin/run_core50_comparisons.py" run --workers 300 >> "$ROOT/api.log" 2>&1
    ;;
  revalidate)
    exec "$PYTHON" -u "$ROOT/bin/revalidate_core50_opus.py" >> "$ROOT/revalidate.log" 2>&1
    ;;
  extend)
    exec "$PYTHON" -u "$ROOT/bin/run_core50_comparisons.py" extend --prepare-workers 50 --workers 300 >> "$ROOT/extension.log" 2>&1
    ;;
  continue)
    exec "$PYTHON" -u "$ROOT/bin/continue_core50_terminal.py" >> "$ROOT/continuation.log" 2>&1
    ;;
  revalidate-comparisons)
    exec "$PYTHON" -u "$ROOT/bin/revalidate_core50_comparison_format.py"
    ;;
  resume-serialization)
    exec "$PYTHON" -u "$ROOT/bin/resume_core50_serialization.py"
    ;;
  minute-prepare)
    exec "$PYTHON" -u "$ROOT/bin/run_core50_minute_opus.py" prepare >> "$RUNTIME/core50_minutes/prepare.log" 2>&1
    ;;
  minute-continue)
    exec "$PYTHON" -u "$ROOT/bin/continue_core50_minutes.py" >> "$RUNTIME/core50_minutes/pipeline.log" 2>&1
    ;;
  *)
    exit 2
    ;;
esac
