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
    exec "$PYTHON" -u "$ROOT/bin/run_core50_comparisons.py" prepare --prepare-workers 12 >> "$ROOT/prepare.log" 2>&1
    ;;
  run)
    exec "$PYTHON" -u "$ROOT/bin/run_core50_comparisons.py" run --workers 150 >> "$ROOT/api.log" 2>&1
    ;;
  revalidate)
    exec "$PYTHON" -u "$ROOT/bin/revalidate_core50_opus.py" >> "$ROOT/revalidate.log" 2>&1
    ;;
  extend)
    exec "$PYTHON" -u "$ROOT/bin/run_core50_comparisons.py" extend --prepare-workers 12 --workers 150 >> "$ROOT/extension.log" 2>&1
    ;;
  *)
    exit 2
    ;;
esac
