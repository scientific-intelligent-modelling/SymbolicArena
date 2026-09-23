set -eu
cd /home/family/workplace/scientific-intelligent-modelling
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$PWD/.agent/work/EXP-001/followup"
exec /home/family/anaconda3/bin/python -u A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/run_core80_followup.py --plan .agent/work/EXP-001/followup/base_plan.jsonl --plan-dir .agent/work/EXP-001/followup/downstream_plans --phase base --workers 32 --recover-interrupted >> .agent/work/EXP-001/followup/downstream_controller.log 2>&1
