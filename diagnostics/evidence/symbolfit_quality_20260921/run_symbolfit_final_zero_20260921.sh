#!/bin/bash
set -eu
root=/tmp/symbolfit_debug_20260921
mkdir -p "$root/final"
cp -a "$root/after/." "$root/final/"
cp /tmp/symbolfit_final_wrapper_20260921.py "$root/final/scientific_intelligent_modelling/algorithms/symbolfit_wrapper/wrapper.py"
cd "$root/final"
export PYTHONPATH=.
export PYTHON_JULIAPKG_PROJECT=/home/zhangziwen/pyjuliapkg_symbolfit
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
exec /home/zhangziwen/anaconda3/envs/sim_symbolfit/bin/python -u diagnostics/symbolfit_quality_probe.py \
  --case zero_mean --seconds 180 --fit-seconds 45 --output "$root/runs/final_zero_mean"
