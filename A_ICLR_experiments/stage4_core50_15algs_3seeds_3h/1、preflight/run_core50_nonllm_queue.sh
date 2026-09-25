#!/bin/bash
set -uo pipefail

ROOT="/home/family/workplace/scientific-intelligent-modelling"
cd "$ROOT"
exec /home/family/anaconda3/bin/python -u \
  A_ICLR_experiments/stage4_core50_15algs_3seeds_3h/1、preflight/run_core50_queue.py \
  --batch-name core50_new15_nonllm_v2 \
  --source-csv A_ICLR_experiments/stage4_core50_15algs_3seeds_3h/1、preflight/core50_new15_training_source.csv \
  --expected-rows 15 \
  --queue-root .agent/work/GOAL-CORE50/queue_nonllm_v2 \
  --params-root .agent/work/EXP-001/controller/params \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 iaaccn48 iaaccn49 iaaccn50 iaaccn51 iaaccn52 iaaccn53 iaaccn55 \
  --tools gplearn pyoperon pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit \
  --seeds 520 521 522 \
  --noise-sigmas 0 0.01 0.05 \
  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \
  --remote-data-root /home/zhangziwen/sim-datasets-data \
  --host-remote-root-overrides iaaccn48=/data1/zhangziwen/sim-runtime/code,iaaccn49=/data3/zhangziwen/sim-runtime/code,iaaccn50=/data1/zhangziwen/sim-runtime/code,iaaccn51=/data1/zhangziwen/sim-runtime/code,iaaccn52=/data1/zhangziwen/sim-runtime/code,iaaccn53=/data1/zhangziwen/sim-runtime/code,iaaccn55=/data1/zhangziwen/sim-runtime/code \
  --host-remote-data-root-overrides iaaccn48=/data1/zhangziwen/sim-datasets-data,iaaccn49=/data3/zhangziwen/sim-datasets-data,iaaccn50=/data1/zhangziwen/sim-datasets-data,iaaccn51=/data1/zhangziwen/sim-datasets-data,iaaccn52=/data1/zhangziwen/sim-datasets-data,iaaccn53=/data1/zhangziwen/sim-datasets-data,iaaccn55=/data1/zhangziwen/sim-datasets-data \
  --max-jobs-per-host 128 \
  --max-new-jobs-per-host-per-poll 32 \
  --max-cpu-used-ratio 0.95 \
  --load-tier-new-jobs 0.50:32,0.70:16,0.80:8,0.90:4,0.95:2 \
  --max-load-ratio 0.95 \
  --max-memory-used-ratio 0.85 \
  --min-free-mem-gb 32 \
  --retry-limit 5 \
  --llm-model-assignment from-params \
  --llm-model-bucket-limits base:0,turbo:0 \
  --condition-dispatch-mode sequential \
  --seed-dispatch-mode sequential \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --session-prefix core50_new15_nonllm_ \
  --host-session-count-prefix core50_new15_nonllm_
