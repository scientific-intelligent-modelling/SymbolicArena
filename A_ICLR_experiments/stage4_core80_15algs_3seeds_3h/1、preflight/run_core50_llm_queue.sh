#!/bin/bash
set -uo pipefail

ROOT="/home/family/workplace/scientific-intelligent-modelling"
cd "$ROOT"
exec /home/family/anaconda3/bin/python -u \
  A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/run_core50_queue.py \
  --batch-name core50_new15_llm_v1 \
  --source-csv A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/core50_new15_training_source.csv \
  --expected-rows 15 \
  --queue-root .agent/work/GOAL-CORE50/queue_llm_v1 \
  --params-root .agent/work/EXP-001/controller/params \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn26 iaaccn27 iaaccn28 iaaccn29 iaaccn48 iaaccn49 iaaccn50 iaaccn51 iaaccn52 iaaccn53 iaaccn55 \
  --tools llmsr drsr \
  --seeds 520 521 522 \
  --noise-sigmas 0 0.01 0.05 \
  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \
  --remote-data-root /home/zhangziwen/sim-datasets-data \
  --host-remote-root-overrides iaaccn48=/data1/zhangziwen/sim-runtime/code,iaaccn49=/data3/zhangziwen/sim-runtime/code,iaaccn50=/data1/zhangziwen/sim-runtime/code,iaaccn51=/data1/zhangziwen/sim-runtime/code,iaaccn52=/data1/zhangziwen/sim-runtime/code,iaaccn53=/data1/zhangziwen/sim-runtime/code,iaaccn55=/data1/zhangziwen/sim-runtime/code \
  --host-remote-data-root-overrides iaaccn48=/data1/zhangziwen/sim-datasets-data,iaaccn49=/data3/zhangziwen/sim-datasets-data,iaaccn50=/data1/zhangziwen/sim-datasets-data,iaaccn51=/data1/zhangziwen/sim-datasets-data,iaaccn52=/data1/zhangziwen/sim-datasets-data,iaaccn53=/data1/zhangziwen/sim-datasets-data,iaaccn55=/data1/zhangziwen/sim-datasets-data \
  --max-jobs-per-host 128 \
  --max-new-jobs-per-host-per-poll 16 \
  --load-tier-new-jobs 0.50:16,0.70:8,0.80:4,0.90:2,0.95:1 \
  --max-load-ratio 0.95 \
  --max-memory-used-ratio 0.85 \
  --min-free-mem-gb 32 \
  --retry-limit 5 \
  --llm-model-assignment from-params \
  --llm-default-bucket turbo \
  --llm-model-bucket-limits base:0,turbo:100 \
  --condition-dispatch-mode sequential \
  --seed-dispatch-mode sequential \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --skip-support-sync \
  --session-prefix core50_new15_llm_ \
  --host-session-count-prefix core50_new15_
