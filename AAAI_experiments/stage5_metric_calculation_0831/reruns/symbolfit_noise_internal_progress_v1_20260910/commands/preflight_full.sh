#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
ASSET_ROOT="$REPO_ROOT/AAAI_experiments/stage5_metric_calculation_0831/reruns/symbolfit_noise_internal_progress_v1_20260910"
FORMAL_ROOT="$REPO_ROOT/benchmark-runs/formal3h/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658"

cd "$REPO_ROOT"
PYTHONPATH=. python check/run_e1_candidate200_12alg_load_queue.py \
  --batch-name symbolfit_noise_internal_progress_v1_20260910_full \
  --source-csv "$FORMAL_ROOT/queues/ssr50_source.csv" \
  --expected-rows 50 \
  --queue-root "$ASSET_ROOT/queue" \
  --params-root "$FORMAL_ROOT/params" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --tools symbolfit \
  --seeds 520 521 522 \
  --noise-sigmas 0.01 0.05 \
  --controller-host iaaccn22 \
  --use-internal-ips \
  --max-jobs-per-host 64 \
  --max-cpu-used-ratio 0.95 \
  --max-new-jobs-per-host-per-poll 8 \
  --load-tier-new-jobs 0.50:8,0.70:6,0.85:4,0.95:2 \
  --max-load-ratio 0.95 \
  --max-memory-used-ratio 0.90 \
  --min-free-mem-gb 16 \
  --poll-seconds 60 \
  --session-prefix symbolfit_noise_internal_v1_ \
  --host-session-count-prefix symbolfit_noise_internal_v1_ \
  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \
  --remote-data-root /home/zhangziwen/sim-datasets-data \
  --preflight-host-timeout 180 \
  --preflight-report "$ASSET_ROOT/preflight_report.json" \
  --preflight-only
