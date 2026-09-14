#!/usr/bin/env bash
set -uo pipefail

cd /home/family/workplace/scientific-intelligent-modelling

output_root="AAAI_experiments/stage5_metric_calculation_0831/audits/gpt56_full_opus_simplification_review_6800_20260915"
log_path="AAAI_experiments/stage5_metric_calculation_0831/audits/gpt56_full_opus_simplification_review_6800_20260915.run.log"
exit_path="AAAI_experiments/stage5_metric_calculation_0831/audits/gpt56_full_opus_simplification_review_6800_20260915.exit"

printf 'started_at=%s\n' "$(date --iso-8601=seconds)" > "${log_path}"
python check/run_gpt_full_opus_simplification_review.py run \
  --release-root AAAI_experiments/Core50_final_20260914 \
  --output-root "${output_root}" \
  --concurrency 32 \
  --timeout 300 \
  --retry-delay 2 \
  --reuse-anthropic-token-for-openai >> "${log_path}" 2>&1
status=$?
printf 'exit_code=%s\nfinished_at=%s\n' "${status}" "$(date --iso-8601=seconds)" > "${exit_path}"
exit "${status}"
