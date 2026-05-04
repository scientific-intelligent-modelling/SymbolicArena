#!/usr/bin/env bash
set -euo pipefail
BATCH_NAME="${1:-core50_llmsr_drsr_modelsplit_rerun_20260503}"
REMOTE_ROOT="/home/zhangziwen/workplace/scientific-intelligent-modelling"
ASSET_REL="exp-planning/04.Core50正式全量评测/generated/core50_llmsr_drsr_modelsplit_rerun_20260503"
for host in iaaccn23 iaaccn24; do
  rsync -a "exp-planning/04.Core50正式全量评测/generated/core50_llmsr_drsr_modelsplit_rerun_20260503/" "$host:$REMOTE_ROOT/$ASSET_REL/"
done
ssh iaaccn23 "cd $REMOTE_ROOT && tmux new-session -d -s core50_llm_base_20260503 /bin/bash $REMOTE_ROOT/$ASSET_REL/remote_jobs/run_iaaccn23_base.sh $BATCH_NAME"
ssh iaaccn24 "cd $REMOTE_ROOT && tmux new-session -d -s core50_llm_turbo_20260503 /bin/bash $REMOTE_ROOT/$ASSET_REL/remote_jobs/run_iaaccn24_turbo.sh $BATCH_NAME"
echo "launched $BATCH_NAME"
