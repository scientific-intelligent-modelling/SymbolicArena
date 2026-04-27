#!/usr/bin/env bash
set -euo pipefail

BATCH_NAME="${1:-e1_remaining5_full200_v1_$(date +%Y%m%d-%H%M%S)}"
WORKERS="${2:-50}"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"

echo "BATCH_NAME=${BATCH_NAME}"
echo "WORKERS=${WORKERS}"

echo "[start] e2esr on iaaccn24 workers=${WORKERS}"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -J hub zhangziwen@10.10.100.24 'tmux new-session -d -s e1_remaining5_e2esr_24 /bin/bash /home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/bootstrap_env_and_run.sh "'"'sim_e2esr"'"' "'"'/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/e2esr_iaaccn24.sh"'"' "'"'${BATCH_NAME}'"'"' "'"'${WORKERS}'"'"' retry'

echo "[start] iMCTS on iaaccn25 workers=${WORKERS}"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -J hub zhangziwen@10.10.100.25 'tmux new-session -d -s e1_remaining5_imcts_25 /bin/bash /home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/bootstrap_env_and_run.sh "'"'sim_iMCTS"'"' "'"'/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/imcts_iaaccn25.sh"'"' "'"'${BATCH_NAME}'"'"' "'"'${WORKERS}'"'"' retry'

echo "[start] QLattice on iaaccn26 workers=${WORKERS}"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -J hub zhangziwen@10.10.100.26 'tmux new-session -d -s e1_remaining5_qlattice_26 /bin/bash /home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/bootstrap_env_and_run.sh "'"'sim_qLattice"'"' "'"'/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/qlattice_iaaccn26.sh"'"' "'"'${BATCH_NAME}'"'"' "'"'${WORKERS}'"'"' retry'

echo "[start] ragsr on iaaccn27 workers=${WORKERS}"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -J hub zhangziwen@10.10.100.27 'tmux new-session -d -s e1_remaining5_ragsr_27 /bin/bash /home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/bootstrap_env_and_run.sh "'"'sim_ragsr"'"' "'"'/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/ragsr_iaaccn27.sh"'"' "'"'${BATCH_NAME}'"'"' "'"'${WORKERS}'"'"' retry'

echo "[start] udsr on iaaccn28 workers=${WORKERS}"
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -J hub zhangziwen@10.10.100.28 'tmux new-session -d -s e1_remaining5_udsr_28 /bin/bash /home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/bootstrap_env_and_run.sh "'"'sim_dso"'"' "'"'/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/udsr_iaaccn28.sh"'"' "'"'${BATCH_NAME}'"'"' "'"'${WORKERS}'"'"' retry'

