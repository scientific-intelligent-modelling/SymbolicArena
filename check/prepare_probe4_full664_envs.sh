#!/usr/bin/env bash
set -euo pipefail

# 准备 Probe4 full-664 所需 conda 环境；只复制缺失环境，不启动实验任务。

SOURCE_HOST="${SOURCE_HOST:-iaaccn25}"
ENV_ROOT="${ENV_ROOT:-/home/zhangziwen/anaconda3/envs}"
SSH_COMMON=(-o BatchMode=yes -o ConnectTimeout=10)
INNER_SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

copy_env() {
  local env_name="$1"
  local dst_ip="$2"
  local dst_label="$3"
  local src_dir="${ENV_ROOT}/${env_name}"
  local dst_dir="${ENV_ROOT}/${env_name}"
  local tmp_dir="${ENV_ROOT}/${env_name}.probe4_tmp_$(date +%Y%m%d-%H%M%S)"

  echo "[check] ${env_name} source=${SOURCE_HOST}:${src_dir}"
  ssh "${SSH_COMMON[@]}" "${SOURCE_HOST}" "test -d '${src_dir}'"

  echo "[check] ${dst_label} ${env_name}"
  if ssh "${SSH_COMMON[@]}" "${SOURCE_HOST}" \
    "ssh ${INNER_SSH_OPTS} '${dst_ip}' 'test -e \"${dst_dir}\"'"; then
    echo "[skip] ${dst_label} already has ${env_name}"
    return 0
  fi

  echo "[prepare] ${dst_label} tmp=${tmp_dir}"
  ssh "${SSH_COMMON[@]}" "${SOURCE_HOST}" \
    "ssh ${INNER_SSH_OPTS} '${dst_ip}' 'mkdir -p \"${tmp_dir}\"'"

  echo "[rsync] ${env_name} -> ${dst_label}"
  ssh "${SSH_COMMON[@]}" "${SOURCE_HOST}" \
    "rsync -a --info=stats2 -e 'ssh ${INNER_SSH_OPTS}' '${src_dir}/' '${dst_ip}:${tmp_dir}/'"

  echo "[finalize] ${dst_label} ${env_name}"
  ssh "${SSH_COMMON[@]}" "${SOURCE_HOST}" \
    "ssh ${INNER_SSH_OPTS} '${dst_ip}' 'test ! -e \"${dst_dir}\" && mv \"${tmp_dir}\" \"${dst_dir}\" && conda run -n \"${env_name}\" python -V'"
}

copy_env sim_iMCTS 10.10.100.23 iaaccn23
copy_env sim_iMCTS 10.10.100.24 iaaccn24
copy_env sim_iMCTS 10.10.100.26 iaaccn26
copy_env sim_iMCTS 10.10.100.27 iaaccn27
copy_env sim_iMCTS 10.10.100.28 iaaccn28
copy_env sim_iMCTS 10.10.100.29 iaaccn29
copy_env sim_dso 10.10.100.29 iaaccn29

echo "[done] probe4 environment preparation completed"
