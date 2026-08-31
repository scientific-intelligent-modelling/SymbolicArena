#!/usr/bin/env bash

# 在 iaaccn22 上运行：扫描本机并通过内网逐台扫描 23--29。
set -u

base_dir="${1:-/tmp/stage5_metric_0831}"
run_kind="${2:-inventory}"
scanner="${base_dir}/remote_snapshot.py"

if [[ "${run_kind}" == "freeze" ]]; then
    output_prefix="clean_freeze"
    scan_flags="--freeze-raw --outer-only"
elif [[ "${run_kind}" == "inventory" ]]; then
    output_prefix="clean_inventory"
    scan_flags=""
else
    printf '未知运行类型: %s（仅支持 inventory/freeze）\n' "${run_kind}" >&2
    exit 2
fi

failures=()

local_host="iaaccn22"
local_task_file="${base_dir}/${local_host}.jsonl"
local_output_file="${base_dir}/${output_prefix}_${local_host}.jsonl.gz"
local_report_file="${base_dir}/${output_prefix}_${local_host}.report.json"
if ! timeout 1800 python "${scanner}" \
    --tasks "${local_task_file}" \
    --output "${local_output_file}" \
    --report "${local_report_file}" \
    --horizon 180 ${scan_flags}; then
    printf 'SCAN_FAIL %s\n' "${local_host}"
    failures+=("${local_host}")
else
    printf 'SCANNED %s\n' "${local_host}"
fi

for suffix in 23 24 25 26 27 28 29; do
    host="iaaccn${suffix}"
    ip="10.10.100.${suffix}"
    task_file="${base_dir}/${host}.jsonl"
    output_file="${base_dir}/${output_prefix}_${host}.jsonl.gz"
    report_file="${base_dir}/${output_prefix}_${host}.report.json"

    if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
        "mkdir -p '${base_dir}'"; then
        printf 'PREP_FAIL %s\n' "${host}"
        failures+=("${host}")
        continue
    fi

    if ! timeout 120 scp -o BatchMode=yes -o ConnectTimeout=10 \
        "${scanner}" "${task_file}" "${ip}:${base_dir}/"; then
        printf 'SYNC_FAIL %s\n' "${host}"
        failures+=("${host}")
        continue
    fi

    if ! timeout 1800 ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
        "python '${scanner}' --tasks '${task_file}' --output '${output_file}' --report '${report_file}' --horizon 180 ${scan_flags}"; then
        printf 'SCAN_FAIL %s\n' "${host}"
        failures+=("${host}")
        continue
    fi

    if ! timeout 300 scp -o BatchMode=yes -o ConnectTimeout=10 \
        "${ip}:${output_file}" "${ip}:${report_file}" "${base_dir}/"; then
        printf 'FETCH_FAIL %s\n' "${host}"
        failures+=("${host}")
        continue
    fi

    printf 'SCANNED %s\n' "${host}"
done

if (( ${#failures[@]} > 0 )); then
    printf 'FAILED_HOSTS %s\n' "${failures[*]}"
    exit 1
fi
