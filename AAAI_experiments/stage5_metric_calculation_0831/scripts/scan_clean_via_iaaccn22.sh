#!/usr/bin/env bash

# 在 iaaccn22 上运行：通过内网逐台扫描 23--29，单机失败时继续并记录。
set -u

base_dir="${1:-/tmp/stage5_metric_0831}"
scanner="${base_dir}/remote_snapshot.py"

for suffix in 23 24 25 26 27 28 29; do
    host="iaaccn${suffix}"
    ip="10.10.100.${suffix}"
    task_file="${base_dir}/${host}.jsonl"
    output_file="${base_dir}/clean_inventory_${host}.jsonl.gz"
    report_file="${base_dir}/clean_inventory_${host}.report.json"

    if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
        "mkdir -p '${base_dir}'"; then
        printf 'PREP_FAIL %s\n' "${host}"
        continue
    fi

    if ! timeout 120 scp -o BatchMode=yes -o ConnectTimeout=10 \
        "${scanner}" "${task_file}" "${ip}:${base_dir}/"; then
        printf 'SYNC_FAIL %s\n' "${host}"
        continue
    fi

    if ! timeout 1800 ssh -o BatchMode=yes -o ConnectTimeout=10 "${ip}" \
        "python '${scanner}' --tasks '${task_file}' --output '${output_file}' --report '${report_file}' --horizon 180"; then
        printf 'SCAN_FAIL %s\n' "${host}"
        continue
    fi

    if ! timeout 300 scp -o BatchMode=yes -o ConnectTimeout=10 \
        "${ip}:${output_file}" "${ip}:${report_file}" "${base_dir}/"; then
        printf 'FETCH_FAIL %s\n' "${host}"
        continue
    fi

    printf 'SCANNED %s\n' "${host}"
done

