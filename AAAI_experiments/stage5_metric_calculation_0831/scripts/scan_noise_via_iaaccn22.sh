#!/usr/bin/env bash

# 在 iaaccn22 上并行冻结 iaaccn22~29 的 noise001/noise005 分钟轨迹。
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BASE_DIR="${1:-$REPO_ROOT/AAAI_experiments/stage5_metric_calculation_0831/work/noise_trajectory_freeze_v1}"
TASK_DIR="$BASE_DIR/tasks"
COLLECTED_DIR="$BASE_DIR/collected"
LOG_DIR="$BASE_DIR/logs"
SCANNER="$REPO_ROOT/AAAI_experiments/stage5_metric_calculation_0831/pipeline/remote_snapshot.py"
HORIZON="${HORIZON:-180}"
SCAN_TIMEOUT_SECONDS="${SCAN_TIMEOUT_SECONDS:-14400}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

mkdir -p "$COLLECTED_DIR" "$LOG_DIR"
test -f "$SCANNER"

scan_local() {
    local host="iaaccn22"
    local task_file="$TASK_DIR/$host.jsonl"
    local output_file="$COLLECTED_DIR/noise_freeze_$host.jsonl.gz"
    local report_file="$COLLECTED_DIR/noise_freeze_$host.report.json"
    test -f "$task_file"
    timeout "$SCAN_TIMEOUT_SECONDS" python "$SCANNER" \
        --tasks "$task_file" \
        --output "$output_file" \
        --report "$report_file" \
        --horizon "$HORIZON" \
        --freeze-raw
}

scan_remote() {
    local suffix="$1"
    local host="iaaccn$suffix"
    local ip="10.10.100.$suffix"
    local task_file="$TASK_DIR/$host.jsonl"
    local remote_dir="$BASE_DIR/remote/$host"
    local remote_scanner="$remote_dir/remote_snapshot.py"
    local remote_tasks="$remote_dir/$host.jsonl"
    local remote_output="$remote_dir/noise_freeze_$host.jsonl.gz"
    local remote_report="$remote_dir/noise_freeze_$host.report.json"

    test -f "$task_file"
    timeout 30 ssh "${SSH_OPTS[@]}" "$ip" "mkdir -p '$remote_dir'"
    timeout 300 scp "${SSH_OPTS[@]}" "$SCANNER" "$task_file" "$ip:$remote_dir/"
    timeout "$SCAN_TIMEOUT_SECONDS" ssh "${SSH_OPTS[@]}" "$ip" \
        "python '$remote_scanner' --tasks '$remote_tasks' --output '$remote_output' --report '$remote_report' --horizon '$HORIZON' --freeze-raw"
    timeout 900 scp "${SSH_OPTS[@]}" \
        "$ip:$remote_output" "$ip:$remote_report" "$COLLECTED_DIR/"
}

declare -a labels=()
declare -a pids=()

(scan_local) >"$LOG_DIR/iaaccn22.log" 2>&1 &
labels+=("iaaccn22")
pids+=("$!")

for suffix in 23 24 25 26 27 28 29; do
    host="iaaccn$suffix"
    (scan_remote "$suffix") >"$LOG_DIR/$host.log" 2>&1 &
    labels+=("$host")
    pids+=("$!")
done

failures=()
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'SCANNED %s\n' "${labels[$index]}"
    else
        printf 'SCAN_FAIL %s log=%s/%s.log\n' \
            "${labels[$index]}" "$LOG_DIR" "${labels[$index]}"
        failures+=("${labels[$index]}")
    fi
done

if (( ${#failures[@]} > 0 )); then
    printf 'FAILED_HOSTS %s\n' "${failures[*]}" >&2
    exit 1
fi

(
    cd "$BASE_DIR"
    find collected -maxdepth 1 -type f \
        \( -name 'noise_freeze_*.jsonl.gz' -o -name 'noise_freeze_*.report.json' \) \
        -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
printf 'FREEZE_COMPLETE output=%s manifest=%s\n' \
    "$COLLECTED_DIR" "$BASE_DIR/SHA256SUMS"
