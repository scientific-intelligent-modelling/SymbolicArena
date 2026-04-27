#!/usr/bin/env python3
"""Generate assets for the semantic Candidate-200 LLM rerun."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证" / "generated"
CANDIDATE200_CSV = GENERATED_ROOT / "candidate200_unified.csv"
OUTPUT_ROOT = GENERATED_ROOT / "semantic200_llm_v1"
REMOTE_PROJECT_ROOT = "/home/zhangziwen/projects/scientific-intelligent-modelling"
SEED = 1314
WORKERS = 50


PARAMS = {
    "llmsr": {
        "timeout_in_seconds": 3600,
        "progress_snapshot_interval_seconds": 60,
        "niterations": 100000,
        "samples_per_iteration": 4,
        "max_params": 10,
        "inject_prompt_semantics": True,
        "canonical_prompt_variables": True,
        "persist_all_samples": False,
        "llm_config_path": (
            f"{REMOTE_PROJECT_ROOT}/exp-planning/02.E1选择验证/"
            "llm_configs/benchmark_llm.config"
        ),
    },
    "drsr": {
        "timeout_in_seconds": 3600,
        "progress_snapshot_interval_seconds": 60,
        "niterations": 100000,
        "samples_per_iteration": 4,
        "max_params": 10,
        "inject_prompt_semantics": True,
        "canonical_prompt_variables": True,
        "persist_all_samples": False,
        "llm_config_path": (
            f"{REMOTE_PROJECT_ROOT}/exp-planning/02.E1选择验证/"
            "llm_configs/benchmark_llm.config"
        ),
    },
}


def _read_rows() -> list[dict[str, str]]:
    with open(CANDIDATE200_CSV, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 200:
        raise ValueError(f"Expected 200 candidate rows, got {len(rows)}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_params() -> None:
    params_dir = OUTPUT_ROOT / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    for tool, payload in PARAMS.items():
        (params_dir / f"{tool}_semantic.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _job_script(tool: str, host: str) -> str:
    rel_slice = f"exp-planning/02.E1选择验证/generated/semantic200_llm_v1/slices/{host}.csv"
    rel_params = f"exp-planning/02.E1选择验证/generated/semantic200_llm_v1/params/{tool}_semantic.json"
    return f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <BATCH_NAME> <WORKERS> [retry]" >&2
  exit 2
fi

BATCH_NAME="$1"
WORKERS="$2"
RETRY_MODE="${{3:-}}"
REMOTE_ROOT="{REMOTE_PROJECT_ROOT}"
EXTRA_ARGS=()
if [ "$RETRY_MODE" = "retry" ]; then
  EXTRA_ARGS+=(--retry-failed)
fi

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n sim_llm python check/launch_e1_benchmark.py run \\
  --tool {tool} \\
  --slice-csv "$REMOTE_ROOT/{rel_slice}" \\
  --params-json "$REMOTE_ROOT/{rel_params}" \\
  --output-root "$REMOTE_ROOT/experiments/${{BATCH_NAME}}/{tool}/{host}" \\
  --seed {SEED} \\
  --workers "$WORKERS" \\
  "${{EXTRA_ARGS[@]}}"
"""


def _write_jobs() -> None:
    jobs_dir = OUTPUT_ROOT / "remote_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for tool in ("llmsr", "drsr"):
        for host in ("iaaccn22", "iaaccn23"):
            path = jobs_dir / f"{tool}_{host}.sh"
            path.write_text(_job_script(tool, host), encoding="utf-8")
            path.chmod(0o755)


def _queue_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="{REMOTE_PROJECT_ROOT}"
REMOTE_HOST_23="${{REMOTE_HOST_23:-10.10.100.23}}"
STAMP="${{STAMP:-$(date +%Y%m%d-%H%M%S)}}"
BATCH_NAME="${{BATCH_NAME:-semantic200_llm_v1_seed{SEED}_${{STAMP}}}}"
WORKERS="${{WORKERS:-{WORKERS}}}"

echo "BATCH_NAME=${{BATCH_NAME}}"
echo "WORKERS=${{WORKERS}}"

start_local() {{
  local tool="$1"
  local session="semantic200_${{tool}}_22"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_v1/remote_jobs/${{tool}}_iaaccn22.sh"
  chmod +x "$script"
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session" /bin/bash "$script" "$BATCH_NAME" "$WORKERS"
  echo "STARTED iaaccn22 $session"
}}

start_remote23() {{
  local tool="$1"
  local session="semantic200_${{tool}}_23"
  local script="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/semantic200_llm_v1/remote_jobs/${{tool}}_iaaccn23.sh"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST_23" \\
    "chmod +x '$script'; tmux kill-session -t '$session' >/dev/null 2>&1 || true; tmux new-session -d -s '$session' /bin/bash '$script' '$BATCH_NAME' '$WORKERS'"
  echo "STARTED iaaccn23 $session"
}}

wait_local() {{
  local session="$1"
  while tmux has-session -t "$session" >/dev/null 2>&1; do
    sleep 60
  done
  echo "FINISHED iaaccn22 $session"
}}

wait_remote23() {{
  local session="$1"
  while timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST_23" \\
    "tmux has-session -t '$session' >/dev/null 2>&1"; do
    sleep 60
  done
  echo "FINISHED iaaccn23 $session"
}}

run_wave() {{
  local tool="$1"
  start_local "$tool"
  start_remote23 "$tool"
  wait_local "semantic200_${{tool}}_22" &
  local p1=$!
  wait_remote23 "semantic200_${{tool}}_23" &
  local p2=$!
  wait "$p1" "$p2"
  echo "WAVE_DONE $tool"
}}

cd "$REMOTE_ROOT"
run_wave llmsr
run_wave drsr
echo "QUEUE_DONE $BATCH_NAME"
"""


def _write_launch() -> None:
    launch_dir = OUTPUT_ROOT / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    path = launch_dir / "run_semantic200_llm_queue.sh"
    path.write_text(_queue_script(), encoding="utf-8")
    path.chmod(0o755)


def _write_manifest(rows_by_host: dict[str, list[dict[str, str]]]) -> None:
    lines = [
        "# semantic200_llm_v1",
        "",
        f"- seed: {SEED}",
        f"- workers_per_host: {WORKERS}",
        "- queue: llmsr on iaaccn22/iaaccn23, then drsr on iaaccn22/iaaccn23",
        "- prompt policy: x0/x1/.../y prompt variables with metadata semantics",
        "",
    ]
    for host, rows in rows_by_host.items():
        lines.append(f"- {host}: {len(rows)} datasets")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = _read_rows()
    fieldnames = list(rows[0].keys())
    rows_by_host = {
        "iaaccn22": [row for idx, row in enumerate(rows) if idx % 2 == 0],
        "iaaccn23": [row for idx, row in enumerate(rows) if idx % 2 == 1],
    }
    for host, host_rows in rows_by_host.items():
        if len(host_rows) != 100:
            raise ValueError(f"{host} expected 100 rows, got {len(host_rows)}")
        _write_csv(OUTPUT_ROOT / "slices" / f"{host}.csv", host_rows, fieldnames)
    _write_params()
    _write_jobs()
    _write_launch()
    _write_manifest(rows_by_host)
    print(f"Wrote semantic200 assets to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
