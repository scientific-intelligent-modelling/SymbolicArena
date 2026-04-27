#!/usr/bin/env python3
"""生成 E1 剩余 5 算法的 Candidate-200 全量分发资产。"""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "exp-planning/02.E1选择验证/generated/candidate200_unified.csv"
OUTPUT_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证/generated/remaining5_full200_v1"
REMOTE_ROOT = "/home/zhangziwen/projects/scientific-intelligent-modelling"
SEED = 1314
WORKERS = 50


JOBS = [
    {
        "tool": "e2esr",
        "params": "e2esr",
        "host": "iaaccn24",
        "env": "sim_e2esr",
        "session": "e1_remaining5_e2esr_24",
    },
    {
        "tool": "iMCTS",
        "params": "imcts",
        "host": "iaaccn25",
        "env": "sim_iMCTS",
        "session": "e1_remaining5_imcts_25",
    },
    {
        "tool": "QLattice",
        "params": "qlattice",
        "host": "iaaccn26",
        "env": "sim_qLattice",
        "session": "e1_remaining5_qlattice_26",
    },
    {
        "tool": "ragsr",
        "params": "ragsr",
        "host": "iaaccn27",
        "env": "sim_ragsr",
        "session": "e1_remaining5_ragsr_27",
    },
    {
        "tool": "udsr",
        "params": "udsr",
        "host": "iaaccn28",
        "env": "sim_dso",
        "session": "e1_remaining5_udsr_28",
    },
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 200:
        raise ValueError(f"期望 Candidate-200，实际 {len(rows)} 行: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_job_script(job: dict[str, str]) -> Path:
    script_path = OUTPUT_ROOT / "remote_jobs" / f"{job['params']}_{job['host']}.sh"
    slice_rel = f"exp-planning/02.E1选择验证/generated/remaining5_full200_v1/slices/{job['params']}/{job['host']}.csv"
    params_rel = f"exp-planning/02.E1选择验证/generated/params/{job['params']}.json"
    content = f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <BATCH_NAME> <WORKERS> [retry]" >&2
  exit 2
fi

BATCH_NAME="$1"
WORKERS="$2"
RETRY_MODE="${{3:-}}"
REMOTE_ROOT="{REMOTE_ROOT}"
EXTRA_ARGS=()
if [ "$RETRY_MODE" = "retry" ]; then
  EXTRA_ARGS+=(--retry-failed)
fi

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n {job['env']} python check/launch_e1_benchmark.py run \\
  --tool {job['tool']} \\
  --slice-csv "$REMOTE_ROOT/{slice_rel}" \\
  --params-json "$REMOTE_ROOT/{params_rel}" \\
  --output-root "$REMOTE_ROOT/experiments/${{BATCH_NAME}}/{job['params']}/{job['host']}" \\
  --seed {SEED} \\
  --workers "$WORKERS" \\
  "${{EXTRA_ARGS[@]}}"
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _write_bootstrap_script() -> Path:
    script_path = OUTPUT_ROOT / "remote_jobs" / "bootstrap_env_and_run.sh"
    content = f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <ENV_NAME> <JOB_SCRIPT> <BATCH_NAME> <WORKERS> [retry]" >&2
  exit 2
fi

ENV_NAME="$1"
JOB_SCRIPT="$2"
BATCH_NAME="$3"
WORKERS="$4"
RETRY_MODE="${{5:-retry}}"
REMOTE_ROOT="{REMOTE_ROOT}"

cd "$REMOTE_ROOT"
export PYTHONPATH=.
BOOTSTRAP_LOG_DIR="$REMOTE_ROOT/experiments/$BATCH_NAME/_bootstrap_logs"
mkdir -p "$BOOTSTRAP_LOG_DIR"
exec > >(tee -a "$BOOTSTRAP_LOG_DIR/${{ENV_NAME}}_$(hostname).log") 2>&1

if ! conda run -n "$ENV_NAME" python -V; then
  echo "[bootstrap] missing env: $ENV_NAME"
  BOOTSTRAP_PY="/tmp/e1_create_env_${{ENV_NAME}}.py"
  cat > "$BOOTSTRAP_PY" <<'PY'
import sys
from scientific_intelligent_modelling.srkit.conda_env_manager import env_manager

env_name = sys.argv[1]
ok = env_manager.create_environment(env_name)
raise SystemExit(0 if ok else 1)
PY
  PYTHONPATH=. conda run -n sim_base python "$BOOTSTRAP_PY" "$ENV_NAME"
fi

conda run -n "$ENV_NAME" python -V
/bin/bash "$JOB_SCRIPT" "$BATCH_NAME" "$WORKERS" "$RETRY_MODE"
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _write_launch_script() -> None:
    path = OUTPUT_ROOT / "launch" / "run_remaining5_full200.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'BATCH_NAME="${1:-e1_remaining5_full200_v1_$(date +%Y%m%d-%H%M%S)}"',
        f'WORKERS="${{2:-{WORKERS}}}"',
        f'REMOTE_ROOT="{REMOTE_ROOT}"',
        "",
        'echo "BATCH_NAME=${BATCH_NAME}"',
        'echo "WORKERS=${WORKERS}"',
        "",
    ]
    for job in JOBS:
        remote_script = f"{REMOTE_ROOT}/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/{job['params']}_{job['host']}.sh"
        bootstrap_script = f"{REMOTE_ROOT}/exp-planning/02.E1选择验证/generated/remaining5_full200_v1/remote_jobs/bootstrap_env_and_run.sh"
        host_suffix = job["host"].removeprefix("iaaccn")
        ssh_target = f"zhangziwen@10.10.100.{host_suffix}"
        lines.extend(
            [
                f'echo "[start] {job["tool"]} on {job["host"]} workers=${{WORKERS}}"',
                (
                    f"timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -J hub {ssh_target} "
                    f"'tmux new-session -d -s {job['session']} /bin/bash {bootstrap_script} "
                    f"\"'\"'{job['env']}\"'\"' \"'\"'{remote_script}\"'\"' "
                    f"\"'\"'${{BATCH_NAME}}'\"'\"' \"'\"'${{WORKERS}}'\"'\"' retry'"
                ),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _write_manifest(rows: list[dict[str, str]]) -> None:
    path = OUTPUT_ROOT / "remaining5_manifest.csv"
    manifest_rows = []
    for job in JOBS:
        manifest_rows.append(
            {
                "tool": job["tool"],
                "params": job["params"],
                "host": job["host"],
                "conda_env": job["env"],
                "workers": WORKERS,
                "tasks": len(rows),
                "slice_csv": f"slices/{job['params']}/{job['host']}.csv",
                "remote_job": f"remote_jobs/{job['params']}_{job['host']}.sh",
                "tmux_session": job["session"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    rows = _read_rows(SOURCE_CSV)
    for job in JOBS:
        _write_csv(OUTPUT_ROOT / "slices" / job["params"] / f"{job['host']}.csv", rows)
        _write_job_script(job)
    _write_bootstrap_script()
    _write_launch_script()
    _write_manifest(rows)
    readme = OUTPUT_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# E1 remaining5 full Candidate-200 assets",
                "",
                f"- source: `{SOURCE_CSV.relative_to(REPO_ROOT)}`",
                f"- seed: `{SEED}`",
                f"- workers per host: `{WORKERS}`",
                "- scope: remaining integrated algorithms without prior full Candidate-200 E1 run",
                "",
                "## Allocation",
                "",
                "| tool | host | env | tasks |",
                "|---|---:|---|---:|",
                *[
                    f"| `{job['tool']}` | `{job['host']}` | `{job['env']}` | `200` |"
                    for job in JOBS
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
