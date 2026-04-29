#!/usr/bin/env python3
"""生成 Probe-4 全量 664 数据集三种子分发资产。"""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "exp-planning/01.双探针实验/datasets_to_run.csv"
OUTPUT_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1"
REMOTE_ROOT = "/home/zhangziwen/projects/scientific-intelligent-modelling"
SEEDS = (520, 521, 522)
HOSTS = ("iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29")

TOOLS = {
    "udsr": {
        "tool_arg": "udsr",
        "params": "udsr",
        "env": "sim_dso",
        "workers": 15,
    },
    "dso": {
        "tool_arg": "dso",
        "params": "dso",
        "env": "sim_dso",
        "workers": 15,
    },
    "imcts": {
        "tool_arg": "iMCTS",
        "params": "imcts",
        "env": "sim_iMCTS",
        "workers": 24,
    },
    "pyoperon": {
        "tool_arg": "pyoperon",
        "params": "pyoperon",
        "env": "sim_base",
        "workers": 24,
    },
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 664:
        raise ValueError(f"期望 664 个全量数据集，实际 {len(rows)} 行: {path}")
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        dataset_dir = row["dataset_dir"]
        out = dict(row)
        out["global_index"] = str(index)
        out["dataset_rel"] = dataset_dir
        out["basename"] = row.get("dataset_name", "")
        out["pool"] = "full664"
        out["selection_mode"] = "full_pool"
        out["candidate_advantage_side"] = ""
        out["formula_py"] = str(Path(dataset_dir) / "formula.py") if row.get("has_formula_py") == "yes" else ""
        normalized.append(out)
    return normalized


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    splits = {host: [] for host in HOSTS}
    for index, row in enumerate(rows):
        splits[HOSTS[index % len(HOSTS)]].append(row)
    return splits


def _write_remote_job(tool_key: str, seed: int, host: str) -> Path:
    tool = TOOLS[tool_key]
    script_path = OUTPUT_ROOT / "remote_jobs" / tool_key / f"seed{seed}" / f"{tool_key}_seed{seed}_{host}.sh"
    slice_rel = f"exp-planning/02.E1选择验证/generated/probe4_full664_v1/slices/{tool_key}/seed{seed}/{host}.csv"
    params_rel = f"exp-planning/02.E1选择验证/generated/params/{tool['params']}.json"
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

conda run -n {tool['env']} python check/launch_e1_benchmark.py run \\
  --tool {tool['tool_arg']} \\
  --slice-csv "$REMOTE_ROOT/{slice_rel}" \\
  --params-json "$REMOTE_ROOT/{params_rel}" \\
  --output-root "$REMOTE_ROOT/experiments/${{BATCH_NAME}}/{tool_key}/seed{seed}/{host}" \\
  --seed {seed} \\
  --workers "$WORKERS" \\
  "${{EXTRA_ARGS[@]}}"
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _write_tool_seed_launcher() -> None:
    path = OUTPUT_ROOT / "launch" / "run_tool_seed.sh"
    valid_tools = " ".join(TOOLS)
    valid_seeds = " ".join(str(seed) for seed in SEEDS)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'if [ "$#" -lt 2 ]; then',
        '  echo "Usage: $0 <tool: udsr|dso|imcts|pyoperon> <seed: 520|521|522> [batch_name] [retry]" >&2',
        "  exit 2",
        "fi",
        "",
        'TOOL="$1"',
        'SEED="$2"',
        'BATCH_NAME="${3:-probe4_full664_v1_$(date +%Y%m%d-%H%M%S)}"',
        'RETRY_MODE="${4:-retry}"',
        f'REPO_ROOT="{REPO_ROOT}"',
        f'REMOTE_ROOT="{REMOTE_ROOT}"',
        "",
        f'case " {valid_tools} " in *" $TOOL "*) ;; *) echo "invalid tool: $TOOL" >&2; exit 2;; esac',
        f'case " {valid_seeds} " in *" $SEED "*) ;; *) echo "invalid seed: $SEED" >&2; exit 2;; esac',
        "",
        'case "$TOOL" in',
    ]
    for tool_key, tool in TOOLS.items():
        lines.append(f'  {tool_key}) DEFAULT_WORKERS="{tool["workers"]}" ;;')
    lines.extend(
        [
            "esac",
            'WORKERS="${WORKERS:-$DEFAULT_WORKERS}"',
            "",
            'echo "BATCH_NAME=${BATCH_NAME}"',
            'echo "TOOL=${TOOL}"',
            'echo "SEED=${SEED}"',
            'echo "WORKERS=${WORKERS}"',
            "",
            "start_job() {",
            '  local host="$1"',
            '  local session="probe4_full664_${TOOL}_s${SEED}_${host}"',
            '  local rel_slice="exp-planning/02.E1选择验证/generated/probe4_full664_v1/slices/${TOOL}/seed${SEED}/${host}.csv"',
            '  local rel_params="exp-planning/02.E1选择验证/generated/params/${TOOL}.json"',
            '  local remote_job="exp-planning/02.E1选择验证/generated/probe4_full664_v1/remote_jobs/${TOOL}/seed${SEED}/${TOOL}_seed${SEED}_${host}.sh"',
            '  if [ "$TOOL" = "imcts" ]; then rel_params="exp-planning/02.E1选择验证/generated/params/imcts.json"; fi',
            "",
            '  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p \\"$REMOTE_ROOT/check\\" \\"$REMOTE_ROOT/$(dirname "$rel_slice")\\" \\"$REMOTE_ROOT/$(dirname "$rel_params")\\" \\"$REMOTE_ROOT/$(dirname "$remote_job")\\""',
            '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/check/launch_e1_benchmark.py" "$host:$REMOTE_ROOT/check/launch_e1_benchmark.py"',
            '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_slice" "$host:$REMOTE_ROOT/$rel_slice"',
            '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_params" "$host:$REMOTE_ROOT/$rel_params"',
            '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$remote_job" "$host:$REMOTE_ROOT/$remote_job"',
            '  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "chmod +x \\"$REMOTE_ROOT/$remote_job\\" && tmux kill-session -t \\"$session\\" >/dev/null 2>&1 || true; tmux new-session -d -s \\"$session\\" /bin/bash \\"$REMOTE_ROOT/$remote_job\\" \\"$BATCH_NAME\\" \\"$WORKERS\\" \\"$RETRY_MODE\\""',
            '  echo "STARTED ${host} ${session}"',
            "}",
            "",
        ]
    )
    for host in HOSTS:
        lines.append(f'start_job "{host}"')
    lines.append('echo "WAVE_STARTED ${TOOL} seed${SEED}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _write_manifest(rows: list[dict[str, str]]) -> None:
    manifest_rows: list[dict[str, str | int]] = []
    split_sizes = _split_rows(rows)
    for tool_key, tool in TOOLS.items():
        for seed in SEEDS:
            for host in HOSTS:
                manifest_rows.append(
                    {
                        "tool": tool_key,
                        "tool_arg": tool["tool_arg"],
                        "seed": seed,
                        "host": host,
                        "conda_env": tool["env"],
                        "workers": tool["workers"],
                        "tasks": len(split_sizes[host]),
                        "slice_csv": f"slices/{tool_key}/seed{seed}/{host}.csv",
                        "remote_job": f"remote_jobs/{tool_key}/seed{seed}/{tool_key}_seed{seed}_{host}.sh",
                        "tmux_session": f"probe4_full664_{tool_key}_s{seed}_{host}",
                    }
                )
    path = OUTPUT_ROOT / "probe4_full664_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)


def _write_readme(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Probe4 Full-664 三种子分发资产",
        "",
        f"- source: `{SOURCE_CSV.relative_to(REPO_ROOT)}`",
        f"- datasets: `{len(rows)}`",
        f"- tools: `{', '.join(TOOLS)}`",
        f"- seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        f"- hosts: `{', '.join(HOSTS)}`",
        "- 注意：`iaaccn22` 当前不可达，因此本批次不分配任务到 22。",
        "",
        "## 运行方式",
        "",
        "单个 tool + seed 启动一轮，脚本会把该轮 664 个数据集切到 7 台机器上：",
        "",
        "```bash",
        "bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/launch/run_tool_seed.sh dso 520",
        "```",
        "",
        "建议不要一次性启动 12 轮；每台机器同时跑多个 CPU-heavy worker 池会互相抢核。",
        "",
        "## 默认 workers",
        "",
        "| tool | env | workers/host | tasks/host |",
        "|---|---|---:|---:|",
    ]
    split_sizes = _split_rows(rows)
    max_tasks = max(len(v) for v in split_sizes.values())
    for tool_key, tool in TOOLS.items():
        lines.append(f"| `{tool_key}` | `{tool['env']}` | `{tool['workers']}` | `{max_tasks}` |")
    lines.extend(
        [
            "",
            "## 输出路径",
            "",
            "```text",
            f"{REMOTE_ROOT}/experiments/<BATCH_NAME>/<tool>/seed<seed>/<host>/",
            "```",
            "",
            "每轮状态文件：",
            "",
            "```text",
            "__launcher__/task_status.jsonl",
            "```",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = _read_rows(SOURCE_CSV)
    _write_csv(OUTPUT_ROOT / "full664_unified.csv", rows)
    split_rows = _split_rows(rows)
    for tool_key in TOOLS:
        for seed in SEEDS:
            for host, host_rows in split_rows.items():
                _write_csv(OUTPUT_ROOT / "slices" / tool_key / f"seed{seed}" / f"{host}.csv", host_rows)
                _write_remote_job(tool_key, seed, host)
    _write_tool_seed_launcher()
    _write_manifest(rows)
    _write_readme(rows)
    print(f"generated: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
