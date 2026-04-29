#!/usr/bin/env python3
"""生成 Probe4 full-664 启动前 smoke-test 资产。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1/full664_unified.csv"
OUTPUT_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1/smoke"
REMOTE_ROOT = "/home/zhangziwen/projects/scientific-intelligent-modelling"
HOSTS = ("iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26")
DATASET_GLOBAL_INDICES = {
    "iaaccn23": "633",
    "iaaccn24": "637",
    "iaaccn25": "638",
    "iaaccn26": "640",
}
SEED = 990


SMOKE_PARAMS = {
    "dso": {
        "timeout_in_seconds": 180,
        "progress_snapshot_interval_seconds": 60,
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "threshold": 1e-12,
            "protected": False,
        },
        "training": {
            "n_samples": 5000,
            "batch_size": 500,
            "epsilon": 0.05,
            "n_cores_batch": 1,
        },
        "policy_optimizer": {
            "learning_rate": 0.0005,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
        },
        "prior": {
            "length": {"min_": 4, "max_": 64, "on": True},
            "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": True},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
            "domain_range": {"on": False},
        },
    },
    "udsr": {
        "benchmark_variant": "udsr_trunk_dso_poly_gp_meld_smoke",
        "component_flags": {
            "aif": False,
            "dsr": True,
            "lspt": False,
            "gp_meld": True,
            "linear_poly": True,
        },
        "timeout_in_seconds": 180,
        "progress_snapshot_interval_seconds": 60,
        "task": {
            "task_type": "regression",
            "function_set": [
                "add",
                "sub",
                "mul",
                "div",
                "sin",
                "cos",
                "exp",
                "log",
                "sqrt",
                1.0,
                "const",
                "poly",
            ],
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "threshold": 1e-12,
            "protected": False,
            "poly_optimizer_params": {
                "degree": 3,
                "coef_tol": 1e-6,
                "regressor": "dso_least_squares",
                "regressor_params": {
                    "cutoff_p_value": 1.0,
                    "n_max_terms": None,
                    "coef_tol": 1e-6,
                },
            },
        },
        "training": {
            "n_samples": 5000,
            "batch_size": 500,
            "epsilon": 0.05,
            "baseline": "R_e",
            "n_cores_batch": 1,
        },
        "policy_optimizer": {
            "policy_optimizer_type": "pg",
            "learning_rate": 0.0005,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
        },
        "gp_meld": {
            "run_gp_meld": True,
            "population_size": 20,
            "generations": 2,
            "crossover_operator": "cxOnePoint",
            "p_crossover": 0.5,
            "mutation_operator": "multi_mutate",
            "p_mutate": 0.5,
            "tournament_size": 5,
            "train_n": 20,
            "mutate_tree_max": 3,
            "verbose": False,
            "parallel_eval": False,
        },
        "prior": {
            "length": {"min_": 4, "max_": 100, "on": True},
            "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": True},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
            "domain_range": {"on": True},
        },
    },
    "imcts": {
        "timeout_in_seconds": 180,
        "progress_snapshot_interval_seconds": 60,
        "ops": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "R"],
        "max_depth": 6,
        "K": 500,
        "c": 4.0,
        "gamma": 0.5,
        "gp_rate": 0.2,
        "mutation_rate": 0.1,
        "exploration_rate": 0.2,
        "max_single_arity_ops": 999,
        "max_constants": 10,
        "max_expressions": 2000,
        "verbose": False,
        "optimization_method": "LN_NELDERMEAD",
    },
    "pyoperon": {
        "timeout_in_seconds": 180,
        "progress_snapshot_interval_seconds": 60,
        "population_size": 100,
        "pool_size": 100,
        "max_length": 50,
        "max_depth": 10,
        "tournament_size": 5,
        "allowed_symbols": "add,sub,mul,div,aq,exp,log,sin,cos,tanh,sqrt,square,constant,variable",
        "offspring_generator": "basic",
        "reinserter": "keep-best",
        "optimizer": "lm",
        "local_search_probability": 1.0,
        "max_evaluations": 5000,
        "n_threads": 4,
    },
}

TOOL_RUNTIME = {
    "udsr": ("udsr", "sim_dso"),
    "dso": ("dso", "sim_dso"),
    "imcts": ("iMCTS", "sim_iMCTS"),
    "pyoperon": ("pyoperon", "sim_base"),
}


def _load_rows() -> dict[str, dict[str, str]]:
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_index = {row["global_index"]: row for row in rows}
    return {host: by_index[index] for host, index in DATASET_GLOBAL_INDICES.items()}


def _write_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_params() -> None:
    params_dir = OUTPUT_ROOT / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    for tool, params in SMOKE_PARAMS.items():
        (params_dir / f"{tool}.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_remote_job() -> None:
    path = OUTPUT_ROOT / "remote_jobs" / "run_host_smoke.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'if [ "$#" -lt 2 ]; then',
        '  echo "Usage: $0 <host-label> <batch-name>" >&2',
        "  exit 2",
        "fi",
        "",
        'HOST_LABEL="$1"',
        'BATCH_NAME="$2"',
        f'REMOTE_ROOT="{REMOTE_ROOT}"',
        'ASSET_ROOT="$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/probe4_full664_v1/smoke"',
        f'SEED="{SEED}"',
        'LOG_ROOT="$REMOTE_ROOT/experiments/${BATCH_NAME}/__smoke_logs/${HOST_LABEL}"',
        'mkdir -p "$LOG_ROOT"',
        'cd "$REMOTE_ROOT"',
        "export PYTHONPATH=.",
        "",
        "run_tool() {",
        '  local tool="$1"',
        '  local tool_arg="$2"',
        '  local env_name="$3"',
        '  local slice_csv="$ASSET_ROOT/slices/${HOST_LABEL}.csv"',
        '  local params_json="$ASSET_ROOT/params/${tool}.json"',
        '  local out_root="$REMOTE_ROOT/experiments/${BATCH_NAME}/${tool}/${HOST_LABEL}"',
        '  echo "[start] ${HOST_LABEL} ${tool}"',
        '  conda run -n "$env_name" python check/launch_e1_benchmark.py run \\',
        '    --tool "$tool_arg" \\',
        '    --slice-csv "$slice_csv" \\',
        '    --params-json "$params_json" \\',
        '    --output-root "$out_root" \\',
        '    --seed "$SEED" \\',
        '    --workers 1 \\',
        '    --retry-failed > "$LOG_ROOT/${tool}.log" 2>&1 &',
        "}",
        "",
    ]
    for tool, (tool_arg, env_name) in TOOL_RUNTIME.items():
        lines.append(f'run_tool "{tool}" "{tool_arg}" "{env_name}"')
    lines.extend(
        [
            "wait",
            'echo "[done] ${HOST_LABEL} smoke"',
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _write_readme(rows_by_host: dict[str, dict[str, str]]) -> None:
    lines = [
        "# Probe4 Full-664 Smoke Test 资产",
        "",
        "- 目的：正式 full-664 启动前，每台机器跑 4 个算法各 1 个真实数据集任务。",
        "- 范围：`iaaccn23~26`。",
        "- 算法：`udsr`, `dso`, `imcts`, `pyoperon`。",
        f"- seed: `{SEED}`。",
        "- 预算：每任务 `timeout_in_seconds=180`，仅用于集成 smoke，不作为正式实验结果。",
        "",
        "| host | dataset | global_index |",
        "|---|---|---|",
    ]
    for host in HOSTS:
        row = rows_by_host[host]
        lines.append(f"| `{host}` | `{row['dataset_name']}` | `{row['global_index']}` |")
    lines.extend(
        [
            "",
            "远端启动脚本：",
            "",
            "```bash",
            "bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/smoke/remote_jobs/run_host_smoke.sh <host-label> <batch-name>",
            "```",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows_by_host = _load_rows()
    for host, row in rows_by_host.items():
        _write_csv(OUTPUT_ROOT / "slices" / f"{host}.csv", row)
    _write_params()
    _write_remote_job()
    _write_readme(rows_by_host)
    print(f"wrote smoke assets: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
