#!/usr/bin/env python3
"""生成 E1 正式切片 CSV 与 wave 启动脚本。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE200_CSV = REPO_ROOT / "experiment-results/benchmark_selection_dossier_20260422/tables/stage1_candidate200_flat.csv"
OUTPUT_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证"
GENERATED_ROOT = OUTPUT_ROOT / "generated"
REMOTE_PROJECT_ROOT = "/home/zhangziwen/projects/scientific-intelligent-modelling"
REMOTE_DATA_ROOT = "/home/zhangziwen/sim-datasets-data"
SEED = 1314

NEUTRAL_SR_BACKGROUND = (
    "This is a symbolic regression task. "
    "Find a compact mathematical equation that predicts the target from the observed variables."
)


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"候选 CSV 为空: {csv_path}")
    return rows


def _to_remote_dataset_dir(dataset_dir: str) -> str:
    path = Path(dataset_dir)
    if path.is_absolute():
        return str(path)
    parts = list(path.parts)
    if parts and parts[0] == "sim-datasets-data":
        return str(Path(REMOTE_DATA_ROOT, *parts[1:]))
    return str(Path(REMOTE_DATA_ROOT, *parts))


def _build_base_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    base_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        dataset_rel = row["dataset_dir"]
        remote_dataset_dir = Path(_to_remote_dataset_dir(dataset_rel))
        base_rows.append(
            {
                "global_index": idx,
                "dataset_name": row.get("dataset") or remote_dataset_dir.name,
                "dataset_dir": str(remote_dataset_dir),
                "dataset_rel": dataset_rel,
                "family": row.get("family", ""),
                "subgroup": row.get("subgroup", ""),
                "basename": row.get("basename", ""),
                "pool": row.get("pool", ""),
                "selection_mode": row.get("selection_mode", ""),
                "candidate_advantage_side": row.get("advantage_side", ""),
                "train_csv": str(remote_dataset_dir / "train.csv"),
                "valid_csv": str(remote_dataset_dir / "valid.csv"),
                "id_test_csv": str(remote_dataset_dir / "id_test.csv"),
                "ood_test_csv": str(remote_dataset_dir / "ood_test.csv"),
                "metadata_yaml": str(remote_dataset_dir / "metadata.yaml"),
                "formula_py": str(remote_dataset_dir / "formula.py"),
            }
        )
    return base_rows


WAVE_CONFIGS = [
    {
        "wave": "W1",
        "tool": "gplearn",
        "machines": ["iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25"],
        "sizes": [50, 50, 50, 50],
        "conda_env": "sim_base",
        "default_workers": 32,
        "params": {
            "timeout_in_seconds": 3600,
            "progress_snapshot_interval_seconds": 60,
            "population_size": 1000,
            "generations": 1000000,
            "tournament_size": 20,
            "stopping_criteria": 0.0,
            "const_range": [-1.0, 1.0],
            "init_depth": [2, 6],
            "init_method": "half and half",
            "function_set": "add,sub,mul,div,sqrt,log,sin,cos",
            "metric": "mean absolute error",
            "parsimony_coefficient": 0.001,
            "p_crossover": 0.9,
            "p_subtree_mutation": 0.01,
            "p_hoist_mutation": 0.01,
            "p_point_mutation": 0.01,
            "p_point_replace": 0.05,
            "max_samples": 1.0,
            "n_jobs": 4,
            "low_memory": False,
            "warm_start": False,
            "verbose": 0,
        },
    },
    {
        "wave": "W1",
        "tool": "llmsr",
        "machines": ["iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29"],
        "sizes": [50, 50, 50, 50],
        "conda_env": "sim_llm",
        "default_workers": 25,
        "params": {
            "timeout_in_seconds": 3600,
            "progress_snapshot_interval_seconds": 60,
            "niterations": 100000,
            "samples_per_iteration": 4,
            "max_params": 10,
            "inject_prompt_semantics": False,
            "background": NEUTRAL_SR_BACKGROUND,
            "persist_all_samples": False,
            "llm_config_path": f"{REMOTE_PROJECT_ROOT}/exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config",
        },
    },
    {
        "wave": "W2",
        "tool": "pyoperon",
        "machines": ["iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25"],
        "sizes": [50, 50, 50, 50],
        "conda_env": "sim_base",
        "default_workers": 24,
        "params": {
            "timeout_in_seconds": 3600,
            "progress_snapshot_interval_seconds": 60,
            "population_size": 500,
            "pool_size": 500,
            "max_length": 50,
            "max_depth": 10,
            "tournament_size": 5,
            "allowed_symbols": "add,sub,mul,div,aq,exp,log,sin,cos,tanh,sqrt,square,constant,variable",
            "offspring_generator": "basic",
            "reinserter": "keep-best",
            "optimizer": "lm",
            "local_search_probability": 1.0,
            "max_evaluations": 500000,
            "n_threads": 4,
        },
    },
    {
        "wave": "W2",
        "tool": "drsr",
        "machines": ["iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29"],
        "sizes": [50, 50, 50, 50],
        "conda_env": "sim_llm",
        "default_workers": 20,
        "params": {
            "timeout_in_seconds": 3600,
            "progress_snapshot_interval_seconds": 60,
            "niterations": 100000,
            "samples_per_iteration": 4,
            "max_params": 10,
            "persist_all_samples": False,
            "background": NEUTRAL_SR_BACKGROUND,
            "llm_config_path": f"{REMOTE_PROJECT_ROOT}/exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config",
        },
    },
    {
        "wave": "W3",
        "tool": "pysr",
        "machines": ["iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25"],
        "sizes": [50, 50, 50, 50],
        "conda_env": "sim_base",
        "default_workers": 32,
        "params": {
            "timeout_in_seconds": 3600,
            "progress_snapshot_interval_seconds": 60,
            "niterations": 10_000_000,
            "population_size": 64,
            "populations": 8,
            "ncycles_per_iteration": 500,
            "maxsize": 30,
            "maxdepth": 10,
            "parsimony": 1e-3,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["square", "cube", "exp", "log", "sin", "cos"],
            "constraints": {
                "/": [-1, 9],
                "square": 9,
                "cube": 9,
                "exp": 7,
                "log": 7,
                "sin": 9,
                "cos": 9,
            },
            "nested_constraints": {
                "exp": {"exp": 0, "log": 1},
                "log": {"exp": 0, "log": 0},
                "square": {"square": 1, "cube": 1, "exp": 0, "log": 0},
                "cube": {"square": 1, "cube": 1, "exp": 0, "log": 0},
            },
            "complexity_of_operators": {
                "/": 2,
                "square": 2,
                "cube": 3,
                "sin": 2,
                "cos": 2,
                "exp": 3,
                "log": 3,
            },
            "complexity_of_constants": 2,
            "complexity_of_variables": 1,
            "precision": 32,
            "deterministic": True,
            "parallelism": "serial",
            "model_selection": "best",
            "progress": True,
            "verbosity": 1,
            "procs": 1,
        },
    },
    {
        "wave": "W4",
        "tool": "dso",
        "machines": ["iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28"],
        "sizes": [29, 29, 29, 29, 28, 28, 28],
        "conda_env": "sim_dso",
        "default_workers": 15,
        "params": {
            "timeout_in_seconds": 3600,
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
                "n_samples": 2_000_000,
                "batch_size": 1000,
                "epsilon": 0.05,
                "n_cores_batch": 4,
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
    },
    {
        "wave": "W5",
        "tool": "tpsr",
        "machines": ["iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25"],
        "sizes": [50, 50, 50, 50],
        "conda_env": "sim_tpsr",
        "default_workers": 25,
        "params": {
            "timeout_in_seconds": 3600,
            "progress_snapshot_interval_seconds": 60,
            "backbone_model": "e2e",
            "max_input_points": 200,
            "max_number_bags": 10,
            "stop_refinement_after": 1,
            "n_trees_to_refine": 10,
            "rescale": True,
            "beam_size": 10,
            "beam_type": "sampling",
            "no_seq_cache": False,
            "no_prefix_cache": True,
            "width": 3,
            "num_beams": 1,
            "rollout": 3,
            "horizon": 200,
            "lam": 0.1,
            "train_value": False,
            "seed": 23,
            "reward_sample_limit": 2048,
            "cpu": True,
            "cpu_num_threads": 4,
            "cpu_interop_threads": 1,
        },
    },
]


def _slice_rows(rows: list[dict[str, str]], sizes: list[int]) -> list[list[dict[str, str]]]:
    if sum(sizes) != len(rows):
        raise ValueError(f"切片大小之和 {sum(sizes)} 与总行数 {len(rows)} 不一致")
    out: list[list[dict[str, str]]] = []
    start = 0
    for size in sizes:
        out.append(rows[start : start + size])
        start += size
    return out


def _write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _render_remote_job_script(
    *,
    tool: str,
    conda_env: str,
    rel_slice_csv: str,
    rel_params_json: str,
    host: str,
) -> str:
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

conda run -n {conda_env} python check/launch_e1_benchmark.py run \\
  --tool {tool} \\
  --slice-csv "$REMOTE_ROOT/{rel_slice_csv}" \\
  --params-json "$REMOTE_ROOT/{rel_params_json}" \\
  --output-root "$REMOTE_ROOT/experiments/${{BATCH_NAME}}/{host}" \\
  --seed {SEED} \\
  --workers "$WORKERS" \\
  "${{EXTRA_ARGS[@]}}"
"""


def _render_wave_launcher(
    *,
    wave: str,
    jobs: list[dict[str, str | int]],
) -> str:
    wave_lower = wave.lower()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'REPO_ROOT="{REPO_ROOT}"',
        f'REMOTE_ROOT="{REMOTE_PROJECT_ROOT}"',
        'STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"',
        f'BATCH_NAME="${{BATCH_NAME:-e1_candidate200_seed{SEED}_{wave_lower}_$STAMP}}"',
        'echo "BATCH_NAME=${BATCH_NAME}"',
        "",
        "start_job() {",
        '  local host="$1"',
        '  local session="$2"',
        '  local remote_job="$3"',
        '  local rel_slice="$4"',
        '  local rel_params="$5"',
        '  local workers="$6"',
        '  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p \\"$REMOTE_ROOT/check\\" \\"$REMOTE_ROOT/$(dirname "$rel_slice")\\" \\"$REMOTE_ROOT/$(dirname "$rel_params")\\" \\"$REMOTE_ROOT/$(dirname "$remote_job")\\""',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/check/launch_e1_benchmark.py" "$host:$REMOTE_ROOT/check/launch_e1_benchmark.py"',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_slice" "$host:$REMOTE_ROOT/$rel_slice"',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$rel_params" "$host:$REMOTE_ROOT/$rel_params"',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$remote_job" "$host:$REMOTE_ROOT/$remote_job"',
        '  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "chmod +x \\"$REMOTE_ROOT/$remote_job\\" && tmux kill-session -t \\"$session\\" >/dev/null 2>&1 || true; tmux new-session -d -s \\"$session\\" /bin/bash \\"$REMOTE_ROOT/$remote_job\\" \\"$BATCH_NAME\\" \\"$workers\\""',
        '  echo "STARTED ${host} ${session}"',
        "}",
        "",
    ]
    for job in jobs:
        lines.append(
            f'start_job "{job["host"]}" "{job["session"]}" "{job["remote_job_rel"]}" "{job["slice_rel"]}" "{job["params_rel"]}" "{job["workers"]}"'
        )
    lines.extend(["", 'echo "WAVE_DONE ' + wave + '"'])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    (GENERATED_ROOT / "params").mkdir(parents=True, exist_ok=True)
    (GENERATED_ROOT / "slices").mkdir(parents=True, exist_ok=True)
    (GENERATED_ROOT / "launch").mkdir(parents=True, exist_ok=True)
    (GENERATED_ROOT / "remote_jobs").mkdir(parents=True, exist_ok=True)

    source_rows = _load_rows(CANDIDATE200_CSV)
    base_rows = _build_base_rows(source_rows)
    _write_csv(GENERATED_ROOT / "candidate200_unified.csv", base_rows)

    allocation_rows: list[dict[str, str | int]] = []
    wave_to_jobs: dict[str, list[dict[str, str | int]]] = {}

    for cfg in WAVE_CONFIGS:
        tool = cfg["tool"]
        wave = cfg["wave"]
        params_path = GENERATED_ROOT / "params" / f"{tool}.json"
        _dump_json(params_path, cfg["params"])

        slices = _slice_rows(base_rows, cfg["sizes"])
        for machine, slice_rows in zip(cfg["machines"], slices, strict=True):
            machine_rows: list[dict[str, str | int]] = []
            for slice_index, row in enumerate(slice_rows, start=1):
                machine_rows.append(
                    {
                        "tool": tool,
                        "machine": machine,
                        "wave": wave,
                        "global_index": row["global_index"],
                        "slice_index": slice_index,
                        **row,
                    }
                )
            slice_path = GENERATED_ROOT / "slices" / wave / tool / f"{machine}.csv"
            _write_csv(slice_path, machine_rows)

            remote_job_path = GENERATED_ROOT / "remote_jobs" / wave / f"{tool}_{machine}.sh"
            remote_job_path.parent.mkdir(parents=True, exist_ok=True)
            remote_job_path.write_text(
                _render_remote_job_script(
                    tool=tool,
                    conda_env=cfg["conda_env"],
                    rel_slice_csv=_rel(slice_path),
                    rel_params_json=_rel(params_path),
                    host=machine,
                ),
                encoding="utf-8",
            )
            remote_job_path.chmod(0o755)

            session = f"e1_{wave.lower()}_{tool.lower()}_{machine.replace('iaaccn', '')}"
            job_info = {
                "wave": wave,
                "tool": tool,
                "host": machine,
                "task_count": len(machine_rows),
                "workers": cfg["default_workers"],
                "conda_env": cfg["conda_env"],
                "slice_rel": _rel(slice_path),
                "params_rel": _rel(params_path),
                "remote_job_rel": _rel(remote_job_path),
                "session": session,
            }
            allocation_rows.append(job_info)
            wave_to_jobs.setdefault(wave, []).append(job_info)

    _write_csv(GENERATED_ROOT / "wave_manifest.csv", allocation_rows)

    for wave, jobs in sorted(wave_to_jobs.items()):
        launch_path = GENERATED_ROOT / "launch" / f"run_{wave.lower()}.sh"
        launch_path.write_text(_render_wave_launcher(wave=wave, jobs=jobs), encoding="utf-8")
        launch_path.chmod(0o755)

    readme = OUTPUT_ROOT / "README.md"
    readme.write_text(
        f"""# E1 正式切片与启动资产

本目录由 `check/generate_e1_formal_assets.py` 生成。

## 输入

- Candidate-200 源表：
  - `{CANDIDATE200_CSV.relative_to(REPO_ROOT)}`

## 输出

- 统一任务表：
  - `generated/candidate200_unified.csv`
- 工具参数：
  - `generated/params/*.json`
- 各 wave / 各机器切片：
  - `generated/slices/<wave>/<tool>/<host>.csv`
- 逐主机远端 job 脚本：
  - `generated/remote_jobs/<wave>/<tool>_<host>.sh`
- 逐 wave 本地启动脚本：
  - `generated/launch/run_w1.sh` ~ `run_w5.sh`
- manifest：
  - `generated/wave_manifest.csv`

## 约定

- 远端代码根：
  - `{REMOTE_PROJECT_ROOT}`
- 远端数据根：
  - `{REMOTE_DATA_ROOT}`
- 固定 seed：
  - `{SEED}`
""",
        encoding="utf-8",
    )

    print(GENERATED_ROOT / "candidate200_unified.csv")
    print(GENERATED_ROOT / "wave_manifest.csv")
    for wave in sorted(wave_to_jobs):
        print(GENERATED_ROOT / "launch" / f"run_{wave.lower()}.sh")


if __name__ == "__main__":
    main()
