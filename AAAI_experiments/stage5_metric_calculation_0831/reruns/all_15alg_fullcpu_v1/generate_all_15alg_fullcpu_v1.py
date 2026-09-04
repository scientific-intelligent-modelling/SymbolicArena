#!/usr/bin/env python3
"""生成 SSR-50 十五算法全条件 CPU 饱和重跑资产，不启动远端任务。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASSET_ROOT = Path(__file__).resolve().parent
SCHEDULER = REPO_ROOT / "check/run_e1_candidate200_12alg_load_queue.py"

FORMAL_ROOT = (
    REPO_ROOT
    / "benchmark-runs/formal3h/"
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658"
)
DRSR_ROOT = (
    REPO_ROOT
    / "benchmark-runs/drsr3h/"
    "drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355"
)
LLMSR_ROOT = (
    REPO_ROOT
    / "benchmark-runs/llmsr3h/"
    "llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442"
)
EXCLUSION_ROOT = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/reruns/all_conditions_cpu_v2"
)

TOOLS = (
    "gplearn",
    "llmsr",
    "pyoperon",
    "drsr",
    "pysr",
    "dso",
    "tpsr",
    "e2esr",
    "fepysr",
    "jaxsr",
    "qlattice",
    "imcts",
    "udsr",
    "ragsr",
    "symbolfit",
)
FORMAL_TOOLS = tuple(tool for tool in TOOLS if tool not in {"drsr", "llmsr"})
CONDITIONS = ("clean", "noise001", "noise005")
CONDITION_SIGMA = {"clean": 0.0, "noise001": 0.01, "noise005": 0.05}
SEEDS = (520, 521, 522)
HOSTS = tuple(f"iaaccn{number}" for number in range(22, 30))
CPU_WEIGHTS = {
    "gplearn": 1,
    "llmsr": 1,
    "pyoperon": 4,
    "drsr": 1,
    "pysr": 4,
    "dso": 4,
    "tpsr": 4,
    "e2esr": 1,
    "fepysr": 4,
    "jaxsr": 1,
    "qlattice": 4,
    "imcts": 1,
    "udsr": 1,
    "ragsr": 4,
    "symbolfit": 1,
}

EXPECTED_MOTHER = 6750
EXPECTED_EXCLUDED = 143
EXPECTED_QUEUE = 6607
TIMEOUT_SECONDS = 10800
SNAPSHOT_SECONDS = 60
LOGICAL_CPUS_PER_HOST = 256
PHYSICAL_CPUS_PER_HOST = 128
CPU_WEIGHT_BUDGET_PER_HOST = 115
MAX_CPU_USED_RATIO = CPU_WEIGHT_BUDGET_PER_HOST / LOGICAL_CPUS_PER_HOST
MAX_JOBS_PER_HOST = 115
MAX_NEW_JOBS_PER_HOST_PER_POLL = 115
LOAD_TIERS = "0.50:115,0.70:115,0.85:115,0.90:115"
MAX_LOAD_RATIO = 0.90
MAX_MEMORY_USED_RATIO = 0.90
MIN_FREE_MEM_GB = 32
POLL_SECONDS = 30
SESSION_PREFIX = "all_conditions_cpu_v2_"
BATCH_NAME = "all_15alg_fullcpu_v1"

LEDGER_FIELDS = (
    "task_id",
    "algorithm",
    "dataset_id",
    "global_index",
    "seed",
    "condition",
    "noise_sigma",
    "params_name",
    "cpu_weight",
    "excluded_by_all_conditions_cpu_v2",
    "execution_set",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields or not rows:
        raise ValueError(f"CSV 为空: {path}")
    return fields, rows


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"拒绝写空 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(materialized)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return _sha256(destination)


def _source_specs() -> tuple[tuple[str, Path], ...]:
    return (
        ("formal3h_13alg", FORMAL_ROOT / "queues/ssr50_source.csv"),
        ("drsr3h", DRSR_ROOT / "queues/ssr50_source.csv"),
        ("llmsr3h", LLMSR_ROOT / "queues/ssr50_source.csv"),
    )


def _freeze_and_validate_sources(
    root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    frozen: dict[str, Any] = {}
    canonical_fields: list[str] | None = None
    canonical_rows: list[dict[str, str]] | None = None
    digests: set[str] = set()
    for label, source in _source_specs():
        fields, rows = _read_csv(source)
        if len(rows) != 50:
            raise ValueError(f"{label} source 应为 50 行，实际 {len(rows)}")
        if canonical_fields is None:
            canonical_fields, canonical_rows = fields, rows
        elif fields != canonical_fields or rows != canonical_rows:
            raise ValueError("formal3h、DRSR、LLMSR 三份 source CSV 内容不一致")
        destination = root / f"frozen_inputs/sources/{label}_ssr50_source.csv"
        copied_sha = _copy(source, destination)
        source_sha = _sha256(source)
        if copied_sha != source_sha:
            raise ValueError(f"source 复制哈希不一致: {label}")
        digests.add(source_sha)
        frozen[label] = {
            "source_path": str(source.resolve()),
            "frozen_path": str(destination.resolve()),
            "sha256": source_sha,
            "rows": len(rows),
        }
    if len(digests) != 1:
        raise ValueError(f"三份 source CSV SHA 不一致: {sorted(digests)}")
    assert canonical_rows is not None
    indices = [int(row["global_index"]) for row in canonical_rows]
    datasets = [row["dataset_id"] for row in canonical_rows]
    if sorted(indices) != list(range(1, 51)) or len(set(datasets)) != 50:
        raise ValueError("SSR-50 source 的 global_index/dataset_id 网格无效")
    return canonical_rows, {
        "sources": frozen,
        "all_three_sha_equal": True,
        "shared_sha256": next(iter(digests)),
        "rows": 50,
    }


def _param_specs() -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    for tool in FORMAL_TOOLS:
        for condition in CONDITIONS:
            filename = f"{tool}__{condition}.json"
            specs.append((tool, condition, FORMAL_ROOT / "params" / filename))
    for tool, source_root in (("drsr", DRSR_ROOT), ("llmsr", LLMSR_ROOT)):
        for condition in CONDITIONS:
            filename = f"{tool}__{condition}.json"
            specs.append((tool, condition, source_root / "params" / filename))
    return specs


def _freeze_and_validate_params(root: Path) -> dict[str, Any]:
    specs = _param_specs()
    if len(specs) != 45 or len({(tool, condition) for tool, condition, _ in specs}) != 45:
        raise ValueError("condition 参数源必须恰为 15 x 3 = 45")
    report: dict[str, Any] = {}
    for tool, condition, source in specs:
        destination = root / "params" / source.name
        copied_sha = _copy(source, destination)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"参数 JSON 顶层不是 object: {source}")
        sigma = float(payload.get("train_label_noise_sigma", math.nan))
        enabled = payload.get("train_label_noise_enabled")
        if int(payload.get("timeout_in_seconds") or 0) != TIMEOUT_SECONDS:
            raise ValueError(f"timeout 不是 {TIMEOUT_SECONDS}: {source}")
        if int(payload.get("progress_snapshot_interval_seconds") or 0) != SNAPSHOT_SECONDS:
            raise ValueError(f"snapshot interval 不是 {SNAPSHOT_SECONDS}: {source}")
        if not math.isclose(sigma, CONDITION_SIGMA[condition], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"noise sigma 与文件条件不一致: {source}")
        expected_enabled = condition != "clean"
        if enabled is not expected_enabled:
            raise ValueError(f"noise enabled 与文件条件不一致: {source}")
        if copied_sha != _sha256(source):
            raise ValueError(f"参数复制哈希不一致: {source}")
        report[source.name] = {
            "tool": tool,
            "condition": condition,
            "source_family": (
                "drsr3h" if tool == "drsr" else "llmsr3h" if tool == "llmsr" else "formal3h_13alg"
            ),
            "source_path": str(source.resolve()),
            "frozen_path": str(destination.resolve()),
            "sha256": copied_sha,
            "timeout_in_seconds": TIMEOUT_SECONDS,
            "progress_snapshot_interval_seconds": SNAPSHOT_SECONDS,
            "train_label_noise_sigma": sigma,
            "train_label_noise_enabled": enabled,
        }
    return {
        "count": len(report),
        "all_valid": True,
        "by_source_family": dict(
            sorted(Counter(item["source_family"] for item in report.values()).items())
        ),
        "files": report,
    }


def _load_excluded_task_ids(root: Path) -> tuple[set[str], dict[str, Any]]:
    source = EXCLUSION_ROOT / "manifests/all_conditions_noise_task_allowlist.csv"
    fields, rows = _read_csv(source)
    if "task_id" not in fields:
        raise ValueError("all_conditions_cpu_v2 allowlist 缺少 task_id")
    task_ids = {row["task_id"].strip() for row in rows if row["task_id"].strip()}
    if len(rows) != EXPECTED_EXCLUDED or len(task_ids) != EXPECTED_EXCLUDED:
        raise ValueError(
            f"排除 allowlist 应恰为 {EXPECTED_EXCLUDED} 个唯一 task_id，实际 {len(rows)}/{len(task_ids)}"
        )
    destination = root / "frozen_inputs/all_conditions_cpu_v2_excluded_143.csv"
    copied_sha = _copy(source, destination)
    return task_ids, {
        "source_path": str(source.resolve()),
        "frozen_path": str(destination.resolve()),
        "sha256": copied_sha,
        "rows": len(rows),
        "unique_task_ids": len(task_ids),
    }


def _build_ledgers(
    source_rows: list[dict[str, str]], excluded: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    mother: list[dict[str, object]] = []
    for tool in TOOLS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                for source in source_rows:
                    global_index = int(source["global_index"])
                    task_id = f"{tool}_s{seed}_{condition}_g{global_index:04d}"
                    mother.append(
                        {
                            "task_id": task_id,
                            "algorithm": tool,
                            "dataset_id": source["dataset_id"],
                            "global_index": global_index,
                            "seed": seed,
                            "condition": condition,
                            "noise_sigma": f"{CONDITION_SIGMA[condition]:.17g}",
                            "params_name": f"{tool}__{condition}",
                            "cpu_weight": CPU_WEIGHTS[tool],
                            "excluded_by_all_conditions_cpu_v2": (
                                "true" if task_id in excluded else "false"
                            ),
                            "execution_set": (
                                "all_conditions_cpu_v2_143"
                                if task_id in excluded
                                else "all_15alg_fullcpu_v1_6607"
                            ),
                        }
                    )
    mother.sort(key=lambda row: str(row["task_id"]))
    ids = [str(row["task_id"]) for row in mother]
    if len(mother) != EXPECTED_MOTHER or len(set(ids)) != EXPECTED_MOTHER:
        raise ValueError("6750 逻辑母集不完整或 task_id 不唯一")
    mother_ids = set(ids)
    unknown_excluded = sorted(excluded.difference(mother_ids))
    if unknown_excluded:
        raise ValueError(f"143 allowlist 含母集外任务: {unknown_excluded[:5]}")
    queue = [row for row in mother if row["task_id"] not in excluded]
    queue_ids = {str(row["task_id"]) for row in queue}
    excluded_in_mother = mother_ids.intersection(excluded)
    if (
        len(queue) != EXPECTED_QUEUE
        or len(queue_ids) != EXPECTED_QUEUE
        or len(excluded_in_mother) != EXPECTED_EXCLUDED
        or queue_ids.intersection(excluded)
        or queue_ids.union(excluded) != mother_ids
    ):
        raise ValueError("143 + 6607 的 composite ledger 集合恒等式不成立")
    report = {
        "mother_count": len(mother),
        "excluded_count": len(excluded),
        "new_queue_count": len(queue),
        "identity": "143 + 6607 = 6750",
        "excluded_and_queue_disjoint": True,
        "excluded_union_queue_equals_mother": True,
        "mother_task_ids_unique": True,
        "queue_task_ids_unique": True,
        "mother_algorithm_counts": dict(
            sorted(Counter(str(row["algorithm"]) for row in mother).items())
        ),
        "queue_algorithm_counts": dict(
            sorted(Counter(str(row["algorithm"]) for row in queue).items())
        ),
        "mother_condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in mother).items())
        ),
        "queue_condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in queue).items())
        ),
        "mother_cpu_weight_counts": dict(
            sorted(Counter(str(row["cpu_weight"]) for row in mother).items())
        ),
        "queue_cpu_weight_counts": dict(
            sorted(Counter(str(row["cpu_weight"]) for row in queue).items())
        ),
        "mother_cpu_weight_total": sum(int(row["cpu_weight"]) for row in mother),
        "queue_cpu_weight_total": sum(int(row["cpu_weight"]) for row in queue),
    }
    return mother, queue, report


def _command(root: Path, mode: str) -> str:
    del root
    suffix = "preflight" if mode == "preflight" else "dry_run" if mode == "dry_run" else "formal"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        'ASSET_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"',
        'REPO_ROOT="${SIM_REPO_ROOT:-$(cd -- "$ASSET_ROOT/../../../.." && pwd)}"',
        'mkdir -p "$ASSET_ROOT/logs" "$ASSET_ROOT/queues/' + suffix + '"',
        'exec > >(tee -a "$ASSET_ROOT/logs/controller_' + suffix + '.log") 2>&1',
        'cd "$REPO_ROOT"',
        "export PYTHONPATH=.",
        "export SIM_QUEUE_CONTROLLER_IS_LOCAL=1",
        'python "$REPO_ROOT/check/run_e1_candidate200_12alg_load_queue.py" \\',
        f"  --batch-name {BATCH_NAME}_{suffix} \\",
        '  --source-csv "$ASSET_ROOT/frozen_inputs/sources/formal3h_13alg_ssr50_source.csv" \\',
        "  --expected-rows 50 \\",
        f'  --queue-root "$ASSET_ROOT/queues/{suffix}" \\',
        '  --params-root "$ASSET_ROOT/params" \\',
        '  --task-id-allowlist-csv "$ASSET_ROOT/manifests/new_queue_6607.csv" \\',
        "  --hosts " + " ".join(HOSTS) + " \\",
        "  --tools " + " ".join(TOOLS) + " \\",
        "  --seeds " + " ".join(str(seed) for seed in SEEDS) + " \\",
        "  --noise-sigmas 0 0.01 0.05 \\",
        "  --controller-host iaaccn22 \\",
        "  --use-internal-ips \\",
        "  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \\",
        "  --remote-data-root /home/zhangziwen/sim-datasets-data \\",
        f"  --max-jobs-per-host {MAX_JOBS_PER_HOST} \\",
        f"  --max-cpu-used-ratio {MAX_CPU_USED_RATIO:.17g} \\",
        f"  --max-new-jobs-per-host-per-poll {MAX_NEW_JOBS_PER_HOST_PER_POLL} \\",
        f'  --load-tier-new-jobs "{LOAD_TIERS}" \\',
        f"  --max-load-ratio {MAX_LOAD_RATIO:.2f} \\",
        f"  --max-memory-used-ratio {MAX_MEMORY_USED_RATIO:.2f} \\",
        f"  --min-free-mem-gb {MIN_FREE_MEM_GB} \\",
        f"  --session-prefix {SESSION_PREFIX} \\",
        f"  --host-session-count-prefix {SESSION_PREFIX} \\",
        f"  --poll-seconds {POLL_SECONDS} \\",
        "  --retry-limit 1 \\",
        "  --prioritize-llm \\",
        "  --llm-model-assignment from-params \\",
        "  --llm-default-bucket turbo \\",
        "  --llm-model-bucket-limits base:0,turbo:30",
    ]
    if mode == "preflight":
        lines[-1] += ' \\\n  --preflight-only \\\n  --preflight-report "$ASSET_ROOT/reports/preflight_report.json"'
    elif mode == "dry_run":
        lines[-1] += " \\\n  --dry-run \\\n  --once"
    else:
        lines[-1] += " \\\n  --force-rerun-existing"
    return "\n".join(lines) + "\n"


def _write_commands(root: Path) -> list[Path]:
    paths = []
    for mode in ("preflight", "dry_run", "formal"):
        path = root / "commands" / f"run_all_15alg_{mode}.sh"
        _atomic_text(path, _command(root, mode))
        path.chmod(0o755)
        paths.append(path)
    return paths


def _runtime_paths() -> list[Path]:
    wrappers = {
        "gplearn": "scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py",
        "llmsr": "scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py",
        "pyoperon": "scientific_intelligent_modelling/algorithms/pyoperon_wrapper/wrapper.py",
        "drsr": "scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py",
        "pysr": "scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py",
        "dso": "scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py",
        "tpsr": "scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py",
        "e2esr": "scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py",
        "fepysr": "scientific_intelligent_modelling/algorithms/fepysr_wrapper/wrapper.py",
        "jaxsr": "scientific_intelligent_modelling/algorithms/jaxsr_wrapper/wrapper.py",
        "qlattice": "scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py",
        "imcts": "scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py",
        "udsr": "scientific_intelligent_modelling/algorithms/udsr_wrapper/wrapper.py",
        "ragsr": "scientific_intelligent_modelling/algorithms/ragsr_wrapper/wrapper.py",
        "symbolfit": "scientific_intelligent_modelling/algorithms/symbolfit_wrapper/wrapper.py",
    }
    relative_paths = [
        "check/run_e1_candidate200_12alg_load_queue.py",
        "check/launch_e1_benchmark.py",
        "scientific_intelligent_modelling/benchmarks/runner.py",
        "scientific_intelligent_modelling/srkit/subprocess_runner.py",
        "scientific_intelligent_modelling/benchmarks/artifact_schema.py",
        "scientific_intelligent_modelling/benchmarks/normalizers.py",
        "scientific_intelligent_modelling/config/toolbox_config.json",
        "scientific_intelligent_modelling/algorithms/iMCTS_wrapper/MCTS-4-SR/iMCTS/regressor.py",
        *wrappers.values(),
    ]
    paths = [REPO_ROOT / relative for relative in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"运行时代码指纹文件缺失: {missing}")
    if len(paths) != 23 or len({path.resolve() for path in paths}) != 23:
        raise ValueError("运行时代码指纹应恰含23个唯一文件")
    return paths


def _write_runtime_fingerprints(root: Path) -> tuple[Path, dict[str, Any]]:
    entries = []
    for path in _runtime_paths():
        entries.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema": "all_15alg_fullcpu_v1.runtime_code_fingerprints.v1",
        "repo_root": str(REPO_ROOT.resolve()),
        "file_count": len(entries),
        "files": entries,
        "coverage": {
            "scheduler": 1,
            "launcher": 1,
            "runner": 1,
            "subprocess_runner": 1,
            "artifact_schema": 1,
            "normalizers": 1,
            "wrappers": 15,
            "imcts_native_regressor": 1,
            "toolbox_config": 1,
        },
    }
    path = root / "manifests/runtime_code_fingerprints.json"
    _write_json(path, payload)
    return path, payload


def _shell_array(values: Sequence[str]) -> str:
    return "\n".join(f"  {value!r}" for value in values)


def _write_deployment_scripts(root: Path, runtime_paths: Sequence[Path]) -> list[Path]:
    runtime_rel = [path.relative_to(REPO_ROOT).as_posix() for path in runtime_paths]
    # 部署目标始终是仓库内正式资产路径；测试可在仓库外临时生成资产。
    asset_rel = DEFAULT_ASSET_ROOT.relative_to(REPO_ROOT).as_posix()
    fanout = root / "commands/fanout_from_iaaccn22.sh"
    fanout_text = f"""#!/usr/bin/env bash
set -u
REMOTE_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
FILES=(
{_shell_array(runtime_rel)}
)
IPS=(10.10.100.23 10.10.100.24 10.10.100.25 10.10.100.26 10.10.100.27 10.10.100.28 10.10.100.29)
cd "$REMOTE_ROOT" || exit 2
failed=0
for ip in "${{IPS[@]}}"; do
  timeout 600 rsync -a --relative "${{FILES[@]}}" "$ip:$REMOTE_ROOT/" || {{ echo "FANOUT_CODE_FAIL $ip"; failed=1; }}
  timeout 600 rsync -a --delete "{asset_rel}/" "$ip:$REMOTE_ROOT/{asset_rel}/" || {{ echo "FANOUT_ASSET_FAIL $ip"; failed=1; }}
done
exit "$failed"
"""
    _atomic_text(fanout, fanout_text)
    fanout.chmod(0o755)

    deploy = root / "commands/deploy_via_iaaccn22.sh"
    deploy_text = f"""#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/../../../.." && pwd)"
REMOTE_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
FILES=(
{_shell_array(runtime_rel)}
)
cd "$REPO_ROOT"
timeout 600 rsync -a --relative "${{FILES[@]}}" "iaaccn22:$REMOTE_ROOT/"
timeout 600 rsync -a --delete "{asset_rel}/" "iaaccn22:$REMOTE_ROOT/{asset_rel}/"
timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 "/bin/bash '$REMOTE_ROOT/{asset_rel}/commands/fanout_from_iaaccn22.sh'"
"""
    _atomic_text(deploy, deploy_text)
    deploy.chmod(0o755)
    return [deploy, fanout]


def _scheduler_args(root: Path) -> list[str]:
    return [
        "python",
        str(SCHEDULER),
        "--batch-name",
        f"{BATCH_NAME}_dry_run_validation",
        "--source-csv",
        str(root / "frozen_inputs/sources/formal3h_13alg_ssr50_source.csv"),
        "--expected-rows",
        "50",
        "--queue-root",
        str(root / "queues/dry_run_validation"),
        "--params-root",
        str(root / "params"),
        "--task-id-allowlist-csv",
        str(root / "manifests/new_queue_6607.csv"),
        "--hosts",
        *HOSTS,
        "--tools",
        *TOOLS,
        "--seeds",
        *(str(seed) for seed in SEEDS),
        "--noise-sigmas",
        "0",
        "0.01",
        "0.05",
        "--controller-host",
        "iaaccn22",
        "--use-internal-ips",
        "--max-jobs-per-host",
        str(MAX_JOBS_PER_HOST),
        "--max-cpu-used-ratio",
        f"{MAX_CPU_USED_RATIO:.17g}",
        "--max-new-jobs-per-host-per-poll",
        str(MAX_NEW_JOBS_PER_HOST_PER_POLL),
        "--load-tier-new-jobs",
        LOAD_TIERS,
        "--max-load-ratio",
        str(MAX_LOAD_RATIO),
        "--max-memory-used-ratio",
        str(MAX_MEMORY_USED_RATIO),
        "--min-free-mem-gb",
        str(MIN_FREE_MEM_GB),
        "--session-prefix",
        SESSION_PREFIX,
        "--host-session-count-prefix",
        SESSION_PREFIX,
        "--poll-seconds",
        str(POLL_SECONDS),
        "--retry-limit",
        "1",
        "--prioritize-llm",
        "--llm-model-assignment",
        "from-params",
        "--llm-default-bucket",
        "turbo",
        "--llm-model-bucket-limits",
        "base:0,turbo:30",
        "--dry-run",
        "--once",
    ]


def _run_and_validate_dry_run(root: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    command = _scheduler_args(root)
    if "--dry-run" not in command or "--preflight-only" in command:
        raise ValueError("本地验证命令必须是纯 dry-run")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"scheduler dry-run 失败: rc={result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    start = None
    for line in result.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "queue_start":
            start = payload
            break
    if not isinstance(start, dict) or start.get("tasks") != EXPECTED_QUEUE:
        raise ValueError(f"scheduler dry-run 未证明 6607 条任务: {start}")
    state_path = (
        root
        / "queues/dry_run_validation/state"
        / f"{BATCH_NAME}_dry_run_validation.state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tasks = state.get("tasks") or {}
    if len(tasks) != EXPECTED_QUEUE or len(set(tasks)) != EXPECTED_QUEUE:
        raise ValueError("dry-run state 不是 6607 个唯一任务")
    algorithm_counts = dict(sorted(Counter(task["tool"] for task in tasks.values()).items()))
    condition_counts = dict(
        sorted(Counter(task["noise_tag"] for task in tasks.values()).items())
    )
    cpu_weight_counts = dict(
        sorted(Counter(str(task["cpu_weight"]) for task in tasks.values()).items())
    )
    if algorithm_counts != ledger["queue_algorithm_counts"]:
        raise ValueError("dry-run 算法计数与 ledger 不一致")
    if condition_counts != ledger["queue_condition_counts"]:
        raise ValueError("dry-run condition 计数与 ledger 不一致")
    if cpu_weight_counts != ledger["queue_cpu_weight_counts"]:
        raise ValueError("dry-run CPU weight 计数与 ledger 不一致")
    report = {
        "status": "passed",
        "mode": "local_scheduler_dry_run",
        "remote_connection_attempted": False,
        "task_count": len(tasks),
        "task_ids_unique": True,
        "algorithm_counts": algorithm_counts,
        "condition_counts": condition_counts,
        "cpu_weight_counts": cpu_weight_counts,
        "cpu_weight_total": sum(int(task["cpu_weight"]) for task in tasks.values()),
        "queue_start": start,
        "state_path": str(state_path.resolve()),
        "state_sha256": _sha256(state_path),
        "command": command,
    }
    _write_json(root / "reports/dry_run_validation.json", report)
    return report


def _readme(report: Mapping[str, Any]) -> str:
    algorithms = ", ".join(TOOLS)
    return f"""# all_15alg_fullcpu_v1

这是 SSR-50 十五算法、三随机种子、三条件的全量 CPU 重跑队列资产。生成资产和 dry-run 不会连接远端；只有 `run_all_15alg_preflight.sh` 与 `run_all_15alg_formal.sh` 会访问服务器。

## 范围与排除

- 逻辑母集：`15 x 50 x 3 x 3 = 6750`。
- 已由 `all_conditions_cpu_v2` 覆盖且从新队列排除：`143`。
- 新队列：`6607`。
- `manifests/composite_ledger_6750.csv` 逐行记录母集归属，严格证明 `143 + 6607 = 6750`、集合无交集且并集等于母集。
- 算法：{algorithms}。
- 条件：`clean`、`noise001`、`noise005`；种子：520、521、522。

## CPU 饱和口径

机器固定为 `iaaccn22~29`，每机实测 `256` 个逻辑 CPU、`128` 个物理核。每机 CPU weight 预算为 `115`，相当于保留约 10% 物理核余量；scheduler 按逻辑 CPU 计算比例，因此配置 `--max-cpu-used-ratio 115/256 = {MAX_CPU_USED_RATIO:.17g}`。

这里的“打满”是持续滚动补位到约 `115` 个物理核等价 CPU weight，不是占满 `256` 个逻辑线程。每机同时最多 115 个任务，每轮最多补 115 个；load tier 在低于 0.90 时允许补至该上限，但最终仍由 CPU weight、load、内存三重门限收口。JAXSR/DRSR 当前权重均为 1；若后续实测其单任务持续占用明显超过 1 个物理核，应在下一版资产提高权重，本版不修改全局 scheduler。

资源保护：load 与内存使用率上限均为 0.90，每机至少保留 32 GB 可用内存，轮询间隔 30 秒。LLMSR 与 DRSR 共用 turbo 全局并发上限 30。

新旧 controller 的 `--session-prefix` 与 `--host-session-count-prefix` 都精确使用 `all_conditions_cpu_v2_`。因此新 controller 会把旧 143 个会话计入资源占用；同时 143 个 task_id 已从新 allowlist 排除，不会产生同名 session 冲突。

## 冻结输入

三份 source CSV 均为 50 行且 SHA256 完全一致。参数来自 formal3h 的 13 个算法、DRSR 和 LLMSR，共 `45` 个 condition 参数；每份均验证 `timeout_in_seconds=10800`、`progress_snapshot_interval_seconds=60` 及噪声字段。

`manifests/runtime_code_fingerprints.json` 固化 scheduler、launcher、runner、subprocess runner、artifact schema、normalizers、15 个 wrapper、iMCTS native regressor 与 toolbox config 的哈希。资产只复制算法参数，不复制 LLM config、API 密钥或其它凭证。

## 执行顺序

1. `commands/run_all_15alg_dry_run.sh`：仅本地构建/检查 6607 任务，不连接远端。
2. `commands/deploy_via_iaaccn22.sh`：先从本机同步到 iaaccn22，再由 iaaccn22 经内网同步到 23~29；本生成器不会执行它。
3. `commands/run_all_15alg_preflight.sh`：对八台 CPU 服务器执行正式前检查。
4. `commands/run_all_15alg_formal.sh`：单 controller 位于 iaaccn22，启用 `--force-rerun-existing` 并持续滚动补位。

生成器不会启动远端。资产哈希见 `asset_manifest.json` 与 `asset_manifest.sha256`。
"""


def _asset_files(root: Path) -> list[Path]:
    excluded = {
        "asset_manifest.json",
        "asset_manifest.sha256",
        "reports/generation_report.json",
    }
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
        and path.relative_to(root).as_posix() not in excluded
    )


def _write_manifest(root: Path, metadata: Mapping[str, Any]) -> tuple[Path, str]:
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _asset_files(root)
    }
    manifest = root / "asset_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "all_15alg_fullcpu_v1.asset_manifest.v1",
            "metadata": metadata,
            "files": files,
        },
    )
    digest = _sha256(manifest)
    _atomic_text(root / "asset_manifest.sha256", f"{digest}  asset_manifest.json\n")
    return manifest, digest


def generate_assets(
    *, asset_root: Path | str | None = None, run_dry_run: bool = True
) -> dict[str, Any]:
    root = Path(asset_root or DEFAULT_ASSET_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    _atomic_text(root / "logs/.gitkeep", "")

    source_rows, source_report = _freeze_and_validate_sources(root)
    params_report = _freeze_and_validate_params(root)
    excluded, exclusion_report = _load_excluded_task_ids(root)
    mother, queue, ledger_report = _build_ledgers(source_rows, excluded)
    _write_csv(root / "manifests/composite_ledger_6750.csv", mother, LEDGER_FIELDS)
    _write_csv(root / "manifests/new_queue_6607.csv", queue, LEDGER_FIELDS)
    commands = _write_commands(root)
    runtime_fingerprint_path, runtime_fingerprints = _write_runtime_fingerprints(root)
    deployment_commands = _write_deployment_scripts(root, _runtime_paths())
    commands.extend(deployment_commands)

    report: dict[str, Any] = {
        "status": "passed",
        "asset_root": str(root),
        "tools": list(TOOLS),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "hosts": list(HOSTS),
        "source_validation": source_report,
        "parameter_validation": params_report,
        "exclusion_input": exclusion_report,
        "composite_ledger": ledger_report,
        "resources": {
            "logical_cpus_per_host": LOGICAL_CPUS_PER_HOST,
            "physical_cpus_per_host": PHYSICAL_CPUS_PER_HOST,
            "cpu_weight_budget_per_host": CPU_WEIGHT_BUDGET_PER_HOST,
            "physical_core_reserve_fraction": 0.10,
            "max_cpu_used_ratio": MAX_CPU_USED_RATIO,
            "max_jobs_per_host": MAX_JOBS_PER_HOST,
            "max_new_jobs_per_host_per_poll": MAX_NEW_JOBS_PER_HOST_PER_POLL,
            "load_tiers": LOAD_TIERS,
            "max_load_ratio": MAX_LOAD_RATIO,
            "max_memory_used_ratio": MAX_MEMORY_USED_RATIO,
            "min_free_mem_gb": MIN_FREE_MEM_GB,
            "poll_seconds": POLL_SECONDS,
            "llm_turbo_global_limit": 30,
        },
        "session_coordination": {
            "session_prefix": SESSION_PREFIX,
            "host_session_count_prefix": SESSION_PREFIX,
            "old_143_counted_by_new_controller": True,
            "old_143_excluded_from_new_queue": True,
            "session_name_collision": False,
        },
        "commands": [str(path.resolve()) for path in commands],
        "runtime_code_fingerprints": {
            "path": str(runtime_fingerprint_path.resolve()),
            "sha256": _sha256(runtime_fingerprint_path),
            "file_count": runtime_fingerprints["file_count"],
            "coverage": runtime_fingerprints["coverage"],
        },
        "deployment": {
            "route": "local -> iaaccn22 -> 10.10.100.23~29",
            "executed": False,
            "scripts": [str(path.resolve()) for path in deployment_commands],
            "copies_llm_config_or_secrets": False,
        },
        "dry_run": None,
        "weight_advisory": {
            "current": {"jaxsr": 1, "drsr": 1},
            "recommendation": "若实测持续占用超过1个物理核，在下一版资产提高权重；本版不改全局scheduler。",
        },
    }
    if run_dry_run:
        report["dry_run"] = _run_and_validate_dry_run(root, ledger_report)
    _atomic_text(root / "README.md", _readme(report))
    _write_json(root / "reports/generation_report.json", report)
    manifest_path, manifest_sha = _write_manifest(root, report)
    report["manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": manifest_sha,
        "self_hash_path": str((root / "asset_manifest.sha256").resolve()),
        "generation_report_excluded_to_avoid_hash_cycle": True,
    }
    _write_json(root / "reports/generation_report.json", report)
    return report


def main() -> int:
    report = generate_assets()
    print(
        json.dumps(
            {
                "status": report["status"],
                "mother": report["composite_ledger"]["mother_count"],
                "excluded": report["composite_ledger"]["excluded_count"],
                "new_queue": report["composite_ledger"]["new_queue_count"],
                "dry_run": (report["dry_run"] or {}).get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
