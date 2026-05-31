from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_DEPLOY_SCRIPTS = [
    "00_sync_code_and_batch_to_iaaccn22.sh",
    "01_preflight_from_iaaccn22.sh",
    "02_smoke_dispatch_from_iaaccn22.sh",
    "03_full_dispatch_from_iaaccn22.sh",
]
REQUIRED_SYNC_ITEMS = [
    "check/run_e1_candidate200_12alg_load_queue.py",
    "check/launch_e1_benchmark.py",
    "scientific_intelligent_modelling/",
    "benchmark-control/compliance/",
    "exp-planning/02.E1选择验证/generated/params/",
]
REQUIRED_RSYNC_FILTERS = [
    "--exclude=.git/",
    "--exclude=__pycache__/",
    "--exclude=*.pyc",
    "--exclude=*.pyo",
]


FORMAL24H_ALGORITHMS = {
    "gplearn",
    "pyoperon",
    "pysr",
    "dso",
    "tpsr",
    "e2esr",
    "fepysr",
    "jaxsr",
    "QLattice",
    "iMCTS",
    "udsr",
    "ragsr",
    "symbolfit",
}
FORMAL24H_PARAM_KEYS = {
    "gplearn",
    "pyoperon",
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
}


def check_readiness(*, batch_dir: Path, profile: str = "stage1_1h") -> dict[str, Any]:
    issues: list[str] = []
    tasks = _read_csv_if_exists(batch_dir / "manifest" / "tasks.csv", issues, "manifest/tasks.csv")
    datasets = _read_csv_if_exists(batch_dir / "manifest" / "datasets.csv", issues, "manifest/datasets.csv")
    algorithms = _read_algorithms(batch_dir / "manifest" / "algorithms.json", issues)
    noise_levels = _read_csv_if_exists(batch_dir / "manifest" / "noise_levels.csv", [], "manifest/noise_levels.csv")
    ssr50_rows = _read_csv_if_exists(batch_dir / "queues" / "ssr50_source.csv", issues, "queues/ssr50_source.csv")
    smoke_rows = _read_csv_if_exists(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        issues,
        "queues/smoke_2datasets_source.csv",
    )

    expected = _expected_contract(profile)
    _expect_count("manifest/tasks.csv", tasks, expected["tasks"], issues)
    _expect_count("manifest/datasets.csv", datasets, 50, issues)
    _expect_count("manifest/algorithms.json", algorithms, expected["algorithms"], issues)
    _expect_count("queues/ssr50_source.csv", ssr50_rows, 50, issues)
    _expect_count("queues/smoke_2datasets_source.csv", smoke_rows, 2, issues)
    _check_task_contract(tasks, issues, profile=profile)
    if profile == "formal24h_13alg_3seed_3noise":
        _expect_count("manifest/noise_levels.csv", noise_levels, 3, issues)
        _check_formal24h_algorithms(algorithms, issues)
        _check_formal24h_params(batch_dir, issues)
    _check_deploy_scripts(batch_dir, issues)

    summary = {
        "ready": not issues,
        "total_tasks": len(tasks),
        "total_datasets": len(datasets),
        "total_algorithms": len(algorithms),
        "total_noise_levels": len(noise_levels) if noise_levels else 1,
        "ssr50_queue_rows": len(ssr50_rows),
        "smoke_queue_rows": len(smoke_rows),
        "issues": issues,
    }
    readiness_dir = batch_dir / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    (readiness_dir / "readiness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _expected_contract(profile: str) -> dict[str, int]:
    if profile == "formal24h_13alg_3seed_3noise":
        return {"tasks": 5850, "algorithms": 13}
    return {"tasks": 750, "algorithms": 15}


def _read_csv_if_exists(path: Path, issues: list[str], label: str) -> list[dict[str, str]]:
    if not path.exists():
        issues.append(f"{label} missing")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_algorithms(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append("manifest/algorithms.json missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"manifest/algorithms.json invalid json: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append("manifest/algorithms.json must be an object")
        return {}
    return payload


def _expect_count(label: str, rows: object, expected: int, issues: list[str]) -> None:
    actual = len(rows) if hasattr(rows, "__len__") else 0
    if actual != expected:
        issues.append(f"{label} expected {expected}, got {actual}")


def _check_task_contract(tasks: list[dict[str, str]], issues: list[str], *, profile: str) -> None:
    seeds = {row.get("seed") for row in tasks}
    timeouts = {row.get("timeout_in_seconds") for row in tasks}
    intervals = {row.get("progress_snapshot_interval_seconds") for row in tasks}
    min_runtimes = {row.get("min_runtime_seconds") for row in tasks if "min_runtime_seconds" in row}
    noise_tags = {row.get("noise_tag") for row in tasks if "noise_tag" in row}
    noise_sigmas = {row.get("noise_sigma") for row in tasks if "noise_sigma" in row}
    if profile == "formal24h_13alg_3seed_3noise":
        if tasks and seeds != {"520", "521", "522"}:
            issues.append(f"manifest/tasks.csv seed set must be {{520,521,522}}, got {sorted(seeds)}")
        if tasks and noise_tags != {"clean", "noise001", "noise005"}:
            issues.append(f"manifest/tasks.csv noise_tag set must be {{clean,noise001,noise005}}, got {sorted(noise_tags)}")
        if tasks and noise_sigmas != {"0", "0.01", "0.05"}:
            issues.append(f"manifest/tasks.csv noise_sigma set must be {{0,0.01,0.05}}, got {sorted(noise_sigmas)}")
        if tasks and timeouts != {"86400"}:
            issues.append(f"manifest/tasks.csv timeout must be 86400, got {sorted(timeouts)}")
        if tasks and min_runtimes != {"82800"}:
            issues.append(f"manifest/tasks.csv min_runtime_seconds must be 82800, got {sorted(min_runtimes)}")
    else:
        if tasks and seeds != {"520"}:
            issues.append(f"manifest/tasks.csv seed set must be {{520}}, got {sorted(seeds)}")
        if tasks and timeouts != {"3600"}:
            issues.append(f"manifest/tasks.csv timeout must be 3600, got {sorted(timeouts)}")
    if tasks and intervals != {"60"}:
        issues.append(f"manifest/tasks.csv progress interval must be 60, got {sorted(intervals)}")
    task_ids = [row.get("task_id", "") for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        issues.append("manifest/tasks.csv contains duplicate task_id")


def _check_formal24h_algorithms(algorithms: dict[str, Any], issues: list[str]) -> None:
    actual = set(algorithms)
    if actual != FORMAL24H_ALGORITHMS:
        issues.append(f"manifest/algorithms.json formal24h algorithms mismatch: {sorted(actual)}")


def _check_formal24h_params(batch_dir: Path, issues: list[str]) -> None:
    expected_sigmas = {"clean": 0.0, "noise001": 0.01, "noise005": 0.05}
    for algorithm in sorted(FORMAL24H_PARAM_KEYS):
        for noise_tag, sigma in expected_sigmas.items():
            path = batch_dir / "params" / f"{algorithm}__{noise_tag}.json"
            if not path.exists():
                issues.append(f"params/{algorithm}__{noise_tag}.json missing")
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(f"params/{algorithm}__{noise_tag}.json invalid json: {exc}")
                continue
            if payload.get("timeout_in_seconds") != 86400:
                issues.append(f"params/{algorithm}__{noise_tag}.json timeout_in_seconds must be 86400")
            if payload.get("progress_snapshot_interval_seconds") != 60:
                issues.append(f"params/{algorithm}__{noise_tag}.json progress_snapshot_interval_seconds must be 60")
            if float(payload.get("train_label_noise_sigma", -1)) != sigma:
                issues.append(f"params/{algorithm}__{noise_tag}.json train_label_noise_sigma must be {sigma}")


def _check_deploy_scripts(batch_dir: Path, issues: list[str]) -> None:
    for script in REQUIRED_DEPLOY_SCRIPTS:
        path = batch_dir / "deploy" / script
        if not path.exists():
            issues.append(f"deploy/{script} missing")
            continue
        content = path.read_text(encoding="utf-8")
        if script.startswith("00_sync"):
            _check_sync_script_contract(script, content, issues)
        else:
            _check_stage_script_contract(script, content, issues)


def _check_sync_script_contract(script: str, content: str, issues: list[str]) -> None:
    if "rsync -aR" not in content:
        issues.append(f"deploy/{script} does not call rsync")
    if '"${RSYNC_FILTERS[@]}"' not in content:
        issues.append(f"deploy/{script} does not use RSYNC_FILTERS array")
    if '"${SYNC_ITEMS[@]}"' not in content:
        issues.append(f"deploy/{script} does not use SYNC_ITEMS array")
    for item in REQUIRED_SYNC_ITEMS:
        if item not in content:
            issues.append(f"deploy/{script} missing sync item {item}")
    for item in REQUIRED_RSYNC_FILTERS:
        if item not in content:
            issues.append(f"deploy/{script} missing rsync filter {item}")


def _check_stage_script_contract(script: str, content: str, issues: list[str]) -> None:
    if "run_e1_candidate200_12alg_load_queue.py" not in content:
        issues.append(f"deploy/{script} does not call load queue scheduler")
    if script.startswith("01_preflight") and "check_preflight_report.py" not in content:
        issues.append(f"deploy/{script} does not call preflight gate")
    if script.startswith("02_smoke") and (
        "preflight_gate_summary.json" not in content or "ready_for_smoke" not in content
    ):
        issues.append(f"deploy/{script} missing preflight gate check")
    if script.startswith("03_full") and (
        "smoke/audit/audit_gate_summary.json" not in content or "audit_passed" not in content
    ):
        issues.append(f"deploy/{script} missing smoke audit gate check")
