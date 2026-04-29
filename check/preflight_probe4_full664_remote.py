#!/usr/bin/env python3
"""Probe4 full-664 远端启动前只读预检。

该脚本只检查文件、数据和环境，不启动 benchmark 任务。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


TOOLS = ("udsr", "dso", "imcts", "pyoperon")
SEEDS = (520, 521, 522)
HOSTS = ("iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29")
ENV_BY_TOOL = {
    "udsr": "sim_dso",
    "dso": "sim_dso",
    "imcts": "sim_iMCTS",
    "pyoperon": "sim_base",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_dataset_dir(dataset_dir: str, data_root: Path, repo_root: Path) -> Path:
    path = Path(dataset_dir)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "sim-datasets-data":
        return data_root / Path(*path.parts[1:])
    return repo_root / path


def _check_assets(repo_root: Path) -> dict[str, Any]:
    asset_root = repo_root / "exp-planning/02.E1选择验证/generated/probe4_full664_v1"
    full = asset_root / "full664_unified.csv"
    manifest = asset_root / "probe4_full664_manifest.csv"
    report: dict[str, Any] = {
        "asset_root": str(asset_root),
        "full664_exists": full.exists(),
        "manifest_exists": manifest.exists(),
        "full664_rows": None,
        "manifest_rows": None,
        "missing_slice_files": [],
        "bad_slice_counts": [],
        "missing_remote_jobs": [],
    }
    if full.exists():
        report["full664_rows"] = len(_read_csv(full))
    if manifest.exists():
        manifest_rows = _read_csv(manifest)
        report["manifest_rows"] = len(manifest_rows)
        for row in manifest_rows:
            slice_path = asset_root / str(row["slice_csv"])
            job_path = asset_root / str(row["remote_job"])
            if not slice_path.exists():
                report["missing_slice_files"].append(str(slice_path))
            else:
                actual_rows = len(_read_csv(slice_path))
                expected_rows = int(row["tasks"])
                if actual_rows != expected_rows:
                    report["bad_slice_counts"].append(
                        {
                            "slice": str(slice_path),
                            "expected": expected_rows,
                            "actual": actual_rows,
                        }
                    )
            if not job_path.exists():
                report["missing_remote_jobs"].append(str(job_path))
    return report


def _check_data(repo_root: Path, data_root: Path) -> dict[str, Any]:
    full = repo_root / "exp-planning/02.E1选择验证/generated/probe4_full664_v1/full664_unified.csv"
    rows = _read_csv(full) if full.exists() else []
    missing: list[dict[str, str]] = []
    required = ("metadata_yaml", "train_csv", "valid_csv", "id_test_csv", "ood_test_csv")
    for row in rows:
        dataset_dir = _resolve_dataset_dir(row.get("dataset_dir", ""), data_root, repo_root)
        if not dataset_dir.exists():
            missing.append({"dataset_name": row.get("dataset_name", ""), "missing": str(dataset_dir)})
            continue
        for key in required:
            value = row.get(key) or ""
            path = _resolve_dataset_dir(value, data_root, repo_root)
            if not path.exists():
                missing.append({"dataset_name": row.get("dataset_name", ""), "missing": str(path)})
    return {
        "dataset_rows": len(rows),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }


def _run(cmd: list[str], cwd: Path, timeout: int = 45) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-1200:],
        "stderr": proc.stderr.strip()[-1200:],
    }


def _check_envs(repo_root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    envs = sorted(set(ENV_BY_TOOL.values()))
    for env in envs:
        checks[f"{env}:python"] = _run(["conda", "run", "-n", env, "python", "-V"], repo_root)
    checks["sim_base:runner_import"] = _run(
        [
            "conda",
            "run",
            "-n",
            "sim_base",
            "python",
            "-c",
            "from scientific_intelligent_modelling.benchmarks.runner import run_benchmark_task; print('runner_ok')",
        ],
        repo_root,
    )
    checks["sim_dso:tool_import"] = _run(
        [
            "conda",
            "run",
            "-n",
            "sim_dso",
            "python",
            "-c",
            "from scientific_intelligent_modelling import SymbolicRegressor; print('dso_udsr_ok')",
        ],
        repo_root,
    )
    checks["sim_iMCTS:tool_import"] = _run(
        [
            "conda",
            "run",
            "-n",
            "sim_iMCTS",
            "python",
            "-c",
            "from scientific_intelligent_modelling import SymbolicRegressor; print('imcts_ok')",
        ],
        repo_root,
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe4 full664 远端预检")
    parser.add_argument("--repo-root", default="/home/zhangziwen/projects/scientific-intelligent-modelling")
    parser.add_argument("--data-root", default="/home/zhangziwen/sim-datasets-data")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    report = {
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "assets": _check_assets(repo_root),
        "data": _check_data(repo_root, data_root),
        "envs": _check_envs(repo_root),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
