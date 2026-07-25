#!/usr/bin/env python3
"""准备 NeurIPS rebuttal 新增三算法 full-664 clean 1h 实验资产。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = (
    REPO_ROOT
    / "exp-planning/02.E1选择验证/generated/probe4_full664_v1/full664_unified.csv"
)
DEFAULT_AAAI_PARAMS_ROOT = (
    REPO_ROOT
    / "benchmark-runs/formal3h/"
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658/params"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
)

TOOLS = ("fepysr", "jaxsr", "symbolfit")
SEEDS = (520, 521, 522)
EXPECTED_DATASETS = 664
TIMEOUT_SECONDS = 3600
MIN_RUNTIME_SECONDS = 3300
SNAPSHOT_INTERVAL_SECONDS = 60
SMOKE_TIMEOUT_SECONDS = 600
SMOKE_MIN_RUNTIME_SECONDS = 540
SMOKE_DATASETS = 2

ALGORITHM_CONFIG = {
    "fepysr": {"env": "sim_fepysr", "regressor": "FePySRRegressor"},
    "jaxsr": {"env": "sim_jaxsr", "regressor": "JAXSRRegressor"},
    "symbolfit": {"env": "sim_symbolfit", "regressor": "SymbolFitRegressor"},
}

DATASET_MANIFEST_FIELDS = (
    "dataset_id",
    "global_index",
    "dataset_name",
    "family",
    "subgroup",
    "dataset_dir",
    "dataset_rel",
    "basename",
    "formula_py",
)

TASK_FIELDS = (
    "task_id",
    "algorithm",
    "dataset_id",
    "global_index",
    "dataset_name",
    "dataset_dir",
    "dataset_rel",
    "seed",
    "noise_tag",
    "noise_sigma",
    "params_name",
    "timeout_in_seconds",
    "min_runtime_seconds",
    "progress_snapshot_interval_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"审计来源路径位于仓库根目录之外: {resolved}"
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...] | list[str]) -> None:
    if not rows:
        raise ValueError(f"拒绝写空 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_and_validate_datasets(
    source_csv: Path,
    *,
    expected_datasets: int,
) -> tuple[list[dict[str, str]], list[str]]:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    required = {
        "global_index",
        "dataset_name",
        "dataset_dir",
        "dataset_rel",
        "family",
        "subgroup",
        "basename",
        "formula_py",
    }
    missing = sorted(required - set(source_fieldnames))
    if missing:
        raise ValueError(f"full-664 清单缺少字段: {missing}")
    if len(rows) != expected_datasets:
        raise ValueError(
            f"full-664 清单行数错误: expected={expected_datasets}, actual={len(rows)}"
        )

    indexes: list[int] = []
    for row in rows:
        try:
            index = int(row["global_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"非法 global_index: {row.get('global_index')!r}") from exc
        indexes.append(index)
        row["global_index"] = str(index)
        row["dataset_id"] = f"g{index:04d}"

    expected_indexes = list(range(1, expected_datasets + 1))
    if sorted(indexes) != expected_indexes:
        raise ValueError(
            "global_index 必须唯一且连续覆盖 "
            f"1..{expected_datasets}; actual_min={min(indexes, default=None)}, "
            f"actual_max={max(indexes, default=None)}, unique={len(set(indexes))}"
        )

    for field in ("dataset_dir", "dataset_rel"):
        values = [row[field].strip() for row in rows]
        if any(not value for value in values):
            raise ValueError(f"{field} 不允许为空")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} 必须在 full-664 中唯一")

    rows.sort(key=lambda row: int(row["global_index"]))
    queue_fieldnames = list(source_fieldnames)
    if "dataset_id" not in queue_fieldnames:
        insert_at = queue_fieldnames.index("global_index") + 1
        queue_fieldnames.insert(insert_at, "dataset_id")
    return rows, queue_fieldnames


def _prepare_params(
    *,
    aaai_params_root: Path,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    fingerprints: dict[str, dict[str, str]] = {}
    for tool in TOOLS:
        source_path = aaai_params_root / f"{tool}__clean.json"
        if not source_path.exists():
            raise FileNotFoundError(f"AAAI 参数文件不存在: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"AAAI 参数必须是 JSON object: {source_path}")
        if int(payload.get("timeout_in_seconds", 0)) != 10800:
            raise ValueError(f"AAAI 参数预算不是 3h: {source_path}")
        if int(payload.get("progress_snapshot_interval_seconds", 0)) != SNAPSHOT_INTERVAL_SECONDS:
            raise ValueError(f"AAAI 参数快照间隔不是 60s: {source_path}")
        if float(payload.get("train_label_noise_sigma", 0.0)) != 0.0:
            raise ValueError(f"AAAI clean 参数的 noise sigma 非零: {source_path}")
        if bool(payload.get("train_label_noise_enabled", False)):
            raise ValueError(f"AAAI clean 参数错误启用了噪声: {source_path}")

        provenance_path = output_dir / "provenance/aaai_params_3h" / source_path.name
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, provenance_path)

        full_payload = dict(payload)
        full_payload["timeout_in_seconds"] = TIMEOUT_SECONDS
        full_payload["progress_snapshot_interval_seconds"] = SNAPSHOT_INTERVAL_SECONDS
        full_payload["train_label_noise_sigma"] = 0.0
        full_payload["train_label_noise_enabled"] = False
        _write_json(output_dir / "params" / source_path.name, full_payload)

        smoke_payload = dict(full_payload)
        smoke_payload["timeout_in_seconds"] = SMOKE_TIMEOUT_SECONDS
        _write_json(output_dir / "params_smoke" / source_path.name, smoke_payload)
        _write_json(output_dir / "smoke/params" / source_path.name, smoke_payload)

        fingerprints[tool] = {
            "path": _repo_relative(source_path),
            "sha256": _sha256(source_path),
            "archived_path": _repo_relative(provenance_path),
        }
    return fingerprints


def _dataset_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {field: row.get(field, "") for field in DATASET_MANIFEST_FIELDS}
        for row in rows
    ]


def _task_rows(
    rows: list[dict[str, str]],
    *,
    timeout_seconds: int,
    min_runtime_seconds: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for tool in TOOLS:
        for seed in SEEDS:
            for row in rows:
                dataset_id = row["dataset_id"]
                tasks.append(
                    {
                        "task_id": f"{tool}__seed{seed}__clean__{dataset_id}",
                        "algorithm": tool,
                        "dataset_id": dataset_id,
                        "global_index": row["global_index"],
                        "dataset_name": row["dataset_name"],
                        "dataset_dir": row["dataset_dir"],
                        "dataset_rel": row["dataset_rel"],
                        "seed": seed,
                        "noise_tag": "clean",
                        "noise_sigma": 0,
                        "params_name": f"{tool}__clean",
                        "timeout_in_seconds": timeout_seconds,
                        "min_runtime_seconds": min_runtime_seconds,
                        "progress_snapshot_interval_seconds": SNAPSHOT_INTERVAL_SECONDS,
                    }
                )
    return tasks


def _budget_payload(*, datasets: int, timeout_seconds: int, min_runtime_seconds: int) -> dict[str, Any]:
    return {
        "timeout_in_seconds": timeout_seconds,
        "min_runtime_seconds": min_runtime_seconds,
        "progress_snapshot_interval_seconds": SNAPSHOT_INTERVAL_SECONDS,
        "total_algorithms": len(TOOLS),
        "total_datasets": datasets,
        "total_seeds": len(SEEDS),
        "total_noise_levels": 1,
        "total_tasks": len(TOOLS) * datasets * len(SEEDS),
        "algorithms": list(TOOLS),
        "seeds": list(SEEDS),
        "noise_levels": [{"tag": "clean", "sigma": 0.0}],
    }


def _write_manifest_set(
    *,
    manifest_dir: Path,
    dataset_rows: list[dict[str, str]],
    task_rows: list[dict[str, Any]],
    budget: dict[str, Any],
    git_revision: str,
    source_fingerprints: dict[str, Any],
) -> None:
    _write_json(manifest_dir / "algorithms.json", ALGORITHM_CONFIG)
    _write_csv(
        manifest_dir / "datasets.csv",
        dataset_rows,
        list(DATASET_MANIFEST_FIELDS),
    )
    _write_csv(
        manifest_dir / "noise_levels.csv",
        [{"noise_tag": "clean", "noise_sigma": 0}],
        ["noise_tag", "noise_sigma"],
    )
    _write_json(manifest_dir / "budget.json", budget)
    _write_csv(manifest_dir / "tasks.csv", task_rows, list(TASK_FIELDS))
    (manifest_dir / "git_revision.txt").write_text(
        git_revision.strip() + "\n",
        encoding="utf-8",
    )
    _write_json(manifest_dir / "source_fingerprints.json", source_fingerprints)


def prepare_batch(
    *,
    source_csv: Path,
    aaai_params_root: Path,
    output_dir: Path,
    expected_datasets: int = EXPECTED_DATASETS,
    git_revision: str | None = None,
) -> dict[str, int]:
    source_csv = source_csv.resolve()
    aaai_params_root = aaai_params_root.resolve()
    output_dir = output_dir.resolve()
    if git_revision is None:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()

    rows, queue_fieldnames = _read_and_validate_datasets(
        source_csv,
        expected_datasets=expected_datasets,
    )
    aaai_fingerprints = _prepare_params(
        aaai_params_root=aaai_params_root,
        output_dir=output_dir,
    )
    source_fingerprints = {
        "schema_version": 1,
        "path_base": "repository_root",
        "git_revision": git_revision,
        "source_csv": {
            "path": _repo_relative(source_csv),
            "sha256": _sha256(source_csv),
        },
        "aaai_params": aaai_fingerprints,
    }

    dataset_rows = _dataset_manifest_rows(rows)
    tasks = _task_rows(
        rows,
        timeout_seconds=TIMEOUT_SECONDS,
        min_runtime_seconds=MIN_RUNTIME_SECONDS,
    )
    _write_manifest_set(
        manifest_dir=output_dir / "manifest",
        dataset_rows=dataset_rows,
        task_rows=tasks,
        budget=_budget_payload(
            datasets=len(rows),
            timeout_seconds=TIMEOUT_SECONDS,
            min_runtime_seconds=MIN_RUNTIME_SECONDS,
        ),
        git_revision=git_revision,
        source_fingerprints=source_fingerprints,
    )

    queue_rows = [{field: row.get(field, "") for field in queue_fieldnames} for row in rows]
    _write_csv(output_dir / "queues/full664_source.csv", queue_rows, queue_fieldnames)
    _write_csv(
        output_dir / "queues/smoke_2datasets_source.csv",
        queue_rows[:SMOKE_DATASETS],
        queue_fieldnames,
    )

    smoke_rows = rows[:SMOKE_DATASETS]
    smoke_dataset_rows = _dataset_manifest_rows(smoke_rows)
    smoke_tasks = _task_rows(
        smoke_rows,
        timeout_seconds=SMOKE_TIMEOUT_SECONDS,
        min_runtime_seconds=SMOKE_MIN_RUNTIME_SECONDS,
    )
    _write_manifest_set(
        manifest_dir=output_dir / "smoke/manifest",
        dataset_rows=smoke_dataset_rows,
        task_rows=smoke_tasks,
        budget=_budget_payload(
            datasets=len(smoke_rows),
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
            min_runtime_seconds=SMOKE_MIN_RUNTIME_SECONDS,
        ),
        git_revision=git_revision,
        source_fingerprints=source_fingerprints,
    )
    _write_csv(
        output_dir / "smoke/queues/smoke_2datasets_source.csv",
        queue_rows[:SMOKE_DATASETS],
        queue_fieldnames,
    )

    summary = {
        "algorithms": len(TOOLS),
        "datasets": len(rows),
        "seeds": len(SEEDS),
        "noise_levels": 1,
        "tasks": len(tasks),
        "smoke_datasets": len(smoke_rows),
        "smoke_tasks": len(smoke_tasks),
    }
    _write_json(output_dir / "manifest/preparation_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--aaai-params-root", default=str(DEFAULT_AAAI_PARAMS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--expected-datasets", type=int, default=EXPECTED_DATASETS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = prepare_batch(
        source_csv=Path(args.source_csv),
        aaai_params_root=Path(args.aaai_params_root),
        output_dir=Path(args.output_dir),
        expected_datasets=args.expected_datasets,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
