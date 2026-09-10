#!/usr/bin/env python3
"""导出九算法 Full-664 的逐运行原始 ID/OOD NMSE 与 Core-50 绑定。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL7 = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "01_new3algs_full664_3seeds_clean_1h"
    / "analysis"
    / "full664_7alg_run_level.csv"
)
DEFAULT_STAGE3_RAW = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage3_664dats_4probes_3seeds_1h"
    / "probe4_current_run_level_raw_digest_7968.csv"
)
DEFAULT_PROBE2 = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage1_664dats_2probes_1seed_1h"
    / "01_probe_run_results"
    / "one_seed_probe_task_results_1328.csv"
)
DEFAULT_CORE50 = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage4_core50_12algs_5seeds_4noise_1h"
    / "core50_manifest"
    / "core50_datasets.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "02_core50_full664_rank_correlation_9algs"
)

STAGE3_ALGORITHMS = ("dso", "imcts", "pyoperon", "udsr")
HELDOUT_ALGORITHMS = ("fepysr", "jaxsr", "symbolfit")
THREE_SEED_ALGORITHMS = (*STAGE3_ALGORITHMS, *HELDOUT_ALGORITHMS)
PROBE2_ALGORITHMS = ("llmsr", "pysr")
ALL_ALGORITHMS = (*THREE_SEED_ALGORITHMS, *PROBE2_ALGORITHMS)
ALGORITHM_DISPLAY = {
    "dso": "DSO",
    "fepysr": "FePySR",
    "imcts": "iMCTS",
    "jaxsr": "JAXSR",
    "llmsr": "LLM-SR",
    "pyoperon": "PyOperon",
    "pysr": "PySR",
    "symbolfit": "SymbolFit",
    "udsr": "uDSR",
}
RUN_FIELDS = [
    "algorithm",
    "algorithm_key",
    "dataset_id",
    "global_index",
    "dataset_name",
    "dataset_dir",
    "seed",
    "id_nmse",
    "ood_nmse",
    "metric_complete",
    "valid_output",
    "valid_output_source",
    "canonical_artifact_present",
    "status",
    "is_core50",
    "source_group",
    "nmse_source",
    "task_status",
    "result_status",
    "failure_reason",
    "result_path",
]
CORE_FIELDS = [
    "core50_index",
    "dataset_id",
    "global_index",
    "dataset_name",
    "dataset_dir",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "match"}


def _int(value: Any) -> int | None:
    try:
        return int(float(_text(value)))
    except (TypeError, ValueError):
        return None


def _nmse(value: Any) -> float | None:
    try:
        number = float(_text(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _normalize_dataset_dir(value: Any) -> str:
    text = _text(value).replace("\\", "/").rstrip("/")
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _dataset_reference(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[int, str]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_dir: dict[str, str] = {}
    by_index: dict[int, str] = {}
    for row in rows:
        dataset_id = _text(row.get("dataset_id"))
        global_index = _int(row.get("global_index"))
        dataset_dir = _normalize_dataset_dir(row.get("dataset_rel"))
        dataset_name = _text(row.get("dataset_name") or row.get("dataset"))
        if not dataset_id or global_index is None or not dataset_dir:
            raise ValueError(f"Full-664 数据集标识不完整: {dataset_id!r}")
        metadata = {
            "dataset_id": dataset_id,
            "global_index": global_index,
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir,
        }
        if dataset_id in by_id and by_id[dataset_id] != metadata:
            raise ValueError(f"dataset_id 元数据冲突: {dataset_id}")
        if dataset_dir in by_dir and by_dir[dataset_dir] != dataset_id:
            raise ValueError(f"dataset_dir 映射冲突: {dataset_dir}")
        if global_index in by_index and by_index[global_index] != dataset_id:
            raise ValueError(f"global_index 映射冲突: {global_index}")
        by_id[dataset_id] = metadata
        by_dir[dataset_dir] = dataset_id
        by_index[global_index] = dataset_id
    return by_id, by_dir, by_index


def _stage3_raw_by_key(
    rows: Sequence[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, str]]:
    output: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        algorithm = _text(row.get("method") or row.get("algorithm")).lower()
        dataset_id = _text(row.get("dataset_id"))
        seed = _int(row.get("seed"))
        if algorithm not in STAGE3_ALGORITHMS or not dataset_id or seed is None:
            raise ValueError("Stage3 raw digest 包含非法运行键")
        key = (algorithm, dataset_id, seed)
        if key in output:
            raise ValueError(f"Stage3 raw digest 重复运行键: {key}")
        output[key] = row
    return output


def _core50_by_dir(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_dir = _normalize_dataset_dir(row.get("dataset_dir"))
        core50_index = _int(row.get("core50_index"))
        if not dataset_dir or core50_index is None or dataset_dir in output:
            raise ValueError(f"Core-50 清单包含空值或重复路径: {dataset_dir!r}")
        output[dataset_dir] = {
            "core50_index": core50_index,
            "dataset_name": _text(row.get("dataset_name")),
        }
    return output


def _full7_export_rows(
    rows: Sequence[dict[str, str]],
    stage3_raw: dict[tuple[str, str, int], dict[str, str]],
    core50: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        algorithm = _text(row.get("algorithm")).lower()
        dataset_id = _text(row.get("dataset_id"))
        seed = _int(row.get("seed"))
        dataset_dir = _normalize_dataset_dir(row.get("dataset_rel"))
        if algorithm not in THREE_SEED_ALGORITHMS or not dataset_id or seed is None:
            raise ValueError("七算法来源包含非法运行键")

        raw: dict[str, str] = {}
        if algorithm in STAGE3_ALGORITHMS:
            key = (algorithm, dataset_id, seed)
            if key not in stage3_raw:
                raise ValueError(f"Stage3 raw digest 缺少运行键: {key}")
            raw = stage3_raw[key]
            id_nmse = _nmse(raw.get("result_id_test_nmse"))
            ood_nmse = _nmse(raw.get("result_ood_test_nmse"))
            nmse_source = "stage3_raw_digest"
        else:
            id_nmse = _nmse(row.get("id_test_nmse"))
            ood_nmse = _nmse(row.get("ood_test_nmse"))
            nmse_source = "full664_7alg_run_level"

        result_status = _text(raw.get("result_status") or row.get("status")).lower()
        status = _text(row.get("status") or result_status).lower() or "missing"
        output.append(
            {
                "algorithm": ALGORITHM_DISPLAY[algorithm],
                "algorithm_key": algorithm,
                "dataset_id": dataset_id,
                "global_index": _int(row.get("global_index")),
                "dataset_name": _text(row.get("dataset_name") or row.get("dataset")),
                "dataset_dir": dataset_dir,
                "seed": seed,
                "id_nmse": id_nmse,
                "ood_nmse": ood_nmse,
                "metric_complete": id_nmse is not None and ood_nmse is not None,
                "valid_output": _bool(row.get("valid_output")),
                "valid_output_source": "full664_7alg_run_level.valid_output",
                "canonical_artifact_present": "",
                "status": status,
                "is_core50": dataset_dir in core50,
                "source_group": _text(row.get("source_group")),
                "nmse_source": nmse_source,
                "task_status": "",
                "result_status": result_status,
                "failure_reason": _text(row.get("failure_reason") or raw.get("result_error")),
                "result_path": _text(raw.get("result_result_path") or row.get("result_path")),
            }
        )
    return output


def _probe2_export_rows(
    rows: Sequence[dict[str, str]],
    dataset_by_id: dict[str, dict[str, Any]],
    dataset_by_dir: dict[str, str],
    dataset_by_index: dict[int, str],
    core50: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        algorithm = _text(row.get("method")).lower()
        seed = _int(row.get("seed"))
        global_index = _int(row.get("global_index"))
        dataset_dir = _normalize_dataset_dir(row.get("dataset_dir"))
        if algorithm not in PROBE2_ALGORITHMS or seed is None or global_index is None:
            raise ValueError("Probe-2 来源包含非法运行键")
        dataset_id = dataset_by_dir.get(dataset_dir)
        if dataset_id is None or dataset_by_index.get(global_index) != dataset_id:
            raise ValueError(
                f"Probe-2 数据集无法绑定 Full-664: {algorithm}, {global_index}, {dataset_dir}"
            )
        metadata = dataset_by_id[dataset_id]
        id_nmse = _nmse(row.get("id_nmse"))
        ood_nmse = _nmse(row.get("ood_nmse"))
        metric_complete = id_nmse is not None and ood_nmse is not None
        task_status = _text(row.get("task_status")).lower()
        result_status = _text(row.get("result_status")).lower()
        output.append(
            {
                "algorithm": ALGORITHM_DISPLAY[algorithm],
                "algorithm_key": algorithm,
                "dataset_id": dataset_id,
                "global_index": global_index,
                "dataset_name": _text(row.get("dataset_name"))
                or metadata["dataset_name"],
                "dataset_dir": dataset_dir,
                "seed": seed,
                "id_nmse": id_nmse,
                "ood_nmse": ood_nmse,
                "metric_complete": metric_complete,
                "valid_output": _bool(row.get("canonical_artifact_present"))
                and metric_complete,
                "valid_output_source": "derived:canonical_artifact_present_and_metric_complete",
                "canonical_artifact_present": _bool(
                    row.get("canonical_artifact_present")
                ),
                "status": result_status or task_status or "missing",
                "is_core50": dataset_dir in core50,
                "source_group": "stage1_probe2",
                "nmse_source": "stage1_probe2_run_results",
                "task_status": task_status,
                "result_status": result_status,
                "failure_reason": _text(row.get("result_error") or row.get("task_error")),
                "result_path": _text(row.get("result_path")),
            }
        )
    return output


def _validate_grid(
    rows: Sequence[dict[str, Any]],
    *,
    expected_datasets: int,
    expected_core: int,
) -> dict[str, Any]:
    keys = {
        (str(row["algorithm_key"]), str(row["dataset_id"]), int(row["seed"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("导出结果存在重复 algorithm × dataset_id × seed")
    algorithms = {str(row["algorithm_key"]) for row in rows}
    if algorithms != set(ALL_ALGORITHMS):
        raise ValueError(f"九算法集合不完整: {sorted(algorithms)}")

    reference_datasets = {
        str(row["dataset_id"])
        for row in rows
        if str(row["algorithm_key"]) == ALL_ALGORITHMS[0]
    }
    if len(reference_datasets) != expected_datasets:
        raise ValueError("Full-664 数据集数量不正确")
    for algorithm in ALL_ALGORITHMS:
        algorithm_rows = [row for row in rows if row["algorithm_key"] == algorithm]
        datasets = {str(row["dataset_id"]) for row in algorithm_rows}
        if datasets != reference_datasets:
            raise ValueError(f"{algorithm} 数据集键空间不一致")
        expected_seeds = {520, 521, 522} if algorithm in THREE_SEED_ALGORITHMS else {1314}
        seeds = {int(row["seed"]) for row in algorithm_rows}
        if seeds != expected_seeds:
            raise ValueError(f"{algorithm} 种子集合不正确: {sorted(seeds)}")
        if len(algorithm_rows) != expected_datasets * len(expected_seeds):
            raise ValueError(f"{algorithm} 运行数量不正确")

    core_dataset_ids = {
        str(row["dataset_id"]) for row in rows if bool(row["is_core50"])
    }
    if len(core_dataset_ids) != expected_core:
        raise ValueError(f"Core-50 绑定数量不正确: {len(core_dataset_ids)}")

    return {
        "run_rows": len(rows),
        "unique_run_keys": len(keys),
        "algorithm_count": len(algorithms),
        "dataset_count": len(reference_datasets),
        "core_dataset_count": len(core_dataset_ids),
        "core_run_rows": sum(bool(row["is_core50"]) for row in rows),
        "runs_by_algorithm": {
            algorithm: sum(row["algorithm_key"] == algorithm for row in rows)
            for algorithm in ALL_ALGORITHMS
        },
    }


def export_run_level(
    *,
    full7_path: Path,
    stage3_raw_path: Path,
    probe2_path: Path,
    core50_path: Path,
    output_dir: Path,
    expected_datasets: int = 664,
    expected_core: int = 50,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    full7_source = _read_csv(full7_path)
    stage3_raw_source = _read_csv(stage3_raw_path)
    probe2_source = _read_csv(probe2_path)
    core50_source = _read_csv(core50_path)
    dataset_by_id, dataset_by_dir, dataset_by_index = _dataset_reference(full7_source)
    core50 = _core50_by_dir(core50_source)
    if len(core50) != expected_core or not set(core50) <= set(dataset_by_dir):
        raise ValueError("Core-50 清单无法完整绑定 Full-664 数据集")

    # 四个 Stage3 算法从 raw digest 取真实 NMSE，不从聚合 log-NMSE 反推。
    stage3_raw = _stage3_raw_by_key(stage3_raw_source)
    expected_stage3_keys = {
        (algorithm, dataset_id, seed)
        for algorithm in STAGE3_ALGORITHMS
        for dataset_id in dataset_by_id
        for seed in (520, 521, 522)
    }
    if set(stage3_raw) != expected_stage3_keys:
        raise ValueError("Stage3 raw digest 与 Full-664 四算法运行网格不一致")

    rows = [
        *_full7_export_rows(full7_source, stage3_raw, core50),
        *_probe2_export_rows(
            probe2_source,
            dataset_by_id,
            dataset_by_dir,
            dataset_by_index,
            core50,
        ),
    ]
    algorithm_order = {algorithm: index for index, algorithm in enumerate(ALL_ALGORITHMS)}
    rows.sort(
        key=lambda row: (
            algorithm_order[str(row["algorithm_key"])],
            int(row["global_index"]),
            int(row["seed"]),
        )
    )
    coverage = _validate_grid(
        rows,
        expected_datasets=expected_datasets,
        expected_core=expected_core,
    )

    core_tasks = []
    for dataset_dir, core_metadata in sorted(
        core50.items(), key=lambda item: int(item[1]["core50_index"])
    ):
        dataset_id = dataset_by_dir[dataset_dir]
        metadata = dataset_by_id[dataset_id]
        core_tasks.append(
            {
                "core50_index": core_metadata["core50_index"],
                "dataset_id": dataset_id,
                "global_index": metadata["global_index"],
                "dataset_name": metadata["dataset_name"],
                "dataset_dir": dataset_dir,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_output = output_dir / "full664_run_level_9alg.csv"
    core_output = output_dir / "core50_task_ids.csv"
    summary_output = output_dir / "full664_run_level_9alg_summary.json"
    _write_csv(run_output, rows, RUN_FIELDS)
    _write_csv(core_output, core_tasks, CORE_FIELDS)

    status_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        status_by_algorithm[str(row["algorithm_key"])][str(row["status"])] += 1
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": {
            "full7": {"path": _repo_relative(full7_path, repo_root), "sha256": _sha256(full7_path)},
            "stage3_raw": {
                "path": _repo_relative(stage3_raw_path, repo_root),
                "sha256": _sha256(stage3_raw_path),
            },
            "probe2": {"path": _repo_relative(probe2_path, repo_root), "sha256": _sha256(probe2_path)},
            "core50": {"path": _repo_relative(core50_path, repo_root), "sha256": _sha256(core50_path)},
        },
        "semantics": {
            "id_nmse": "raw finite nonnegative ID NMSE from the recorded run; blank when absent",
            "ood_nmse": "raw finite nonnegative OOD NMSE from the recorded run; blank when absent",
            "missing_policy": "blank; no log inversion and no penalty imputation",
            "valid_output_full7": "source valid_output field",
            "valid_output_probe2": "canonical_artifact_present and both raw NMSE values present",
            "valid_output_source": "row-level field stating whether valid_output is observed or derived",
            "core50_binding": "normalized dataset_dir membership in the frozen Core-50 manifest",
        },
        "coverage": coverage,
        "quality": {
            "metric_complete_by_algorithm": {
                algorithm: sum(
                    row["algorithm_key"] == algorithm and bool(row["metric_complete"])
                    for row in rows
                )
                for algorithm in ALL_ALGORITHMS
            },
            "valid_output_by_algorithm": {
                algorithm: sum(
                    row["algorithm_key"] == algorithm and bool(row["valid_output"])
                    for row in rows
                )
                for algorithm in ALL_ALGORITHMS
            },
            "missing_id_nmse_by_algorithm": {
                algorithm: sum(
                    row["algorithm_key"] == algorithm and row["id_nmse"] is None
                    for row in rows
                )
                for algorithm in ALL_ALGORITHMS
            },
            "missing_ood_nmse_by_algorithm": {
                algorithm: sum(
                    row["algorithm_key"] == algorithm and row["ood_nmse"] is None
                    for row in rows
                )
                for algorithm in ALL_ALGORITHMS
            },
            "status_by_algorithm": {
                algorithm: dict(sorted(status_by_algorithm[algorithm].items()))
                for algorithm in ALL_ALGORITHMS
            },
        },
        "outputs": {
            "run_level": _repo_relative(run_output, repo_root),
            "core50_task_ids": _repo_relative(core_output, repo_root),
            "summary": _repo_relative(summary_output, repo_root),
        },
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full7", type=Path, default=DEFAULT_FULL7)
    parser.add_argument("--stage3-raw", type=Path, default=DEFAULT_STAGE3_RAW)
    parser.add_argument("--probe2", type=Path, default=DEFAULT_PROBE2)
    parser.add_argument("--core50", type=Path, default=DEFAULT_CORE50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-datasets", type=int, default=664)
    parser.add_argument("--expected-core", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_run_level(
        full7_path=args.full7,
        stage3_raw_path=args.stage3_raw,
        probe2_path=args.probe2,
        core50_path=args.core50,
        output_dir=args.output_dir,
        expected_datasets=args.expected_datasets,
        expected_core=args.expected_core,
    )
    print(json.dumps(summary["coverage"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
