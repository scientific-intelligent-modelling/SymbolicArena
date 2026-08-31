from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


NOISE_ORDER = {
    "clean": 0,
    "noise001": 1,
    "noise005": 2,
}

EXPECTED_SEEDS = (520, 521, 522)
EXPECTED_NOISE_TAGS = ("clean", "noise001", "noise005")
EXPECTED_STAGE4_ROWS = 15 * 50 * 3 * 3
EXPECTED_GT_ROWS = 50


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int(value: str, *, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 不是合法整数: {value!r}") from exc


def _parse_nonnegative_finite_float(value: str, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 不是合法浮点数: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} 不是有限值: {value!r}")
    if parsed < 0.0:
        raise ValueError(f"{field_name} 必须非负: {value!r}")
    return parsed


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _stable_run_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        NOISE_ORDER[row["noise_tag"]],
        row["algorithm"],
        row["dataset_id"],
        row["seed"],
        row["task_id"],
        row["host"],
        row["path"],
    )


def _stable_gt_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["core50_index"],
        row["basename"],
        row["dataset_name"],
    )


def load_stage4_rows(stage4_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with stage4_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "batch",
            "algorithm",
            "host",
            "seed",
            "noise_tag",
            "task_id",
            "dataset_id",
            "status",
            "seconds",
            "id_nmse",
            "ood_nmse",
            "id_r2",
            "ood_r2",
            "id_acc",
            "ood_acc",
            "path",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Stage4 CSV 缺少字段: {missing}")
        for row_index, raw_row in enumerate(reader, start=1):
            row = dict(raw_row)
            row["row_index"] = row_index
            row["seed"] = _parse_int(raw_row["seed"], field_name="seed")
            row["seconds"] = _parse_nonnegative_finite_float(raw_row["seconds"], field_name="seconds")
            row["id_nmse"] = _parse_nonnegative_finite_float(raw_row["id_nmse"], field_name="id_nmse")
            row["ood_nmse"] = _parse_nonnegative_finite_float(raw_row["ood_nmse"], field_name="ood_nmse")
            row["id_r2"] = float(raw_row["id_r2"])
            row["ood_r2"] = float(raw_row["ood_r2"])
            row["id_acc"] = float(raw_row["id_acc"])
            row["ood_acc"] = float(raw_row["ood_acc"])
            row["noise_order"] = NOISE_ORDER.get(raw_row["noise_tag"], -1)
            row["logical_key"] = (
                f"{raw_row['algorithm']}::{raw_row['dataset_id']}::"
                f"s{row['seed']}::{raw_row['noise_tag']}"
            )
            rows.append(row)
    return rows


def validate_stage4_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != EXPECTED_STAGE4_ROWS:
        errors.append(f"Stage4 行数应为 {EXPECTED_STAGE4_ROWS}，实际为 {len(rows)}")

    algorithms = sorted({row["algorithm"] for row in rows})
    dataset_ids = sorted({row["dataset_id"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    noise_tags = sorted({row["noise_tag"] for row in rows}, key=lambda item: NOISE_ORDER.get(item, 99))
    batches = sorted({row["batch"] for row in rows})

    statuses = Counter(row["status"] for row in rows)
    if set(statuses) != {"ok"}:
        errors.append(f"Stage4 status 只允许 ok，实际为 {dict(statuses)}")

    seen: set[tuple[str, str, int, str]] = set()
    duplicates: list[str] = []
    for row in rows:
        if row["noise_order"] < 0:
            errors.append(f"未知 noise_tag: {row['noise_tag']}")
        key = (row["algorithm"], row["dataset_id"], row["seed"], row["noise_tag"])
        if key in seen:
            duplicates.append("::".join([key[0], key[1], f"s{key[2]}", key[3]]))
        seen.add(key)
    if duplicates:
        errors.append(f"发现重复逻辑键 {len(duplicates)} 条")

    expected_seed_set = set(EXPECTED_SEEDS)
    if set(seeds) != expected_seed_set:
        errors.append(f"seed 集合不符，期望 {sorted(expected_seed_set)}，实际 {seeds}")
    expected_noise_set = set(EXPECTED_NOISE_TAGS)
    if set(noise_tags) != expected_noise_set:
        errors.append(f"noise_tag 集合不符，期望 {sorted(expected_noise_set)}，实际 {noise_tags}")
    if len(dataset_ids) != 50:
        errors.append(f"dataset_id 数量应为 50，实际为 {len(dataset_ids)}")
    if len(algorithms) != 15:
        errors.append(f"algorithm 数量应为 15，实际为 {len(algorithms)}")

    per_algorithm = Counter(row["algorithm"] for row in rows)
    bad_algorithm_counts = {name: count for name, count in per_algorithm.items() if count != 450}
    if bad_algorithm_counts:
        errors.append(f"存在算法覆盖数不是 450: {bad_algorithm_counts}")

    per_noise = Counter(row["noise_tag"] for row in rows)
    bad_noise_counts = {
        noise_tag: count for noise_tag, count in per_noise.items() if count != 2250
    }
    if bad_noise_counts:
        errors.append(f"存在条件覆盖数不是 2250: {bad_noise_counts}")

    per_algorithm_noise = Counter((row["algorithm"], row["noise_tag"]) for row in rows)
    bad_algorithm_noise = {
        f"{algorithm}::{noise_tag}": count
        for (algorithm, noise_tag), count in per_algorithm_noise.items()
        if count != 150
    }
    if bad_algorithm_noise:
        errors.append(f"存在算法-条件覆盖数不是 150: {bad_algorithm_noise}")

    sorted_rows = sorted(rows, key=_stable_run_sort_key)
    return {
        "ok": not errors,
        "errors": errors,
        "summary": {
            "row_count": len(rows),
            "algorithm_count": len(algorithms),
            "dataset_count": len(dataset_ids),
            "seed_count": len(seeds),
            "noise_count": len(noise_tags),
            "algorithms": algorithms,
            "datasets": dataset_ids,
            "seeds": seeds,
            "noise_tags": noise_tags,
            "batches": batches,
            "status_counts": dict(sorted(statuses.items())),
            "per_algorithm": dict(sorted(per_algorithm.items())),
            "per_noise_tag": {tag: per_noise[tag] for tag in EXPECTED_NOISE_TAGS},
            "per_algorithm_noise_tag": {
                f"{algorithm}::{noise_tag}": per_algorithm_noise[(algorithm, noise_tag)]
                for algorithm in algorithms
                for noise_tag in EXPECTED_NOISE_TAGS
            },
            "first_logical_keys": [row["logical_key"] for row in sorted_rows[:5]],
            "last_logical_keys": [row["logical_key"] for row in sorted_rows[-5:]],
        },
    }


def build_source_runs_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=_stable_run_sort_key):
        manifest_rows.append(
            {
                "batch": row["batch"],
                "noise_order": row["noise_order"],
                "algorithm": row["algorithm"],
                "dataset_id": row["dataset_id"],
                "seed": row["seed"],
                "noise_tag": row["noise_tag"],
                "task_id": row["task_id"],
                "host": row["host"],
                "status": row["status"],
                "seconds": f"{row['seconds']:.6f}",
                "id_nmse": f"{row['id_nmse']:.17g}",
                "ood_nmse": f"{row['ood_nmse']:.17g}",
                "id_r2": f"{row['id_r2']:.17g}",
                "ood_r2": f"{row['ood_r2']:.17g}",
                "id_acc": f"{row['id_acc']:.17g}",
                "ood_acc": f"{row['ood_acc']:.17g}",
                "path": row["path"],
                "logical_key": row["logical_key"],
            }
        )
    return manifest_rows


def load_ground_truth_rows(gt_csv: Path, *, repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gt_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "core50_index",
            "dataset_name",
            "dataset_dir",
            "family",
            "subgroup",
            "basename",
            "target_name",
            "feature_count",
            "metadata_yaml",
            "train_csv",
            "valid_csv",
            "id_test_csv",
            "ood_test_csv",
            "formula_py",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Ground Truth CSV 缺少字段: {missing}")
        for raw_row in reader:
            row = dict(raw_row)
            row["core50_index"] = _parse_int(raw_row["core50_index"], field_name="core50_index")
            row["feature_count"] = _parse_int(raw_row["feature_count"], field_name="feature_count")
            resolved = {
                name: _resolve_repo_path(repo_root, raw_row[name])
                for name in (
                    "metadata_yaml",
                    "train_csv",
                    "valid_csv",
                    "id_test_csv",
                    "ood_test_csv",
                    "formula_py",
                )
            }
            for field_name, path in resolved.items():
                if not path.exists():
                    raise FileNotFoundError(f"{field_name} 不存在: {path}")
                row[f"{field_name}_abs"] = str(path)
                row[f"{field_name}_sha256"] = sha256_file(path)
            rows.append(row)
    return sorted(rows, key=_stable_gt_sort_key)


def validate_ground_truth_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != EXPECTED_GT_ROWS:
        errors.append(f"Ground Truth 行数应为 {EXPECTED_GT_ROWS}，实际为 {len(rows)}")
    indexes = [row["core50_index"] for row in rows]
    if indexes != list(range(1, EXPECTED_GT_ROWS + 1)):
        errors.append("Ground Truth core50_index 必须连续为 1..50")
    basenames = [row["basename"] for row in rows]
    if len(set(basenames)) != len(basenames):
        errors.append("Ground Truth basename 存在重复")
    dataset_names = [row["dataset_name"] for row in rows]
    if len(set(dataset_names)) != len(dataset_names):
        errors.append("Ground Truth dataset_name 存在重复")
    return {
        "ok": not errors,
        "errors": errors,
        "summary": {
            "row_count": len(rows),
            "first_basenames": basenames[:5],
            "last_basenames": basenames[-5:],
        },
    }


def validate_source_ground_truth_alignment(
    source_rows: list[dict[str, Any]],
    ground_truth_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """验证运行清单与 Ground Truth 引用的是同一组 benchmark 任务。"""

    source_datasets = {str(row["dataset_id"]) for row in source_rows}
    ground_truth_datasets = {str(row["basename"]) for row in ground_truth_rows}
    missing_from_source = sorted(ground_truth_datasets - source_datasets)
    missing_from_ground_truth = sorted(source_datasets - ground_truth_datasets)
    return {
        "ok": not missing_from_source and not missing_from_ground_truth,
        "source_dataset_count": len(source_datasets),
        "ground_truth_dataset_count": len(ground_truth_datasets),
        "missing_from_source": missing_from_source,
        "missing_from_ground_truth": missing_from_ground_truth,
    }


def build_ground_truth_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        manifest_rows.append(
            {
                "core50_index": row["core50_index"],
                "dataset_name": row["dataset_name"],
                "basename": row["basename"],
                "family": row["family"],
                "subgroup": row["subgroup"],
                "target_name": row["target_name"],
                "feature_count": row["feature_count"],
                "dataset_dir": row["dataset_dir"],
                "metadata_yaml": row["metadata_yaml"],
                "metadata_yaml_sha256": row["metadata_yaml_sha256"],
                "train_csv": row["train_csv"],
                "train_csv_sha256": row["train_csv_sha256"],
                "valid_csv": row["valid_csv"],
                "valid_csv_sha256": row["valid_csv_sha256"],
                "id_test_csv": row["id_test_csv"],
                "id_test_csv_sha256": row["id_test_csv_sha256"],
                "ood_test_csv": row["ood_test_csv"],
                "ood_test_csv_sha256": row["ood_test_csv_sha256"],
                "formula_py": row["formula_py"],
                "formula_py_sha256": row["formula_py_sha256"],
            }
        )
    return manifest_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"不允许写空 CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_source_preflight(
    *,
    stage4_csv: Path,
    gt_csv: Path,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    stage4_rows = load_stage4_rows(stage4_csv)
    stage4_validation = validate_stage4_rows(stage4_rows)
    if not stage4_validation["ok"]:
        raise ValueError("; ".join(stage4_validation["errors"]))

    gt_rows = load_ground_truth_rows(gt_csv, repo_root=repo_root)
    gt_validation = validate_ground_truth_rows(gt_rows)
    if not gt_validation["ok"]:
        raise ValueError("; ".join(gt_validation["errors"]))

    alignment = validate_source_ground_truth_alignment(stage4_rows, gt_rows)
    if not alignment["ok"]:
        raise ValueError(
            "Stage4 与 Ground Truth 数据集集合不一致: "
            f"source 缺失={alignment['missing_from_source']}, "
            f"GT 缺失={alignment['missing_from_ground_truth']}"
        )

    source_manifest = build_source_runs_manifest(stage4_rows)
    ground_truth_manifest = build_ground_truth_manifest(gt_rows)
    report = {
        "generated_at": None,
        "repo_root": str(repo_root),
        "inputs": {
            "stage4_csv": str(stage4_csv),
            "ground_truth_csv": str(gt_csv),
            "stage4_csv_sha256": sha256_file(stage4_csv),
            "ground_truth_csv_sha256": sha256_file(gt_csv),
        },
        "stage4": stage4_validation["summary"],
        "ground_truth": gt_validation["summary"],
        "source_ground_truth_alignment": alignment,
        "outputs": {
            "source_runs_row_count": len(source_manifest),
            "ground_truth_row_count": len(ground_truth_manifest),
        },
    }
    return source_manifest, ground_truth_manifest, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    default_stage4 = repo_root / "AAAI_experiments/stage4_ssr50_15algs_3seeds_3noise_3h/selected_runs_with_fepysr_rerun.csv"
    default_gt = repo_root / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
    default_manifest_dir = repo_root / "AAAI_experiments/stage5_metric_calculation_0831/manifests"
    default_report = repo_root / "AAAI_experiments/stage5_metric_calculation_0831/reports/source_preflight.json"
    parser = argparse.ArgumentParser(description="生成 Stage5 来源冻结 manifest")
    parser.add_argument("--stage4-csv", type=Path, default=default_stage4)
    parser.add_argument("--gt-csv", type=Path, default=default_gt)
    parser.add_argument("--manifest-dir", type=Path, default=default_manifest_dir)
    parser.add_argument("--report-json", type=Path, default=default_report)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_rows, gt_rows, report = build_source_preflight(
        stage4_csv=args.stage4_csv.resolve(),
        gt_csv=args.gt_csv.resolve(),
    )
    report["generated_at"] = "2026-08-31"
    if not args.dry_run:
        write_csv(args.manifest_dir / "source_runs.csv", source_rows)
        write_csv(args.manifest_dir / "ground_truth.csv", gt_rows)
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
