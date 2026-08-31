"""从已冻结的 clean 最终结果计算 run 级 ID/OOD 与算法聚合分数。"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import phi_nmse
from .trajectories import canonical_expression


class NumericPreparationError(ValueError):
    """冻结来源或数值输入不满足正式指标契约。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite_nonnegative(value: object, *, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise NumericPreparationError(f"{field} 缺失或不是数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericPreparationError(f"{field} 不是合法数值: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise NumericPreparationError(f"{field} 必须是非负有限数值: {value!r}")
    return number


def _approximately_equal(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-12 + 1.0e-9 * abs(right)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericPreparationError(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NumericPreparationError(f"{path} 顶层必须是 JSON object")
    return payload


def _load_clean_manifest(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "batch",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "status",
        "id_nmse",
        "ood_nmse",
        "path",
        "logical_key",
    }
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise NumericPreparationError(f"source_runs 缺少字段: {missing}")
        for raw in reader:
            if raw["noise_tag"] != "clean":
                continue
            row = dict(raw)
            key = row["logical_key"]
            expected_key = (
                f"{row['algorithm']}::{row['dataset_id']}::s{int(row['seed'])}::clean"
            )
            if key != expected_key:
                raise NumericPreparationError(
                    f"source_runs logical_key 不一致: {key!r} != {expected_key!r}"
                )
            if key in rows:
                raise NumericPreparationError(f"source_runs 出现重复 clean 逻辑键: {key}")
            if row["status"] != "ok":
                raise NumericPreparationError(f"{key} 的 Stage4 status 不是 ok")
            _finite_nonnegative(row["id_nmse"], field=f"{key}.id_nmse")
            _finite_nonnegative(row["ood_nmse"], field=f"{key}.ood_nmse")
            rows[key] = row
    return rows


def _binding_bundle_hashes(path: Path) -> tuple[dict[str, str], int]:
    payload = _read_json_object(path)
    if payload.get("contract_ok") is not True:
        raise NumericPreparationError("freeze_binding.contract_ok 不是 true")
    counts = payload.get("freeze_counts")
    if not isinstance(counts, Mapping):
        raise NumericPreparationError("freeze_binding 缺少 freeze_counts")
    try:
        task_count = int(counts["tasks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NumericPreparationError("freeze_binding.freeze_counts.tasks 非法") from exc
    input_files = payload.get("input_files")
    records = input_files.get("freeze_records") if isinstance(input_files, Mapping) else None
    if not isinstance(records, list) or not records:
        raise NumericPreparationError("freeze_binding 缺少 freeze_records")
    hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise NumericPreparationError("freeze_records 项不是 object")
        name = Path(str(record.get("path", ""))).name
        sha256 = record.get("sha256")
        if not name or not isinstance(sha256, str) or len(sha256) != 64:
            raise NumericPreparationError("freeze_records 的 path/sha256 非法")
        if name in hashes:
            raise NumericPreparationError(f"freeze_binding 出现重复 bundle 文件名: {name}")
        hashes[name] = sha256
    return hashes, task_count


def _iter_bundle(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = gzip.open(path, "rt", encoding="utf-8")
    except OSError as exc:
        raise NumericPreparationError(f"无法打开冻结 bundle: {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NumericPreparationError(
                    f"{path}:{line_number} 不是合法 JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise NumericPreparationError(f"{path}:{line_number} 顶层不是 object")
            yield line_number, payload


def _split_nmse(payload: Mapping[str, Any], split: str, *, key: str) -> float:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        raise NumericPreparationError(f"{key}.{split} 缺失")
    return _finite_nonnegative(block.get("nmse"), field=f"{key}.{split}.nmse")


def _validate_source(
    source: Mapping[str, Any], manifest: Mapping[str, str]
) -> tuple[str, int]:
    try:
        seed = int(source.get("seed"))
    except (TypeError, ValueError) as exc:
        raise NumericPreparationError("freeze source.seed 非法") from exc
    key = f"{source.get('algorithm')}::{source.get('dataset_id')}::s{seed}::{source.get('noise_tag')}"
    checks = {
        "batch": source.get("batch"),
        "algorithm": source.get("algorithm"),
        "dataset_id": source.get("dataset_id"),
        "seed": str(seed),
        "noise_tag": source.get("noise_tag"),
        "task_id": source.get("task_id"),
        "host": source.get("host"),
        "path": source.get("path"),
    }
    for field, actual in checks.items():
        if str(actual) != str(manifest[field]):
            raise NumericPreparationError(
                f"{key} 的 freeze source.{field} 与 source_runs 不一致"
            )
    return key, seed


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_clean_numeric(
    *,
    source_runs_csv: Path,
    freeze_paths: Sequence[Path],
    freeze_binding_json: Path,
    run_csv: Path,
    algorithm_csv: Path,
    report_json: Path,
    expected_runs: int = 2250,
    expected_algorithms: int = 15,
    expected_runs_per_algorithm: int = 150,
) -> dict[str, Any]:
    """严格绑定冻结来源，逐 run 映射后再按算法平均。"""

    manifest_rows = _load_clean_manifest(source_runs_csv)
    if len(manifest_rows) != expected_runs:
        raise NumericPreparationError(
            f"clean source_runs 应为 {expected_runs} 条，实际为 {len(manifest_rows)}"
        )
    expected_hashes, binding_task_count = _binding_bundle_hashes(freeze_binding_json)
    if binding_task_count != expected_runs:
        raise NumericPreparationError(
            f"freeze_binding tasks 应为 {expected_runs}，实际为 {binding_task_count}"
        )
    paths_by_name = {path.name: path for path in freeze_paths}
    if len(paths_by_name) != len(freeze_paths):
        raise NumericPreparationError("freeze_paths 出现重复文件名")
    if set(paths_by_name) != set(expected_hashes):
        raise NumericPreparationError("freeze_paths 与 freeze_binding 记录集合不一致")
    bundle_inputs: list[dict[str, Any]] = []
    for name in sorted(paths_by_name):
        path = paths_by_name[name]
        actual_sha = _sha256_file(path)
        if actual_sha != expected_hashes[name]:
            raise NumericPreparationError(
                f"冻结 bundle SHA-256 漂移: {name}: {actual_sha} != {expected_hashes[name]}"
            )
        bundle_inputs.append(
            {"path": str(path.resolve()), "sha256": actual_sha, "size_bytes": path.stat().st_size}
        )

    run_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in sorted(paths_by_name):
        path = paths_by_name[name]
        for line_number, record in _iter_bundle(path):
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise NumericPreparationError(f"{path}:{line_number} 缺少 source")
            try:
                source_seed = int(source.get("seed"))
            except (TypeError, ValueError) as exc:
                raise NumericPreparationError(f"{path}:{line_number} source.seed 非法") from exc
            key = (
                f"{source.get('algorithm')}::{source.get('dataset_id')}::"
                f"s{source_seed}::{source.get('noise_tag')}"
            )
            manifest = manifest_rows.get(key)
            if manifest is None:
                raise NumericPreparationError(f"冻结 bundle 出现未声明逻辑键: {key}")
            if key in seen:
                raise NumericPreparationError(f"冻结 bundle 出现重复逻辑键: {key}")
            key, seed = _validate_source(source, manifest)
            result = record.get("result")
            if not isinstance(result, Mapping) or result.get("status") != "ok":
                raise NumericPreparationError(f"{key} 的冻结 result 不可用")
            raw_text = result.get("raw_text")
            expected_result_sha = result.get("sha256")
            if not isinstance(raw_text, str) or _sha256_text(raw_text) != expected_result_sha:
                raise NumericPreparationError(f"{key} 的 result.raw_text SHA-256 校验失败")
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise NumericPreparationError(f"{key} 的 result.raw_text 不是合法 JSON") from exc
            if not isinstance(payload, Mapping):
                raise NumericPreparationError(f"{key} 的最终结果顶层不是 object")

            id_nmse = _split_nmse(payload, "id_test", key=key)
            ood_nmse = _split_nmse(payload, "ood_test", key=key)
            source_id_nmse = _finite_nonnegative(manifest["id_nmse"], field=f"{key}.source_id_nmse")
            source_ood_nmse = _finite_nonnegative(
                manifest["ood_nmse"], field=f"{key}.source_ood_nmse"
            )
            if not _approximately_equal(id_nmse, source_id_nmse):
                raise NumericPreparationError(f"{key} 的 ID NMSE 与 source_runs 不一致")
            if not _approximately_equal(ood_nmse, source_ood_nmse):
                raise NumericPreparationError(f"{key} 的 OOD NMSE 与 source_runs 不一致")

            expression = canonical_expression(payload)
            valid_output = payload.get("status") == "ok" and bool(expression)
            id_quality = phi_nmse(id_nmse) if valid_output else 0.0
            ood_quality = phi_nmse(ood_nmse) if valid_output else 0.0
            run_rows.append(
                {
                    "logical_key": key,
                    "algorithm": manifest["algorithm"],
                    "dataset_id": manifest["dataset_id"],
                    "seed": seed,
                    "noise_tag": "clean",
                    "task_id": manifest["task_id"],
                    "host": manifest["host"],
                    "result_sha256": expected_result_sha,
                    "valid_output": str(valid_output).lower(),
                    "formula_source": "canonical_artifact"
                    if isinstance(payload.get("canonical_artifact"), Mapping)
                    else "equation",
                    "id_nmse": f"{id_nmse:.17g}",
                    "ood_nmse": f"{ood_nmse:.17g}",
                    "id_quality": f"{id_quality:.17g}",
                    "ood_quality": f"{ood_quality:.17g}",
                }
            )
            seen.add(key)

    missing = sorted(set(manifest_rows) - seen)
    if missing:
        raise NumericPreparationError(f"冻结 bundle 缺少 {len(missing)} 个 clean 逻辑键")
    if len(run_rows) != expected_runs:
        raise NumericPreparationError(
            f"clean run 输出应为 {expected_runs} 条，实际为 {len(run_rows)}"
        )
    run_rows.sort(key=lambda row: (str(row["algorithm"]), str(row["dataset_id"]), int(row["seed"])))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["algorithm"])].append(row)
    if len(grouped) != expected_algorithms:
        raise NumericPreparationError(
            f"算法数应为 {expected_algorithms}，实际为 {len(grouped)}"
        )
    bad_counts = {
        algorithm: len(rows)
        for algorithm, rows in grouped.items()
        if len(rows) != expected_runs_per_algorithm
    }
    if bad_counts:
        raise NumericPreparationError(f"算法 clean run 覆盖数不符: {bad_counts}")

    algorithm_rows: list[dict[str, Any]] = []
    for algorithm in sorted(grouped):
        rows = grouped[algorithm]
        id_score = 100.0 * sum(float(row["id_quality"]) for row in rows) / len(rows)
        ood_score = 100.0 * sum(float(row["ood_quality"]) for row in rows) / len(rows)
        algorithm_rows.append(
            {
                "algorithm": algorithm,
                "run_count": len(rows),
                "ID": f"{id_score:.17g}",
                "OOD": f"{ood_score:.17g}",
            }
        )

    run_fields = list(run_rows[0])
    algorithm_fields = list(algorithm_rows[0])
    _write_csv(run_csv, run_fields, run_rows)
    _write_csv(algorithm_csv, algorithm_fields, algorithm_rows)
    validity = Counter(row["valid_output"] for row in run_rows)
    report: dict[str, Any] = {
        "status": "ok",
        "contract_ok": True,
        "metric_definition": "phi-per-task-seed-then-empirical-mean",
        "counts": {
            "runs": len(run_rows),
            "algorithms": len(algorithm_rows),
            "valid_outputs": validity["true"],
            "invalid_outputs": validity["false"],
        },
        "inputs": {
            "source_runs_csv": str(source_runs_csv.resolve()),
            "source_runs_sha256": _sha256_file(source_runs_csv),
            "freeze_binding_json": str(freeze_binding_json.resolve()),
            "freeze_binding_sha256": _sha256_file(freeze_binding_json),
            "freeze_bundles": bundle_inputs,
        },
        "outputs": {
            "run_csv": str(run_csv.resolve()),
            "run_csv_sha256": _sha256_file(run_csv),
            "run_csv_row_count": len(run_rows),
            "algorithm_csv": str(algorithm_csv.resolve()),
            "algorithm_csv_sha256": _sha256_file(algorithm_csv),
            "algorithm_csv_row_count": len(algorithm_rows),
        },
        "algorithm_scores": algorithm_rows,
    }
    _atomic_write_text(
        report_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    stage_root = repo_root / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description="准备正式 clean ID/OOD run 级结果")
    parser.add_argument("--source-runs-csv", type=Path, default=stage_root / "manifests/source_runs.csv")
    parser.add_argument(
        "--freeze-glob",
        default=str(stage_root / "source_snapshot/trajectory_freeze/clean_freeze_*.jsonl.gz"),
    )
    parser.add_argument(
        "--freeze-binding-json", type=Path, default=stage_root / "reports/freeze_binding.json"
    )
    parser.add_argument(
        "--run-csv", type=Path, default=stage_root / "results/clean_numeric_run_metrics.csv"
    )
    parser.add_argument(
        "--algorithm-csv",
        type=Path,
        default=stage_root / "results/clean_numeric_algorithm_metrics.csv",
    )
    parser.add_argument(
        "--report-json", type=Path, default=stage_root / "reports/clean_numeric_preparation.json"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze_paths = [Path(path).resolve() for path in sorted(glob.glob(args.freeze_glob))]
    report = prepare_clean_numeric(
        source_runs_csv=args.source_runs_csv.resolve(),
        freeze_paths=freeze_paths,
        freeze_binding_json=args.freeze_binding_json.resolve(),
        run_csv=args.run_csv.resolve(),
        algorithm_csv=args.algorithm_csv.resolve(),
        report_json=args.report_json.resolve(),
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
