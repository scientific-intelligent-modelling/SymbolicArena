from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import yaml


MANIFEST_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831/manifests/ground_truth.csv")
DEFAULT_JSONL_RELATIVE = Path(
    "AAAI_experiments/stage5_metric_calculation_0831/reports/dataset_probes.jsonl"
)
DEFAULT_SUMMARY_RELATIVE = Path(
    "AAAI_experiments/stage5_metric_calculation_0831/reports/dataset_probes.summary.json"
)
REQUIRED_MANIFEST_FIELDS = {
    "core50_index",
    "dataset_name",
    "basename",
    "target_name",
    "feature_count",
    "dataset_dir",
    "metadata_yaml",
    "metadata_yaml_sha256",
    "id_test_csv",
    "id_test_csv_sha256",
    "ood_test_csv",
    "ood_test_csv_sha256",
}
SPLIT_ORDER = ("id_test", "ood_test")
SPLIT_FIELD_NAMES = {
    "id_test": "id_test_csv",
    "ood_test": "ood_test_csv",
}
SCHEMA_VERSION = "dataset_probes_v1"


class DatasetProbeError(ValueError):
    """数据集探针冻结契约错误。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise DatasetProbeError(f"{field_name} 不是合法整数: {value!r}") from exc
    if parsed <= 0:
        raise DatasetProbeError(f"{field_name} 必须为正整数: {value!r}")
    return parsed


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _format_real(value: float) -> float:
    return float(format(float(value), ".17g"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _require_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise DatasetProbeError(
            f"{label} sha256 不匹配，期望 {expected_sha256}，实际 {actual_sha256}"
        )
    return actual_sha256


def _load_ground_truth_manifest(
    manifest_csv: Path,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(reader.fieldnames or []))
        if missing:
            raise DatasetProbeError(f"ground_truth.csv 缺少字段: {missing}")
        for raw_row in reader:
            row = dict(raw_row)
            row["core50_index"] = _parse_positive_int(raw_row["core50_index"], field_name="core50_index")
            row["feature_count"] = _parse_positive_int(raw_row["feature_count"], field_name="feature_count")
            dataset_name = str(row["dataset_name"])
            for field_name in ("metadata_yaml", "id_test_csv", "ood_test_csv"):
                resolved = _resolve_repo_path(repo_root, raw_row[field_name])
                if not resolved.exists():
                    raise DatasetProbeError(f"{dataset_name}: {field_name} 不存在: {resolved}")
                row[f"{field_name}_abs"] = str(resolved)
                row[f"{field_name}_verified_sha256"] = _require_sha256(
                    resolved,
                    raw_row[f"{field_name}_sha256"],
                    label=f"{dataset_name}:{field_name}",
                )
            rows.append(row)
    rows.sort(key=lambda item: (item["core50_index"], item["basename"], item["dataset_name"]))
    return rows


def _load_dataset_contract(row: dict[str, Any]) -> tuple[list[str], str]:
    dataset_name = str(row["dataset_name"])
    metadata_path = Path(str(row["metadata_yaml_abs"]))
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise DatasetProbeError(f"{dataset_name}: metadata.yaml 缺少 dataset 根节点")
    features = dataset.get("features")
    if not isinstance(features, list) or not features:
        raise DatasetProbeError(f"{dataset_name}: metadata.yaml 缺少 features 列表")
    feature_names: list[str] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            raise DatasetProbeError(f"{dataset_name}: metadata.yaml 第 {index} 个 feature 不是对象")
        name = feature.get("name")
        if not isinstance(name, str) or not name:
            raise DatasetProbeError(f"{dataset_name}: metadata.yaml 第 {index} 个 feature 缺少合法 name")
        feature_names.append(name)
    if len(set(feature_names)) != len(feature_names):
        raise DatasetProbeError(f"{dataset_name}: metadata.yaml features 含重复列名")
    target = dataset.get("target")
    if not isinstance(target, dict):
        raise DatasetProbeError(f"{dataset_name}: metadata.yaml 缺少 target 节点")
    target_name = target.get("name")
    if not isinstance(target_name, str) or not target_name:
        raise DatasetProbeError(f"{dataset_name}: metadata.yaml 缺少合法 target.name")
    if target_name in feature_names:
        raise DatasetProbeError(f"{dataset_name}: target.name 不能与 feature 重名")
    if target_name != row["target_name"]:
        raise DatasetProbeError(
            f"{dataset_name}: manifest target_name={row['target_name']!r} 与 metadata target.name={target_name!r} 不一致"
        )
    if len(feature_names) != int(row["feature_count"]):
        raise DatasetProbeError(
            f"{dataset_name}: manifest feature_count={row['feature_count']} 与 metadata features={len(feature_names)} 不一致"
        )
    return feature_names, target_name


def _coerce_finite_float(
    value: str | None,
    *,
    dataset_name: str,
    split_name: str,
    row_index: int,
    column_name: str,
) -> float:
    if value is None:
        raise DatasetProbeError(
            f"{dataset_name}: {split_name} 第 {row_index} 行 {column_name} 缺失"
        )
    try:
        numeric = float(value)
    except ValueError as exc:
        raise DatasetProbeError(
            f"{dataset_name}: {split_name} 第 {row_index} 行 {column_name} 不是合法数值: {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise DatasetProbeError(
            f"{dataset_name}: {split_name} 第 {row_index} 行 {column_name} 不是有限实数: {value!r}"
        )
    return _format_real(numeric)


def _validate_header(
    header: list[str],
    *,
    dataset_name: str,
    split_name: str,
    feature_names: list[str],
    target_name: str,
) -> None:
    if len(set(header)) != len(header):
        raise DatasetProbeError(f"{dataset_name}: {split_name} CSV 表头存在重复列")
    expected_header = [*feature_names, target_name]
    if header != expected_header:
        raise DatasetProbeError(
            f"{dataset_name}: {split_name} CSV 表头不匹配，期望 {expected_header}，实际 {header}"
        )


def _deterministic_sample_rows(
    csv_path: Path,
    *,
    dataset_name: str,
    split_name: str,
    feature_names: list[str],
    target_name: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    reservoir: list[dict[str, object]] = []
    rng = random.Random(seed)
    seen_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if not header:
            raise DatasetProbeError(f"{dataset_name}: {split_name} CSV 缺少表头")
        _validate_header(
            header,
            dataset_name=dataset_name,
            split_name=split_name,
            feature_names=feature_names,
            target_name=target_name,
        )
        for row_index, raw_row in enumerate(reader):
            if None in raw_row:
                raise DatasetProbeError(
                    f"{dataset_name}: {split_name} 第 {row_index} 行列数异常"
                )
            values = {
                name: _coerce_finite_float(
                    raw_row.get(name),
                    dataset_name=dataset_name,
                    split_name=split_name,
                    row_index=row_index,
                    column_name=name,
                )
                for name in feature_names
            }
            _coerce_finite_float(
                raw_row.get(target_name),
                dataset_name=dataset_name,
                split_name=split_name,
                row_index=row_index,
                column_name=target_name,
            )
            point = {
                "split": split_name,
                "row_index": row_index,
                "values": values,
            }
            seen_rows += 1
            if len(reservoir) < sample_size:
                reservoir.append(point)
            else:
                replacement_index = rng.randrange(seen_rows)
                if replacement_index < sample_size:
                    reservoir[replacement_index] = point
    if seen_rows == 0:
        raise DatasetProbeError(f"{dataset_name}: {split_name} CSV 没有数据行")
    reservoir.sort(key=lambda item: int(item["row_index"]))
    return reservoir


def _dataset_seed(schema_version: str, dataset_name: str, core50_index: int) -> int:
    seed_material = f"{schema_version}::{core50_index}::{dataset_name}"
    return int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)


def build_dataset_probe_record(
    row: dict[str, Any],
    *,
    schema_version: str = SCHEMA_VERSION,
    sample_size_per_split: int = 8,
) -> dict[str, object]:
    if sample_size_per_split <= 0:
        raise DatasetProbeError("sample_size_per_split 必须为正整数")
    feature_names, target_name = _load_dataset_contract(row)
    dataset_name = str(row["dataset_name"])
    dataset_seed = _dataset_seed(schema_version, dataset_name, int(row["core50_index"]))
    split_points: list[dict[str, object]] = []
    for split_offset, split_name in enumerate(SPLIT_ORDER):
        field_name = SPLIT_FIELD_NAMES[split_name]
        split_seed = _dataset_seed(schema_version, f"{dataset_name}::{split_name}", dataset_seed + split_offset)
        split_points.extend(
            _deterministic_sample_rows(
                Path(str(row[f"{field_name}_abs"])),
                dataset_name=dataset_name,
                split_name=split_name,
                feature_names=feature_names,
                target_name=target_name,
                sample_size=sample_size_per_split,
                seed=split_seed,
            )
        )
    payload = {
        "schema_version": schema_version,
        "core50_index": int(row["core50_index"]),
        "dataset_name": dataset_name,
        "basename": row["basename"],
        "dataset_dir": row["dataset_dir"],
        "target_name": target_name,
        "variables": list(feature_names),
        "point_count": len(split_points),
        "points": split_points,
        "source_sha256": {
            "metadata_yaml": row["metadata_yaml_verified_sha256"],
            "id_test_csv": row["id_test_csv_verified_sha256"],
            "ood_test_csv": row["ood_test_csv_verified_sha256"],
        },
    }
    payload["sample_sha256"] = _sha256_text(
        _canonical_json(
            {
                "schema_version": schema_version,
                "dataset_name": dataset_name,
                "variables": payload["variables"],
                "points": payload["points"],
            }
        )
    )
    payload["evidence_sha256"] = _sha256_text(
        _canonical_json({key: value for key, value in payload.items() if key != "evidence_sha256"})
    )
    return payload


def build_dataset_probes(
    *,
    manifest_csv: Path | None = None,
    output_jsonl: Path | None = None,
    output_summary: Path | None = None,
    schema_version: str = SCHEMA_VERSION,
    sample_size_per_split: int = 8,
    dry_run: bool = False,
) -> dict[str, object]:
    repo_root = _repo_root()
    manifest_path = manifest_csv or (repo_root / MANIFEST_RELATIVE)
    output_jsonl_path = output_jsonl or (repo_root / DEFAULT_JSONL_RELATIVE)
    output_summary_path = output_summary or (repo_root / DEFAULT_SUMMARY_RELATIVE)
    rows = _load_ground_truth_manifest(manifest_path, repo_root=repo_root)
    records = [
        build_dataset_probe_record(
            row,
            schema_version=schema_version,
            sample_size_per_split=sample_size_per_split,
        )
        for row in rows
    ]
    jsonl_lines = [_canonical_json(record) for record in records]
    jsonl_text = ("\n".join(jsonl_lines) + "\n") if jsonl_lines else ""
    summary = {
        "schema_version": schema_version,
        "manifest_csv": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(records),
        "dataset_names": [record["dataset_name"] for record in records],
        "total_point_count": sum(int(record["point_count"]) for record in records),
        "output_jsonl": str(output_jsonl_path),
        "output_jsonl_sha256": _sha256_text(jsonl_text),
        "output_summary": str(output_summary_path),
        "dry_run": dry_run,
    }
    result = {
        "rows": records,
        "jsonl_text": jsonl_text,
        "summary": summary,
    }
    if not dry_run:
        _atomic_write_text(output_jsonl_path, jsonl_text)
        _atomic_write_json(output_summary_path, summary)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结数据集定义域探针点")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--schema-version", default=SCHEMA_VERSION)
    parser.add_argument("--sample-size-per-split", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_dataset_probes(
        manifest_csv=args.manifest_csv,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
        schema_version=args.schema_version,
        sample_size_per_split=args.sample_size_per_split,
        dry_run=args.dry_run,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
