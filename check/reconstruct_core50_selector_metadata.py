"""Rebuild auditable Core-50 selector metadata from frozen Probe-4 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "AAAI_experiments/stage3_664dats_4probes_3seeds_1h/"
    "probe4_postprocess_dataset_level.csv"
)
DEFAULT_OUTPUT = ROOT / (
    "AAAI_experiments/stage3_664dats_4probes_3seeds_1h/"
    "historical_selector_replay_20260918"
)
SOURCE_FIELDS = (
    "dataset_id", "dataset_rel", "family", "subgroup", "basename",
    "semantic_duplicate_group", "feature_count", "train_samples",
    "valid_samples", "id_test_samples", "ood_test_samples",
    "formula_char_count", "formula_operator_count", "operator_group",
    "eligible_class", "metadata_class",
)
PROVENANCE_FIELDS = (
    "ood_type", "dummy_variable_flag", "ood_range_layout",
    "ood_type_source", "dummy_variable_source", "metadata_yaml_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_numeric_pair(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return False
    return all(math.isfinite(item) for item in value)


def _is_interval(value: Any) -> bool:
    return _is_numeric_pair(value) and value[0] <= value[1]


def ood_range_layout(features: list[dict[str, Any]]) -> str:
    if not features:
        raise ValueError("metadata has no feature definitions")
    layouts: set[str] = set()
    for feature in features:
        value = feature.get("ood_range")
        if _is_numeric_pair(value):
            if not _is_interval(value):
                return "invalid_bounds"
            layouts.add("single_interval")
        elif isinstance(value, list) and value and all(_is_numeric_pair(item) for item in value):
            if not all(_is_interval(item) for item in value):
                return "invalid_bounds"
            layouts.add("multiple_intervals")
        else:
            raise ValueError(f"unsupported ood_range: {value!r}")
    if len(layouts) == 1:
        return next(iter(layouts))
    return "mixed_intervals"


def declared_dummy_flag(
    family: str, metadata_class: str, features: list[dict[str, Any]]
) -> tuple[int, str]:
    descriptions = [str(feature.get("description") or "").casefold() for feature in features]
    marked_features = any(
        any(marker in text for marker in ("meaningless", "dummy", "distractor"))
        for text in descriptions
    )
    family = family.casefold()
    metadata_class = metadata_class.casefold()
    if family == "srsd":
        if "non-dummy" in metadata_class:
            flag = 0
        elif "dummy" in metadata_class:
            flag = 1
        else:
            raise ValueError(f"SRSD task lacks an explicit dummy class: {metadata_class}")
        if bool(flag) != marked_features:
            raise ValueError(f"SRSD dummy class conflicts with feature descriptions: {metadata_class}")
        return flag, "srsd_class_confirmed_by_feature_descriptions"
    if marked_features:
        raise ValueError(f"non-SRSD task declares a dummy feature: {family}")
    return 0, "no_declared_dummy_feature_outside_srsd"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(SOURCE_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"dataset-level table lacks fields: {sorted(missing)}")
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(source: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = _read_rows(source)
    keys = [row["dataset_id"] for row in source_rows]
    paths = [row["dataset_rel"] for row in source_rows]
    if len(source_rows) != 664 or len(set(keys)) != 664 or len(set(paths)) != 664:
        raise ValueError("expected 664 unique dataset IDs and paths")
    evidence_rows: list[dict[str, str]] = []
    proxy_rows: list[dict[str, str]] = []
    range_counts: Counter[str] = Counter()
    dummy_counts: Counter[str] = Counter()
    unresolved_ood_ids: list[str] = []
    for source_row in source_rows:
        dataset_path = (ROOT / source_row["dataset_rel"]).resolve()
        if not dataset_path.is_relative_to(ROOT):
            raise ValueError(f"dataset path escapes the repository: {dataset_path}")
        metadata_path = dataset_path / "metadata.yaml"
        payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        dataset = payload.get("dataset", payload)
        features = dataset.get("features")
        if not isinstance(features, list) or not all(isinstance(item, dict) for item in features):
            raise ValueError(f"invalid feature metadata: {metadata_path}")
        if len(features) != int(source_row["feature_count"]):
            raise ValueError(f"feature count differs from dataset-level table: {metadata_path}")
        layout = ood_range_layout(features)
        dummy_flag, dummy_source = declared_dummy_flag(
            source_row["family"], source_row["metadata_class"], features
        )
        base = {field: source_row[field] for field in SOURCE_FIELDS}
        base.update({
            "dummy_variable_flag": str(dummy_flag),
            "ood_range_layout": layout,
            "dummy_variable_source": dummy_source,
            "metadata_yaml_sha256": sha256(metadata_path),
        })
        evidence_rows.append({
            **base,
            "ood_type": "",
            "ood_type_source": "not_recorded_in_source_metadata",
        })
        proxy_rows.append({
            **base,
            "ood_type": f"proxy_range_layout:{layout}" if layout != "invalid_bounds" else "",
            "ood_type_source": (
                "derived_ood_range_layout_not_historical_type"
                if layout != "invalid_bounds" else "unresolved_invalid_source_bounds"
            ),
        })
        range_counts[layout] += 1
        dummy_counts[f"{source_row['family']}:{dummy_flag}"] += 1
        if layout == "invalid_bounds":
            unresolved_ood_ids.append(source_row["dataset_id"])
    fields = list(SOURCE_FIELDS + PROVENANCE_FIELDS)
    evidence_rows = [{field: row[field] for field in fields} for row in evidence_rows]
    proxy_rows = [{field: row[field] for field in fields} for row in proxy_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "metadata_evidence_664.csv"
    proxy_path = output_dir / "metadata_selector_proxy_664.csv"
    audit_path = output_dir / "metadata_reconstruction_audit.json"
    if any(path.exists() for path in (evidence_path, proxy_path, audit_path)):
        raise FileExistsError("metadata reconstruction outputs already exist")
    _write_csv(evidence_path, evidence_rows)
    _write_csv(proxy_path, proxy_rows)
    audit = {
        "schema_version": "core50_selector_metadata_reconstruction.v1",
        "task_count": len(source_rows),
        "source_dataset_level": str(source),
        "source_dataset_level_sha256": sha256(source),
        "evidence_csv": str(evidence_path),
        "evidence_csv_sha256": sha256(evidence_path),
        "proxy_csv": str(proxy_path),
        "proxy_csv_sha256": sha256(proxy_path),
        "ood_range_layout_counts": dict(range_counts),
        "unresolved_ood_dataset_ids": unresolved_ood_ids,
        "dummy_flag_counts_by_family": dict(dummy_counts),
        "historical_ood_type_recovered": False,
        "historical_dummy_flag_recovered": False,
        "dummy_flag_interpretation": "explicitly declared dummy feature; zero outside SRSD means not declared, not a proof of mathematical relevance",
        "proxy_warning": "metadata_selector_proxy_664.csv uses range layout in ood_type. This is not the historical OOD shift taxonomy and must not be called an exact selector replay.",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-level", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset_level, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
