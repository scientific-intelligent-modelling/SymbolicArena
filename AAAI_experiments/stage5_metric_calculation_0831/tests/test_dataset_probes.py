from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.dataset_probes import (
    DatasetProbeError,
    build_dataset_probes,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_manifest_fixture(
    tmp_path: Path,
    *,
    dataset_name: str = "toy_dataset",
    feature_names: list[str] | None = None,
    target_name: str = "target",
    id_rows: list[list[object]] | None = None,
    ood_rows: list[list[object]] | None = None,
) -> Path:
    feature_names = feature_names or ["x0", "x1"]
    id_rows = id_rows or [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    ood_rows = ood_rows or [
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0],
    ]
    dataset_dir = tmp_path / "datasets" / dataset_name
    metadata_path = dataset_dir / "metadata.yaml"
    id_path = dataset_dir / "id_test.csv"
    ood_path = dataset_dir / "ood_test.csv"
    metadata = {
        "dataset": {
            "features": [{"name": name, "type": "continuous"} for name in feature_names],
            "target": {"name": target_name, "type": "continuous"},
        }
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    header = [*feature_names, target_name]
    _write_csv(id_path, header, id_rows)
    _write_csv(ood_path, header, ood_rows)

    manifest_path = tmp_path / "ground_truth.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "core50_index": 1,
                "dataset_name": dataset_name,
                "basename": dataset_name,
                "target_name": target_name,
                "feature_count": len(feature_names),
                "dataset_dir": str(dataset_dir),
                "metadata_yaml": str(metadata_path),
                "metadata_yaml_sha256": _sha256_file(metadata_path),
                "id_test_csv": str(id_path),
                "id_test_csv_sha256": _sha256_file(id_path),
                "ood_test_csv": str(ood_path),
                "ood_test_csv_sha256": _sha256_file(ood_path),
            }
        )
    return manifest_path


def test_build_dataset_probes_is_deterministic_and_does_not_freeze_target(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)
    output_jsonl = tmp_path / "dataset_probes.jsonl"
    output_summary = tmp_path / "dataset_probes.summary.json"
    first = build_dataset_probes(
        manifest_csv=manifest_path,
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        sample_size_per_split=2,
        dry_run=True,
    )
    second = build_dataset_probes(
        manifest_csv=manifest_path,
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        sample_size_per_split=2,
        dry_run=True,
    )
    assert first["summary"]["dry_run"] is True
    assert first["jsonl_text"] == second["jsonl_text"]
    assert first["summary"]["output_jsonl_sha256"] == second["summary"]["output_jsonl_sha256"]
    assert output_jsonl.exists() is False
    assert output_summary.exists() is False

    row = first["rows"][0]
    assert row["variables"] == ["x0", "x1"]
    assert row["point_count"] == 4
    assert {point["split"] for point in row["points"]} == {"id_test", "ood_test"}
    assert all(set(point["values"]) == {"x0", "x1"} for point in row["points"])
    assert all("target" not in point["values"] for point in row["points"])
    json.loads(first["jsonl_text"].splitlines()[0])


def test_build_dataset_probes_rejects_sha_drift(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    rows[0]["id_test_csv_sha256"] = "0" * 64
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetProbeError, match="sha256 不匹配"):
        build_dataset_probes(manifest_csv=manifest_path, dry_run=True)


def test_build_dataset_probes_rejects_header_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)
    manifest_row = next(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    id_path = Path(manifest_row["id_test_csv"])
    _write_csv(id_path, ["x1", "x0", "target"], [[1.0, 2.0, 3.0]])
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    rows[0]["id_test_csv_sha256"] = _sha256_file(id_path)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetProbeError, match="表头不匹配"):
        build_dataset_probes(manifest_csv=manifest_path, dry_run=True)


def test_real_ground_truth_manifest_builds_all_50_dataset_probes() -> None:
    result = build_dataset_probes(sample_size_per_split=1, dry_run=True)
    assert result["summary"]["row_count"] == 50
    assert len(result["rows"]) == 50
    assert result["summary"]["total_point_count"] == 100
    assert len(result["summary"]["dataset_names"]) == 50
    assert all(row["point_count"] == 2 for row in result["rows"])
