from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.result_freeze_adapter import (  # noqa: E402
    ResultFreezeAdapterError,
    build_result_freeze_bundle,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    result_path = tmp_path / "results" / "noise001" / "DemoAlg" / "demoalg_s520_noise001_g0001" / "result.json"
    result_path.parent.mkdir(parents=True)
    payload = {
        "status": "ok",
        "equation": "x0 + 1",
        "feature_names": ["x0"],
        "target_name": "y",
        "train_label_noise": {"enabled": True, "requested": True, "sigma": 0.01},
        "id_test": {"nmse": 0.125},
        "ood_test": {"nmse": 0.25},
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    source_path = tmp_path / "source_runs.csv"
    source_row = {
        "batch": "batch1",
        "noise_order": 1,
        "algorithm": "DemoAlg",
        "dataset_id": "DemoGT",
        "seed": 520,
        "noise_tag": "noise001",
        "task_id": "demoalg_s520_noise001_g0001",
        "host": "iaaccn22",
        "status": "ok",
        "seconds": "10",
        "id_nmse": "0.125",
        "ood_nmse": "0.25",
        "id_r2": "0",
        "ood_r2": "0",
        "id_acc": "1",
        "ood_acc": "1",
        "path": "/remote/result.json",
        "logical_key": "DemoAlg::DemoGT::s520::noise001",
    }
    _write_csv(source_path, [source_row])
    index_path = tmp_path / "result_index_noise001.csv"
    index_row = {
        "logical_key": source_row["logical_key"],
        "batch": "batch1",
        "noise_tag": "noise001",
        "algorithm": "DemoAlg",
        "dataset_id": "DemoGT",
        "seed": 520,
        "task_id": source_row["task_id"],
        "host": "iaaccn22",
        "remote_result_path": source_row["path"],
        "local_result_path": str(result_path),
        "local_sha256": _sha(result_path),
        "status": "ok",
        "equation_present": "True",
        "canonical_artifact_present": "False",
        "id_nmse_source": "0.125",
        "id_nmse_result": "0.125",
        "ood_nmse_source": "0.25",
        "ood_nmse_result": "0.25",
        "metrics_match": "True",
    }
    _write_csv(index_path, [index_row])
    return source_path, index_path, result_path


def test_adapter_writes_deterministic_atomic_bundle_and_reports_equation_fallback(tmp_path: Path) -> None:
    source, index, _ = _fixture(tmp_path)
    output = tmp_path / "persistent" / "noise001_freeze.jsonl.gz"
    report_path = tmp_path / "persistent" / "noise001_freeze_report.json"
    numeric_path = tmp_path / "persistent" / "noise001_numeric.csv"

    report = build_result_freeze_bundle(
        source_runs_csv=source,
        result_index_csv=index,
        condition="noise001",
        output_jsonl_gz=output,
        report_json=report_path,
        expected_count=1,
        numeric_metrics_csv=numeric_path,
    )
    first_sha = _sha(output)
    build_result_freeze_bundle(
        source_runs_csv=source,
        result_index_csv=index,
        condition="noise001",
        output_jsonl_gz=output,
        report_json=report_path,
        expected_count=1,
        numeric_metrics_csv=numeric_path,
    )

    assert _sha(output) == first_sha
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    assert row["source"]["noise_tag"] == "noise001"
    assert row["source"]["task_id"] == "demoalg_s520_noise001_g0001"
    assert hashlib.sha256(row["result"]["raw_text"].encode()).hexdigest() == row["result"]["sha256"]
    assert report["formula_source_counts"] == {"equation": 1}
    assert report["canonical_artifact_missing_count"] == 1
    assert report["unresolved_parameter_count"] == 0
    numeric = list(csv.DictReader(numeric_path.open(encoding="utf-8")))
    assert numeric[0]["noise_tag"] == "noise001"
    assert numeric[0]["result_sha256"] == row["result"]["sha256"]
    assert numeric[0]["valid_output"] == "true"
    assert json.loads(report_path.read_text())["output_sha256"] == first_sha


def test_adapter_rejects_sha_status_and_nmse_drift(tmp_path: Path) -> None:
    source, index, result = _fixture(tmp_path)
    output = tmp_path / "freeze.jsonl.gz"
    report = tmp_path / "report.json"
    result.write_text(result.read_text().replace('"status": "ok"', '"status": "error"'), encoding="utf-8")

    with pytest.raises(ResultFreezeAdapterError, match="local_sha256"):
        build_result_freeze_bundle(
            source_runs_csv=source,
            result_index_csv=index,
            condition="noise001",
            output_jsonl_gz=output,
            report_json=report,
            expected_count=1,
        )


def test_adapter_rejects_aborted_fullcpu_source(tmp_path: Path) -> None:
    source, index, _ = _fixture(tmp_path)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[0]["batch"] = "all_15alg_fullcpu_v1_formal"
    rows[0]["path"] = "/experiments/all_15alg_fullcpu_v1_formal/result.json"
    _write_csv(source, rows)

    with pytest.raises(ResultFreezeAdapterError, match="已中止"):
        build_result_freeze_bundle(
            source_runs_csv=source,
            result_index_csv=index,
            condition="noise001",
            output_jsonl_gz=tmp_path / "freeze.jsonl.gz",
            report_json=tmp_path / "report.json",
            expected_count=1,
        )


@pytest.mark.parametrize(
    ("replacement", "expected_error"),
    [
        (('"status": "ok"', '"status": "error"'), "status"),
        (('"nmse": 0.125', '"nmse": 0.5'), "NMSE"),
    ],
)
def test_adapter_rejects_status_or_nmse_drift_after_sha_refresh(
    tmp_path: Path,
    replacement: tuple[str, str],
    expected_error: str,
) -> None:
    source, index, result = _fixture(tmp_path)
    result.write_text(result.read_text().replace(*replacement, 1), encoding="utf-8")
    rows = list(csv.DictReader(index.open(encoding="utf-8")))
    rows[0]["local_sha256"] = _sha(result)
    _write_csv(index, rows)

    with pytest.raises(ResultFreezeAdapterError, match=expected_error):
        build_result_freeze_bundle(
            source_runs_csv=source,
            result_index_csv=index,
            condition="noise001",
            output_jsonl_gz=tmp_path / "freeze.jsonl.gz",
            report_json=tmp_path / "report.json",
            expected_count=1,
        )
