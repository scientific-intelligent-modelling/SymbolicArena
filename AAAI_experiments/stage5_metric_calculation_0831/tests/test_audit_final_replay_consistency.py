from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_final_replay_consistency import (
    FinalReplayAuditError,
    audit_final_replay_consistency,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        {
            "logical_key": "demo::case::s520::clean",
            "algorithm": "demo",
            "dataset_id": "case",
            "seed": "520",
            "noise_tag": "clean",
            "result_sha256": "",
            "evaluation_status": "valid",
            "valid_output": "true",
            "native_id_nmse": "1",
            "native_ood_nmse": "2",
            "id_nmse": "1",
            "ood_nmse": "2",
            "id_quality_delta_from_native": "0",
            "ood_quality_delta_from_native": "0",
            "replay_error": "",
            "invalid_reason": "",
        },
        {
            "logical_key": "demo::case::s521::clean",
            "algorithm": "demo",
            "dataset_id": "case",
            "seed": "521",
            "noise_tag": "clean",
            "result_sha256": "",
            "evaluation_status": "valid",
            "valid_output": "true",
            "native_id_nmse": "100",
            "native_ood_nmse": "100",
            "id_nmse": "1",
            "ood_nmse": "1",
            "id_quality_delta_from_native": "0.142857142857",
            "ood_quality_delta_from_native": "0.142857142857",
            "replay_error": "",
            "invalid_reason": "",
        },
    ]
    records = []
    for index, row in enumerate(rows):
        payload = {
            "status": "ok",
            "recovered_from_timeout": index == 1,
            "timeout_type": "budget_exhausted_with_output" if index == 1 else "not_timeout",
            "termination_reason": "budget_exhausted_with_output" if index == 1 else "completed",
        }
        raw_text = json.dumps(payload, sort_keys=True)
        result_sha = hashlib.sha256(raw_text.encode()).hexdigest()
        row["result_sha256"] = result_sha
        records.append(
            {
                "source": {
                    "algorithm": "demo",
                    "dataset_id": "case",
                    "seed": 520 + index,
                    "noise_tag": "clean",
                },
                "result": {"sha256": result_sha, "raw_text": raw_text},
            }
        )

    run_csv = tmp_path / "runs.csv"
    with run_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    freeze = tmp_path / "freeze.jsonl.gz"
    with gzip.open(freeze, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return run_csv, freeze


def test_audit_separates_native_from_timeout_recovery(tmp_path: Path) -> None:
    run_csv, freeze = _write_fixture(tmp_path)
    output_csv = tmp_path / "audit.csv"
    report = audit_final_replay_consistency(
        run_csv=run_csv,
        freeze_paths=[freeze],
        condition="clean",
        output_csv=output_csv,
        output_report=tmp_path / "audit.json",
        expected_runs=2,
        expected_algorithm_count=1,
        expected_dataset_count=1,
        expected_seeds=(520, 521),
    )

    assert report["contract_ok"] is True
    assert report["summary"] == {
        "runs": 2,
        "algorithms": 1,
        "native_predict_runs": 1,
        "canonical_recovery_runs": 1,
        "native_formal_failures": 0,
        "native_formal_unavailable": 0,
        "replay_unavailable": 0,
        "native_formal_ready": True,
    }
    algorithm = report["algorithms"][0]
    assert algorithm["native_strict_pass"] == 1
    assert algorithm["native_formal_pass"] == 1
    assert algorithm["recovery_strict_fail"] == 1
    assert algorithm["recovery_formal_fail"] == 1
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8", newline="")))
    assert rows[0]["reference_source"] == "native_predict"
    assert rows[0]["native_fidelity_applicable"] == "true"
    assert rows[1]["reference_source"] == "canonical_recovery_timeout"
    assert rows[1]["native_fidelity_applicable"] == "false"


def test_audit_rejects_result_sha_drift(tmp_path: Path) -> None:
    run_csv, freeze = _write_fixture(tmp_path)
    rows = list(csv.DictReader(run_csv.open(encoding="utf-8", newline="")))
    rows[0]["result_sha256"] = "0" * 64
    with run_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(FinalReplayAuditError, match="result SHA"):
        audit_final_replay_consistency(
            run_csv=run_csv,
            freeze_paths=[freeze],
            condition="clean",
            output_csv=tmp_path / "audit.csv",
            output_report=tmp_path / "audit.json",
            expected_runs=2,
            expected_algorithm_count=1,
            expected_dataset_count=1,
            expected_seeds=(520, 521),
        )


def _rewrite_csv(path: Path, mutate) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_freeze(path: Path, mutate) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    mutate(records)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            payload = record["result"].pop("payload", None)
            if payload is not None:
                raw_text = json.dumps(payload, sort_keys=True)
                record["result"]["raw_text"] = raw_text
                record["result"]["sha256"] = hashlib.sha256(raw_text.encode()).hexdigest()
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@pytest.mark.parametrize("field", ["logical_key", "algorithm", "noise_tag", "seed"])
def test_audit_rejects_logical_key_field_drift(tmp_path: Path, field: str) -> None:
    run_csv, freeze = _write_fixture(tmp_path)

    def mutate(rows):
        rows[0][field] = {
            "logical_key": "demo::other::s520::clean",
            "algorithm": "other",
            "noise_tag": "noise001",
            "seed": "999",
        }[field]

    _rewrite_csv(run_csv, mutate)
    with pytest.raises(FinalReplayAuditError, match="logical_key|condition|seed"):
        audit_final_replay_consistency(
            run_csv=run_csv,
            freeze_paths=[freeze],
            condition="clean",
            output_csv=tmp_path / "audit.csv",
            output_report=tmp_path / "audit.json",
            expected_runs=2,
            expected_algorithm_count=1,
            expected_dataset_count=1,
            expected_seeds=(520, 521),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_status": "unknown"},
        {"valid_output": ""},
        {
            "evaluation_status": "invalid_output",
            "valid_output": "false",
            "invalid_reason": "domain error",
            "id_nmse": "1",
            "ood_nmse": "",
        },
        {
            "evaluation_status": "replay_unavailable",
            "valid_output": "",
            "id_nmse": "",
            "ood_nmse": "",
            "id_quality_delta_from_native": "",
            "ood_quality_delta_from_native": "",
            "replay_error": "",
        },
    ],
)
def test_audit_rejects_invalid_evaluation_status_field_combinations(
    tmp_path: Path, changes: dict[str, str]
) -> None:
    run_csv, freeze = _write_fixture(tmp_path)
    _rewrite_csv(run_csv, lambda rows: rows[0].update(changes))
    with pytest.raises(FinalReplayAuditError, match="evaluation_status|字段组合"):
        audit_final_replay_consistency(
            run_csv=run_csv,
            freeze_paths=[freeze],
            condition="clean",
            output_csv=tmp_path / "audit.csv",
            output_report=tmp_path / "audit.json",
            expected_runs=2,
            expected_algorithm_count=1,
            expected_dataset_count=1,
            expected_seeds=(520, 521),
        )


def test_audit_rejects_non_boolean_recovery_marker(tmp_path: Path) -> None:
    run_csv, freeze = _write_fixture(tmp_path)

    def mutate(records):
        payload = json.loads(records[0]["result"]["raw_text"])
        payload["recovered_from_timeout"] = "false"
        records[0]["result"]["payload"] = payload

    _rewrite_freeze(freeze, mutate)
    _rewrite_csv(
        run_csv,
        lambda rows: rows[0].update(result_sha256=_freeze_result_sha(freeze, 0)),
    )
    with pytest.raises(FinalReplayAuditError, match="JSON bool"):
        audit_final_replay_consistency(
            run_csv=run_csv,
            freeze_paths=[freeze],
            condition="clean",
            output_csv=tmp_path / "audit.csv",
            output_report=tmp_path / "audit.json",
            expected_runs=2,
            expected_algorithm_count=1,
            expected_dataset_count=1,
            expected_seeds=(520, 521),
        )


def _freeze_result_sha(path: Path, index: int) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.loads(list(handle)[index])["result"]["sha256"]


def test_report_is_not_ready_when_replay_is_unavailable(tmp_path: Path) -> None:
    run_csv, freeze = _write_fixture(tmp_path)

    def mutate(rows):
        rows[0].update(
            evaluation_status="replay_unavailable",
            valid_output="",
            invalid_reason="",
            replay_error="temporary replay failure",
            id_nmse="",
            ood_nmse="",
            id_quality_delta_from_native="",
            ood_quality_delta_from_native="",
        )

    _rewrite_csv(run_csv, mutate)
    report = audit_final_replay_consistency(
        run_csv=run_csv,
        freeze_paths=[freeze],
        condition="clean",
        output_csv=tmp_path / "audit.csv",
        output_report=tmp_path / "audit.json",
        expected_runs=2,
        expected_algorithm_count=1,
        expected_dataset_count=1,
        expected_seeds=(520, 521),
    )
    assert report["status"] == "error"
    assert report["contract_ok"] is False
    assert report["summary"]["replay_unavailable"] == 1
    assert report["summary"]["native_formal_ready"] is False


def test_audit_rejects_non_cartesian_grid_contract(tmp_path: Path) -> None:
    run_csv, freeze = _write_fixture(tmp_path)
    with pytest.raises(FinalReplayAuditError, match="网格"):
        audit_final_replay_consistency(
            run_csv=run_csv,
            freeze_paths=[freeze],
            condition="clean",
            output_csv=tmp_path / "audit.csv",
            output_report=tmp_path / "audit.json",
            expected_runs=2,
            expected_algorithm_count=1,
            expected_dataset_count=1,
            expected_seeds=(520, 521, 522),
        )
