from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import prepare_noise_numeric as module
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_noise_numeric import (
    NoiseNumericPreparationError,
    prepare_noise_numeric,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path], dict[str, Path]]:
    source_path = tmp_path / "manifests/source_runs.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch",
        "noise_order",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "status",
        "seconds",
        "id_nmse",
        "ood_nmse",
        "id_r2",
        "ood_r2",
        "id_acc",
        "ood_acc",
        "path",
        "logical_key",
    ]
    source_rows: list[dict[str, object]] = []
    bundle_rows: dict[str, list[dict[str, object]]] = {"noise001": [], "noise005": []}
    algorithms = ("DemoA", "DemoB")
    seeds = (520, 521, 522)
    for condition, noise_order in (("noise001", 1), ("noise005", 2)):
        for algorithm in algorithms:
            for seed in seeds:
                task_id = f"{algorithm.lower()}_s{seed}_{condition}_g0001"
                logical_key = f"{algorithm}::DemoGT::s{seed}::{condition}"
                source = {
                    "batch": "batch1",
                    "noise_order": noise_order,
                    "algorithm": algorithm,
                    "dataset_id": "DemoGT",
                    "seed": seed,
                    "noise_tag": condition,
                    "task_id": task_id,
                    "host": "iaaccn22",
                    "status": "ok",
                    "seconds": "1",
                    "id_nmse": "0.25",
                    "ood_nmse": "4.0",
                    "id_r2": "0",
                    "ood_r2": "0",
                    "id_acc": "1",
                    "ood_acc": "1",
                    "path": f"/remote/{task_id}/result.json",
                    "logical_key": logical_key,
                }
                source_rows.append(source)
                source_public = {
                    "algorithm": algorithm,
                    "batch": "batch1",
                    "dataset_id": "DemoGT",
                    "host": "iaaccn22",
                    "noise_tag": condition,
                    "path": source["path"],
                    "seed": str(seed),
                    "source_row_sha256": hashlib.sha256(
                        _canonical_json({k: str(v) for k, v in source.items()}).encode("utf-8")
                    ).hexdigest(),
                    "task_id": task_id,
                }
                payload = {
                    "status": "ok",
                    "equation": "x0 + 1",
                    "dataset_dir": "sim-datasets-data/ssr50/datasets/demo/DemoGT",
                    "expected_dataset_rel": "sim-datasets-data/ssr50/datasets/demo/DemoGT",
                    "feature_names": ["x"],
                    "target_name": "y",
                    "id_test": {"nmse": 0.25},
                    "ood_test": {"nmse": 4.0},
                    "train_label_noise": {
                        "enabled": True,
                        "requested": True,
                        "sigma": 0.01 if condition == "noise001" else 0.05,
                    },
                }
                raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                bundle_rows[condition].append(
                    {
                        "source": source_public,
                        "result": {
                            "raw_text": raw_text,
                            "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                        },
                    }
                )
    _write_csv(source_path, source_rows)

    bundle_paths: dict[str, Path] = {}
    report_paths: dict[str, Path] = {}
    for condition, rows in bundle_rows.items():
        bundle = tmp_path / f"source_snapshot/result_freeze/{condition}_results.jsonl.gz"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(bundle, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
        bundle_paths[condition] = bundle
        report = tmp_path / f"reports/{condition}_result_freeze_adapter.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "condition": condition,
                    "output_jsonl_gz": str(bundle.resolve()),
                    "output_sha256": _sha256(bundle),
                    "source_runs_csv": str(source_path.resolve()),
                    "source_runs_sha256": _sha256(source_path),
                    "row_count": len(rows),
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        report_paths[condition] = report
    recovery_paths: dict[str, Path] = {}
    for condition in ("noise001", "noise005"):
        recovery = source_path.parent / f"{condition}_formula_recovery.v1.json"
        recovery.write_text(
            json.dumps(
                {
                    "schema_version": "formula_recovery.v1",
                    "condition": condition,
                    "entries": [],
                }
            ),
            encoding="utf-8",
        )
        recovery_paths[condition] = recovery
    return source_path, bundle_paths, report_paths, recovery_paths


def test_prepare_noise_numeric_replays_both_conditions_and_preserves_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, bundles, bundle_reports, recovery = _fixture(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_replay(payload, *, algorithm, repo_root, cache, **kwargs):
        calls.append((algorithm, str(repo_root)))
        return {
            "evaluation_path": "canonical_replay.v1",
            "dataset_dir": str(repo_root / "sim-datasets-data/ssr50/datasets/demo/DemoGT"),
            "canonical_artifact": {"instantiated_expression": "x0 + 1"},
            "canonical_artifact_sha256": "a" * 64,
            "artifact_rebuilt": algorithm == "DemoB",
            "id_test": {"nmse": 0.5},
            "ood_test": {"nmse": 8.0},
            "id_quality": 0.8,
            "ood_quality": 0.7,
            "valid_output": True,
            "native_id_nmse": float(payload["id_test"]["nmse"]),
            "native_ood_nmse": float(payload["ood_test"]["nmse"]),
            "error": None,
        }

    monkeypatch.setattr(module, "replay_payload_performance", fake_replay)
    outputs = {
        "noise001": tmp_path / "results/noise001_numeric_run_metrics.csv",
        "noise005": tmp_path / "results/noise005_numeric_run_metrics.csv",
    }
    report_path = tmp_path / "reports/noise_numeric_preparation.json"

    report = prepare_noise_numeric(
        source_runs_csv=source_path,
        noise001_bundle=bundles["noise001"],
        noise005_bundle=bundles["noise005"],
        noise001_run_csv=outputs["noise001"],
        noise005_run_csv=outputs["noise005"],
        report_json=report_path,
        noise001_bundle_report_json=bundle_reports["noise001"],
        noise005_bundle_report_json=bundle_reports["noise005"],
        noise001_formula_recovery_json=recovery["noise001"],
        noise005_formula_recovery_json=recovery["noise005"],
        expected_runs=6,
        expected_algorithms=2,
        expected_datasets=1,
        repo_root=tmp_path,
    )

    assert report["contract_ok"] is True
    assert report["status"] == "ok"
    assert report["evaluation_path"] == "canonical_replay.v1"
    assert report["inputs"]["formula_recovery_manifests"]["noise005"] == {
        "path": str(recovery["noise005"].resolve()),
        "sha256": _sha256(recovery["noise005"]),
        "condition": "noise005",
        "entry_count": 0,
    }
    assert report["counts"] == {
        "noise001": {
            "runs": 6,
            "valid_outputs": 6,
            "canonical_invalid_outputs": 0,
            "replay_unavailable": 0,
            "artifact_rebuilt": 3,
            "replay_errors": 0,
        },
        "noise005": {
            "runs": 6,
            "valid_outputs": 6,
            "canonical_invalid_outputs": 0,
            "replay_unavailable": 0,
            "artifact_rebuilt": 3,
            "replay_errors": 0,
        },
    }
    assert len(calls) == 12
    for condition, output in outputs.items():
        rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
        assert len(rows) == 6
        assert {row["noise_tag"] for row in rows} == {condition}
        assert {row["logical_key"] for row in rows} == {
            f"{algorithm}::DemoGT::s{seed}::{condition}"
            for algorithm in ("DemoA", "DemoB")
            for seed in (520, 521, 522)
        }
        assert rows[0]["evaluation_path"] == "canonical_replay.v1"
        assert rows[0]["canonical_artifact_sha256"] == "a" * 64
        assert rows[0]["artifact_rebuilt"] in {"true", "false"}
        assert float(rows[0]["native_id_nmse"]) == pytest.approx(0.25)
        assert float(rows[0]["id_nmse"]) == pytest.approx(0.5)
        assert float(rows[0]["id_quality_delta_from_native"]) != 0.0
        required = {
            "logical_key",
            "algorithm",
            "dataset_id",
            "seed",
            "noise_tag",
            "task_id",
            "host",
            "valid_output",
            "id_quality",
            "ood_quality",
        }
        assert required.issubset(rows[0])
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_prepare_noise_numeric_counts_invalid_outputs_without_replay_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, bundles, bundle_reports, recovery = _fixture(tmp_path)

    def fake_replay(payload, *, algorithm, repo_root, cache, **kwargs):
        valid = algorithm == "DemoA"
        return {
            "evaluation_path": "canonical_replay.v1",
            "canonical_artifact": {"instantiated_expression": "x0 + 1"},
            "canonical_artifact_sha256": "c" * 64,
            "artifact_rebuilt": False,
            "id_test": {"nmse": 0.5} if valid else None,
            "ood_test": {"nmse": 8.0} if valid else None,
            "id_quality": 0.8 if valid else 0.0,
            "ood_quality": 0.7 if valid else 0.0,
            "valid_output": valid,
            "invalid_reason": "canonical prediction 含 NaN/Inf" if not valid else None,
            "error": None,
        }

    monkeypatch.setattr(module, "replay_payload_performance", fake_replay)
    outputs = {
        "noise001": tmp_path / "noise001.csv",
        "noise005": tmp_path / "noise005.csv",
    }
    report = prepare_noise_numeric(
        source_runs_csv=source_path,
        noise001_bundle=bundles["noise001"],
        noise005_bundle=bundles["noise005"],
        noise001_run_csv=outputs["noise001"],
        noise005_run_csv=outputs["noise005"],
        report_json=tmp_path / "report.json",
        noise001_bundle_report_json=bundle_reports["noise001"],
        noise005_bundle_report_json=bundle_reports["noise005"],
        noise001_formula_recovery_json=recovery["noise001"],
        noise005_formula_recovery_json=recovery["noise005"],
        expected_runs=6,
        expected_algorithms=2,
        expected_datasets=1,
        repo_root=tmp_path,
    )

    assert report["contract_ok"] is True
    for condition, output in outputs.items():
        assert report["counts"][condition]["canonical_invalid_outputs"] == 3
        assert report["counts"][condition]["replay_unavailable"] == 0
        assert report["counts"][condition]["replay_errors"] == 0
        rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
        invalid = [row for row in rows if row["valid_output"] == "false"]
        assert len(invalid) == 3
        assert all(row["invalid_reason"] for row in invalid)
        assert all(row["replay_error"] == "" for row in invalid)
        assert all(float(row["id_quality"]) == 0.0 for row in invalid)
        assert all(row["evaluation_status"] == "invalid_output" for row in invalid)


def test_prepare_noise_numeric_marks_unavailable_recovery_as_replay_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, bundles, bundle_reports, recovery = _fixture(tmp_path)
    with gzip.open(bundles["noise005"], "rt", encoding="utf-8") as handle:
        target_record = json.loads(next(handle))
    target_task = target_record["source"]["task_id"]
    target_result_sha = target_record["result"]["sha256"]
    target_payload = json.loads(target_record["result"]["raw_text"])
    target_equation = target_payload["equation"]
    recovery["noise005"].write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": "noise005",
                "entries": [
                    {
                        "task_id": target_task,
                        "resolution": "unavailable",
                        "frozen_result_sha256": target_result_sha,
                        "equation_sha256": hashlib.sha256(
                            target_equation.encode("utf-8")
                        ).hexdigest(),
                        "reason": "no_matching_history_candidate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_replay(
        payload,
        *,
        algorithm,
        repo_root,
        cache,
        recovery_manifest,
        task_id,
        condition,
        result_sha256,
    ):
        if task_id == target_task:
            recovery_manifest.params_for(
                task_id=task_id,
                condition=condition,
                result_sha256=result_sha256,
                equation=payload["equation"],
            )
        return {
            "evaluation_path": "canonical_replay.v1",
            "canonical_artifact": {"instantiated_expression": "x0 + 1"},
            "canonical_artifact_sha256": "d" * 64,
            "artifact_rebuilt": False,
            "id_test": {"nmse": 0.5},
            "ood_test": {"nmse": 8.0},
            "valid_output": True,
            "invalid_reason": None,
            "error": None,
        }

    monkeypatch.setattr(module, "replay_payload_performance", fake_replay)
    output005 = tmp_path / "noise005.csv"
    report = prepare_noise_numeric(
        source_runs_csv=source_path,
        noise001_bundle=bundles["noise001"],
        noise005_bundle=bundles["noise005"],
        noise001_run_csv=tmp_path / "noise001.csv",
        noise005_run_csv=output005,
        report_json=tmp_path / "report.json",
        noise001_bundle_report_json=bundle_reports["noise001"],
        noise005_bundle_report_json=bundle_reports["noise005"],
        noise001_formula_recovery_json=recovery["noise001"],
        noise005_formula_recovery_json=recovery["noise005"],
        expected_runs=6,
        expected_algorithms=2,
        expected_datasets=1,
        repo_root=tmp_path,
    )

    assert report["contract_ok"] is False
    assert report["counts"]["noise005"]["replay_errors"] == 1
    assert report["counts"]["noise005"]["replay_unavailable"] == 1
    rows = list(csv.DictReader(output005.open(encoding="utf-8", newline="")))
    unavailable = next(row for row in rows if row["task_id"] == target_task)
    assert "unavailable" in unavailable["replay_error"]
    assert unavailable["evaluation_status"] == "replay_unavailable"
    assert unavailable["valid_output"] == ""
    assert unavailable["invalid_reason"] == ""
    for field in (
        "id_nmse",
        "ood_nmse",
        "id_quality",
        "ood_quality",
        "id_nmse_delta_from_native",
        "ood_nmse_delta_from_native",
        "id_quality_delta_from_native",
        "ood_quality_delta_from_native",
    ):
        assert unavailable[field] == ""


def test_prepare_noise_numeric_rejects_bundle_report_sha_drift(tmp_path: Path) -> None:
    source_path, bundles, bundle_reports, recovery = _fixture(tmp_path)
    payload = json.loads(bundle_reports["noise001"].read_text(encoding="utf-8"))
    payload["output_sha256"] = "0" * 64
    bundle_reports["noise001"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NoiseNumericPreparationError, match="SHA-256"):
        prepare_noise_numeric(
            source_runs_csv=source_path,
            noise001_bundle=bundles["noise001"],
            noise005_bundle=bundles["noise005"],
            noise001_run_csv=tmp_path / "noise001.csv",
            noise005_run_csv=tmp_path / "noise005.csv",
            report_json=tmp_path / "report.json",
            noise001_bundle_report_json=bundle_reports["noise001"],
            noise005_bundle_report_json=bundle_reports["noise005"],
            noise001_formula_recovery_json=recovery["noise001"],
            noise005_formula_recovery_json=recovery["noise005"],
            expected_runs=6,
            expected_algorithms=2,
            expected_datasets=1,
            repo_root=tmp_path,
        )


def test_prepare_noise_numeric_rejects_incomplete_bundle_report(tmp_path: Path) -> None:
    source_path, bundles, bundle_reports, recovery = _fixture(tmp_path)
    payload = json.loads(bundle_reports["noise001"].read_text(encoding="utf-8"))
    payload.pop("source_runs_sha256")
    bundle_reports["noise001"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NoiseNumericPreparationError, match="缺少必填字段"):
        prepare_noise_numeric(
            source_runs_csv=source_path,
            noise001_bundle=bundles["noise001"],
            noise005_bundle=bundles["noise005"],
            noise001_run_csv=tmp_path / "noise001.csv",
            noise005_run_csv=tmp_path / "noise005.csv",
            report_json=tmp_path / "report.json",
            noise001_bundle_report_json=bundle_reports["noise001"],
            noise005_bundle_report_json=bundle_reports["noise005"],
            noise001_formula_recovery_json=recovery["noise001"],
            noise005_formula_recovery_json=recovery["noise005"],
            expected_runs=6,
            expected_algorithms=2,
            expected_datasets=1,
            repo_root=tmp_path,
        )


def test_prepare_noise_numeric_can_require_both_bundle_reports(tmp_path: Path) -> None:
    source_path, bundles, _bundle_reports, recovery = _fixture(tmp_path)
    with pytest.raises(NoiseNumericPreparationError, match="必须提供冻结 report"):
        prepare_noise_numeric(
            source_runs_csv=source_path,
            noise001_bundle=bundles["noise001"],
            noise005_bundle=bundles["noise005"],
            noise001_run_csv=tmp_path / "noise001.csv",
            noise005_run_csv=tmp_path / "noise005.csv",
            report_json=tmp_path / "report.json",
            noise001_bundle_report_json=tmp_path / "missing-001.json",
            noise005_bundle_report_json=tmp_path / "missing-005.json",
            noise001_formula_recovery_json=recovery["noise001"],
            noise005_formula_recovery_json=recovery["noise005"],
            expected_runs=6,
            expected_algorithms=2,
            expected_datasets=1,
            repo_root=tmp_path,
            require_bundle_reports=True,
        )


def test_prepare_noise_numeric_rejects_source_manifest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, bundles, bundle_reports, recovery = _fixture(tmp_path)
    rows = list(csv.DictReader(source_path.open(encoding="utf-8", newline="")))
    rows[0]["host"] = "iaaccn99"
    _write_csv(source_path, rows)
    monkeypatch.setattr(module, "replay_payload_performance", lambda **_: {})

    with pytest.raises(NoiseNumericPreparationError, match="source.*一致"):
        prepare_noise_numeric(
            source_runs_csv=source_path,
            noise001_bundle=bundles["noise001"],
            noise005_bundle=bundles["noise005"],
            noise001_run_csv=tmp_path / "noise001.csv",
            noise005_run_csv=tmp_path / "noise005.csv",
            report_json=tmp_path / "report.json",
            noise001_bundle_report_json=None,
            noise005_bundle_report_json=bundle_reports["noise005"],
            noise001_formula_recovery_json=recovery["noise001"],
            noise005_formula_recovery_json=recovery["noise005"],
            expected_runs=6,
            expected_algorithms=2,
            expected_datasets=1,
            repo_root=tmp_path,
        )


def test_noise_numeric_cli_returns_nonzero_when_contract_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "prepare_noise_numeric",
        lambda **_kwargs: {
            "counts": {"noise001": {"replay_unavailable": 1}},
            "contract_ok": False,
        },
    )

    assert module.main([]) == 2
