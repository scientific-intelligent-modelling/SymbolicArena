from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from benchmark_control_compliance_manifest_import import load_for_test


def _write_tasks(batch_dir: Path, dataset_ids: tuple[str, ...] = ("d1", "d2", "d3")) -> None:
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    rows = [
        "task_id,algorithm,dataset_id,dataset_dir,seed,timeout_in_seconds,progress_snapshot_interval_seconds"
    ]
    for dataset_id in dataset_ids:
        rows.append(
            f"alg__seed520__{dataset_id},alg,{dataset_id},/data/{dataset_id},520,3600,60"
        )
    (manifest_dir / "tasks.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_result(
    task_dir: Path,
    *,
    seconds: float,
    metric: float | None,
    artifact: str | None,
    with_progress: bool = True,
) -> None:
    task_dir.mkdir(parents=True)
    payload = {
        "status": "ok",
        "runtime_seconds": seconds,
        "valid": {"nmse": metric},
        "id_test": {"nmse": metric},
        "ood_test": {"nmse": metric},
        "canonical_artifact": {"expression": artifact} if artifact else None,
    }
    (task_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    if with_progress:
        progress_dir = task_dir / "progress"
        progress_dir.mkdir()
        (progress_dir / "minute_0001.json").write_text("{}", encoding="utf-8")
        (progress_dir / "minute_0055.json").write_text("{}", encoding="utf-8")


def test_audit_classifies_success_early_stop_and_missing_result(tmp_path: Path) -> None:
    module = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    _write_tasks(batch_dir)
    _write_result(batch_dir / "runs" / "alg" / "seed520" / "d1", seconds=3501.0, metric=0.1, artifact="x0")
    _write_result(batch_dir / "runs" / "alg" / "seed520" / "d2", seconds=120.0, metric=0.1, artifact="x0")

    summary = module.audit_batch(batch_dir=batch_dir)

    assert summary == {"total_tasks": 3, "failed": 2, "passed": 1}
    with (batch_dir / "audit" / "task_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["failure_class"] for row in rows] == ["", "early_stop", "missing_result"]
    with (batch_dir / "audit" / "failure_cases.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    assert [row["failure_class"] for row in failures] == ["early_stop", "missing_result"]
    with (batch_dir / "audit" / "early_stop_cases.csv").open(newline="", encoding="utf-8") as handle:
        early_stop_rows = list(csv.DictReader(handle))
    assert [row["dataset_id"] for row in early_stop_rows] == ["d2"]


def test_audit_classifies_progress_metric_and_artifact_failures(tmp_path: Path) -> None:
    module = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    _write_tasks(batch_dir)
    _write_result(
        batch_dir / "runs" / "alg" / "seed520" / "d1",
        seconds=3501.0,
        metric=0.1,
        artifact="x0",
        with_progress=False,
    )
    _write_result(
        batch_dir / "runs" / "alg" / "seed520" / "d2",
        seconds=3501.0,
        metric=None,
        artifact="x0",
    )
    _write_result(
        batch_dir / "runs" / "alg" / "seed520" / "d3",
        seconds=3501.0,
        metric=0.1,
        artifact=None,
    )

    module.audit_batch(batch_dir=batch_dir)

    with (batch_dir / "audit" / "task_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["failure_class"] for row in rows] == [
        "missing_progress",
        "metric_invalid",
        "artifact_invalid",
    ]
    with (batch_dir / "audit" / "budget_compliance_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows == [
        {
            "algorithm": "alg",
            "total": "3",
            "passed": "0",
            "failed": "3",
            "compliant": "false",
        }
    ]
    with (batch_dir / "audit" / "metric_failure_cases.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert [row["dataset_id"] for row in metric_rows] == ["d2"]
    assert [row["failure_class"] for row in metric_rows] == ["metric_invalid"]
    with (batch_dir / "audit" / "missing_artifact_cases.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        artifact_rows = list(csv.DictReader(handle))
    assert [row["dataset_id"] for row in artifact_rows] == ["d3"]
    assert [row["failure_class"] for row in artifact_rows] == ["artifact_invalid"]


def test_audit_treats_missing_or_null_nmse_as_metric_invalid(tmp_path: Path) -> None:
    module = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    _write_tasks(batch_dir, ("d1", "d2"))

    task_dir_1 = batch_dir / "runs" / "alg" / "seed520" / "d1"
    task_dir_1.mkdir(parents=True)
    (task_dir_1 / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "runtime_seconds": 3501.0,
                "valid": {"rmse": 0.0, "nmse": None},
                "id_test": {"rmse": 0.0, "nmse": 0.1},
                "ood_test": {"rmse": 0.0, "nmse": 0.1},
                "canonical_artifact": {"expression": "x0"},
            }
        ),
        encoding="utf-8",
    )
    progress_dir_1 = task_dir_1 / "progress"
    progress_dir_1.mkdir()
    (progress_dir_1 / "minute_0001.json").write_text("{}", encoding="utf-8")
    (progress_dir_1 / "minute_0055.json").write_text("{}", encoding="utf-8")

    task_dir_2 = batch_dir / "runs" / "alg" / "seed520" / "d2"
    task_dir_2.mkdir(parents=True)
    (task_dir_2 / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "runtime_seconds": 3501.0,
                "valid": {"rmse": 0.0, "nmse": 0.1},
                "id_test": {"rmse": 0.0},
                "ood_test": {"rmse": 0.0, "nmse": 0.1},
                "canonical_artifact": {"expression": "x0"},
            }
        ),
        encoding="utf-8",
    )
    progress_dir_2 = task_dir_2 / "progress"
    progress_dir_2.mkdir()
    (progress_dir_2 / "minute_0001.json").write_text("{}", encoding="utf-8")
    (progress_dir_2 / "minute_0055.json").write_text("{}", encoding="utf-8")

    module.audit_batch(batch_dir=batch_dir)

    with (batch_dir / "audit" / "task_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["failure_class"] for row in rows] == ["metric_invalid", "metric_invalid"]


def test_audit_reads_noise_aware_run_paths_and_fields(tmp_path: Path) -> None:
    module = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "tasks.csv").write_text(
        "\n".join(
            [
                "task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds",
                "pysr__seed520__noise001__d0,pysr,d0,sim-datasets-data/ssr50/d0,520,noise001,0.01,86400,82800,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_result(
        batch_dir / "runs" / "pysr" / "seed520" / "noise001" / "d0",
        seconds=83000.0,
        metric=0.1,
        artifact="x0",
    )

    summary = module.audit_batch(batch_dir=batch_dir)

    assert summary == {"total_tasks": 1, "failed": 0, "passed": 1}
    with (batch_dir / "audit" / "task_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["noise_tag"] == "noise001"
    assert rows[0]["noise_sigma"] == "0.01"


def test_audit_launcher_runs_and_schema_matches_failure_classes(tmp_path: Path, monkeypatch) -> None:
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "audit_batch.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_audit_batch", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    batch_dir = tmp_path / "batch"
    _write_tasks(batch_dir, ("d1",))
    _write_result(batch_dir / "runs" / "alg" / "seed520" / "d1", seconds=3501.0, metric=0.1, artifact="x0")

    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_batch.py", "--batch-dir", str(batch_dir)],
    )

    assert launcher.main() == 0
    with (batch_dir / "audit" / "failure_cases.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "audit-schemas"
        / "task_audit.schema.json"
    )
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert payload["columns"] == [
        "task_id",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "noise_sigma",
        "status",
        "runtime_seconds",
        "has_result",
        "has_progress",
        "metrics_valid",
        "artifact_valid",
        "failure_class",
        "reason",
    ]
    assert payload["failure_classes"] == [
        "early_stop",
        "missing_result",
        "missing_progress",
        "metric_invalid",
        "artifact_invalid",
        "timeout_unrecovered",
        "runtime_crash",
        "dispatch_failure",
        "unknown",
    ]
