from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from benchmark_control_compliance_manifest_import import load_for_test


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_stage1_batch(batch_dir: Path) -> None:
    algorithms = ["QLattice", *[f"alg{idx:02d}" for idx in range(14)]]
    datasets = [f"dataset_{idx:04d}" for idx in range(3)]
    tasks = [
        {
            "task_id": f"{algorithm}__seed520__{dataset_id}",
            "algorithm": algorithm,
            "dataset_id": dataset_id,
            "dataset_dir": f"sim-datasets-data/ssr50/{dataset_id}",
            "seed": "520",
            "timeout_in_seconds": "3600",
            "progress_snapshot_interval_seconds": "60",
        }
        for algorithm in algorithms
        for dataset_id in datasets
    ]
    _write_csv(batch_dir / "manifest" / "tasks.csv", tasks)
    _write_csv(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        [
            {
                "global_index": str(idx + 1),
                "dataset_id": dataset_id,
                "dataset_name": dataset_id,
                "dataset_dir": f"sim-datasets-data/ssr50/{dataset_id}",
                "dataset_rel": f"sim-datasets-data/ssr50/{dataset_id}",
            }
            for idx, dataset_id in enumerate(datasets[:2])
        ],
    )


def test_prepare_smoke_batch_writes_30_task_manifest(tmp_path: Path) -> None:
    smoke = load_for_test("smoke")
    batch_dir = tmp_path / "batch"
    _write_stage1_batch(batch_dir)

    summary = smoke.prepare_smoke_batch(batch_dir=batch_dir)

    assert summary == {"total_tasks": 30, "total_datasets": 2, "total_algorithms": 15}
    smoke_dir = batch_dir / "smoke"
    with (smoke_dir / "manifest" / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    assert len(tasks) == 30
    assert {row["dataset_id"] for row in tasks} == {"dataset_0000", "dataset_0001"}
    assert (smoke_dir / "queues" / "smoke_2datasets_source.csv").exists()


def test_prepare_formal24h_smoke_manifest_uses_smoke_budget(tmp_path: Path) -> None:
    smoke = load_for_test("smoke")
    batch_dir = tmp_path / "batch"
    _write_csv(
        batch_dir / "manifest" / "tasks.csv",
        [
            {
                "task_id": "gplearn__seed520__clean__dataset_0000",
                "algorithm": "gplearn",
                "dataset_id": "dataset_0000",
                "dataset_dir": "sim-datasets-data/ssr50/dataset_0000",
                "seed": "520",
                "noise_tag": "clean",
                "noise_sigma": "0",
                "timeout_in_seconds": "86400",
                "min_runtime_seconds": "82800",
                "progress_snapshot_interval_seconds": "60",
            },
            {
                "task_id": "gplearn__seed520__clean__dataset_0001",
                "algorithm": "gplearn",
                "dataset_id": "dataset_0001",
                "dataset_dir": "sim-datasets-data/ssr50/dataset_0001",
                "seed": "520",
                "noise_tag": "clean",
                "noise_sigma": "0",
                "timeout_in_seconds": "86400",
                "min_runtime_seconds": "82800",
                "progress_snapshot_interval_seconds": "60",
            },
        ],
    )
    _write_csv(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        [
            {
                "global_index": "1",
                "dataset_id": "dataset_0000",
                "dataset_name": "dataset_0000",
                "dataset_dir": "sim-datasets-data/ssr50/dataset_0000",
                "dataset_rel": "sim-datasets-data/ssr50/dataset_0000",
            },
            {
                "global_index": "2",
                "dataset_id": "dataset_0001",
                "dataset_name": "dataset_0001",
                "dataset_dir": "sim-datasets-data/ssr50/dataset_0001",
                "dataset_rel": "sim-datasets-data/ssr50/dataset_0001",
            },
        ],
    )
    (batch_dir / "params_smoke").mkdir(parents=True)
    (batch_dir / "params_smoke" / "gplearn__clean.json").write_text(
        json.dumps({"timeout_in_seconds": 600}),
        encoding="utf-8",
    )

    smoke.prepare_smoke_batch(batch_dir=batch_dir)

    with (batch_dir / "smoke" / "manifest" / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    assert {row["timeout_in_seconds"] for row in tasks} == {"600"}
    assert {row["min_runtime_seconds"] for row in tasks} == {"540"}


def test_audit_gate_passes_only_when_all_expected_tasks_pass(tmp_path: Path) -> None:
    gate = load_for_test("audit_gate")
    batch_dir = tmp_path / "batch"
    _write_csv(
        batch_dir / "manifest" / "tasks.csv",
        [
            {
                "task_id": "a",
                "algorithm": "alg",
                "dataset_id": "d1",
                "dataset_dir": "d1",
                "seed": "520",
                "timeout_in_seconds": "3600",
                "progress_snapshot_interval_seconds": "60",
            },
            {
                "task_id": "b",
                "algorithm": "alg",
                "dataset_id": "d2",
                "dataset_dir": "d2",
                "seed": "520",
                "timeout_in_seconds": "3600",
                "progress_snapshot_interval_seconds": "60",
            },
        ],
    )
    _write_csv(
        batch_dir / "audit" / "task_audit.csv",
        [
            {"task_id": "a", "failure_class": ""},
            {"task_id": "b", "failure_class": ""},
        ],
    )
    _write_csv(batch_dir / "audit" / "failure_cases.csv", [{"task_id": "", "failure_class": ""}])
    _write_csv(
        batch_dir / "audit" / "budget_compliance_summary.csv",
        [{"algorithm": "alg", "total": "2", "passed": "2", "failed": "0", "compliant": "true"}],
    )

    summary = gate.check_audit_success(batch_dir=batch_dir, expected_total_tasks=2)

    assert summary["audit_passed"] is True
    assert summary["issues"] == []
    saved = json.loads((batch_dir / "audit" / "audit_gate_summary.json").read_text(encoding="utf-8"))
    assert saved["audit_passed"] is True

    _write_csv(batch_dir / "audit" / "failure_cases.csv", [{"task_id": "b", "failure_class": "early_stop"}])
    failed = gate.check_audit_success(batch_dir=batch_dir, expected_total_tasks=2)
    assert failed["audit_passed"] is False
    assert "audit/failure_cases.csv has 1 failure rows" in failed["issues"]


def test_smoke_launchers_resolve_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    prepare_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "prepare_smoke_batch.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_prepare_smoke", prepare_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {prepare_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    fake_root = tmp_path / "repo"
    calls: list[Path] = []

    def fake_prepare_smoke_batch(*, batch_dir: Path) -> dict[str, int]:
        calls.append(batch_dir)
        return {"total_tasks": 30, "total_datasets": 2, "total_algorithms": 15}

    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(launcher, "prepare_smoke_batch", fake_prepare_smoke_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_smoke_batch.py", "--batch-dir", "benchmark-runs/compliance/latest"],
    )

    assert launcher.main() == 0
    assert calls == [fake_root / "benchmark-runs" / "compliance" / "latest"]
