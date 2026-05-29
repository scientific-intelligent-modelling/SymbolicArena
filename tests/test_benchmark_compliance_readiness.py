from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from benchmark_control_compliance_manifest_import import load_for_test


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_ready_batch(batch_dir: Path) -> None:
    tasks = [
        {
            "task_id": f"alg{tool_idx:02d}__seed520__dataset_{dataset_idx:04d}",
            "algorithm": f"alg{tool_idx:02d}",
            "dataset_id": f"dataset_{dataset_idx:04d}",
            "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            "seed": "520",
            "timeout_in_seconds": "3600",
            "progress_snapshot_interval_seconds": "60",
        }
        for tool_idx in range(15)
        for dataset_idx in range(50)
    ]
    datasets = [
        {
            "dataset_id": f"dataset_{dataset_idx:04d}",
            "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
        }
        for dataset_idx in range(50)
    ]
    algorithms = {f"alg{tool_idx:02d}": {"env": "sim_base", "regressor": "Reg"} for tool_idx in range(15)}
    _write_csv(batch_dir / "manifest" / "tasks.csv", tasks)
    _write_csv(batch_dir / "manifest" / "datasets.csv", datasets)
    (batch_dir / "manifest" / "algorithms.json").write_text(json.dumps(algorithms), encoding="utf-8")
    _write_csv(
        batch_dir / "queues" / "ssr50_source.csv",
        [
            {
                "global_index": str(dataset_idx + 1),
                "dataset_id": f"dataset_{dataset_idx:04d}",
                "dataset_name": f"dataset_{dataset_idx:04d}",
                "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
                "dataset_rel": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            }
            for dataset_idx in range(50)
        ],
    )
    _write_csv(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        [
            {
                "global_index": str(dataset_idx + 1),
                "dataset_id": f"dataset_{dataset_idx:04d}",
                "dataset_name": f"dataset_{dataset_idx:04d}",
                "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
                "dataset_rel": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            }
            for dataset_idx in range(2)
        ],
    )
    for script in (
        "01_preflight_from_iaaccn22.sh",
        "02_smoke_dispatch_from_iaaccn22.sh",
        "03_full_dispatch_from_iaaccn22.sh",
    ):
        path = batch_dir / "deploy" / script
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\npython check/run_e1_candidate200_12alg_load_queue.py\n",
            encoding="utf-8",
        )


def test_readiness_passes_for_complete_stage1_batch(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    _write_ready_batch(batch_dir)

    summary = readiness.check_readiness(batch_dir=batch_dir)

    assert summary["ready"] is True
    assert summary["total_tasks"] == 750
    assert summary["total_algorithms"] == 15
    assert summary["total_datasets"] == 50
    assert summary["issues"] == []
    saved = json.loads((batch_dir / "readiness" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert saved["ready"] is True


def test_readiness_reports_missing_full_queue_and_scripts(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    _write_ready_batch(batch_dir)
    (batch_dir / "queues" / "ssr50_source.csv").unlink()
    (batch_dir / "deploy" / "02_smoke_dispatch_from_iaaccn22.sh").unlink()

    summary = readiness.check_readiness(batch_dir=batch_dir)

    assert summary["ready"] is False
    assert "queues/ssr50_source.csv missing" in summary["issues"]
    assert "deploy/02_smoke_dispatch_from_iaaccn22.sh missing" in summary["issues"]


def test_readiness_launcher_resolves_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "check_stage1_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_check_stage1_readiness", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    fake_root = tmp_path / "repo"
    calls: list[Path] = []

    def fake_check_readiness(*, batch_dir: Path) -> dict[str, object]:
        calls.append(batch_dir)
        return {"ready": True, "issues": []}

    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(launcher, "check_readiness", fake_check_readiness)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_stage1_readiness.py",
            "--batch-dir",
            "benchmark-runs/compliance/latest",
        ],
    )

    assert launcher.main() == 0
    assert calls == [fake_root / "benchmark-runs" / "compliance" / "latest"]
