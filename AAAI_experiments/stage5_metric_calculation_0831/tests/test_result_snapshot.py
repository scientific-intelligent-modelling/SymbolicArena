from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.result_snapshot import (  # noqa: E402
    ResultSnapshotError,
    finalize_snapshot,
    prepare_rsync_lists,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_row(*, task_id: str, host: str, path: str, dataset: str) -> dict[str, object]:
    return {
        "batch": "formal",
        "noise_order": "1",
        "algorithm": "DemoAlg",
        "dataset_id": dataset,
        "seed": "520",
        "noise_tag": "noise001",
        "task_id": task_id,
        "host": host,
        "status": "ok",
        "seconds": "3",
        "id_nmse": "0.125",
        "ood_nmse": "0.25",
        "id_r2": "0",
        "ood_r2": "0",
        "id_acc": "1",
        "ood_acc": "1",
        "path": path,
        "logical_key": f"DemoAlg::{dataset}::s520::noise001",
    }


def _payload() -> dict[str, object]:
    return {
        "status": "ok",
        "equation": "x0 + 1",
        "canonical_artifact": {"instantiated_expression": "x0 + 1"},
        "id_test": {"nmse": 0.125},
        "ood_test": {"nmse": 0.25},
    }


def test_prepare_and_finalize_snapshot_without_network(tmp_path: Path) -> None:
    source_csv = tmp_path / "source_runs.csv"
    rows = [
        _source_row(
            task_id="demoalg_s520_noise001_g0001",
            host="iaaccn22",
            path="/remote/a/result.json",
            dataset="D1",
        ),
        _source_row(
            task_id="demoalg_s520_noise001_g0002",
            host="iaaccn23",
            path="/remote/b/result.json",
            dataset="D2",
        ),
    ]
    _write_csv(source_csv, rows)
    lists_root = tmp_path / "rsync_lists"

    report = prepare_rsync_lists(
        source_runs_csv=source_csv,
        condition="noise001",
        output_root=lists_root,
        expected_count=2,
    )

    assert report["row_count"] == 2
    assert (lists_root / "noise001/iaaccn22.txt").read_text() == "remote/a/result.json\n"
    assert (lists_root / "noise001/iaaccn23.txt").read_text() == "remote/b/result.json\n"

    mirror_root = tmp_path / "remote_mirror"
    for row in rows:
        mirror = mirror_root / "noise001" / str(row["host"]) / str(row["path"]).lstrip("/")
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(json.dumps(_payload(), indent=2) + "\n", encoding="utf-8")
    snapshot_root = tmp_path / "snapshot"
    index_path = tmp_path / "result_index_noise001.csv"
    final = finalize_snapshot(
        source_runs_csv=source_csv,
        condition="noise001",
        mirror_root=mirror_root,
        snapshot_root=snapshot_root,
        index_csv=index_path,
        expected_count=2,
    )

    assert final["row_count"] == 2
    index_rows = list(csv.DictReader(index_path.open(encoding="utf-8")))
    assert len(index_rows) == 2
    assert all(row["metrics_match"] == "True" for row in index_rows)
    assert all(row["canonical_artifact_present"] == "True" for row in index_rows)
    for row in index_rows:
        local = Path(row["local_result_path"])
        assert local.is_file()
        assert hashlib.sha256(local.read_bytes()).hexdigest() == row["local_sha256"]


def test_prepare_rejects_unsafe_or_noncanonical_source_identity(tmp_path: Path) -> None:
    source_csv = tmp_path / "source_runs.csv"
    row = _source_row(
        task_id="demoalg_s520_noise001_g0001",
        host="iaaccn22",
        path="/remote/../escape/result.json",
        dataset="D1",
    )
    _write_csv(source_csv, [row])

    with pytest.raises(ResultSnapshotError, match="路径"):
        prepare_rsync_lists(
            source_runs_csv=source_csv,
            condition="noise001",
            output_root=tmp_path / "lists",
            expected_count=1,
        )


def test_prepare_rejects_aborted_fullcpu_source(tmp_path: Path) -> None:
    source_csv = tmp_path / "source_runs.csv"
    row = _source_row(
        task_id="demoalg_s520_noise001_g0001",
        host="iaaccn22",
        path="/experiments/all_15alg_fullcpu_v1_formal/demo/result.json",
        dataset="D1",
    )
    row["batch"] = "all_15alg_fullcpu_v1_formal"
    _write_csv(source_csv, [row])

    with pytest.raises(ResultSnapshotError, match="已中止"):
        prepare_rsync_lists(
            source_runs_csv=source_csv,
            condition="noise001",
            output_root=tmp_path / "lists",
            expected_count=1,
        )


def test_finalize_refuses_existing_drift_and_preserves_it(tmp_path: Path) -> None:
    source_csv = tmp_path / "source_runs.csv"
    row = _source_row(
        task_id="demoalg_s520_noise001_g0001",
        host="iaaccn22",
        path="/remote/a/result.json",
        dataset="D1",
    )
    _write_csv(source_csv, [row])
    mirror = tmp_path / "mirror/noise001/iaaccn22/remote/a/result.json"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(json.dumps(_payload()) + "\n", encoding="utf-8")
    destination = tmp_path / "snapshot/results/noise001/DemoAlg/demoalg_s520_noise001_g0001/result.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("user-owned-drift\n", encoding="utf-8")

    with pytest.raises(ResultSnapshotError, match="漂移"):
        finalize_snapshot(
            source_runs_csv=source_csv,
            condition="noise001",
            mirror_root=tmp_path / "mirror",
            snapshot_root=tmp_path / "snapshot",
            index_csv=tmp_path / "index.csv",
            expected_count=1,
        )
    assert destination.read_text() == "user-owned-drift\n"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("status", "error", "status"), ("id_nmse", 9.0, "NMSE")],
)
def test_finalize_rejects_result_status_or_nmse_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source_csv = tmp_path / "source_runs.csv"
    row = _source_row(
        task_id="demoalg_s520_noise001_g0001",
        host="iaaccn22",
        path="/remote/a/result.json",
        dataset="D1",
    )
    _write_csv(source_csv, [row])
    payload = _payload()
    if field == "status":
        payload["status"] = value
    else:
        payload["id_test"] = {"nmse": value}
    mirror = tmp_path / "mirror/noise001/iaaccn22/remote/a/result.json"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ResultSnapshotError, match=message):
        finalize_snapshot(
            source_runs_csv=source_csv,
            condition="noise001",
            mirror_root=tmp_path / "mirror",
            snapshot_root=tmp_path / "snapshot",
            index_csv=tmp_path / "index.csv",
            expected_count=1,
        )
