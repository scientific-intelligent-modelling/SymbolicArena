from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.scripts.build_noise_trajectory_tasks import (
    build_noise_trajectory_tasks,
)


def _write_index(path: Path, *, noise_tag: str, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "logical_key",
        "batch",
        "noise_tag",
        "algorithm",
        "dataset_id",
        "seed",
        "task_id",
        "host",
        "remote_result_path",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "noise_tag": noise_tag})


def _row(*, algorithm: str, noise_tag: str, host: str, seed: int) -> dict[str, str]:
    task_id = f"{algorithm}_s{seed}_{noise_tag}_g0001"
    return {
        "logical_key": f"{algorithm}::d1::s{seed}::{noise_tag}",
        "batch": "formal",
        "algorithm": algorithm,
        "dataset_id": "d1",
        "seed": str(seed),
        "task_id": task_id,
        "host": host,
        "remote_result_path": f"/remote/{task_id}/result.json",
        "status": "ok",
    }


def test_builds_host_partitioned_noise_tasks(tmp_path: Path) -> None:
    noise001 = tmp_path / "noise001.csv"
    noise005 = tmp_path / "noise005.csv"
    _write_index(
        noise001,
        noise_tag="noise001",
        rows=[_row(algorithm="a", noise_tag="noise001", host="iaaccn22", seed=520)],
    )
    _write_index(
        noise005,
        noise_tag="noise005",
        rows=[_row(algorithm="a", noise_tag="noise005", host="iaaccn23", seed=520)],
    )

    output_dir = tmp_path / "tasks"
    report = build_noise_trajectory_tasks(
        index_paths={"noise001": noise001, "noise005": noise005},
        output_dir=output_dir,
        expected_hosts=("iaaccn22", "iaaccn23"),
        expected_algorithms=("a",),
        expected_rows_per_condition=1,
        expected_runs_per_algorithm_condition=1,
    )

    assert report["status"] == "ok"
    assert report["total_tasks"] == 2
    assert report["condition_counts"] == {"noise001": 1, "noise005": 1}
    assert report["host_counts"] == {"iaaccn22": 1, "iaaccn23": 1}
    rows = [
        json.loads(line)
        for path in sorted(output_dir.glob("iaaccn*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["path"] for row in rows} == {
        "/remote/a_s520_noise001_g0001/result.json",
        "/remote/a_s520_noise005_g0001/result.json",
    }
    assert all(set(row) == {
        "algorithm",
        "batch",
        "dataset_id",
        "host",
        "logical_key",
        "noise_tag",
        "path",
        "seed",
        "task_id",
    } for row in rows)


def test_rejects_duplicate_logical_keys(tmp_path: Path) -> None:
    duplicate = _row(
        algorithm="a", noise_tag="noise001", host="iaaccn22", seed=520
    )
    noise001 = tmp_path / "noise001.csv"
    noise005 = tmp_path / "noise005.csv"
    _write_index(noise001, noise_tag="noise001", rows=[duplicate, duplicate])
    _write_index(
        noise005,
        noise_tag="noise005",
        rows=[_row(algorithm="a", noise_tag="noise005", host="iaaccn23", seed=520)],
    )

    with pytest.raises(ValueError, match="logical_key.*重复"):
        build_noise_trajectory_tasks(
            index_paths={"noise001": noise001, "noise005": noise005},
            output_dir=tmp_path / "tasks",
            expected_hosts=("iaaccn22", "iaaccn23"),
            expected_algorithms=("a",),
            expected_rows_per_condition=None,
            expected_runs_per_algorithm_condition=None,
        )
