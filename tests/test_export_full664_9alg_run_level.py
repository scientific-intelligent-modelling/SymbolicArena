from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "export_full664_9alg_run_level.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_full664_9alg_run_level", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_builds_unique_run_grid_and_core50_binding(tmp_path: Path) -> None:
    module = _load_module()
    datasets = [
        (index, f"g{index:04d}", f"dataset-{index}", f"sim-datasets-data/family/d{index}")
        for index in range(1, 4)
    ]

    full7_rows: list[dict[str, object]] = []
    raw_stage3_rows: list[dict[str, object]] = []
    for algorithm in module.THREE_SEED_ALGORITHMS:
        for global_index, dataset_id, dataset_name, dataset_dir in datasets:
            for seed in (520, 521, 522):
                is_stage3 = algorithm in module.STAGE3_ALGORITHMS
                full7_rows.append(
                    {
                        "algorithm": algorithm,
                        "dataset_id": dataset_id,
                        "global_index": global_index,
                        "dataset_name": dataset_name,
                        "dataset_rel": dataset_dir,
                        "seed": seed,
                        "status": "ok",
                        "valid_output": "True",
                        "metric_complete": "True",
                        "id_test_nmse": "" if is_stage3 else global_index / 100,
                        "ood_test_nmse": "" if is_stage3 else global_index / 10,
                        "source_group": "stage3" if is_stage3 else "new3",
                        "failure_reason": "",
                        "result_path": f"/results/{algorithm}/{dataset_id}/{seed}.json",
                    }
                )
                if is_stage3:
                    raw_stage3_rows.append(
                        {
                            "method": algorithm,
                            "dataset_id": dataset_id,
                            "seed": seed,
                            "result_id_test_nmse": global_index / 1000,
                            "result_ood_test_nmse": global_index / 100,
                            "result_status": "ok",
                            "result_valid_output": "1",
                            "result_result_path": f"/raw/{algorithm}/{dataset_id}/{seed}.json",
                        }
                    )

    probe2_rows: list[dict[str, object]] = []
    for algorithm in module.PROBE2_ALGORITHMS:
        for global_index, _dataset_id, dataset_name, dataset_dir in datasets:
            missing = algorithm == "pysr" and global_index == 3
            probe2_rows.append(
                {
                    "method": algorithm,
                    "seed": 1314,
                    "dataset_name": dataset_name,
                    "dataset_dir": dataset_dir,
                    "global_index": global_index,
                    "task_status": "timed_out" if algorithm == "pysr" else "ok",
                    "task_error": "",
                    "result_status": "timed_out" if algorithm == "pysr" else "ok",
                    "result_error": "",
                    "canonical_artifact_present": "False" if missing else "True",
                    "id_nmse": "" if missing else global_index / 10,
                    "ood_nmse": "" if missing else global_index,
                    "result_path": f"/probe2/{algorithm}/{global_index}.json",
                }
            )

    core_rows = [
        {
            "core50_index": index,
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir,
        }
        for index, (_global_index, _dataset_id, dataset_name, dataset_dir) in enumerate(
            datasets[:2], start=1
        )
    ]
    full7_path = tmp_path / "full7.csv"
    raw_stage3_path = tmp_path / "stage3_raw.csv"
    probe2_path = tmp_path / "probe2.csv"
    core_path = tmp_path / "core50.csv"
    output_dir = tmp_path / "output"
    _write_csv(full7_path, full7_rows)
    _write_csv(raw_stage3_path, raw_stage3_rows)
    _write_csv(probe2_path, probe2_rows)
    _write_csv(core_path, core_rows)

    summary = module.export_run_level(
        full7_path=full7_path,
        stage3_raw_path=raw_stage3_path,
        probe2_path=probe2_path,
        core50_path=core_path,
        output_dir=output_dir,
        expected_datasets=3,
        expected_core=2,
        repo_root=tmp_path,
    )

    rows = _read_csv(output_dir / "full664_run_level_9alg.csv")
    core_tasks = _read_csv(output_dir / "core50_task_ids.csv")
    assert len(rows) == 69
    assert len({(row["algorithm_key"], row["dataset_id"], row["seed"]) for row in rows}) == 69
    assert sum(row["is_core50"] == "True" for row in rows) == 46
    assert len(core_tasks) == 2
    assert {row["dataset_id"] for row in core_tasks} == {"g0001", "g0002"}

    dso = next(
        row
        for row in rows
        if row["algorithm_key"] == "dso"
        and row["dataset_id"] == "g0001"
        and row["seed"] == "520"
    )
    assert float(dso["id_nmse"]) == 0.001
    assert float(dso["ood_nmse"]) == 0.01
    assert dso["nmse_source"] == "stage3_raw_digest"
    assert dso["valid_output_source"] == "full664_7alg_run_level.valid_output"

    pysr_missing = next(
        row
        for row in rows
        if row["algorithm_key"] == "pysr" and row["dataset_id"] == "g0003"
    )
    assert pysr_missing["id_nmse"] == ""
    assert pysr_missing["ood_nmse"] == ""
    assert pysr_missing["valid_output"] == "False"
    assert pysr_missing["canonical_artifact_present"] == "False"
    assert pysr_missing["valid_output_source"].startswith("derived:")
    assert pysr_missing["status"] == "timed_out"

    assert summary["coverage"]["run_rows"] == 69
    assert summary["coverage"]["dataset_count"] == 3
    assert summary["coverage"]["core_dataset_count"] == 2
    written_summary = json.loads((output_dir / "full664_run_level_9alg_summary.json").read_text())
    assert written_summary["coverage"] == summary["coverage"]
