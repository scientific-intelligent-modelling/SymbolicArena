from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "analyze_neurips_rebuttal_full664.py"
TOOLS = ("fepysr", "jaxsr", "symbolfit")
STAGE3_TOOLS = ("dso", "imcts", "pyoperon", "udsr")
SEEDS = (520, 521, 522)


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_neurips_rebuttal_full664", SCRIPT)
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
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_batch(batch_dir: Path, *, datasets: int = 2) -> list[dict[str, object]]:
    dataset_rows = [
        {
            "dataset_id": f"g{index:04d}",
            "global_index": index,
            "dataset_name": f"d{index}",
            "family": "family",
            "subgroup": "group",
            "dataset_dir": f"sim-datasets-data/family/d{index}",
            "dataset_rel": f"sim-datasets-data/family/d{index}",
            "basename": f"d{index}",
            "formula_py": f"sim-datasets-data/family/d{index}/formula.py",
        }
        for index in range(1, datasets + 1)
    ]
    task_rows: list[dict[str, object]] = []
    for tool in TOOLS:
        for seed in SEEDS:
            for dataset in dataset_rows:
                task_rows.append(
                    {
                        "task_id": f"{tool}__seed{seed}__clean__{dataset['dataset_id']}",
                        "algorithm": tool,
                        "dataset_id": dataset["dataset_id"],
                        "global_index": dataset["global_index"],
                        "dataset_name": dataset["dataset_name"],
                        "dataset_dir": dataset["dataset_dir"],
                        "dataset_rel": dataset["dataset_rel"],
                        "seed": seed,
                        "noise_tag": "clean",
                        "noise_sigma": 0,
                        "params_name": f"{tool}__clean",
                        "timeout_in_seconds": 3600,
                        "min_runtime_seconds": 3300,
                        "progress_snapshot_interval_seconds": 60,
                    }
                )
    _write_csv(batch_dir / "manifest" / "datasets.csv", dataset_rows)
    _write_csv(batch_dir / "manifest" / "tasks.csv", task_rows)
    return task_rows


def _write_result(
    batch_dir: Path,
    task: dict[str, object],
    *,
    id_nmse: float = 1e-3,
    ood_nmse: float = 1e-2,
    artifact_valid: bool = True,
) -> Path:
    result_path = (
        batch_dir
        / "runs"
        / str(task["algorithm"])
        / f"seed{task['seed']}"
        / "clean"
        / str(task["dataset_id"])
        / "result.json"
    )
    payload = {
        "tool": task["algorithm"],
        "task_global_index": task["global_index"],
        "dataset": task["dataset_name"],
        "dataset_dir": f"/home/zhangziwen/{task['dataset_rel']}",
        "dataset_identity_check": {"status": "match", "match": True},
        "status": "ok",
        "seed": task["seed"],
        "seconds": 3601.5,
        "runtime_seconds": 3601.5,
        "equation": "x0 + 1",
        "canonical_artifact": {
            "artifact_valid": artifact_valid,
            "ast_node_count": 3,
            "tree_depth": 2,
            "normalized_expression": "x0 + 1",
        },
        "id_test": {"nmse": id_nmse},
        "ood_test": {"nmse": ood_nmse},
        "experiment_dir": "/tmp/experiment",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _write_stage3(path: Path, *, datasets: int = 2, drop_last: bool = False) -> None:
    rows: list[dict[str, object]] = []
    for tool in STAGE3_TOOLS:
        for index in range(1, datasets + 1):
            for seed in SEEDS:
                rows.append(
                    {
                        "dataset_id": f"g{index:04d}",
                        "global_index": index,
                        "dataset_name": f"d{index}",
                        "family": "family",
                        "subgroup": "group",
                        "dataset_rel": f"sim-datasets-data/family/d{index}",
                        "method_norm": tool,
                        "seed_norm": seed,
                        "is_finished_run": "True",
                        "wrong_dataset_flag": "False",
                        "id_log_nmse_used": "-1.0",
                        "ood_log_nmse_used": "0.5",
                        "run_outcome_class": "valid_finite_result",
                        "result_valid_output_raw": "1",
                        "result_metric_complete_raw": "1",
                        "result_has_expression_raw": "1",
                        "result_seconds": "3600",
                        "result_equation": "x0",
                        "result_complexity": "1",
                        "result_tree_depth": "1",
                        "synthetic_missing_row": "False",
                    }
                )
    if drop_last:
        rows.pop()
    _write_csv(path, rows)


def _write_audit_gate(batch_dir: Path, expected: int) -> None:
    path = batch_dir / "audit" / "audit_gate_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "audit_passed": True,
                "expected_total_tasks": expected,
                "task_audit_rows": expected,
                "failure_rows": 0,
                "issues": [],
            }
        ),
        encoding="utf-8",
    )


def test_new3_rows_use_core50_metrics_and_penalize_missing(tmp_path: Path) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    tasks = _make_batch(batch_dir, datasets=1)
    result_path = _write_result(batch_dir, tasks[0])

    rows, diagnostics = module.build_new3_run_rows(
        batch_dir / "manifest" / "tasks.csv",
        batch_dir / "runs",
    )

    present = rows[0]
    assert present["result_path"] == str(result_path.resolve())
    assert present["metric_complete"] is True
    assert present["valid_output"] is True
    assert present["id_log_nmse"] == pytest.approx(-3.0)
    assert present["ood_log_nmse"] == pytest.approx(-2.0)
    assert present["complexity"] == 3
    assert present["tree_depth"] == 2
    assert present["family"] == "family"
    assert present["subgroup"] == "group"

    missing = rows[1]
    assert missing["result_present"] is False
    assert missing["metric_complete"] is False
    assert missing["valid_output"] is False
    assert missing["id_log_nmse"] is None
    assert missing["id_log_nmse_penalized"] == 12.0
    assert missing["ood_log_nmse_penalized"] == 12.0
    assert diagnostics["expected_tasks"] == 9
    assert diagnostics["present_results"] == 1
    assert diagnostics["missing_results"] == 8


def test_incomplete_batch_cannot_be_published_as_final(tmp_path: Path) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _make_batch(batch_dir, datasets=1)

    with pytest.raises(RuntimeError, match="不完整"):
        module.analyze_batch(
            batch_dir=batch_dir,
            stage3_run_level=None,
            output_dir=batch_dir / "analysis",
            allow_incomplete=False,
        )

    summary = module.analyze_batch(
        batch_dir=batch_dir,
        stage3_run_level=None,
        output_dir=batch_dir / "analysis",
        allow_incomplete=True,
    )
    assert summary["final_ready"] is False
    assert summary["new3"]["missing_results"] == 9
    assert (batch_dir / "analysis" / "new3_run_level.csv").is_file()


def test_full_merge_requires_exact_stage3_keyspace_and_writes_seven_algorithms(
    tmp_path: Path,
) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    tasks = _make_batch(batch_dir)
    for task in tasks:
        _write_result(batch_dir, task)
    _write_audit_gate(batch_dir, len(tasks))
    stage3_path = tmp_path / "stage3.csv"
    _write_stage3(stage3_path)

    summary = module.analyze_batch(
        batch_dir=batch_dir,
        stage3_run_level=stage3_path,
        output_dir=batch_dir / "analysis",
        allow_incomplete=False,
    )

    assert summary["final_ready"] is True
    assert summary["new3"]["expected_tasks"] == 18
    assert summary["stage3"]["rows"] == 24
    run_rows = _read_csv(batch_dir / "analysis" / "full664_7alg_run_level.csv")
    leaderboard = _read_csv(batch_dir / "analysis" / "full664_7alg_leaderboard.csv")
    assert len(run_rows) == 42
    assert {row["algorithm"] for row in run_rows} == set(TOOLS + STAGE3_TOOLS)
    assert len(leaderboard) == 7
    assert [int(row["Rank"]) for row in leaderboard] == list(range(1, 8))


def test_stage3_keyspace_mismatch_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    tasks = _make_batch(batch_dir)
    for task in tasks:
        _write_result(batch_dir, task)
    _write_audit_gate(batch_dir, len(tasks))
    stage3_path = tmp_path / "stage3.csv"
    _write_stage3(stage3_path, drop_last=True)

    with pytest.raises(ValueError, match="Stage3 键空间"):
        module.analyze_batch(
            batch_dir=batch_dir,
            stage3_run_level=stage3_path,
            output_dir=batch_dir / "analysis",
            allow_incomplete=False,
        )
