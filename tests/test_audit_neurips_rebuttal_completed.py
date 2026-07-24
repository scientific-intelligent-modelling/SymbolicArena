from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "audit_neurips_rebuttal_completed.py"
DEPLOY_SCRIPT = (
    ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "01_new3algs_full664_3seeds_clean_1h"
    / "deploy"
    / "06_audit_completed_from_iaaccn22.sh"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_neurips_rebuttal_completed",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(task_id: str, host: str = "iaaccn22") -> dict:
    return {
        "tasks": {
            task_id: {
                "task_id": task_id,
                "tool": "fepysr",
                "seed": 520,
                "state": "done",
                "assigned_host": host,
            },
            "pending": {
                "task_id": "pending",
                "tool": "fepysr",
                "seed": 520,
                "state": "pending",
                "assigned_host": host,
            },
        }
    }


def _write_result(
    experiment_root: Path,
    task_id: str,
    *,
    runtime: float = 3601,
    nested: bool = False,
) -> Path:
    base = (
        experiment_root
        / "fepysr"
        / "seed520"
        / "tasks"
        / task_id
        / "iaaccn22"
        / "fepysr"
        / "g0001_d1"
    )
    if nested:
        base = base / "experiments" / "nested"
    path = base / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "runtime_seconds": runtime,
                "equation": "x0",
                "dataset_identity_check": {"match": True},
                "id_test": {"nmse": 0.1},
                "ood_test": {"nmse": 0.2},
                "canonical_artifact": {"artifact_valid": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_audit_completed_uses_outer_result_and_checks_all_contracts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    task_id = "fepysr_s520_clean_g0001"
    root = tmp_path / "experiments"
    outer = _write_result(root, task_id)
    _write_result(root, task_id, runtime=1, nested=True)

    report = module.audit_completed(
        state=_state(task_id),
        experiment_root=root,
        host="iaaccn22",
        min_runtime=3300,
    )

    assert report["passed"] is True
    assert report["done_tasks"] == 1
    assert report["validated_results"] == 1
    assert report["runtime_min"] == 3601
    assert report["issues"] == []
    assert report["result_paths"] == [str(outer)]


def test_audit_completed_reports_budget_and_missing_result_failures(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root = tmp_path / "experiments"
    low_runtime_task = "fepysr_s520_clean_g0001"
    missing_task = "fepysr_s520_clean_g0002"
    _write_result(root, low_runtime_task, runtime=120)
    state = _state(low_runtime_task)
    state["tasks"][missing_task] = {
        "task_id": missing_task,
        "tool": "fepysr",
        "seed": 520,
        "state": "done",
        "assigned_host": "iaaccn22",
    }

    report = module.audit_completed(
        state=state,
        experiment_root=root,
        host="iaaccn22",
        min_runtime=3300,
    )

    assert report["passed"] is False
    assert report["done_tasks"] == 2
    assert report["validated_results"] == 1
    assert {issue["issue"] for issue in report["issues"]} == {
        "contract_failed",
        "result_path_count",
    }
    contract = next(
        issue
        for issue in report["issues"]
        if issue["issue"] == "contract_failed"
    )
    assert contract["failed_checks"] == ["runtime_compliant"]


def test_completed_audit_deploy_uses_one_immutable_state_snapshot() -> None:
    content = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'STATE_SNAPSHOT="$REPORT_DIR/state.snapshot.json"' in content
    assert 'cp "$STATE" "$STATE_SNAPSHOT"' in content
    assert 'state_basename="$(basename "$STATE_SNAPSHOT")"' in content
    assert '--state "$STATE_SNAPSHOT"' in content
    assert '"$STATE_SNAPSHOT" \\' in content
    assert "' \"$STATE_SNAPSHOT\"" in content
