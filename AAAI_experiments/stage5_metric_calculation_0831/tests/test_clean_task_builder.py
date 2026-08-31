from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
CANONICAL_VARIABLE_PATTERN = re.compile(r"\bx\d+\b")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import (  # noqa: E402
    PlannedTask,
    build_clean_task_plan,
    main,
    select_formula_with_source,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    evaluation_key,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (  # noqa: E402
    TaskDefinition as RunnerTaskDefinition,
)


@pytest.fixture(scope="session")
def real_all_plan() -> tuple[list[PlannedTask], dict[str, object]]:
    return build_clean_task_plan(phase="all")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _build_freeze_row(
    *,
    task_id: str,
    dataset_id: str = "DemoSet",
    algorithm: str = "DemoAlg",
    seed: int = 520,
    payload: dict[str, object],
) -> dict[str, object]:
    raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return {
        "source": {
            "algorithm": algorithm,
            "batch": "demo_batch",
            "dataset_id": dataset_id,
            "host": "iaaccn22",
            "noise_tag": "clean",
            "path": f"/tmp/{task_id}/result.json",
            "seed": str(seed),
            "source_row_sha256": "source-sha-demo",
            "task_id": task_id,
        },
        "result": {
            "raw_text": raw_text,
            "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        },
    }


def test_real_counts_and_clean_validation(real_all_plan: tuple[list[PlannedTask], dict[str, object]]) -> None:
    tasks, report = real_all_plan
    assert len(tasks) == 2300
    task_types = [task.task_type for task in tasks]
    assert task_types.count("gt_simplify") == 50
    assert task_types.count("pred_simplify") == 2250

    planning_counts = report["planning_counts"]
    assert planning_counts["gt_simplify_total"] == 50
    assert planning_counts["pred_simplify_total"] == 2250
    assert planning_counts["future_equivalence_max"] == 2250
    assert planning_counts["future_structure_max"] == 2250
    assert planning_counts["clean_total_max"] == 6800
    assert planning_counts["no_call_count"] == 0

    pred_validation = report["validation"]["pred_freeze"]
    assert pred_validation["row_count"] == 2250
    assert pred_validation["unique_clean_keys"] == 2250
    assert pred_validation["duplicate_clean_keys"] == 0
    assert pred_validation["result_raw_sha_valid_count"] == 2250
    assert pred_validation["tasks_with_variables"] == 2250
    assert pred_validation["tasks_without_variables"] == 0
    assert pred_validation["formula_source_counts"] == {
        "canonical_artifact.instantiated_expression": 2238,
        "equation": 12,
    }
    assert report["no_call_counts"] == {"gt": 0, "pred": 0}
    assert report["planning_counts"]["pred_no_call_count"] == 0
    assert report["planning_counts"]["gt_no_call_count"] == 0
    gt_logical_ids = [task.logical_id for task in tasks if task.task_type == "gt_simplify"]
    assert "gt_simplify::II.34.2_1_0" in gt_logical_ids
    assert gt_logical_ids == sorted(gt_logical_ids)


def test_real_build_is_stable(real_all_plan: tuple[list[PlannedTask], dict[str, object]]) -> None:
    first_tasks, first_report = real_all_plan
    second_tasks, second_report = build_clean_task_plan(phase="all")
    assert [task.evaluation_key for task in first_tasks] == [task.evaluation_key for task in second_tasks]
    assert [task.input_hash for task in first_tasks] == [task.input_hash for task in second_tasks]
    assert first_report["contract"] == second_report["contract"]
    first_record = json.dumps(
        first_tasks[0].to_json_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    second_record = json.dumps(
        second_tasks[0].to_json_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first_record == second_record
    assert "api_key" not in first_record
    assert "token" not in first_record.lower()


def test_formula_priority_prefers_instantiated_then_normalized_then_return_then_equation() -> None:
    payload = {
        "equation": "x0 + 3",
        "canonical_artifact": {
            "instantiated_expression": "x0 + 1",
            "normalized_expression": "x0 + 2",
            "return_expression_source": "x0 + 4",
        },
    }
    assert select_formula_with_source(payload) == (
        "x0 + 1",
        "canonical_artifact.instantiated_expression",
    )

    payload["canonical_artifact"]["instantiated_expression"] = ""
    assert select_formula_with_source(payload) == (
        "x0 + 2",
        "canonical_artifact.normalized_expression",
    )

    payload["canonical_artifact"]["normalized_expression"] = ""
    assert select_formula_with_source(payload) == (
        "x0 + 4",
        "canonical_artifact.return_expression_source",
    )

    payload["canonical_artifact"]["return_expression_source"] = ""
    assert select_formula_with_source(payload) == ("x0 + 3", "equation")


def test_missing_formula_emits_explicit_no_call_plan(tmp_path: Path) -> None:
    gt_path = tmp_path / "ground_truth_extract.jsonl"
    _write_jsonl(
        gt_path,
        [
            {
                "dataset_id": "DemoGT",
                "evidence_sha256": "gt-evidence",
                "ordered_variables": ["x0"],
                "normalized_expression_input": "x0 + 0",
                "return_source": "x0 + 0",
                "return_ast_dump": "BinOp(...)",
                "selection_reason": "demo",
                "source_checksums": {"formula_py_sha256": "abc"},
                "target": "y",
            }
        ],
    )
    freeze_path = tmp_path / "clean_freeze_demo.jsonl.gz"
    missing_payload = {
        "status": "error",
        "equation": "",
        "feature_names": ["x0"],
        "canonical_artifact": {
            "instantiated_expression": "",
            "normalized_expression": "",
            "return_expression_source": "",
            "raw_equation_kind": "prefix_expression",
            "raw_equation": "",
            "python_function_source": "",
            "variables": ["x0"],
            "ast_node_count": 0,
            "tree_depth": 0,
            "normalization_mode": "demo",
            "normalization_notes": [],
        },
        "train_label_noise": {"enabled": False, "requested": False, "sigma": 0.0},
    }
    row = _build_freeze_row(
        task_id="demoalg_s520_clean_g0001",
        algorithm="DemoAlg",
        payload=missing_payload,
    )
    with gzip.open(freeze_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    tasks, report = build_clean_task_plan(
        phase="pred",
        ground_truth_jsonl=gt_path,
        freeze_glob=str(freeze_path),
        expected_gt_count=1,
        expected_pred_count=1,
        repo_root=REPO_ROOT,
    )
    assert tasks == []
    assert report["planning_counts"]["selected_phase_task_count"] == 0
    assert report["planning_counts"]["no_call_count"] == 1
    assert report["planning_counts"]["pred_no_call_count"] == 1
    assert report["planning_counts"]["gt_no_call_count"] == 0
    assert report["no_call_counts"] == {"gt": 0, "pred": 1}
    no_call = report["no_call_records"][0]
    assert no_call["logical_id"] == "pred_simplify::demoalg::g0001::s520::clean"
    assert no_call["reason"] == "missing_final_expression"
    assert no_call["request_context"]["expression"] is None


def test_planned_task_runner_adapter_and_evaluation_key(real_all_plan: tuple[list[PlannedTask], dict[str, object]]) -> None:
    task = real_all_plan[0][0]
    assert isinstance(task, PlannedTask)
    runner_definition = task.to_runner_definition()
    assert isinstance(runner_definition, RunnerTaskDefinition)
    assert runner_definition.prompt_path == Path(task.prompt_path)
    assert runner_definition.schema_path == Path(task.schema_path)
    assert runner_definition.prompt_template == task.prompt_template
    assert dict(runner_definition.schema) == task.schema_content
    assert runner_definition.prompt_sha256 == task.prompt_sha256
    assert runner_definition.schema_sha256 == task.schema_sha256
    assert runner_definition.task_kind == "simplify"
    assert runner_definition.task_spec.evaluation_key == evaluation_key(
        task_type=task.task_type,
        logical_id=task.logical_id,
        prompt_version=task.prompt_version,
        schema_version=task.schema_version,
        prompt_sha256=task.prompt_sha256,
        schema_sha256=task.schema_sha256,
        normalized_input=task.normalized_input,
        evidence_hash=task.request["evidence_hash"],
    )


def test_real_pred_tasks_have_variables_and_frozen_formula_sources(
    real_all_plan: tuple[list[PlannedTask], dict[str, object]]
) -> None:
    tasks, report = real_all_plan
    pred_tasks = [task for task in tasks if task.task_type == "pred_simplify"]
    assert len(pred_tasks) == 2250
    allowed_sources = {
        "canonical_artifact.instantiated_expression",
        "canonical_artifact.normalized_expression",
        "canonical_artifact.return_expression_source",
        "equation",
    }
    for task in pred_tasks:
        variables = task.request["variables"]
        assert isinstance(variables, list)
        assert all(isinstance(item, str) and item for item in variables)
        source = task.request["ast_source_evidence"]["selected_expression_source"]
        assert source in allowed_sources
        expression = task.request["expression"]
        assert isinstance(expression, str) and expression.strip()
        if variables:
            continue
        assert not CANONICAL_VARIABLE_PATTERN.findall(expression)
    assert sum(report["validation"]["pred_freeze"]["formula_source_counts"].values()) == 2250


def test_cli_dry_run_reports_contract_hashes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gt_path = tmp_path / "ground_truth_extract.jsonl"
    _write_jsonl(
        gt_path,
        [
            {
                "dataset_id": "DemoGT",
                "evidence_sha256": "gt-evidence",
                "ordered_variables": ["x0"],
                "normalized_expression_input": "x0 + 0",
                "return_source": "x0 + 0",
                "return_ast_dump": "BinOp(...)",
                "selection_reason": "demo",
                "source_checksums": {"formula_py_sha256": "abc"},
                "target": "y",
            }
        ],
    )
    freeze_path = tmp_path / "clean_freeze_demo.jsonl.gz"
    payload = {
        "status": "ok",
        "equation": "x0 + 1",
        "feature_names": ["x0"],
        "canonical_artifact": {
            "instantiated_expression": "x0 + 1",
            "normalized_expression": "x0 + 1",
            "return_expression_source": "x0 + 1",
            "raw_equation_kind": "prefix_expression",
            "raw_equation": "x0 + 1",
            "python_function_source": "def equation(x0): return x0 + 1",
            "variables": ["x0"],
            "ast_node_count": 3,
            "tree_depth": 2,
            "normalization_mode": "demo",
            "normalization_notes": [],
        },
        "train_label_noise": {"enabled": False, "requested": False, "sigma": 0.0},
    }
    row = _build_freeze_row(
        task_id="demoalg_s520_clean_g0001",
        algorithm="DemoAlg",
        payload=payload,
    )
    with gzip.open(freeze_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    exit_code = main(
        [
            "--phase",
            "all",
            "--ground-truth-jsonl",
            str(gt_path),
            "--freeze-glob",
            str(freeze_path),
            "--expected-gt-count",
            "1",
            "--expected-pred-count",
            "1",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    report = json.loads(stdout)
    assert report["contract"]["prompt_sha256"]
    assert report["contract"]["schema_sha256"]
    assert report["planning_counts"]["selected_phase_task_count"] == 2
