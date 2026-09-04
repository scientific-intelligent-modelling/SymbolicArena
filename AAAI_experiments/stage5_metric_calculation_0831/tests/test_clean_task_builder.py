from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
CANONICAL_VARIABLE_PATTERN = re.compile(r"\bx\d+\b")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder as clean_task_builder  # noqa: E402

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import (  # noqa: E402
    CleanTaskBuilderError,
    PlannedTask,
    build_argument_parser,
    build_clean_task_plan,
    extract_expression_body,
    instantiate_parameters,
    load_formula_recovery_manifest,
    main,
    map_indexed_variables,
    select_formula_with_source,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    evaluation_key,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (  # noqa: E402
    TaskDefinition as RunnerTaskDefinition,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.register_symbolic_plan import (  # noqa: E402
    register_symbolic_plan,
)


@pytest.fixture(scope="session")
def real_all_plan() -> tuple[list[PlannedTask], dict[str, object]]:
    return build_clean_task_plan(phase="all")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def test_planned_task_json_record_handles_deep_request_without_mutation() -> None:
    canonical_tree: dict[str, object] = {"leaf": "x0"}
    for _ in range(1100):
        canonical_tree = {"child": canonical_tree}
    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(5000)
        json.dumps(canonical_tree)
    finally:
        sys.setrecursionlimit(previous_limit)

    request: dict[str, object] = {
        "expression": "x0",
        "ast_source_evidence": {
            "canonical_artifact": {"canonical_tree": canonical_tree},
        },
    }
    task = PlannedTask(
        evaluation_key="a" * 64,
        logical_id="pred_simplify::demo::g0001::s520::clean",
        task_type="pred_simplify",
        condition="clean",
        priority=20,
        input_hash="b" * 64,
        prompt_version="simplify.v1",
        prompt_sha256="c" * 64,
        schema_version="simplify.v1",
        schema_sha256="d" * 64,
        dependencies=(),
        prompt_path="prompt.txt",
        schema_path="schema.json",
        prompt_template="REQUEST:\n{{REQUEST_JSON}}",
        schema_content={"type": "object"},
        normalized_input={"request": "demo"},
        request=request,
    )

    record = task.to_json_record()

    assert set(record) == {
        "evaluation_key",
        "logical_id",
        "task_type",
        "condition",
        "priority",
        "input_hash",
        "prompt_version",
        "prompt_sha256",
        "schema_version",
        "schema_sha256",
        "dependencies",
        "prompt_path",
        "schema_path",
        "prompt_template",
        "schema_content",
        "normalized_input",
        "request",
        "task_spec",
        "rendered_prompt",
    }
    assert record["request"] is request
    assert task.request is request
    with pytest.raises(FrozenInstanceError):
        task.request = {}  # type: ignore[misc]
    assert record["dependencies"] == []
    assert "canonical_tree" not in record["rendered_prompt"]
    assert canonical_tree == request["ast_source_evidence"]["canonical_artifact"]["canonical_tree"]  # type: ignore[index]


def test_noise_cli_defaults_defer_condition_specific_inputs() -> None:
    args = build_argument_parser().parse_args(
        ["--phase", "pred", "--condition", "noise005"]
    )

    assert args.formula_recovery_json is None
    assert args.freeze_glob is None
    assert args.output_jsonl is None
    assert args.full_plan_jsonl is None
    assert args.build_workers == 1


def test_noise_default_freeze_glob_targets_result_freeze_bundle() -> None:
    assert clean_task_builder._default_pred_freeze_glob("noise005") == str(
        STAGE_ROOT / "source_snapshot/result_freeze/noise005_results.jsonl.gz"
    )


def _write_probe_jsonl(
    path: Path,
    *,
    dataset_id: str,
    variables: list[str],
    target_name: str,
) -> None:
    points = [
        {
            "split": "id_test",
            "row_index": 0,
            "values": {name: float(index + 1) for index, name in enumerate(variables)},
        }
    ]
    row: dict[str, object] = {
        "schema_version": "dataset_probes_v1",
        "core50_index": 1,
        "dataset_name": dataset_id,
        "basename": dataset_id,
        "dataset_dir": f"/tmp/{dataset_id}",
        "target_name": target_name,
        "variables": variables,
        "point_count": len(points),
        "points": points,
        "source_sha256": {
            "metadata_yaml": "a" * 64,
            "id_test_csv": "b" * 64,
            "ood_test_csv": "c" * 64,
        },
    }
    row["sample_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": row["schema_version"],
                "dataset_name": dataset_id,
                "variables": variables,
                "points": points,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    row["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_jsonl(path, [row])


def _build_freeze_row(
    *,
    task_id: str,
    dataset_id: str = "DemoSet",
    algorithm: str = "DemoAlg",
    seed: int = 520,
    payload: dict[str, object],
    condition: str = "clean",
) -> dict[str, object]:
    raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return {
        "source": {
            "algorithm": algorithm,
            "batch": "demo_batch",
            "dataset_id": dataset_id,
            "host": "iaaccn22",
            "noise_tag": condition,
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


def test_noise_prediction_uses_condition_identity_and_equation_fallback(tmp_path: Path) -> None:
    gt_path = tmp_path / "ground_truth_extract.jsonl"
    _write_jsonl(
        gt_path,
        [{
            "dataset_id": "DemoGT",
            "evidence_sha256": "gt-evidence",
            "ordered_variables": ["x0"],
            "normalized_expression_input": "x0",
            "return_source": "x0",
            "return_ast_dump": "Name(...) ",
            "selection_reason": "demo",
            "source_checksums": {"formula_py_sha256": "abc"},
            "target": "y",
        }],
    )
    probes_path = tmp_path / "dataset_probes.jsonl"
    _write_probe_jsonl(probes_path, dataset_id="DemoGT", variables=["x0"], target_name="y")
    freeze_path = tmp_path / "noise001_freeze.jsonl.gz"
    payload = {
        "status": "ok",
        "equation": "x0 + 1",
        "feature_names": ["x0"],
        "target_name": "y",
        "train_label_noise": {"enabled": True, "requested": True, "sigma": 0.01},
    }
    row = _build_freeze_row(
        task_id="demoalg_s520_noise001_g0001",
        dataset_id="DemoGT",
        algorithm="DemoAlg",
        payload=payload,
        condition="noise001",
    )
    with gzip.open(freeze_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    tasks, report = build_clean_task_plan(
        phase="pred",
        condition="noise001",
        ground_truth_jsonl=gt_path,
        dataset_probes_jsonl=probes_path,
        freeze_glob=str(freeze_path),
        expected_gt_count=1,
        expected_pred_count=1,
        repo_root=REPO_ROOT,
    )

    assert len(tasks) == 1
    assert tasks[0].condition == "noise001"
    assert tasks[0].logical_id == "pred_simplify::demoalg::g0001::s520::noise001"
    assert tasks[0].request["noise_tag"] == "noise001"
    assert tasks[0].request["ast_source_evidence"]["selected_expression_source"] == "equation"
    assert report["validation"]["noise_condition"] == "noise001"
    assert report["validation"]["pred_freeze"]["formula_source_counts"] == {"equation": 1}


def test_noise_prediction_with_unresolved_parameters_is_no_call(tmp_path: Path) -> None:
    gt_path = tmp_path / "ground_truth_extract.jsonl"
    _write_jsonl(
        gt_path,
        [{
            "dataset_id": "DemoGT",
            "evidence_sha256": "gt-evidence",
            "ordered_variables": ["x0"],
            "normalized_expression_input": "x0",
            "return_source": "x0",
            "return_ast_dump": "Name(...) ",
            "selection_reason": "demo",
            "source_checksums": {"formula_py_sha256": "abc"},
            "target": "y",
        }],
    )
    probes_path = tmp_path / "dataset_probes.jsonl"
    _write_probe_jsonl(probes_path, dataset_id="DemoGT", variables=["x0"], target_name="y")
    freeze_path = tmp_path / "noise005_freeze.jsonl.gz"
    payload = {
        "status": "ok",
        "equation": "x0 + params[0]",
        "feature_names": ["x0"],
        "target_name": "y",
        "train_label_noise": {"enabled": True, "requested": True, "sigma": 0.05},
    }
    row = _build_freeze_row(
        task_id="demoalg_s520_noise005_g0001",
        dataset_id="DemoGT",
        algorithm="DemoAlg",
        payload=payload,
        condition="noise005",
    )
    with gzip.open(freeze_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    tasks, report = build_clean_task_plan(
        phase="pred",
        condition="noise005",
        ground_truth_jsonl=gt_path,
        dataset_probes_jsonl=probes_path,
        freeze_glob=str(freeze_path),
        expected_gt_count=1,
        expected_pred_count=1,
        repo_root=REPO_ROOT,
    )

    assert tasks == []
    no_call = report["no_call_records"][0]
    assert no_call["reason"] == "unresolved_parameter_values"
    assert no_call["condition"] == "noise005"
    assert no_call["status"] == "planned_non_applicable"
    assert no_call["task_spec"]["task_type"] == "pred_simplify"
    assert no_call["task_spec"]["condition"] == "noise005"
    assert no_call["request"]["evidence_hash"] == no_call["evidence_sha256"]
    assert no_call["evidence_payload"]["schema_version"] == "symbolic_non_applicable.v1"
    assert no_call["evidence_payload"]["phase"] == "pred"
    assert no_call["evidence_payload"]["reason"] == "unresolved_parameter_values"
    assert report["planning_counts"]["condition_total_max"] == 2251

    evidence_dir = tmp_path / "non_applicable_evidence"
    materialized_tasks, materialized_report = build_clean_task_plan(
        phase="pred",
        condition="noise005",
        ground_truth_jsonl=gt_path,
        dataset_probes_jsonl=probes_path,
        freeze_glob=str(freeze_path),
        expected_gt_count=1,
        expected_pred_count=1,
        repo_root=REPO_ROOT,
        non_applicable_evidence_dir=evidence_dir,
        write_non_applicable_evidence=True,
        build_workers=2,
    )
    callable_plan = tmp_path / "callable.jsonl"
    non_applicable_index = tmp_path / "non_applicable.jsonl"
    _write_jsonl(callable_plan, [task.to_json_record() for task in materialized_tasks])
    _write_jsonl(non_applicable_index, materialized_report["no_call_records"])
    registration = register_symbolic_plan(
        plan_jsonl=callable_plan,
        non_applicable_index_jsonl=non_applicable_index,
        state_db=tmp_path / "state.sqlite3",
    )

    assert registration["counts"]["callable_task_count"] == 0
    assert registration["counts"]["non_applicable_task_count"] == 1
    assert registration["distributions"]["final_state"] == {"non_applicable": 1}

    cli_callable = tmp_path / "cli_callable.jsonl"
    cli_non_applicable = tmp_path / "cli_non_applicable.jsonl"
    cli_full_plan = tmp_path / "cli_full_plan.jsonl"
    cli_report = tmp_path / "cli_report.json"
    assert main(
        [
            "--phase",
            "pred",
            "--condition",
            "noise005",
            "--ground-truth-jsonl",
            str(gt_path),
            "--dataset-probes-jsonl",
            str(probes_path),
            "--freeze-glob",
            str(freeze_path),
            "--expected-gt-count",
            "1",
            "--expected-pred-count",
            "1",
            "--output-jsonl",
            str(cli_callable),
            "--non-applicable-index-jsonl",
            str(cli_non_applicable),
            "--full-plan-jsonl",
            str(cli_full_plan),
            "--non-applicable-evidence-dir",
            str(tmp_path / "cli_evidence"),
            "--report",
            str(cli_report),
        ]
    ) == 0
    assert cli_callable.read_text(encoding="utf-8") == ""
    assert len(cli_non_applicable.read_text(encoding="utf-8").splitlines()) == 1
    assert cli_full_plan.read_text(encoding="utf-8") == cli_non_applicable.read_text(
        encoding="utf-8"
    )
    assert json.loads(cli_report.read_text(encoding="utf-8"))["outputs"][
        "full_plan_usage"
    ] == "frozen_index_only_never_api_runner"


def test_noise_prediction_uses_explicit_condition_recovery_manifest(tmp_path: Path) -> None:
    gt_path = tmp_path / "ground_truth_extract.jsonl"
    _write_jsonl(
        gt_path,
        [{
            "dataset_id": "DemoGT",
            "evidence_sha256": "gt-evidence",
            "ordered_variables": ["x0"],
            "normalized_expression_input": "x0",
            "return_source": "x0",
            "return_ast_dump": "Name(...)",
            "selection_reason": "demo",
            "source_checksums": {"formula_py_sha256": "abc"},
            "target": "y",
        }],
    )
    probes_path = tmp_path / "dataset_probes.jsonl"
    _write_probe_jsonl(probes_path, dataset_id="DemoGT", variables=["x0"], target_name="y")
    equation = "def equation(col0, params):\n    return col0 + params[0]\n"
    payload = {
        "status": "ok",
        "equation": equation,
        "feature_names": ["x0"],
        "target_name": "y",
        "train_label_noise": {"enabled": True, "requested": True, "sigma": 0.01},
    }
    row = _build_freeze_row(
        task_id="demoalg_s520_noise001_g0001",
        dataset_id="DemoGT",
        algorithm="DemoAlg",
        payload=payload,
        condition="noise001",
    )
    freeze_path = tmp_path / "noise001_freeze.jsonl.gz"
    with gzip.open(freeze_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    candidate = {
        "function": "def equation(x0, params):\n    return x0 + params[0]\n",
        "params": [2.5],
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    canonical_params = json.dumps(
        candidate["params"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    recovery_path = tmp_path / "noise001_formula_recovery.v1.json"
    recovery_path.write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": "noise001",
                "entries": [{
                    "task_id": "demoalg_s520_noise001_g0001",
                    "resolution": "recovered_params",
                    "frozen_result_sha256": row["result"]["sha256"],
                    "equation_sha256": hashlib.sha256(equation.encode("utf-8")).hexdigest(),
                    "params": candidate["params"],
                    "source_evidence": {
                        "candidate_path": str(candidate_path),
                        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                        "params_sha256": hashlib.sha256(canonical_params.encode("utf-8")).hexdigest(),
                    },
                }],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tasks, report = build_clean_task_plan(
        phase="pred",
        condition="noise001",
        ground_truth_jsonl=gt_path,
        formula_recovery_json=recovery_path,
        dataset_probes_jsonl=probes_path,
        freeze_glob=str(freeze_path),
        expected_gt_count=1,
        expected_pred_count=1,
        repo_root=REPO_ROOT,
    )

    assert len(tasks) == 1
    assert tasks[0].request["expression"] == "x0 + (2.5)"
    assert tasks[0].request["ast_source_evidence"]["formula_resolution"]["status"] == "recovered_params"
    assert report["no_call_counts"]["pred"] == 0


def test_noise_condition_rejects_gt_and_wrong_training_noise(tmp_path: Path) -> None:
    with pytest.raises(CleanTaskBuilderError, match="noise 条件只允许 phase=pred"):
        build_clean_task_plan(phase="gt", condition="noise001")


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
    assert report["validation"]["dataset_probes"] == {
        "jsonl_sha256": "d939b69ab0a319f975293f8a10dfdf8ef929f62f101166800dddbb31329d8fe1",
        "dataset_count": 50,
        "schema_version": "dataset_probes_v1",
    }


def test_real_build_is_stable(real_all_plan: tuple[list[PlannedTask], dict[str, object]]) -> None:
    first_tasks, first_report = real_all_plan
    first_gt_tasks = [task for task in first_tasks if task.task_type == "gt_simplify"]
    second_tasks, second_report = build_clean_task_plan(phase="gt")
    assert [task.evaluation_key for task in first_gt_tasks] == [
        task.evaluation_key for task in second_tasks
    ]
    assert [task.input_hash for task in first_gt_tasks] == [
        task.input_hash for task in second_tasks
    ]
    assert first_report["contract"] == second_report["contract"]
    first_record = json.dumps(
        first_gt_tasks[0].to_json_record(),
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


def test_gt_v2_uses_distinct_prompt_and_logical_ids() -> None:
    v1_tasks, _ = build_clean_task_plan(phase="gt")
    v2_tasks, report = build_clean_task_plan(
        phase="gt",
        gt_prompt_path=STAGE_ROOT / "config/prompts/simplify.v2.txt",
        gt_logical_id_suffix="v2",
    )

    assert len(v2_tasks) == 50
    assert all(task.logical_id.endswith("::v2") for task in v2_tasks)
    assert {task.logical_id for task in v1_tasks}.isdisjoint(
        {task.logical_id for task in v2_tasks}
    )
    assert {task.evaluation_key for task in v1_tasks}.isdisjoint(
        {task.evaluation_key for task in v2_tasks}
    )
    assert {task.prompt_version for task in v2_tasks} == {"simplify.v2"}
    assert {task.schema_version for task in v2_tasks} == {"simplify.v1"}
    assert report["contract"]["gt_logical_id_suffix"] == "v2"
    assert report["contract"]["prompt_path"].endswith("simplify.v2.txt")


@pytest.mark.parametrize("suffix", ["V2", "v2::retry", "../v2", ""])
def test_gt_logical_id_suffix_rejects_ambiguous_values(suffix: str) -> None:
    with pytest.raises(CleanTaskBuilderError, match="gt_logical_id_suffix"):
        build_clean_task_plan(phase="gt", gt_logical_id_suffix=suffix)


def test_gt_contract_override_is_rejected_outside_gt_phase() -> None:
    with pytest.raises(CleanTaskBuilderError, match="仅允许与 phase=gt"):
        build_clean_task_plan(
            phase="all",
            gt_prompt_path=STAGE_ROOT / "config/prompts/simplify.v2.txt",
            gt_logical_id_suffix="v2",
        )


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


def test_indexed_prediction_variables_map_atomically_to_dataset_feature_names() -> None:
    mapped, mapping = map_indexed_variables(
        "x0*x1 + X1 + col0",
        ["t", "P"],
    )
    assert mapped == "t*P + P + t"
    assert mapping == {"X1": "P", "col0": "t", "x0": "t", "x1": "P"}


def test_indexed_variable_mapping_rejects_referenced_out_of_range_column() -> None:
    with pytest.raises(Exception, match="超出 feature_names"):
        map_indexed_variables("x2 + x0", ["t", "P"])


def test_function_body_extraction_and_parameter_instantiation_are_static() -> None:
    source = '''def equation(x0, params):
    """first"""
    """second"""
    return params[0] * x0 + params[-1]
'''
    body = extract_expression_body(source)
    assert body == "params[0] * x0 + params[-1]"
    assert instantiate_parameters(body, [1.25, -2.5]) == "(1.25) * x0 + (-2.5)"


def test_parameter_instantiation_rejects_dynamic_or_out_of_range_index() -> None:
    with pytest.raises(Exception, match="静态实例化"):
        instantiate_parameters("params[index] + x0", [1.0])
    with pytest.raises(Exception, match="超出长度"):
        instantiate_parameters("params[-2] + x0", [1.0])


def test_real_formula_recovery_manifest_is_strict_and_hashed() -> None:
    entries, manifest_sha256 = load_formula_recovery_manifest(
        STAGE_ROOT / "manifests/formula_recovery.v1.json"
    )
    assert len(entries) == 11
    assert len(manifest_sha256) == 64
    assert {entry["resolution"] for entry in entries.values()} == {"recovered_params"}
    assert all(entry.get("_candidate_function") for entry in entries.values())
    assert all(entry.get("_candidate_file_sha256") for entry in entries.values())


def test_formula_recovery_manifest_rejects_entry_from_other_condition(tmp_path: Path) -> None:
    manifest_path = tmp_path / "mixed_condition.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": "noise005",
                "entries": [
                    {
                        "task_id": "demoalg_s520_noise001_g0001",
                        "resolution": "unavailable",
                        "frozen_result_sha256": "a" * 64,
                        "equation_sha256": "b" * 64,
                        "reason": "demo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CleanTaskBuilderError, match="condition"):
        load_formula_recovery_manifest(manifest_path, expected_condition="noise005")


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
    probes_path = tmp_path / "dataset_probes.jsonl"
    _write_probe_jsonl(
        probes_path,
        dataset_id="DemoGT",
        variables=["x0"],
        target_name="y",
    )
    missing_payload = {
        "status": "error",
        "equation": "",
        "feature_names": ["x0"],
        "target_name": "y",
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
        dataset_id="DemoGT",
        algorithm="DemoAlg",
        payload=missing_payload,
    )
    with gzip.open(freeze_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    tasks, report = build_clean_task_plan(
        phase="pred",
        ground_truth_jsonl=gt_path,
        dataset_probes_jsonl=probes_path,
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
        assert "params[" not in expression
        probe = task.request["dataset_probe_evidence"]
        assert probe["dataset_name"] == task.request["dataset_id"]
        assert probe["variables"] == variables
        assert task.request["probe_points"] == probe["points"]
        assert task.request["probe_sample_sha256"] == probe["sample_sha256"]
        artifact = task.request["deterministic_evidence"]["symbolic_artifact"]
        assert artifact["artifact_sha256"]
        assert artifact["node_count"] >= 1
        assert "sympy_expression" not in artifact
        assert task.request["original_expression"] == expression
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
    probes_path = tmp_path / "dataset_probes.jsonl"
    _write_probe_jsonl(
        probes_path,
        dataset_id="DemoGT",
        variables=["x0"],
        target_name="y",
    )
    payload = {
        "status": "ok",
        "equation": "x0 + 1",
        "feature_names": ["x0"],
        "target_name": "y",
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
        dataset_id="DemoGT",
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
            "--dataset-probes-jsonl",
            str(probes_path),
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
