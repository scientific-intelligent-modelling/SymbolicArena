from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
    render_prompt,
)


SEEDS = (520, 521, 522)
ALGORITHM_COUNT = 15
DATASET_COUNT = 50


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_text(canonical_json(payload))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_probe(dataset_id: str, variables: list[str]) -> dict[str, Any]:
    points = [
        {"split": "id_test", "row_index": 0, "values": {"x0": 1.0, "x1": 2.0}},
        {"split": "id_test", "row_index": 1, "values": {"x0": -1.5, "x1": 0.5}},
        {"split": "ood_test", "row_index": 9, "values": {"x0": 3.0, "x1": -4.0}},
    ]
    sample_payload = {
        "schema_version": "dataset_probes_v1",
        "dataset_name": dataset_id,
        "variables": variables,
        "points": points,
    }
    probe = {
        "schema_version": "dataset_probes_v1",
        "dataset_name": dataset_id,
        "dataset_id": dataset_id,
        "dataset_dir": f"sim-datasets-data/demo/{dataset_id}",
        "basename": dataset_id,
        "core50_index": 1,
        "variables": variables,
        "target_name": "y",
        "points": points,
        "point_count": len(points),
        "sample_sha256": _sha256_json(sample_payload),
        "source_sha256": {
            "metadata_yaml": "1" * 64,
            "id_test_csv": "2" * 64,
            "ood_test_csv": "3" * 64,
        },
    }
    probe["evidence_sha256"] = _sha256_json({key: value for key, value in probe.items() if key != "evidence_sha256"})
    return probe


def _domain_assumptions() -> dict[str, Any]:
    return {
        "number_system": "real",
        "equivalence_domain": "Compare expressions on their common real-valued domain where both sides are defined and finite.",
        "implicit_protected_operators": "none",
        "literal_protected_operators": [],
        "operator_semantics": [
            "Division is ordinary real division; zero denominators are outside the common domain.",
            "log, sqrt, and non-integer powers use ordinary real-domain semantics.",
            "maximum, minimum, clip, and where are literal functions only when written in the expression.",
            "Do not infer hidden clipping, epsilon guards, or fitted constants beyond the frozen expression.",
        ],
    }


def _symbolic_artifact(expression: str, variables: list[str]) -> dict[str, Any]:
    return {
        "artifact_sha256": _sha256_text(f"artifact::{expression}"),
        "canonical_expression": expression,
        "variables": variables,
        "function_set": [],
        "operator_set": ["add"],
        "node_count": 3,
    }


def _write_contract(repo_root: Path) -> None:
    prompt_dir = repo_root / "AAAI_experiments/stage5_metric_calculation_0831/config/prompts"
    schema_dir = repo_root / "AAAI_experiments/stage5_metric_calculation_0831/config/schemas"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "equivalence.v1.txt").write_text("Judge equivalence.\n{{REQUEST_JSON}}\n", encoding="utf-8")
    (prompt_dir / "structure.v1.txt").write_text("Judge structure.\n{{REQUEST_JSON}}\n", encoding="utf-8")
    (prompt_dir / "simplify.v1.txt").write_text("Simplify.\n{{REQUEST_JSON}}\n", encoding="utf-8")
    (schema_dir / "equivalence.v1.json").write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {"type": "string"},
                    "evidence_basis": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "brief_reason": {"type": "string"},
                },
                "required": [
                    "decision",
                    "evidence_basis",
                    "assumptions",
                    "confidence",
                    "brief_reason",
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (schema_dir / "structure.v1.json").write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {"type": "string"},
                    "confidence": {"type": "number"},
                    "brief_reason": {"type": "string"},
                },
                "required": ["decision", "confidence", "brief_reason"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (schema_dir / "simplify.v1.json").write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outcome": {"type": "string"},
                    "simplified_expression": {"type": ["string", "null"]},
                    "equivalence_assessment": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "brief_reason": {"type": "string"},
                },
                "required": [
                    "outcome",
                    "simplified_expression",
                    "equivalence_assessment",
                    "assumptions",
                    "confidence",
                    "brief_reason",
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _simplify_plan_row(
    repo_root: Path,
    *,
    logical_id: str,
    task_type: str,
    priority: int,
    request: dict[str, Any],
) -> dict[str, Any]:
    prompt_path = repo_root / "AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify.v1.txt"
    schema_path = repo_root / "AAAI_experiments/stage5_metric_calculation_0831/config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    schema_content = json.loads(schema_path.read_text(encoding="utf-8"))
    normalized_input = {
        "request": dict(request),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=str(request["evidence_hash"]),
    )
    task_spec = {
        "evaluation_key": key,
        "logical_id": logical_id,
        "task_type": task_type,
        "condition": "clean",
        "priority": priority,
        "input_hash": input_hash,
        "prompt_version": "simplify.v1",
        "schema_version": "simplify.v1",
        "dependencies": [],
    }
    return {
        **task_spec,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_template": prompt_template,
        "schema_content": schema_content,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": task_spec,
        "rendered_prompt": render_prompt(prompt_template, request, schema_content),
    }


def _gt_request(dataset_id: str, *, variables: list[str] | None = None) -> dict[str, Any]:
    variables = variables or ["x0", "x1"]
    probe = _dataset_probe(dataset_id, variables)
    expression = "x0 + x1"
    return {
        "dataset_id": dataset_id,
        "target_name": "y",
        "variables": variables,
        "allowed_functions": ["add"],
        "expression": expression,
        "original_expression": "x0 + x1 + 0",
        "domain_assumptions": _domain_assumptions(),
        "probe_points": probe["points"],
        "probe_source": probe["schema_version"],
        "probe_sample_sha256": probe["sample_sha256"],
        "dataset_probe_evidence": probe,
        "deterministic_evidence": {
            "dataset_probe": probe,
            "symbolic_artifact": _symbolic_artifact(expression, variables),
            "domain_assumptions": _domain_assumptions(),
        },
        "ast_source_evidence": {"return_source": "x0 + x1 + 0"},
        "evidence_hash": f"gt-evidence::{dataset_id}",
    }


def _pred_request(
    *,
    dataset_id: str,
    dataset_index: str,
    algorithm: str,
    algorithm_slug: str,
    seed: int,
) -> dict[str, Any]:
    variables = ["x0", "x1"]
    probe = _dataset_probe(dataset_id, variables)
    expression = "x0 + x1"
    return {
        "dataset_id": dataset_id,
        "dataset_index": dataset_index,
        "algorithm": algorithm,
        "algorithm_slug": algorithm_slug,
        "seed": seed,
        "noise_tag": "clean",
        "task_id": f"{algorithm_slug}_s{seed}_clean_{dataset_index}",
        "variables": variables,
        "allowed_functions": ["add"],
        "expression": expression,
        "original_expression": "x0 + x1 + 0",
        "domain_assumptions": _domain_assumptions(),
        "probe_points": probe["points"],
        "probe_source": probe["schema_version"],
        "probe_sample_sha256": probe["sample_sha256"],
        "dataset_probe_evidence": probe,
        "deterministic_evidence": {
            "dataset_probe": probe,
            "symbolic_artifact": _symbolic_artifact(expression, variables),
            "domain_assumptions": _domain_assumptions(),
        },
        "ast_source_evidence": {"selected_expression_source": "equation"},
        "evidence_hash": f"pred-evidence::{algorithm_slug}::{dataset_index}::{seed}",
    }


def _frozen_index_row(
    *,
    plan_sha256: str,
    evaluation_key_value: str,
    logical_id: str,
    task_type: str,
    priority: int,
    state: str,
    structured_output: dict[str, Any] | None = None,
    reason: str | None = None,
    result_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "plan_sha256": plan_sha256,
        "evaluation_key": evaluation_key_value,
        "logical_id": logical_id,
        "task_type": task_type,
        "task_kind": "simplify",
        "condition": "clean",
        "priority": priority,
        "state": state,
        "attempt_id": None,
        "result_path": None,
        "result_sha256": result_sha256,
        "structured_output": structured_output,
        "non_applicable": None,
    }
    if state == "non_applicable":
        payload["non_applicable"] = {
            "reason": reason or "upstream_missing",
            "evidence_path": "audit.json",
            "evidence_sha256": "a" * 64,
        }
    return payload


def _write_frozen_summary(
    path: Path,
    *,
    output_jsonl: Path,
    plan_jsonl: Path,
    frozen_count: int,
    non_applicable_count: int,
) -> None:
    _write_json(
        path,
        {
            "output_jsonl": str(output_jsonl),
            "output_sha256": _sha256_file(output_jsonl),
            "plan_jsonl": str(plan_jsonl),
            "plan_sha256": _sha256_file(plan_jsonl),
            "row_count": frozen_count + non_applicable_count,
            "state_counts": {"frozen": frozen_count, "non_applicable": non_applicable_count},
            "status": "ok",
        },
    )


def _result_sha(seed: int, suffix: int) -> str:
    nibble = f"{(seed + suffix) % 16:x}"
    return nibble * 64


def _make_fixture(
    tmp_path: Path,
    *,
    full_counts: bool,
    gt_unavailable_dataset: str | None = None,
    pred_overrides: dict[tuple[str, str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    _write_contract(repo_root)
    pred_overrides = pred_overrides or {}

    datasets = [
        (f"Dataset{index:02d}", f"g{index:04d}")
        for index in range(1, DATASET_COUNT + 1 if full_counts else 2)
    ]
    algorithms = [
        (f"Algorithm{index:02d}", f"alg{index:02d}")
        for index in range(ALGORITHM_COUNT if full_counts else 1)
    ]

    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for dataset_id, dataset_index in datasets:
        gt_rows.append(
            _simplify_plan_row(
                repo_root,
                logical_id=f"gt_simplify::{dataset_id}",
                task_type="gt_simplify",
                priority=10,
                request=_gt_request(dataset_id),
            )
        )
        for algorithm, algorithm_slug in algorithms:
            for seed in SEEDS:
                pred_rows.append(
                    _simplify_plan_row(
                        repo_root,
                        logical_id=f"pred_simplify::{algorithm_slug}::{dataset_index}::s{seed}::clean",
                        task_type="pred_simplify",
                        priority=20,
                        request=_pred_request(
                            dataset_id=dataset_id,
                            dataset_index=dataset_index,
                            algorithm=algorithm,
                            algorithm_slug=algorithm_slug,
                            seed=seed,
                        ),
                    )
                )

    gt_plan = tmp_path / "clean_gt_simplify_tasks.jsonl"
    pred_plan = tmp_path / "clean_pred_simplify_tasks.jsonl"
    _write_jsonl(gt_plan, gt_rows)
    _write_jsonl(pred_plan, pred_rows)
    gt_plan_sha256 = _sha256_file(gt_plan)
    pred_plan_sha256 = _sha256_file(pred_plan)

    gt_frozen_rows: list[dict[str, Any]] = []
    gt_frozen_count = 0
    gt_non_applicable_count = 0
    for row in gt_rows:
        dataset_id = row["request"]["dataset_id"]
        if dataset_id == gt_unavailable_dataset:
            gt_frozen_rows.append(
                _frozen_index_row(
                    plan_sha256=gt_plan_sha256,
                    evaluation_key_value=row["evaluation_key"],
                    logical_id=row["logical_id"],
                    task_type="gt_simplify",
                    priority=10,
                    state="non_applicable",
                    reason="gt_missing",
                )
            )
            gt_non_applicable_count += 1
        else:
            gt_frozen_rows.append(
                _frozen_index_row(
                    plan_sha256=gt_plan_sha256,
                    evaluation_key_value=row["evaluation_key"],
                    logical_id=row["logical_id"],
                    task_type="gt_simplify",
                    priority=10,
                    state="frozen",
                    result_sha256=_result_sha(0, gt_frozen_count + 1),
                    structured_output={
                        "outcome": "simplified",
                        "simplified_expression": "x0 + x1",
                        "equivalence_assessment": "preserved",
                        "assumptions": [],
                        "confidence": 1.0,
                        "brief_reason": "fixture gt",
                    },
                )
            )
            gt_frozen_count += 1
    gt_frozen = tmp_path / "gt_frozen_index.jsonl"
    _write_jsonl(gt_frozen, gt_frozen_rows)

    pred_frozen_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, str]] = []
    pred_frozen_count = 0
    pred_non_applicable_count = 0
    for row_index, row in enumerate(pred_rows, start=1):
        request = row["request"]
        key = (request["algorithm_slug"], request["dataset_index"], request["seed"])
        override = pred_overrides.get(key, {})
        state = override.get("state", "frozen")
        outcome = override.get("outcome", "simplified")
        valid_output = bool(override.get("valid_output", True))
        result_sha256 = str(override.get("result_sha256", _result_sha(request["seed"], row_index)))

        if state == "non_applicable":
            pred_frozen_rows.append(
                _frozen_index_row(
                    plan_sha256=pred_plan_sha256,
                    evaluation_key_value=row["evaluation_key"],
                    logical_id=row["logical_id"],
                    task_type="pred_simplify",
                    priority=20,
                    state="non_applicable",
                    reason="pred_missing",
                )
            )
            pred_non_applicable_count += 1
        else:
            simplified_expression = override.get("simplified_expression")
            if simplified_expression is None:
                simplified_expression = None if outcome == "unable" else f"x0 + x1 + {request['seed'] - 520}"
            pred_frozen_rows.append(
                _frozen_index_row(
                    plan_sha256=pred_plan_sha256,
                    evaluation_key_value=row["evaluation_key"],
                    logical_id=row["logical_id"],
                    task_type="pred_simplify",
                    priority=20,
                    state="frozen",
                    result_sha256=result_sha256,
                    structured_output={
                        "outcome": outcome,
                        "simplified_expression": simplified_expression,
                        "equivalence_assessment": "undetermined" if outcome == "unable" else "preserved",
                        "assumptions": [],
                        "confidence": 0.2 if outcome == "unable" else 1.0,
                        "brief_reason": "fixture pred",
                    },
                )
            )
            pred_frozen_count += 1
        numeric_rows.append(
            {
                "logical_key": f"{request['algorithm']}::{request['dataset_id']}::s{request['seed']}::clean",
                "algorithm": request["algorithm"],
                "dataset_id": request["dataset_id"],
                "seed": str(request["seed"]),
                "noise_tag": "clean",
                "task_id": request["task_id"],
                "host": "fixture-host",
                "result_sha256": result_sha256,
                "valid_output": str(valid_output).lower(),
                "formula_source": "canonical_artifact",
                "id_nmse": "0.1",
                "ood_nmse": "0.2",
                "id_quality": "0.9",
                "ood_quality": "0.8",
            }
        )

    pred_frozen = tmp_path / "pred_frozen_index.jsonl"
    _write_jsonl(pred_frozen, pred_frozen_rows)
    clean_run_metrics_csv = tmp_path / "clean_numeric_run_metrics.csv"
    _write_csv(
        clean_run_metrics_csv,
        [
            "logical_key",
            "algorithm",
            "dataset_id",
            "seed",
            "noise_tag",
            "task_id",
            "host",
            "result_sha256",
            "valid_output",
            "formula_source",
            "id_nmse",
            "ood_nmse",
            "id_quality",
            "ood_quality",
        ],
        numeric_rows,
    )

    gt_summary = tmp_path / "clean_gt_frozen_index_summary.json"
    pred_summary = tmp_path / "clean_pred_frozen_index_summary.json"
    _write_frozen_summary(
        gt_summary,
        output_jsonl=gt_frozen,
        plan_jsonl=gt_plan,
        frozen_count=gt_frozen_count,
        non_applicable_count=gt_non_applicable_count,
    )
    _write_frozen_summary(
        pred_summary,
        output_jsonl=pred_frozen,
        plan_jsonl=pred_plan,
        frozen_count=pred_frozen_count,
        non_applicable_count=pred_non_applicable_count,
    )

    return {
        "repo_root": repo_root,
        "gt_plan": gt_plan,
        "pred_plan": pred_plan,
        "gt_frozen": gt_frozen,
        "pred_frozen": pred_frozen,
        "gt_summary": gt_summary,
        "pred_summary": pred_summary,
        "clean_run_metrics_csv": clean_run_metrics_csv,
        "non_applicable_index_jsonl": tmp_path / "clean_symbolic_non_applicable.jsonl",
        "non_applicable_evidence_dir": tmp_path / "clean_symbolic_non_applicable",
    }


def test_full_plan_uses_numeric_validity_and_closes_2250_per_phase(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(
        tmp_path,
        full_counts=True,
        pred_overrides={
            ("alg00", "g0001", 521): {"outcome": "unable", "valid_output": True},
            ("alg00", "g0002", 520): {"outcome": "simplified", "valid_output": False},
        },
    )
    tasks, report = build_symbolic_task_plan(
        gt_frozen_index_jsonl=fixture["gt_frozen"],
        pred_frozen_index_jsonl=fixture["pred_frozen"],
        gt_frozen_summary_json=fixture["gt_summary"],
        pred_frozen_summary_json=fixture["pred_summary"],
        gt_plan_jsonl=fixture["gt_plan"],
        pred_plan_jsonl=fixture["pred_plan"],
        clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
        non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
        repo_root=fixture["repo_root"],
        expected_gt_count=50,
        expected_pred_count=2250,
        expected_pair_count=2250,
    )

    equivalence_tasks = [task for task in tasks if task.task_type == "equivalence"]
    structure_tasks = [task for task in tasks if task.task_type == "stab_structure"]
    assert len(equivalence_tasks) == 2249
    assert len(structure_tasks) == 2246
    assert report["planning_counts"]["equivalence_no_call_count"] == 1
    assert report["planning_counts"]["structure_no_call_count"] == 4
    assert report["planning_counts"]["equivalence_closed_total"] == 2250
    assert report["planning_counts"]["structure_closed_total"] == 2250
    assert report["validation"]["clean_run_metrics"]["sha256"] == _sha256_file(fixture["clean_run_metrics_csv"])
    assert report["validation"]["clean_run_metrics"]["row_count"] == 2250
    assert report["validation"]["required_seed_pairs"] == [[520, 521], [520, 522], [521, 522]]

    eq_task = next(task for task in equivalence_tasks if task.logical_id == "equivalence::alg00::g0001::s520::clean")
    eq_evidence = eq_task.request["deterministic_evidence"]
    assert eq_evidence["lhs_binding"]["frozen_plan_sha256"] == _sha256_file(fixture["gt_plan"])
    assert eq_evidence["rhs_binding"]["frozen_plan_sha256"] == _sha256_file(fixture["pred_plan"])
    assert eq_evidence["pair_evidence"]["symbolic_difference"]["numeric_probes"]
    assert eq_evidence["evidence_sha256"] == eq_task.request["evidence_hash"]

    structure_task = next(task for task in structure_tasks if task.logical_id == "stab_structure::alg00::g0001::s520-s522")
    structure_evidence = structure_task.request["deterministic_pair_evidence"]
    assert structure_evidence["lhs_binding"]["frozen_result_sha256"] == _result_sha(520, 1)
    assert structure_evidence["rhs_binding"]["frozen_result_sha256"]
    assert structure_evidence["pair_evidence"]["probe_count"] >= 1

    no_call_records = report["no_call_records"]
    assert {row["reason"] for row in no_call_records} == {
        "upstream_pred_unavailable",
        "invalid_seed_or_expression",
    }
    for row in no_call_records:
        evidence_path = Path(row["evidence_path"])
        assert evidence_path.exists()
        assert evidence_path.parent == fixture["non_applicable_evidence_dir"]
        assert evidence_path.stem == row["evaluation_key"]
        assert _sha256_file(evidence_path) == row["evidence_sha256"]
        assert row["request"]["evidence_hash"] == row["evidence_sha256"]


def test_equivalence_gt_unavailable_uses_frozen_reason_enum(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(tmp_path, full_counts=False, gt_unavailable_dataset="Dataset01")
    tasks, report = build_symbolic_task_plan(
        gt_frozen_index_jsonl=fixture["gt_frozen"],
        pred_frozen_index_jsonl=fixture["pred_frozen"],
        gt_frozen_summary_json=fixture["gt_summary"],
        pred_frozen_summary_json=fixture["pred_summary"],
        gt_plan_jsonl=fixture["gt_plan"],
        pred_plan_jsonl=fixture["pred_plan"],
        clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
        non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
        repo_root=fixture["repo_root"],
        expected_gt_count=1,
        expected_pred_count=3,
        expected_pair_count=3,
    )

    assert len([task for task in tasks if task.task_type == "equivalence"]) == 0
    assert len([task for task in tasks if task.task_type == "stab_structure"]) == 3
    assert report["planning_counts"]["equivalence_no_call_count"] == 3
    assert {row["reason"] for row in report["no_call_records"] if row["phase"] == "equivalence"} == {
        "upstream_gt_unavailable"
    }


def test_summary_and_frozen_result_sha_are_required(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        SymbolicTaskBuilderError,
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(tmp_path, full_counts=False)
    with pytest.raises(SymbolicTaskBuilderError, match="必填"):
        build_symbolic_task_plan(  # type: ignore[arg-type]
            gt_frozen_index_jsonl=fixture["gt_frozen"],
            pred_frozen_index_jsonl=fixture["pred_frozen"],
            gt_frozen_summary_json=None,
            pred_frozen_summary_json=fixture["pred_summary"],
            gt_plan_jsonl=fixture["gt_plan"],
            pred_plan_jsonl=fixture["pred_plan"],
            clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
            non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
            repo_root=fixture["repo_root"],
            expected_gt_count=1,
            expected_pred_count=3,
            expected_pair_count=3,
        )

    pred_rows = [json.loads(line) for line in fixture["pred_frozen"].read_text(encoding="utf-8").splitlines()]
    pred_rows[0]["result_sha256"] = "not-a-sha"
    _write_jsonl(fixture["pred_frozen"], pred_rows)
    _write_frozen_summary(
        fixture["pred_summary"],
        output_jsonl=fixture["pred_frozen"],
        plan_jsonl=fixture["pred_plan"],
        frozen_count=3,
        non_applicable_count=0,
    )
    with pytest.raises(SymbolicTaskBuilderError, match="result_sha256"):
        build_symbolic_task_plan(
            gt_frozen_index_jsonl=fixture["gt_frozen"],
            pred_frozen_index_jsonl=fixture["pred_frozen"],
            gt_frozen_summary_json=fixture["gt_summary"],
            pred_frozen_summary_json=fixture["pred_summary"],
            gt_plan_jsonl=fixture["gt_plan"],
            pred_plan_jsonl=fixture["pred_plan"],
            clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
            non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
            repo_root=fixture["repo_root"],
            expected_gt_count=1,
            expected_pred_count=3,
            expected_pair_count=3,
        )


def test_summary_and_numeric_identity_are_strict(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        SymbolicTaskBuilderError,
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(tmp_path, full_counts=False)
    summary_payload = json.loads(fixture["pred_summary"].read_text(encoding="utf-8"))
    summary_payload["plan_sha256"] = "0" * 64
    _write_json(fixture["pred_summary"], summary_payload)
    with pytest.raises(SymbolicTaskBuilderError, match="plan_sha256"):
        build_symbolic_task_plan(
            gt_frozen_index_jsonl=fixture["gt_frozen"],
            pred_frozen_index_jsonl=fixture["pred_frozen"],
            gt_frozen_summary_json=fixture["gt_summary"],
            pred_frozen_summary_json=fixture["pred_summary"],
            gt_plan_jsonl=fixture["gt_plan"],
            pred_plan_jsonl=fixture["pred_plan"],
            clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
            non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
            repo_root=fixture["repo_root"],
            expected_gt_count=1,
            expected_pred_count=3,
            expected_pair_count=3,
        )

    fixture = _make_fixture(tmp_path / "bad_numeric", full_counts=False)
    numeric_rows = list(csv.DictReader(fixture["clean_run_metrics_csv"].open("r", encoding="utf-8")))
    numeric_rows[0]["task_id"] = "drifted_task_id"
    _write_csv(
        fixture["clean_run_metrics_csv"],
        list(numeric_rows[0].keys()),
        numeric_rows,
    )
    with pytest.raises(SymbolicTaskBuilderError, match="numeric task_id"):
        build_symbolic_task_plan(
            gt_frozen_index_jsonl=fixture["gt_frozen"],
            pred_frozen_index_jsonl=fixture["pred_frozen"],
            gt_frozen_summary_json=fixture["gt_summary"],
            pred_frozen_summary_json=fixture["pred_summary"],
            gt_plan_jsonl=fixture["gt_plan"],
            pred_plan_jsonl=fixture["pred_plan"],
            clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
            non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
            repo_root=fixture["repo_root"],
            expected_gt_count=1,
            expected_pred_count=3,
            expected_pair_count=3,
        )


def test_simplify_plan_logical_id_must_match_request_identity(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        SymbolicTaskBuilderError,
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(tmp_path, full_counts=False)
    pred_rows = [
        json.loads(line)
        for line in fixture["pred_plan"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pred_rows[0]["logical_id"] = "pred_simplify::wrong::g0001::s520::clean"
    _write_jsonl(fixture["pred_plan"], pred_rows)

    with pytest.raises(SymbolicTaskBuilderError, match="logical_id 非 canonical"):
        build_symbolic_task_plan(
            gt_frozen_index_jsonl=fixture["gt_frozen"],
            pred_frozen_index_jsonl=fixture["pred_frozen"],
            gt_frozen_summary_json=fixture["gt_summary"],
            pred_frozen_summary_json=fixture["pred_summary"],
            gt_plan_jsonl=fixture["gt_plan"],
            pred_plan_jsonl=fixture["pred_plan"],
            clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
            non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
            repo_root=fixture["repo_root"],
            expected_gt_count=1,
            expected_pred_count=3,
            expected_pair_count=3,
        )


def test_gt_v2_logical_id_is_preserved_across_downstream_symbolic_plan(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(tmp_path, full_counts=False)
    repo_root = fixture["repo_root"]
    gt_row = _simplify_plan_row(
        repo_root,
        logical_id="gt_simplify::Dataset01::v2",
        task_type="gt_simplify",
        priority=10,
        request=_gt_request("Dataset01"),
    )
    _write_jsonl(fixture["gt_plan"], [gt_row])
    _write_jsonl(
        fixture["gt_frozen"],
        [
            _frozen_index_row(
                plan_sha256=_sha256_file(fixture["gt_plan"]),
                evaluation_key_value=gt_row["evaluation_key"],
                logical_id=gt_row["logical_id"],
                task_type="gt_simplify",
                priority=10,
                state="frozen",
                result_sha256=_result_sha(0, 1),
                structured_output={
                    "outcome": "simplified",
                    "simplified_expression": "x0 + x1",
                    "equivalence_assessment": "preserved",
                    "assumptions": [],
                    "confidence": 1.0,
                    "brief_reason": "fixture gt v2",
                },
            )
        ],
    )
    _write_frozen_summary(
        fixture["gt_summary"],
        output_jsonl=fixture["gt_frozen"],
        plan_jsonl=fixture["gt_plan"],
        frozen_count=1,
        non_applicable_count=0,
    )

    tasks, _ = build_symbolic_task_plan(
        gt_frozen_index_jsonl=fixture["gt_frozen"],
        pred_frozen_index_jsonl=fixture["pred_frozen"],
        gt_frozen_summary_json=fixture["gt_summary"],
        pred_frozen_summary_json=fixture["pred_summary"],
        gt_plan_jsonl=fixture["gt_plan"],
        pred_plan_jsonl=fixture["pred_plan"],
        clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
        non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
        repo_root=repo_root,
        expected_gt_count=1,
        expected_pred_count=3,
        expected_pair_count=3,
    )

    eq_task = next(task for task in tasks if task.task_type == "equivalence")
    assert eq_task.request["ground_truth_logical_id"] == "gt_simplify::Dataset01::v2"
    assert eq_task.request["deterministic_evidence"]["lhs_binding"]["frozen_logical_id"] == "gt_simplify::Dataset01::v2"


def test_gt_unknown_suffix_is_rejected_in_symbolic_plan_input(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import (
        SymbolicTaskBuilderError,
        build_symbolic_task_plan,
    )

    fixture = _make_fixture(tmp_path, full_counts=False)
    gt_rows = [
        json.loads(line)
        for line in fixture["gt_plan"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gt_rows[0]["logical_id"] = "gt_simplify::Dataset01::v3"
    _write_jsonl(fixture["gt_plan"], gt_rows)
    _write_frozen_summary(
        fixture["gt_summary"],
        output_jsonl=fixture["gt_frozen"],
        plan_jsonl=fixture["gt_plan"],
        frozen_count=1,
        non_applicable_count=0,
    )

    with pytest.raises(SymbolicTaskBuilderError, match="GT logical_id 非 canonical"):
        build_symbolic_task_plan(
            gt_frozen_index_jsonl=fixture["gt_frozen"],
            pred_frozen_index_jsonl=fixture["pred_frozen"],
            gt_frozen_summary_json=fixture["gt_summary"],
            pred_frozen_summary_json=fixture["pred_summary"],
            gt_plan_jsonl=fixture["gt_plan"],
            pred_plan_jsonl=fixture["pred_plan"],
            clean_run_metrics_csv=fixture["clean_run_metrics_csv"],
            non_applicable_evidence_dir=fixture["non_applicable_evidence_dir"],
            repo_root=fixture["repo_root"],
            expected_gt_count=1,
            expected_pred_count=3,
            expected_pair_count=3,
        )


def test_cli_dry_run_does_not_pollute_output_or_evidence_dir(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import main

    fixture = _make_fixture(
        tmp_path,
        full_counts=False,
        pred_overrides={("alg00", "g0001", 521): {"outcome": "unable", "valid_output": True}},
    )
    output_jsonl = tmp_path / "symbolic_tasks.jsonl"
    report_json = tmp_path / "symbolic_report.json"
    exit_code = main(
        [
            "--gt-frozen-index-jsonl",
            str(fixture["gt_frozen"]),
            "--pred-frozen-index-jsonl",
            str(fixture["pred_frozen"]),
            "--gt-frozen-summary-json",
            str(fixture["gt_summary"]),
            "--pred-frozen-summary-json",
            str(fixture["pred_summary"]),
            "--gt-plan-jsonl",
            str(fixture["gt_plan"]),
            "--pred-plan-jsonl",
            str(fixture["pred_plan"]),
            "--clean-run-metrics-csv",
            str(fixture["clean_run_metrics_csv"]),
            "--repo-root",
            str(fixture["repo_root"]),
            "--expected-gt-count",
            "1",
            "--expected-pred-count",
            "3",
            "--expected-pair-count",
            "3",
            "--output-jsonl",
            str(output_jsonl),
            "--non-applicable-index-jsonl",
            str(fixture["non_applicable_index_jsonl"]),
            "--non-applicable-evidence-dir",
            str(fixture["non_applicable_evidence_dir"]),
            "--report-json",
            str(report_json),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output_jsonl.exists()
    assert not report_json.exists()
    assert not fixture["non_applicable_index_jsonl"].exists()
    assert not fixture["non_applicable_evidence_dir"].exists()


def test_cli_materializes_phase_specific_callable_no_call_and_full_plans(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import load_plan_jsonl
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import main

    fixture = _make_fixture(
        tmp_path,
        full_counts=False,
        pred_overrides={("alg00", "g0001", 521): {"outcome": "unable", "valid_output": True}},
    )
    all_callable_jsonl = tmp_path / "symbolic_tasks.jsonl"
    all_non_applicable_jsonl = tmp_path / "symbolic_non_applicable.jsonl"
    report_json = tmp_path / "symbolic_report.json"
    eq_callable_jsonl = tmp_path / "equivalence_callable.jsonl"
    eq_non_applicable_jsonl = tmp_path / "equivalence_non_applicable.jsonl"
    eq_full_plan_jsonl = tmp_path / "equivalence_full_plan.jsonl"
    structure_callable_jsonl = tmp_path / "structure_callable.jsonl"
    structure_non_applicable_jsonl = tmp_path / "structure_non_applicable.jsonl"
    structure_full_plan_jsonl = tmp_path / "structure_full_plan.jsonl"

    exit_code = main(
        [
            "--gt-frozen-index-jsonl",
            str(fixture["gt_frozen"]),
            "--pred-frozen-index-jsonl",
            str(fixture["pred_frozen"]),
            "--gt-frozen-summary-json",
            str(fixture["gt_summary"]),
            "--pred-frozen-summary-json",
            str(fixture["pred_summary"]),
            "--gt-plan-jsonl",
            str(fixture["gt_plan"]),
            "--pred-plan-jsonl",
            str(fixture["pred_plan"]),
            "--clean-run-metrics-csv",
            str(fixture["clean_run_metrics_csv"]),
            "--repo-root",
            str(fixture["repo_root"]),
            "--expected-gt-count",
            "1",
            "--expected-pred-count",
            "3",
            "--expected-pair-count",
            "3",
            "--output-jsonl",
            str(all_callable_jsonl),
            "--non-applicable-index-jsonl",
            str(all_non_applicable_jsonl),
            "--non-applicable-evidence-dir",
            str(fixture["non_applicable_evidence_dir"]),
            "--report-json",
            str(report_json),
            "--equivalence-output-jsonl",
            str(eq_callable_jsonl),
            "--equivalence-non-applicable-index-jsonl",
            str(eq_non_applicable_jsonl),
            "--equivalence-full-plan-jsonl",
            str(eq_full_plan_jsonl),
            "--structure-output-jsonl",
            str(structure_callable_jsonl),
            "--structure-non-applicable-index-jsonl",
            str(structure_non_applicable_jsonl),
            "--structure-full-plan-jsonl",
            str(structure_full_plan_jsonl),
        ]
    )
    assert exit_code == 0

    all_callable_rows = [json.loads(line) for line in all_callable_jsonl.read_text(encoding="utf-8").splitlines()]
    all_non_applicable_rows = [
        json.loads(line) for line in all_non_applicable_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(all_callable_rows) == 3
    assert len(all_non_applicable_rows) == 3

    eq_callable_rows = [json.loads(line) for line in eq_callable_jsonl.read_text(encoding="utf-8").splitlines()]
    eq_non_applicable_rows = [
        json.loads(line) for line in eq_non_applicable_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    structure_callable_rows = [
        json.loads(line) for line in structure_callable_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    structure_non_applicable_rows = [
        json.loads(line)
        for line in structure_non_applicable_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(eq_callable_rows) == 2
    assert len(eq_non_applicable_rows) == 1
    assert len(structure_callable_rows) == 1
    assert len(structure_non_applicable_rows) == 2
    assert {row["phase"] for row in eq_non_applicable_rows} == {"equivalence"}
    assert {row["phase"] for row in structure_non_applicable_rows} == {"structure"}

    assert len(load_plan_jsonl(eq_full_plan_jsonl).entries) == 3
    assert len(load_plan_jsonl(structure_full_plan_jsonl).entries) == 3

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["phase_outputs"]["equivalence"]["callable_task_count"] == 2
    assert report["phase_outputs"]["equivalence"]["non_applicable_count"] == 1
    assert report["phase_outputs"]["equivalence"]["full_plan_count"] == 3
    assert report["phase_outputs"]["equivalence"]["callable_output_sha256"] == hashlib.sha256(
        eq_callable_jsonl.read_bytes()
    ).hexdigest()
    assert report["phase_outputs"]["equivalence"]["non_applicable_index_sha256"] == hashlib.sha256(
        eq_non_applicable_jsonl.read_bytes()
    ).hexdigest()
    assert report["phase_outputs"]["equivalence"]["full_plan_sha256"] == hashlib.sha256(
        eq_full_plan_jsonl.read_bytes()
    ).hexdigest()
    assert report["phase_outputs"]["structure"]["callable_task_count"] == 1
    assert report["phase_outputs"]["structure"]["non_applicable_count"] == 2
    assert report["phase_outputs"]["structure"]["full_plan_count"] == 3


def test_cli_rejects_partial_phase_materialization_request(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import main

    fixture = _make_fixture(tmp_path, full_counts=False)
    exit_code = main(
        [
            "--gt-frozen-index-jsonl",
            str(fixture["gt_frozen"]),
            "--pred-frozen-index-jsonl",
            str(fixture["pred_frozen"]),
            "--gt-frozen-summary-json",
            str(fixture["gt_summary"]),
            "--pred-frozen-summary-json",
            str(fixture["pred_summary"]),
            "--gt-plan-jsonl",
            str(fixture["gt_plan"]),
            "--pred-plan-jsonl",
            str(fixture["pred_plan"]),
            "--clean-run-metrics-csv",
            str(fixture["clean_run_metrics_csv"]),
            "--repo-root",
            str(fixture["repo_root"]),
            "--expected-gt-count",
            "1",
            "--expected-pred-count",
            "3",
            "--expected-pair-count",
            "3",
            "--equivalence-output-jsonl",
            str(tmp_path / "equivalence_callable.jsonl"),
        ]
    )
    assert exit_code == 2


def test_cli_selected_phase_filters_standard_non_applicable_index(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import main

    fixture = _make_fixture(
        tmp_path,
        full_counts=False,
        pred_overrides={("alg00", "g0001", 521): {"outcome": "unable", "valid_output": True}},
    )
    output_jsonl = tmp_path / "equivalence_tasks.jsonl"
    non_applicable_jsonl = tmp_path / "equivalence_non_applicable.jsonl"
    report_json = tmp_path / "equivalence_report.json"
    exit_code = main(
        [
            "--gt-frozen-index-jsonl",
            str(fixture["gt_frozen"]),
            "--pred-frozen-index-jsonl",
            str(fixture["pred_frozen"]),
            "--gt-frozen-summary-json",
            str(fixture["gt_summary"]),
            "--pred-frozen-summary-json",
            str(fixture["pred_summary"]),
            "--gt-plan-jsonl",
            str(fixture["gt_plan"]),
            "--pred-plan-jsonl",
            str(fixture["pred_plan"]),
            "--clean-run-metrics-csv",
            str(fixture["clean_run_metrics_csv"]),
            "--repo-root",
            str(fixture["repo_root"]),
            "--expected-gt-count",
            "1",
            "--expected-pred-count",
            "3",
            "--expected-pair-count",
            "3",
            "--phase",
            "equivalence",
            "--output-jsonl",
            str(output_jsonl),
            "--non-applicable-index-jsonl",
            str(non_applicable_jsonl),
            "--non-applicable-evidence-dir",
            str(fixture["non_applicable_evidence_dir"]),
            "--report-json",
            str(report_json),
        ]
    )
    assert exit_code == 0
    callable_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    non_applicable_rows = [json.loads(line) for line in non_applicable_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(callable_rows) == 2
    assert len(non_applicable_rows) == 1
    assert {row["phase"] for row in non_applicable_rows} == {"equivalence"}
