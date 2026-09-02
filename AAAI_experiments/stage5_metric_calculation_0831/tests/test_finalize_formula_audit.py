from __future__ import annotations

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
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_retry_plan import (  # noqa: E402
    build_formula_audit_retry_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.finalize_formula_audit import (  # noqa: E402
    FormulaAuditFinalizationError,
    _offline_severity,
    _resolve_decision,
    finalize_formula_audit,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def _model_output(
    *,
    reference: str,
    simplification: str = "preserved",
    quality: str = "strictly_simpler",
    severity: str = "pass",
) -> dict[str, Any]:
    return {
        "simplification_equivalence": simplification,
        "simplification_quality": quality,
        "suggested_simplified_expression": None,
        "reference_equivalence": reference,
        "counterexample": None,
        "severity": severity,
        "assumptions": [],
        "confidence": 0.95,
        "brief_reason": "fixture",
    }


def _plan_row(
    *,
    logical_id: str,
    scope: str,
    review_round: int,
    prompt_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256(prompt_path.read_bytes())
    schema_sha256 = _sha256(schema_path.read_bytes())
    request_without_hash: dict[str, Any] = {
        "audit_scope": scope,
        "audit_binding_sha256": _sha256(logical_id.removesuffix("::v2").encode()),
        "variables": ["x0"],
        "allowed_functions": [],
        "domain_assumptions": {"variable_domain": "real"},
        "original_expression": "x0 + 0",
        "candidate_simplified_expression": "x0",
        "reference_simplified_expression": "x0" if scope == "prediction_formula" else None,
        "review_round": review_round,
    }
    request = dict(request_without_hash)
    request["evidence_hash"] = _sha256(canonical_json(request_without_hash).encode())
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256(canonical_json(normalized_input).encode())
    key = evaluation_key(
        task_type="formula_audit",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type="formula_audit",
        condition="clean",
        priority=50,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=(),
    )
    return {
        "evaluation_key": key,
        "logical_id": logical_id,
        "task_type": "formula_audit",
        "task_kind": "formula_audit",
        "condition": "clean",
        "priority": 50,
        "input_hash": input_hash,
        "prompt_version": prompt_path.stem,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_path.stem,
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_template": prompt_template,
        "schema_content": schema,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
    }


def _spec(row: dict[str, Any]) -> TaskSpec:
    payload = row["task_spec"]
    return TaskSpec(
        evaluation_key=payload["evaluation_key"],
        logical_id=payload["logical_id"],
        task_type=payload["task_type"],
        condition=payload["condition"],
        priority=payload["priority"],
        input_hash=payload["input_hash"],
        prompt_version=payload["prompt_version"],
        schema_version=payload["schema_version"],
        dependencies=tuple(payload["dependencies"]),
    )


def _usage(*, input_tokens: int, output_tokens: int, cache_read: int, cache_creation: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def _finish_attempt(
    *,
    store: TaskStateStore,
    row: dict[str, Any],
    attempts_dir: Path,
    frozen_dir: Path,
    usage: dict[str, int],
    output: dict[str, Any] | None,
    now: float,
) -> None:
    spec = _spec(row)
    lease = store.reserve_attempt(spec.evaluation_key, now=now)
    attempt_payload = {
        "attempt_id": lease.attempt_id,
        "evaluation_key": spec.evaluation_key,
        "request": row["request"],
        "metadata": {"usage": usage},
    }
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts_dir / f"{lease.attempt_id}.json"
    attempt_path.write_text(json.dumps(attempt_payload, sort_keys=True) + "\n")
    if output is None:
        store.finish_failure(
            lease.attempt_id,
            error_class="api_http_transient",
            retryable=False,
            now=now + 0.01,
        )
        return
    frozen_payload = {
        **attempt_payload,
        "logical_id": spec.logical_id,
        "task_type": spec.task_type,
        "task_kind": "formula_audit",
        "structured_output": output,
        "validation": {"ok": True, "structured_output": output},
    }
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = frozen_dir / f"{spec.evaluation_key}.json"
    frozen_path.write_text(json.dumps(frozen_payload, sort_keys=True) + "\n")
    store.freeze_result(
        lease.attempt_id,
        result_path=str(frozen_path),
        result_sha256=_sha256(frozen_path.read_bytes()),
        now=now + 0.01,
    )


def _local_check(
    decision: str,
    *,
    proof_basis: str = "none",
    counterexample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "proof_basis": proof_basis,
        "counterexample": counterexample,
        "probe_count": 1 if counterexample else 0,
        "max_abs_error": "1" if counterexample else "0",
        "max_rel_error": "1" if counterexample else "0",
        "evidence_sha256": _sha256(
            canonical_json([decision, proof_basis, counterexample]).encode()
        ),
        "error": None,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    config = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    prompt = config / "prompts/formula_audit.v1.txt"
    schema = config / "schemas/formula_audit.v1.json"
    definitions = [
        ("prediction::alg_a::g1::s520::clean", "prediction_formula", "alg_a", "clean", 520),
        ("prediction::alg_b::g2::s521::noise001", "prediction_formula", "alg_b", "noise001", 521),
        ("prediction::alg_a::g3::s522::noise005", "prediction_formula", "alg_a", "noise005", 522),
        ("ground_truth::g4", "ground_truth_simplification", None, "clean", None),
    ]
    round1 = [
        _plan_row(
            logical_id=f"formula_audit::{name}",
            scope=scope,
            review_round=1,
            prompt_path=prompt,
            schema_path=schema,
        )
        for name, scope, _, _, _ in definitions
    ]
    round2 = [
        _plan_row(
            logical_id=f"{round1[index]['logical_id']}::v2",
            scope=definitions[index][1],
            review_round=2,
            prompt_path=prompt,
            schema_path=schema,
        )
        for index in range(3)
    ]
    round1_plan = tmp_path / "round1.jsonl"
    round2_plan = tmp_path / "round2.jsonl"
    _write_jsonl(round1_plan, round1)
    _write_jsonl(round2_plan, round2)

    manifest = []
    comparisons = []
    for index, (row, definition) in enumerate(zip(round1, definitions, strict=True)):
        _, scope, algorithm, condition, seed = definition
        manifest.append(
            {
                "audit_logical_id": row["logical_id"],
                "audit_evaluation_key": row["evaluation_key"],
                "audit_scope": scope,
                "algorithm": algorithm,
                "dataset_id": f"g{index:04d}",
                "condition": condition,
                "seed": seed,
                "logical_key": f"source::{index}" if algorithm else None,
                "source_formula_logical_id": f"formula::{index}",
                "source_formula_evaluation_key": _sha256(f"formula::{index}".encode()),
                "source_formula_result_sha256": _sha256(f"result::{index}".encode()),
                "stored_simplification_outcome": "simplified",
                "stored_equivalence_decision": (
                    "not_applicable"
                    if scope == "ground_truth_simplification"
                    else ("not_equivalent" if index == 1 else "equivalent")
                ),
            }
        )
        comparisons.append(
            {
                "audit_logical_id": row["logical_id"],
                "round1_evaluation_key": row["evaluation_key"],
                "requires_round2": index < 3,
            }
        )
    manifest_path = tmp_path / "manifest.jsonl"
    comparisons_path = tmp_path / "comparisons.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(comparisons_path, comparisons)

    local = [
        {
            "audit_logical_id": round1[0]["logical_id"],
            "round1_evaluation_key": round1[0]["evaluation_key"],
            "local_simplification": _local_check(
                "not_equivalent",
                counterexample={"values": {"x0": "1"}, "abs_error": "1"},
            ),
            "local_reference": _local_check("equivalent", proof_basis="artifact_identity"),
        },
        {
            "audit_logical_id": round1[1]["logical_id"],
            "round1_evaluation_key": round1[1]["evaluation_key"],
            "local_simplification": _local_check("undetermined"),
            "local_reference": _local_check("undetermined"),
        },
        {
            "audit_logical_id": round1[2]["logical_id"],
            "round1_evaluation_key": round1[2]["evaluation_key"],
            "local_simplification": _local_check("undetermined"),
            "local_reference": _local_check("undetermined"),
        },
    ]
    for row in local:
        row["local_evidence_hash"] = _sha256(canonical_json(row).encode())
    local_path = tmp_path / "local.jsonl"
    _write_jsonl(local_path, local)

    round1_state = tmp_path / "round1_state.sqlite"
    round1_store = TaskStateStore(
        round1_state, attempt_cap=20, logical_task_cap=20, max_attempts_per_task=1
    )
    round1_store.register_tasks([_spec(row) for row in round1], now=1.0)
    round1_outputs = [
        _model_output(reference="equivalent", severity="pass"),
        _model_output(reference="not_equivalent", severity="critical"),
        _model_output(reference="equivalent"),
        _model_output(reference="not_applicable", severity="critical"),
    ]
    for index, (row, output) in enumerate(zip(round1, round1_outputs, strict=True)):
        _finish_attempt(
            store=round1_store,
            row=row,
            attempts_dir=tmp_path / "round1_attempts",
            frozen_dir=tmp_path / "round1_frozen",
            usage=_usage(input_tokens=100, output_tokens=10, cache_read=5, cache_creation=7),
            output=output,
            now=2.0 + index,
        )

    round2_state = tmp_path / "round2_state.sqlite"
    round2_store = TaskStateStore(
        round2_state, attempt_cap=20, logical_task_cap=20, max_attempts_per_task=1
    )
    round2_store.register_tasks([_spec(row) for row in round2], now=10.0)
    round2_outputs = [
        _model_output(reference="equivalent"),
        _model_output(reference="equivalent"),
        None,
    ]
    for index, (row, output) in enumerate(zip(round2, round2_outputs, strict=True)):
        _finish_attempt(
            store=round2_store,
            row=row,
            attempts_dir=tmp_path / "round2_attempts",
            frozen_dir=tmp_path / "round2_frozen",
            usage=_usage(input_tokens=50, output_tokens=5, cache_read=2, cache_creation=3),
            output=output,
            now=11.0 + index,
        )
    return {
        "round1_plan": round1_plan,
        "round2_plan": round2_plan,
        "manifest": manifest_path,
        "comparisons": comparisons_path,
        "local": local_path,
        "round1_state": round1_state,
        "round2_state": round2_state,
        "round1_attempts": tmp_path / "round1_attempts",
        "round2_attempts": tmp_path / "round2_attempts",
    }


def _add_retry_fixture(
    fixture: dict[str, Any],
    tmp_path: Path,
    *,
    output: dict[str, Any] | None,
) -> dict[str, Any]:
    retry_plan = tmp_path / "retry.jsonl"
    build_formula_audit_retry_plan(
        round2_plan_jsonl=fixture["round2_plan"],
        round2_state_db=fixture["round2_state"],
        output_jsonl=retry_plan,
        report_json=tmp_path / "retry_report.json",
    )
    retry_rows = _read_jsonl(retry_plan)
    assert len(retry_rows) == 1
    retry_state = tmp_path / "retry_state.sqlite"
    retry_store = TaskStateStore(
        retry_state, attempt_cap=2, logical_task_cap=2, max_attempts_per_task=1
    )
    retry_store.register_tasks([_spec(row) for row in retry_rows], now=20.0)
    _finish_attempt(
        store=retry_store,
        row=retry_rows[0],
        attempts_dir=tmp_path / "retry_attempts",
        frozen_dir=tmp_path / "retry_frozen",
        usage=_usage(input_tokens=40, output_tokens=4, cache_read=1, cache_creation=2),
        output=output,
        now=21.0,
    )
    return {
        "plan": retry_plan,
        "state": retry_state,
        "attempts": tmp_path / "retry_attempts",
    }


def test_finalize_resolves_rounds_local_override_usage_and_dimensions(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = finalize_formula_audit(
        round1_plan_jsonl=fixture["round1_plan"],
        round1_comparisons_jsonl=fixture["comparisons"],
        round1_state_db=fixture["round1_state"],
        round2_plan_jsonl=fixture["round2_plan"],
        round2_state_db=fixture["round2_state"],
        round1_attempts_dir=fixture["round1_attempts"],
        round2_attempts_dir=fixture["round2_attempts"],
        local_verification_jsonl=fixture["local"],
        sample_manifest_jsonl=fixture["manifest"],
        external_attempt_count=1,
        maximum_total_physical_attempts=10,
        output_root=tmp_path / "final",
    )

    assert report["final_count"] == 4
    assert report["budget"]["total_physical_attempt_count"] == 8
    assert report["budget"]["round1"]["input_tokens"] == 400
    assert report["budget"]["round2"]["output_tokens"] == 15
    assert report["budget"]["cache_creation_input_tokens_unpriced"] == 37
    assert report["budget"]["priced_cost_cny"] == pytest.approx(0.0024828)
    final = _read_jsonl(report["outputs"]["final_jsonl"])
    by_algorithm_condition = {
        (row["algorithm"], row["condition"]): row for row in final
    }
    local_override = by_algorithm_condition[("alg_a", "clean")]
    assert local_override["final_simplification_decision"] == "not_preserved"
    assert local_override["simplification_resolution_source"] == "local_counterexample"
    assert local_override["offline_severity"] == "critical"
    assert local_override["dataset_id"] == "g0000"
    assert local_override["expressions"] == {
        "original": "x0 + 0",
        "candidate_simplified": "x0",
        "reference_simplified": "x0",
    }
    assert local_override["stored_decisions"] == {
        "simplification_outcome": "simplified",
        "simplification_decision": "preserved",
        "reference_equivalence": "equivalent",
    }
    assert local_override["round1_model_evidence"]["brief_reason"] == "fixture"
    assert local_override["round1_model_evidence"]["confidence"] == 0.95
    assert local_override["round1_model_evidence"]["counterexample"] is None
    assert local_override["round2_model_evidence"]["brief_reason"] == "fixture"
    round2_success = by_algorithm_condition[("alg_b", "noise001")]
    assert round2_success["final_reference_decision"] == "equivalent"
    assert round2_success["reference_resolution_source"] == "round2"
    assert round2_success["round_disagreement"]["reference"] is True
    exhausted = by_algorithm_condition[("alg_a", "noise005")]
    assert exhausted["round2_state"] == "exhausted"
    assert exhausted["final_reference_decision"] == "equivalent"
    assert exhausted["reference_resolution_source"] == "round1"
    assert "round2_result_unavailable:exhausted" in exhausted["issue_types"]
    ground_truth = by_algorithm_condition[(None, "clean")]
    assert ground_truth["offline_severity"] == "pass"
    assert ground_truth["model_reported_severity"]["round1"] == "critical"
    assert ground_truth["model_reported_severity"]["used_for_final"] is False

    issues = _read_jsonl(report["outputs"]["issues_jsonl"])
    issue_by_logical = {row["audit_logical_id"]: row for row in issues}
    self_contained_issue = issue_by_logical[local_override["audit_logical_id"]]
    assert self_contained_issue["algorithm"] == "alg_a"
    assert self_contained_issue["condition"] == "clean"
    assert self_contained_issue["seed"] == 520
    assert self_contained_issue["dataset_id"] == "g0000"
    assert self_contained_issue["expressions"] == local_override["expressions"]
    assert self_contained_issue["round1_model_evidence"]["brief_reason"] == "fixture"
    assert self_contained_issue["round2_model_evidence"]["confidence"] == 0.95
    assert (
        self_contained_issue["local_criteria"]["simplification"]["counterexample"]
        is not None
    )
    assert self_contained_issue["resolution"]["simplification"] == "not_preserved"

    statistics_document = json.loads(
        Path(report["outputs"]["statistics_json"]).read_text(encoding="utf-8")
    )
    dimensions = statistics_document["statistics"]["dimensions"]
    assert dimensions["algorithm"]["alg_a"]["total"] == 2
    assert dimensions["algorithm"]["not_applicable"]["total"] == 1
    assert dimensions["condition"]["clean"]["total"] == 2
    assert dimensions["seed"]["520"]["total"] == 1
    assert dimensions["audit_scope"]["prediction_formula"]["total"] == 3
    statistics_csv_path = Path(report["outputs"]["statistics_csv"])
    assert statistics_csv_path.read_text().startswith(
        "dimension,value,total,issue_count"
    )
    assert b"\r" not in statistics_csv_path.read_bytes()
    markdown = Path(report["outputs"]["markdown_report"]).read_text(encoding="utf-8")
    assert "Model-reported severity is never used" in markdown


def test_finalize_rejects_total_physical_attempt_budget_overrun(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(FormulaAuditFinalizationError, match="突破硬上限"):
        finalize_formula_audit(
            round1_plan_jsonl=fixture["round1_plan"],
            round1_comparisons_jsonl=fixture["comparisons"],
            round1_state_db=fixture["round1_state"],
            round2_plan_jsonl=fixture["round2_plan"],
            round2_state_db=fixture["round2_state"],
            round1_attempts_dir=fixture["round1_attempts"],
            round2_attempts_dir=fixture["round2_attempts"],
            local_verification_jsonl=fixture["local"],
            sample_manifest_jsonl=fixture["manifest"],
            external_attempt_count=4,
            maximum_total_physical_attempts=10,
            output_root=tmp_path / "over-budget",
        )


def test_finalize_uses_successful_retry_as_effective_round2(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    retry = _add_retry_fixture(
        fixture,
        tmp_path,
        output=_model_output(reference="not_equivalent", severity="critical"),
    )

    report = finalize_formula_audit(
        round1_plan_jsonl=fixture["round1_plan"],
        round1_comparisons_jsonl=fixture["comparisons"],
        round1_state_db=fixture["round1_state"],
        round2_plan_jsonl=fixture["round2_plan"],
        round2_state_db=fixture["round2_state"],
        retry_plan_jsonl=retry["plan"],
        retry_state_db=retry["state"],
        round1_attempts_dir=fixture["round1_attempts"],
        round2_attempts_dir=fixture["round2_attempts"],
        retry_attempts_dir=retry["attempts"],
        local_verification_jsonl=fixture["local"],
        sample_manifest_jsonl=fixture["manifest"],
        external_attempt_count=1,
        maximum_total_physical_attempts=10,
        output_root=tmp_path / "retry-success",
    )

    assert report["budget"]["total_physical_attempt_count"] == 9
    assert report["budget"]["round2_retry"]["physical_attempt_count"] == 1
    assert report["budget"]["round2_retry"]["output_tokens"] == 4
    final = _read_jsonl(report["outputs"]["final_jsonl"])
    recovered = next(row for row in final if row["condition"] == "noise005")
    assert recovered["round2_state"] == "exhausted"
    assert recovered["round2_model_evidence"] is None
    assert recovered["round2_retry_state"] == "frozen"
    assert recovered["round2_retry_model_evidence"]["reference_equivalence"] == "not_equivalent"
    assert recovered["effective_round2_source"] == "round2_retry"
    assert recovered["effective_round2_model_evidence"]["reference_equivalence"] == "not_equivalent"
    assert recovered["final_reference_decision"] == "not_equivalent"
    assert recovered["reference_resolution_source"] == "round2_retry"
    assert "round2_result_unavailable:exhausted" in recovered["issue_types"]
    assert "round2_retry_recovered" in recovered["issue_types"]


def test_finalize_retry_failure_conservatively_falls_back_to_round1(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    retry = _add_retry_fixture(fixture, tmp_path, output=None)

    report = finalize_formula_audit(
        round1_plan_jsonl=fixture["round1_plan"],
        round1_comparisons_jsonl=fixture["comparisons"],
        round1_state_db=fixture["round1_state"],
        round2_plan_jsonl=fixture["round2_plan"],
        round2_state_db=fixture["round2_state"],
        retry_plan_jsonl=retry["plan"],
        retry_state_db=retry["state"],
        round1_attempts_dir=fixture["round1_attempts"],
        round2_attempts_dir=fixture["round2_attempts"],
        retry_attempts_dir=retry["attempts"],
        local_verification_jsonl=fixture["local"],
        sample_manifest_jsonl=fixture["manifest"],
        external_attempt_count=1,
        maximum_total_physical_attempts=10,
        output_root=tmp_path / "retry-failure",
    )

    final = _read_jsonl(report["outputs"]["final_jsonl"])
    unresolved_retry = next(row for row in final if row["condition"] == "noise005")
    assert unresolved_retry["round2_retry_state"] == "exhausted"
    assert unresolved_retry["round2_retry_model_evidence"] is None
    assert unresolved_retry["effective_round2_source"] is None
    assert unresolved_retry["final_reference_decision"] == "equivalent"
    assert unresolved_retry["reference_resolution_source"] == "round1"
    assert "round2_retry_result_unavailable:exhausted" in unresolved_retry["issue_types"]


def test_finalize_rejects_retry_predecessor_identity_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    retry = _add_retry_fixture(
        fixture,
        tmp_path,
        output=_model_output(reference="not_equivalent"),
    )
    retry_rows = _read_jsonl(retry["plan"])
    retry_rows[0]["retry_predecessor_evaluation_key"] = "0" * 64
    _write_jsonl(retry["plan"], retry_rows)

    with pytest.raises(FormulaAuditFinalizationError, match="predecessor evaluation_key 漂移"):
        finalize_formula_audit(
            round1_plan_jsonl=fixture["round1_plan"],
            round1_comparisons_jsonl=fixture["comparisons"],
            round1_state_db=fixture["round1_state"],
            round2_plan_jsonl=fixture["round2_plan"],
            round2_state_db=fixture["round2_state"],
            retry_plan_jsonl=retry["plan"],
            retry_state_db=retry["state"],
            round1_attempts_dir=fixture["round1_attempts"],
            round2_attempts_dir=fixture["round2_attempts"],
            retry_attempts_dir=retry["attempts"],
            local_verification_jsonl=fixture["local"],
            sample_manifest_jsonl=fixture["manifest"],
            external_attempt_count=0,
            output_root=tmp_path / "retry-drift",
        )


def test_finalize_rejects_retry_request_semantic_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    round2_rows = _read_jsonl(fixture["round2_plan"])
    predecessor = round2_rows[2]
    config = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    retry_row = _plan_row(
        logical_id=f"{predecessor['logical_id']}::retry1",
        scope="prediction_formula",
        review_round=2,
        prompt_path=config / "prompts/formula_audit.v3.txt",
        schema_path=config / "schemas/formula_audit.v1.json",
    )
    retry_row["retry_predecessor_evaluation_key"] = predecessor["evaluation_key"]
    retry_plan = tmp_path / "retry_request_drift.jsonl"
    _write_jsonl(retry_plan, [retry_row])
    retry_state = tmp_path / "retry_request_drift.sqlite"
    retry_store = TaskStateStore(
        retry_state, attempt_cap=1, logical_task_cap=1, max_attempts_per_task=1
    )
    retry_store.register_tasks([_spec(retry_row)], now=30.0)
    _finish_attempt(
        store=retry_store,
        row=retry_row,
        attempts_dir=tmp_path / "retry_request_drift_attempts",
        frozen_dir=tmp_path / "retry_request_drift_frozen",
        usage=_usage(input_tokens=1, output_tokens=1, cache_read=0, cache_creation=0),
        output=None,
        now=31.0,
    )

    with pytest.raises(FormulaAuditFinalizationError, match="request 与原 round2 不完全一致"):
        finalize_formula_audit(
            round1_plan_jsonl=fixture["round1_plan"],
            round1_comparisons_jsonl=fixture["comparisons"],
            round1_state_db=fixture["round1_state"],
            round2_plan_jsonl=fixture["round2_plan"],
            round2_state_db=fixture["round2_state"],
            retry_plan_jsonl=retry_plan,
            retry_state_db=retry_state,
            round1_attempts_dir=fixture["round1_attempts"],
            round2_attempts_dir=fixture["round2_attempts"],
            retry_attempts_dir=tmp_path / "retry_request_drift_attempts",
            local_verification_jsonl=fixture["local"],
            sample_manifest_jsonl=fixture["manifest"],
            external_attempt_count=0,
            output_root=tmp_path / "retry-request-drift",
        )


def test_symbolic_zero_proof_respects_simplification_domain_semantics() -> None:
    local_zero_proof = _local_check(
        "equivalent", proof_basis="symbolic_difference_zero"
    )

    assert _resolve_decision(
        local=local_zero_proof,
        round2="not_preserved",
        round1="preserved",
        simplification=True,
    ) == ("not_preserved", "round2")
    assert _resolve_decision(
        local=local_zero_proof,
        round2="not_equivalent",
        round1="not_equivalent",
        simplification=False,
    ) == ("equivalent", "local_strict_proof")


def test_abstention_resolution_is_not_a_major_conflict() -> None:
    assert _offline_severity(
        scope="prediction_formula",
        final_simplification="preserved",
        final_reference="not_equivalent",
        final_quality="strictly_simpler",
        stored_simplification="undetermined",
        stored_reference="not_equivalent",
    ) == ("pass", ["stored_simplification_abstention_resolved"])
    assert _offline_severity(
        scope="prediction_formula",
        final_simplification="preserved",
        final_reference="not_equivalent",
        final_quality="strictly_simpler",
        stored_simplification="preserved",
        stored_reference="undetermined",
    ) == ("pass", ["stored_equivalence_abstention_resolved"])


def test_finalize_lists_coverage_recovery_and_minor_diagnostics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = _read_jsonl(fixture["manifest"])
    recovered = next(row for row in manifest if row["algorithm"] == "alg_b")
    recovered["stored_simplification_outcome"] = "unable"
    recovered["stored_equivalence_decision"] = "undetermined"
    _write_jsonl(fixture["manifest"], manifest)

    retry = _add_retry_fixture(
        fixture,
        tmp_path,
        output=_model_output(reference="equivalent", quality="more_complex"),
    )
    report = finalize_formula_audit(
        round1_plan_jsonl=fixture["round1_plan"],
        round1_comparisons_jsonl=fixture["comparisons"],
        round1_state_db=fixture["round1_state"],
        round2_plan_jsonl=fixture["round2_plan"],
        round2_state_db=fixture["round2_state"],
        retry_plan_jsonl=retry["plan"],
        retry_state_db=retry["state"],
        round1_attempts_dir=fixture["round1_attempts"],
        round2_attempts_dir=fixture["round2_attempts"],
        retry_attempts_dir=retry["attempts"],
        local_verification_jsonl=fixture["local"],
        sample_manifest_jsonl=fixture["manifest"],
        external_attempt_count=0,
        maximum_total_physical_attempts=10,
        output_root=tmp_path / "coverage-and-minor",
    )

    final = _read_jsonl(report["outputs"]["final_jsonl"])
    coverage = next(row for row in final if row["algorithm"] == "alg_b")
    assert coverage["offline_severity"] == "pass"
    assert "coverage_recovery:stored_simplification_unable" in coverage["issue_types"]
    assert "coverage_recovery:stored_equivalence_undetermined" in coverage["issue_types"]
    minor = next(row for row in final if row["condition"] == "noise005")
    assert minor["offline_severity"] == "minor"
    assert "offline_severity:minor" in minor["issue_types"]
