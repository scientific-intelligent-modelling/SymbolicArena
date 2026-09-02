from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_formula_audit import (  # noqa: E402
    aggregate_formula_audit,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (  # noqa: E402
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)


def _sha256_bytes(payload: bytes) -> str:
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


def _source_identity(logical_id: str, scope: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "source_formula_logical_id": f"source::{logical_id}",
        "source_formula_evaluation_key": _sha256_bytes(f"eval::{logical_id}".encode()),
        "source_formula_result_sha256": _sha256_bytes(f"result::{logical_id}".encode()),
    }
    if scope == "prediction_formula":
        identity.update(
            {
                "source_gt_artifact_sha256": _sha256_bytes(f"gt::{logical_id}".encode()),
                "source_pred_artifact_sha256": _sha256_bytes(f"pred::{logical_id}".encode()),
                "expression_resolution": "llm_simplified_expression",
            }
        )
    return identity


def _audit_binding(logical_id: str, scope: str, *, stored_reference: str) -> str:
    identity = _source_identity(logical_id, scope)
    if scope == "ground_truth_simplification":
        payload = {
            "source_evaluation_key": identity["source_formula_evaluation_key"],
            "source_result_sha256": identity["source_formula_result_sha256"],
            "stored_simplification_outcome": "simplified",
        }
    else:
        payload = {
            "source_prediction_evaluation_key": identity["source_formula_evaluation_key"],
            "source_prediction_result_sha256": identity["source_formula_result_sha256"],
            "source_gt_artifact_sha256": identity["source_gt_artifact_sha256"],
            "source_prediction_artifact_sha256": identity["source_pred_artifact_sha256"],
            "stored_simplification_outcome": "simplified",
            "expression_resolution": identity["expression_resolution"],
            "stored_equivalence_decision": stored_reference,
        }
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _plan_row(
    *,
    logical_id: str,
    scope: str,
    prompt_path: Path,
    schema_path: Path,
    stored_reference: str | None = None,
) -> dict[str, Any]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256_bytes(prompt_path.read_bytes())
    schema_sha256 = _sha256_bytes(schema_path.read_bytes())
    request_without_hash: dict[str, Any] = {
        "audit_scope": scope,
        "audit_binding_sha256": _audit_binding(
            logical_id,
            scope,
            stored_reference=stored_reference
            or ("not_applicable" if scope == "ground_truth_simplification" else "equivalent"),
        ),
        "variables": ["x0"],
        "allowed_functions": [],
        "domain_assumptions": {"variable_domain": "real"},
        "original_expression": "x0 + 0",
        "candidate_simplified_expression": "x0",
        "reference_simplified_expression": "x0" if scope == "prediction_formula" else None,
        "review_round": 1,
    }
    request = dict(request_without_hash)
    request["evidence_hash"] = _sha256_bytes(
        canonical_json(request_without_hash).encode("utf-8")
    )
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_bytes(canonical_json(normalized_input).encode("utf-8"))
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


def _output(
    *,
    simplification: str = "preserved",
    quality: str = "strictly_simpler",
    reference: str = "equivalent",
    severity: str = "pass",
    confidence: float = 0.95,
) -> dict[str, Any]:
    return {
        "simplification_equivalence": simplification,
        "simplification_quality": quality,
        "suggested_simplified_expression": None,
        "reference_equivalence": reference,
        "counterexample": "x0=1" if reference == "not_equivalent" else None,
        "severity": severity,
        "assumptions": [],
        "confidence": confidence,
        "brief_reason": "Synthetic audit result.",
    }


def _freeze(
    *,
    store: TaskStateStore,
    row: dict[str, Any],
    output: dict[str, Any],
    frozen_dir: Path,
    now: float,
) -> None:
    spec_payload = row["task_spec"]
    spec = TaskSpec(
        evaluation_key=spec_payload["evaluation_key"],
        logical_id=spec_payload["logical_id"],
        task_type=spec_payload["task_type"],
        condition=spec_payload["condition"],
        priority=spec_payload["priority"],
        input_hash=spec_payload["input_hash"],
        prompt_version=spec_payload["prompt_version"],
        schema_version=spec_payload["schema_version"],
        dependencies=tuple(spec_payload["dependencies"]),
    )
    store.register_task(spec, now=now)
    lease = store.reserve_attempt(spec.evaluation_key, now=now + 0.1)
    payload = {
        "attempt_id": lease.attempt_id,
        "evaluation_key": spec.evaluation_key,
        "logical_id": spec.logical_id,
        "task_type": spec.task_type,
        "task_kind": "formula_audit",
        "request": row["request"],
        "structured_output": output,
        "validation": {
            "ok": True,
            "error_class": None,
            "error_message": None,
            "structured_output": output,
        },
    }
    frozen_path = frozen_dir / f"{spec.evaluation_key}.json"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    store.freeze_result(
        lease.attempt_id,
        result_path=str(frozen_path),
        result_sha256=_sha256_bytes(frozen_path.read_bytes()),
        now=now + 0.2,
    )


def test_aggregate_recomputes_severity_and_builds_budgeted_blind_round2(
    tmp_path: Path,
) -> None:
    config_root = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    prompt_path = config_root / "prompts/formula_audit.v1.txt"
    schema_path = config_root / "schemas/formula_audit.v1.json"
    rows = [
        _plan_row(
            logical_id="formula_audit::ground_truth::g0001",
            scope="ground_truth_simplification",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
        _plan_row(
            logical_id="formula_audit::prediction::alg::g0001::s520::clean",
            scope="prediction_formula",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
        _plan_row(
            logical_id="formula_audit::prediction::alg::g0002::s520::clean",
            scope="prediction_formula",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
        _plan_row(
            logical_id="formula_audit::prediction::alg::g0003::s520::clean",
            scope="prediction_formula",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
    ]
    plan_path = tmp_path / "round1.jsonl"
    _write_jsonl(plan_path, rows)

    manifests = []
    for index, row in enumerate(rows):
        scope = row["request"]["audit_scope"]
        manifests.append(
            {
                "audit_logical_id": row["logical_id"],
                "audit_evaluation_key": row["evaluation_key"],
                "audit_scope": scope,
                "condition": "clean",
                "dataset_id": f"g{index + 1:04d}",
                "source_formula_logical_id": f"source::{index}",
                **_source_identity(str(row["logical_id"]), str(scope)),
                "stored_simplification_outcome": "simplified",
                "stored_equivalence_decision": (
                    "not_applicable" if scope == "ground_truth_simplification" else "equivalent"
                ),
            }
        )
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_path, manifests)

    outputs = [
        # 模型自报 critical 仅是证据；离线重算仍为 pass。
        _output(reference="not_applicable", severity="critical"),
        # stored equivalence 冲突，离线重算为 major。
        _output(reference="not_equivalent", severity="pass"),
        # 低置信度触发复判，但离线严重度仍为 pass。
        _output(confidence=0.5),
        # 没有触发条件；prediction 的模型自报 critical 不影响结论。
        _output(severity="critical"),
    ]
    state_path = tmp_path / "state.sqlite"
    store = TaskStateStore(
        state_path,
        attempt_cap=7,
        logical_task_cap=8,
        max_attempts_per_task=1,
    )
    for index, (row, output) in enumerate(zip(rows, outputs, strict=True)):
        _freeze(
            store=store,
            row=row,
            output=output,
            frozen_dir=tmp_path / "frozen",
            now=float(index + 1),
        )

    report = aggregate_formula_audit(
        round1_plan_jsonl=plan_path,
        sample_manifest_jsonl=manifest_path,
        state_db=state_path,
        output_root=tmp_path / "audit",
        expected_round1_task_count=4,
        maximum_total_attempts=7,
    )

    assert report["model_invoked"] is False
    assert report["budget"]["round1_initial_task_count"] == 4
    assert report["budget"]["round1_physical_attempt_count"] == 4
    assert report["budget"]["maximum_total_physical_attempts"] == 7
    assert report["budget"]["remaining_physical_attempt_budget"] == 3
    assert report["budget"]["round2_plan_count"] == 3
    assert report["budget"]["round2_budget_sufficient"] is True
    assert report["budget"]["round2_within_remaining_budget"] is True
    assert report["counts"]["round2_trigger_count"] == 3
    assert report["counts"]["round2_deferred_by_budget_count"] == 0
    comparisons = _read_jsonl(report["outputs"]["comparisons_jsonl"])
    assert comparisons[0]["model_reported_severity"] == "critical"
    assert comparisons[0]["offline_final_severity"] == "pass"
    assert comparisons[1]["offline_final_severity"] == "major"
    assert comparisons[3]["requires_round2"] is False

    round2_rows = _read_jsonl(report["outputs"]["round2_plan_jsonl"])
    assert len(round2_rows) == 3
    assert len(load_plan_jsonl(report["outputs"]["round2_plan_jsonl"]).entries) == 3
    round2_request = round2_rows[0]["request"]
    assert round2_request["review_round"] == 2
    assert round2_rows[0]["prompt_version"] == "formula_audit.v2"
    round1_by_base = {str(row["logical_id"]): row for row in rows}
    round1_base = str(round2_rows[0]["logical_id"]).removesuffix("::v2")
    assert (
        round2_request["audit_binding_sha256"]
        == round1_by_base[round1_base]["request"]["audit_binding_sha256"]
    )
    forbidden = {
        "stored_simplification_outcome",
        "stored_equivalence_decision",
        "round1_structured_output",
        "model_reported_severity",
        "offline_final_severity",
        "trigger_reasons",
    }
    assert not forbidden.intersection(round2_request)
    serialized_request = canonical_json(round2_request)
    assert "Synthetic audit result" not in serialized_request


def test_round2_triggers_all_required_uncertainty_and_simplification_cases(
    tmp_path: Path,
) -> None:
    config_root = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    row = _plan_row(
        logical_id="formula_audit::prediction::alg::g0001::s520::clean",
        scope="prediction_formula",
        prompt_path=config_root / "prompts/formula_audit.v1.txt",
        schema_path=config_root / "schemas/formula_audit.v1.json",
        stored_reference="undetermined",
    )
    plan_path = tmp_path / "round1.jsonl"
    _write_jsonl(plan_path, [row])
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "audit_logical_id": row["logical_id"],
                "audit_evaluation_key": row["evaluation_key"],
                "audit_scope": "prediction_formula",
                "condition": "clean",
                "dataset_id": "g0001",
                "source_formula_logical_id": "source::1",
                **_source_identity(str(row["logical_id"]), "prediction_formula"),
                "stored_simplification_outcome": "simplified",
                "stored_equivalence_decision": "undetermined",
            }
        ],
    )
    output = _output(
        simplification="not_preserved",
        quality="undetermined",
        reference="undetermined",
        severity="undetermined",
        confidence=0.79,
    )
    state_path = tmp_path / "state.sqlite"
    store = TaskStateStore(
        state_path,
        attempt_cap=2,
        logical_task_cap=2,
        max_attempts_per_task=1,
    )
    _freeze(
        store=store,
        row=row,
        output=output,
        frozen_dir=tmp_path / "frozen",
        now=1.0,
    )

    report = aggregate_formula_audit(
        round1_plan_jsonl=plan_path,
        sample_manifest_jsonl=manifest_path,
        state_db=state_path,
        output_root=tmp_path / "audit",
        expected_round1_task_count=1,
        maximum_total_attempts=2,
    )
    comparison = _read_jsonl(report["outputs"]["comparisons_jsonl"])[0]
    reasons = set(comparison["round2_trigger_reasons"])
    assert comparison["offline_final_severity"] == "critical"
    assert {
        "undetermined:simplification_quality",
        "undetermined:reference_equivalence",
        "undetermined:severity",
        "confidence_below_threshold",
        "simplification_not_preserved",
        "stored_simplification_conflict",
    }.issubset(reasons)


def test_attempt_cap_with_unfrozen_task_is_classified_as_budget_closure(
    tmp_path: Path,
) -> None:
    config_root = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    row = _plan_row(
        logical_id="formula_audit::prediction::alg::g0001::s520::clean",
        scope="prediction_formula",
        prompt_path=config_root / "prompts/formula_audit.v1.txt",
        schema_path=config_root / "schemas/formula_audit.v1.json",
    )
    plan_path = tmp_path / "round1.jsonl"
    _write_jsonl(plan_path, [row])
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "audit_logical_id": row["logical_id"],
                "audit_evaluation_key": row["evaluation_key"],
                "audit_scope": "prediction_formula",
                "condition": "clean",
                "dataset_id": "g0001",
                "source_formula_logical_id": "source::1",
                **_source_identity(str(row["logical_id"]), "prediction_formula"),
                "stored_simplification_outcome": "simplified",
                "stored_equivalence_decision": "equivalent",
            }
        ],
    )
    spec_payload = row["task_spec"]
    spec = TaskSpec(
        evaluation_key=spec_payload["evaluation_key"],
        logical_id=spec_payload["logical_id"],
        task_type=spec_payload["task_type"],
        condition=spec_payload["condition"],
        priority=spec_payload["priority"],
        input_hash=spec_payload["input_hash"],
        prompt_version=spec_payload["prompt_version"],
        schema_version=spec_payload["schema_version"],
        dependencies=tuple(spec_payload["dependencies"]),
    )
    state_path = tmp_path / "state.sqlite"
    store = TaskStateStore(
        state_path,
        attempt_cap=1,
        logical_task_cap=1,
        max_attempts_per_task=1,
    )
    store.register_task(spec, now=1.0)
    lease = store.reserve_attempt(spec.evaluation_key, now=1.1)
    assert (
        store.finish_failure(
            lease.attempt_id,
            error_class="api_http_transient",
            retryable=True,
            now=1.2,
        )
        == "exhausted"
    )

    report = aggregate_formula_audit(
        round1_plan_jsonl=plan_path,
        sample_manifest_jsonl=manifest_path,
        state_db=state_path,
        output_root=tmp_path / "audit",
        expected_round1_task_count=1,
        maximum_total_attempts=1,
    )
    assert report["status"] == "round2_budget_insufficient"
    assert (
        report["round1_execution_classification"]
        == "physical_budget_exhausted_with_unresolved_tasks"
    )
    assert report["budget"]["remaining_physical_attempt_budget"] == 0
    assert report["budget"]["round2_budget_shortfall"] == 1
    assert report["budget"]["round2_plan_withheld_on_shortfall"] is True
    assert report["counts"]["round2_trigger_count"] == 1
    assert report["counts"]["round2_planned_count"] == 0
    comparison = _read_jsonl(report["outputs"]["comparisons_jsonl"])[0]
    assert comparison["round1_state"] == "exhausted"
    assert comparison["offline_final_severity"] == "undetermined"
