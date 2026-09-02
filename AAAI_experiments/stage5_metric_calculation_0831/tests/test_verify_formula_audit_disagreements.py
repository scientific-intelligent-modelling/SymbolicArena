from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.verify_formula_audit_disagreements import (  # noqa: E402
    verify_formula_audit_disagreements,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def _plan_row(
    *,
    logical_id: str,
    original: str,
    candidate: str,
    reference: str,
    prompt_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256(prompt_path.read_bytes())
    schema_sha256 = _sha256(schema_path.read_bytes())
    request_without_hash: dict[str, Any] = {
        "audit_scope": "prediction_formula",
        "audit_binding_sha256": _sha256(logical_id.encode("utf-8")),
        "variables": ["x0"],
        "allowed_functions": [],
        "domain_assumptions": {"variable_domain": "real"},
        "original_expression": original,
        "candidate_simplified_expression": candidate,
        "reference_simplified_expression": reference,
        "review_round": 1,
    }
    request = dict(request_without_hash)
    request["evidence_hash"] = _sha256(
        canonical_json(request_without_hash).encode("utf-8")
    )
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256(canonical_json(normalized_input).encode("utf-8"))
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


def test_local_verification_is_compact_deterministic_and_covers_parse_errors(
    tmp_path: Path,
) -> None:
    config_root = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    prompt_path = config_root / "prompts/formula_audit.v1.txt"
    schema_path = config_root / "schemas/formula_audit.v1.json"
    rows = [
        _plan_row(
            logical_id="formula_audit::prediction::alg::g0001::s520::clean",
            original="x0 + 0",
            candidate="x0",
            reference="x0",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
        _plan_row(
            logical_id="formula_audit::prediction::alg::g0002::s520::clean",
            original="x0 + 0",
            candidate="x0",
            reference="x0 + 1",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
        _plan_row(
            logical_id="formula_audit::prediction::alg::g0003::s520::clean",
            original="x0",
            candidate="x0 + (",
            reference="x0",
            prompt_path=prompt_path,
            schema_path=schema_path,
        ),
    ]
    plan_path = tmp_path / "round1.jsonl"
    _write_jsonl(plan_path, rows)
    manifests = [
        {
            "audit_logical_id": row["logical_id"],
            "audit_evaluation_key": row["evaluation_key"],
            "audit_scope": "prediction_formula",
            "logical_key": f"source-key-{index}",
            "condition": "clean",
            "algorithm": "alg",
            "dataset_id": f"g{index:04d}",
            "seed": 520,
            "source_formula_logical_id": f"pred::{index}",
            "source_formula_evaluation_key": _sha256(f"pred::{index}".encode()),
            "source_formula_result_sha256": _sha256(f"result::{index}".encode()),
        }
        for index, row in enumerate(rows, start=1)
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_path, manifests)
    comparisons = [
        {
            "audit_logical_id": row["logical_id"],
            "round1_evaluation_key": row["evaluation_key"],
            "round1_state": "exhausted" if index == 3 else "frozen",
            "requires_round2": True,
        }
        for index, row in enumerate(rows, start=1)
    ]
    comparisons_path = tmp_path / "comparisons.jsonl"
    _write_jsonl(comparisons_path, comparisons)
    triggers = [
        {
            "audit_logical_id": row["logical_id"],
            "round1_evaluation_key": row["evaluation_key"],
            "trigger_reasons": ["fixture"],
            "selected_for_round2": True,
        }
        for row in rows
    ]
    triggers_path = tmp_path / "triggers.jsonl"
    _write_jsonl(triggers_path, triggers)

    first = verify_formula_audit_disagreements(
        round1_plan_jsonl=plan_path,
        sample_manifest_jsonl=manifest_path,
        comparisons_jsonl=comparisons_path,
        triggers_jsonl=triggers_path,
        output_jsonl=tmp_path / "first.jsonl",
    )
    second = verify_formula_audit_disagreements(
        round1_plan_jsonl=plan_path,
        sample_manifest_jsonl=manifest_path,
        comparisons_jsonl=comparisons_path,
        triggers_jsonl=triggers_path,
        output_jsonl=tmp_path / "second.jsonl",
        workers=2,
    )

    assert first["workers"] == 2
    assert first["verified_count"] == 3
    assert first["output_sha256"] == second["output_sha256"]
    output = _read_jsonl(first["output_jsonl"])
    assert output[0]["local_simplification"]["decision"] == "equivalent"
    assert output[0]["local_reference"]["decision"] == "equivalent"
    assert output[1]["local_simplification"]["decision"] == "equivalent"
    assert output[1]["local_reference"]["decision"] == "not_equivalent"
    assert output[1]["local_reference"]["counterexample"] is not None
    assert output[2]["local_simplification"]["decision"] == "error"
    assert output[2]["local_reference"]["decision"] == "error"
    assert output[2]["local_simplification"]["error"]["class"] == "SyntaxError"
    assert output[2]["source_identity"]["source_formula_logical_id"] == "pred::3"
    serialized = Path(first["output_jsonl"]).read_text(encoding="utf-8")
    assert "canonical_tree" not in serialized
    assert "numeric_probes" not in serialized
    assert "lhs_artifact" not in serialized
