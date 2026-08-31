from __future__ import annotations

import hashlib
import json
import sqlite3
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
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index import (  # noqa: E402
    FrozenResultIndexError,
    build_frozen_result_index,
    main,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_plan_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_kind_for(task_type: str) -> str:
    if task_type.endswith("simplify"):
        return "simplify"
    if task_type.endswith("equivalence"):
        return "equivalence"
    if task_type.endswith("structure"):
        return "structure"
    raise AssertionError(f"未知测试 task_type: {task_type}")


def _schema_for(task_kind: str) -> dict[str, Any]:
    if task_kind == "equivalence":
        return {
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
        }
    if task_kind == "structure":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string"},
                "confidence": {"type": "number"},
                "brief_reason": {"type": "string"},
            },
            "required": ["decision", "confidence", "brief_reason"],
        }
    raise AssertionError(f"当前测试未覆盖 task_kind={task_kind}")


def _valid_structured_output(task_kind: str) -> dict[str, Any]:
    if task_kind == "equivalence":
        return {
            "decision": "equivalent",
            "evidence_basis": "symbolic_proof",
            "assumptions": [],
            "confidence": 1.0,
            "brief_reason": "deterministic test fixture",
        }
    if task_kind == "structure":
        return {
            "decision": "same_canonical_structure",
            "confidence": 1.0,
            "brief_reason": "deterministic test fixture",
        }
    raise AssertionError(f"当前测试未覆盖 task_kind={task_kind}")


def _build_plan_row(
    tmp_path: Path,
    logical_id: str,
    *,
    task_type: str = "equivalence",
    priority: int = 1,
    request: dict[str, Any] | None = None,
    dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    task_kind = _task_kind_for(task_type)
    contract_dir = tmp_path / "contract" / logical_id.replace("::", "__")
    contract_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = contract_dir / f"{task_kind}.v1.txt"
    schema_path = contract_dir / f"{task_kind}.v1.json"
    prompt_template = f"Judge {task_kind}.\n{{{{REQUEST_JSON}}}}\n"
    schema_content = _schema_for(task_kind)
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(
        json.dumps(schema_content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prompt_sha256 = _sha256_file(prompt_path)
    schema_sha256 = _sha256_file(schema_path)
    request_payload = dict(
        request
        or {
            "lhs": "x0 + x1",
            "rhs": "x1 + x0",
            "evidence_hash": _sha256_text(f"evidence::{logical_id}"),
        }
    )
    normalized_input = {
        "request": dict(request_payload),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=f"{task_kind}.v1",
        schema_version=f"{task_kind}.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request_payload["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type=task_type,
        condition="clean",
        priority=priority,
        input_hash=input_hash,
        prompt_version=f"{task_kind}.v1",
        schema_version=f"{task_kind}.v1",
        dependencies=dependencies,
    )
    return {
        "evaluation_key": key,
        "logical_id": logical_id,
        "task_type": task_type,
        "task_kind": task_kind,
        "condition": "clean",
        "priority": priority,
        "input_hash": input_hash,
        "prompt_version": f"{task_kind}.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": f"{task_kind}.v1",
        "schema_sha256": schema_sha256,
        "dependencies": list(dependencies),
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_template": prompt_template,
        "schema_content": schema_content,
        "normalized_input": normalized_input,
        "request": request_payload,
        "task_spec": json.loads(spec.canonical_json()),
        "rendered_prompt": render_prompt(prompt_template, request_payload, schema_content),
    }


def _task_spec_from_plan_row(row: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        evaluation_key=str(row["evaluation_key"]),
        logical_id=str(row["logical_id"]),
        task_type=str(row["task_type"]),
        condition=str(row["condition"]),
        priority=int(row["priority"]),
        input_hash=str(row["input_hash"]),
        prompt_version=str(row["prompt_version"]),
        schema_version=str(row["schema_version"]),
        dependencies=tuple(row["dependencies"]),
    )


def _build_non_applicable_fixture(
    tmp_path: Path,
    *,
    logical_id: str,
    task_type: str = "stab_structure",
    reason: str = "invalid_seed_pair",
    payload_overrides: dict[str, Any] | None = None,
    write_evidence: bool = True,
) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    phase = "structure" if task_type.endswith("structure") else "equivalence"
    request_context = {"lhs": "x0 + x1", "rhs": "x1 + x0"}
    payload: dict[str, Any] = {
        "schema_version": "symbolic_non_applicable.v1",
        "logical_id": logical_id,
        "task_type": task_type,
        "phase": phase,
        "condition": "clean",
        "reason": reason,
        "dependencies": [],
        "request_context": request_context,
    }
    if payload_overrides:
        payload.update(payload_overrides)
    evidence_path = tmp_path / "audit" / f"{logical_id.replace('::', '__')}.json"
    evidence_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    evidence_sha256 = _sha256_text(evidence_text)
    if write_evidence:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(evidence_text, encoding="utf-8")
    row = _build_plan_row(
        tmp_path,
        logical_id,
        task_type=task_type,
        request={**request_context, "evidence_hash": evidence_sha256},
    )
    return row, evidence_path, evidence_sha256, payload


def _freeze_task(
    store: TaskStateStore,
    row: dict[str, Any],
    result_path: Path,
    *,
    structured_output: dict[str, Any] | None = None,
    payload_overrides: dict[str, Any] | None = None,
) -> None:
    spec = _task_spec_from_plan_row(row)
    store.register_task(spec)
    lease = store.reserve_attempt(spec.evaluation_key, now=1.0, lease_seconds=60.0)
    payload = {
        "attempt_id": lease.attempt_id,
        "evaluation_key": spec.evaluation_key,
        "logical_id": spec.logical_id,
        "task_type": spec.task_type,
        "task_kind": str(row["task_kind"]),
        "structured_output": structured_output or _valid_structured_output(str(row["task_kind"])),
    }
    if payload_overrides:
        payload.update(payload_overrides)
    _write_json(result_path, payload)
    store.freeze_result(
        lease.attempt_id,
        result_path=str(result_path),
        result_sha256=_sha256_file(result_path),
        now=2.0,
    )


def test_build_frozen_result_index_cli_writes_index_and_summary(tmp_path: Path) -> None:
    frozen_row = _build_plan_row(tmp_path, "equivalence::frozen", task_type="equivalence", priority=1)
    non_applicable_row, evidence_path, evidence_sha256, _ = _build_non_applicable_fixture(
        tmp_path,
        logical_id="structure::skipped",
        task_type="stab_structure",
    )
    plan_path = tmp_path / "plan.jsonl"
    plan_sha256 = _write_plan_jsonl(plan_path, [frozen_row, non_applicable_row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    _freeze_task(store, frozen_row, tmp_path / "frozen" / "equivalence.json")
    store.register_task(_task_spec_from_plan_row(non_applicable_row))
    store.mark_non_applicable(
        str(non_applicable_row["evaluation_key"]),
        reason="invalid_seed_pair",
        evidence_path=str(evidence_path),
        evidence_sha256=evidence_sha256,
        now=3.0,
    )

    output_jsonl = tmp_path / "frozen_index.jsonl"
    summary_json = tmp_path / "frozen_index.summary.json"
    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["logical_id"] for row in rows] == [
        "equivalence::frozen",
        "structure::skipped",
    ]
    assert rows[0]["state"] == "frozen"
    assert rows[0]["plan_sha256"] == plan_sha256
    assert rows[0]["result_sha256"] == _sha256_file(Path(rows[0]["result_path"]))
    assert rows[0]["structured_output"]["decision"] == "equivalent"
    assert rows[0]["non_applicable"] is None
    assert rows[1]["state"] == "non_applicable"
    assert rows[1]["structured_output"] is None
    assert rows[1]["non_applicable"] == {
        "reason": "invalid_seed_pair",
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha256,
    }

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary == {
        "output_jsonl": str(output_jsonl),
        "output_sha256": _sha256_file(output_jsonl),
        "plan_jsonl": str(plan_path),
        "plan_sha256": plan_sha256,
        "row_count": 2,
        "state_counts": {"frozen": 1, "non_applicable": 1},
        "state_db": str(state_db),
        "status": "ok",
    }


def test_duplicate_logical_id_or_evaluation_key_hard_fails(tmp_path: Path) -> None:
    duplicate_row = _build_plan_row(tmp_path, "equivalence::dup", task_type="equivalence")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [duplicate_row, duplicate_row])

    with pytest.raises(FrozenResultIndexError, match="重复"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=tmp_path / "state.sqlite3",
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_non_terminal_state_breaks_closed_loop(tmp_path: Path) -> None:
    row = _build_plan_row(tmp_path, "equivalence::pending", task_type="equivalence")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(_task_spec_from_plan_row(row))

    with pytest.raises(FrozenResultIndexError, match="闭环"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_sha_drift_hard_fails_without_writing_outputs(tmp_path: Path) -> None:
    row = _build_plan_row(tmp_path, "equivalence::drift", task_type="equivalence")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    frozen_path = tmp_path / "frozen" / "drift.json"
    _freeze_task(store, row, frozen_path)
    frozen_path.write_text("{\"tampered\":true}\n", encoding="utf-8")

    output_jsonl = tmp_path / "out.jsonl"
    summary_json = tmp_path / "summary.json"
    with pytest.raises(FrozenResultIndexError, match="SHA256"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=output_jsonl,
            summary_json=summary_json,
        )
    assert not output_jsonl.exists()
    assert not summary_json.exists()


def test_invalid_structured_output_hard_fails(tmp_path: Path) -> None:
    row = _build_plan_row(tmp_path, "equivalence::bad_output", task_type="equivalence")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    _freeze_task(
        store,
        row,
        tmp_path / "frozen" / "bad_output.json",
        structured_output={
            "decision": "equivalent",
            "evidence_basis": "symbolic_proof",
            "assumptions": [],
            "confidence": 1.0,
            "brief_reason": "bad fixture",
            "extra": True,
        },
    )

    with pytest.raises(FrozenResultIndexError, match="structured_output"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_non_applicable_requires_existing_evidence_file(tmp_path: Path) -> None:
    row, evidence_path, missing_evidence_sha256, _ = _build_non_applicable_fixture(
        tmp_path,
        logical_id="structure::missing_evidence",
        write_evidence=False,
    )
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(_task_spec_from_plan_row(row))
    store.mark_non_applicable(
        str(row["evaluation_key"]),
        reason="invalid_seed_pair",
        evidence_path=str(evidence_path),
        evidence_sha256=missing_evidence_sha256,
        now=3.0,
    )

    with pytest.raises(FrozenResultIndexError, match="证据文件不存在"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_non_applicable_evidence_sha_drift_hard_fails(tmp_path: Path) -> None:
    row, evidence_path, evidence_sha256, _ = _build_non_applicable_fixture(
        tmp_path,
        logical_id="structure::evidence_drift",
    )
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(_task_spec_from_plan_row(row))
    evidence_path.write_text("{\"tampered\":true}\n", encoding="utf-8")
    store.mark_non_applicable(
        str(row["evaluation_key"]),
        reason="invalid_seed_pair",
        evidence_path=str(evidence_path),
        evidence_sha256=evidence_sha256,
        now=3.0,
    )

    with pytest.raises(FrozenResultIndexError, match="证据文件 SHA256 漂移"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_missing_frozen_result_sha256_hard_fails(tmp_path: Path) -> None:
    row = _build_plan_row(tmp_path, "equivalence::missing_sha", task_type="equivalence")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    _freeze_task(store, row, tmp_path / "frozen" / "missing_sha.json")
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            "UPDATE frozen_results SET result_sha256 = '' WHERE evaluation_key = ?",
            (str(row["evaluation_key"]),),
        )
        connection.commit()

    with pytest.raises(FrozenResultIndexError, match="result_sha256"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_non_applicable_evidence_identity_drift_hard_fails_even_if_state_sha_is_synced(
    tmp_path: Path,
) -> None:
    row, evidence_path, original_sha256, original_payload = _build_non_applicable_fixture(
        tmp_path,
        logical_id="structure::identity_drift",
    )
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(_task_spec_from_plan_row(row))
    store.mark_non_applicable(
        str(row["evaluation_key"]),
        reason="invalid_seed_pair",
        evidence_path=str(evidence_path),
        evidence_sha256=original_sha256,
        now=3.0,
    )

    drifted_payload = {**original_payload, "reason": "tampered_reason"}
    _write_json(evidence_path, drifted_payload)
    drifted_sha256 = _sha256_file(evidence_path)
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            "UPDATE non_applicable_results SET evidence_sha256 = ? WHERE evaluation_key = ?",
            (drifted_sha256, str(row["evaluation_key"])),
        )
        connection.commit()

    with pytest.raises(FrozenResultIndexError, match="plan_request.evidence_hash"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_non_applicable_plan_evidence_hash_must_bind_actual_file(tmp_path: Path) -> None:
    original_row, evidence_path, evidence_sha256, _ = _build_non_applicable_fixture(
        tmp_path,
        logical_id="structure::plan_evidence_drift",
    )
    row = _build_plan_row(
        tmp_path,
        str(original_row["logical_id"]),
        task_type="stab_structure",
        request={
            "lhs": "x0 + x1",
            "rhs": "x1 + x0",
            "evidence_hash": "0" * 64,
        },
    )
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(_task_spec_from_plan_row(row))
    store.mark_non_applicable(
        str(row["evaluation_key"]),
        reason="invalid_seed_pair",
        evidence_path=str(evidence_path),
        evidence_sha256=evidence_sha256,
        now=3.0,
    )

    with pytest.raises(FrozenResultIndexError, match="plan_request.evidence_hash"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


@pytest.mark.parametrize(
    ("payload_overrides", "error_pattern"),
    [
        ({"reason": "tampered_reason"}, "evidence_json.reason"),
        ({"phase": "equivalence"}, "evidence_json.phase"),
        ({"dependencies": ["unexpected"]}, "evidence_json.dependencies"),
        ({"request_context": {"lhs": "tampered"}}, "evidence_json.request_context"),
    ],
)
def test_non_applicable_evidence_context_binding_hard_fails(
    tmp_path: Path,
    payload_overrides: dict[str, Any],
    error_pattern: str,
) -> None:
    row, evidence_path, evidence_sha256, _ = _build_non_applicable_fixture(
        tmp_path,
        logical_id="structure::context_drift",
        payload_overrides=payload_overrides,
    )
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(_task_spec_from_plan_row(row))
    store.mark_non_applicable(
        str(row["evaluation_key"]),
        reason="invalid_seed_pair",
        evidence_path=str(evidence_path),
        evidence_sha256=evidence_sha256,
        now=3.0,
    )

    with pytest.raises(FrozenResultIndexError, match=error_pattern):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )


def test_frozen_result_plan_sha_identity_drift_hard_fails(tmp_path: Path) -> None:
    row = _build_plan_row(tmp_path, "equivalence::plan_sha_drift", task_type="equivalence")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    _freeze_task(
        store,
        row,
        tmp_path / "frozen" / "plan_sha_drift.json",
        payload_overrides={"plan_sha256": "0" * 64},
    )

    with pytest.raises(FrozenResultIndexError, match="plan_sha256"):
        build_frozen_result_index(
            plan_jsonl=plan_path,
            state_db=state_db,
            output_jsonl=tmp_path / "out.jsonl",
            summary_json=tmp_path / "summary.json",
        )
