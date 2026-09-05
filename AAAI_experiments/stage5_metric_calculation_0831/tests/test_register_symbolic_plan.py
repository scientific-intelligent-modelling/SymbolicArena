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
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.register_symbolic_plan import (  # noqa: E402
    RegisterSymbolicPlanError,
    main,
    register_symbolic_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    StateContractError,
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    return _sha256_file(path)


def _write_contract_files(tmp_path: Path, task_kind: str) -> tuple[Path, str, Path, str, dict[str, Any], str]:
    contract_dir = tmp_path / "contract" / task_kind
    contract_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = contract_dir / f"{task_kind}.v1.txt"
    schema_path = contract_dir / f"{task_kind}.v1.json"
    prompt_template = f"Judge {task_kind}.\n{{{{REQUEST_JSON}}}}\n"
    if task_kind == "equivalence":
        schema_content = {
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
    else:
        raise AssertionError(f"未覆盖 task_kind={task_kind}")
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(
        json.dumps(schema_content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        prompt_path,
        _sha256_file(prompt_path),
        schema_path,
        _sha256_file(schema_path),
        schema_content,
        prompt_template,
    )


def _build_task_row(
    tmp_path: Path,
    logical_id: str,
    *,
    task_type: str = "equivalence",
    priority: int = 30,
    request: dict[str, Any] | None = None,
    dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    task_kind = "equivalence"
    (
        prompt_path,
        prompt_sha256,
        schema_path,
        schema_sha256,
        schema_content,
        prompt_template,
    ) = _write_contract_files(tmp_path, task_kind)
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
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=str(request_payload["evidence_hash"]),
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type=task_type,
        condition="clean",
        priority=priority,
        input_hash=input_hash,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
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
        "prompt_version": "equivalence.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": "equivalence.v1",
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


def _build_non_applicable_row(
    tmp_path: Path,
    logical_id: str,
    *,
    task_type: str = "equivalence",
    priority: int = 30,
    phase: str = "equivalence",
    reason: str = "upstream_pred_unavailable",
    dependencies: tuple[str, ...] = (),
    evidence_payload_override: dict[str, Any] | None = None,
    skip_write_evidence: bool = False,
) -> dict[str, Any]:
    request_context = {
        "lhs": "x0 + x1",
        "rhs": "x1 + x0",
        "dataset_id": logical_id.replace("::", "__"),
    }
    evidence_payload = {
        "schema_version": "symbolic_non_applicable.v1",
        "logical_id": logical_id,
        "task_type": task_type,
        "phase": phase,
        "condition": "clean",
        "reason": reason,
        "dependencies": list(dependencies),
        "request_context": dict(request_context),
    }
    if evidence_payload_override:
        evidence_payload.update(evidence_payload_override)
    evidence_path = tmp_path / "evidence" / f"{logical_id.replace('::', '__')}.json"
    evidence_bytes = (canonical_json(evidence_payload) + "\n").encode("utf-8")
    if not skip_write_evidence:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_bytes(evidence_bytes)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    request = {**request_context, "evidence_hash": evidence_sha256}
    row = _build_task_row(
        tmp_path,
        logical_id,
        task_type=task_type,
        priority=priority,
        request=request,
        dependencies=dependencies,
    )
    row.update(
        {
            "phase": phase,
            "reason": reason,
            "evidence_hash": evidence_sha256,
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha256,
            "request_context": request_context,
            "evidence_payload": evidence_payload,
            "status": "planned_non_applicable",
        }
    )
    return row


def _table_count(path: Path, table: str) -> int:
    if not path.exists():
        return 0
    with sqlite3.connect(path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_register_symbolic_plan_uses_bulk_registration_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_row = _build_task_row(tmp_path, "equivalence::call")
    no_call_row = _build_non_applicable_row(tmp_path, "equivalence::skip")
    plan_path = tmp_path / "plan.jsonl"
    no_call_path = tmp_path / "no_call.jsonl"
    state_db = tmp_path / "control" / "state.sqlite3"
    report_json = tmp_path / "reports" / "register.json"
    _write_jsonl(plan_path, [call_row])
    _write_jsonl(no_call_path, [no_call_row])

    recorded_batches: list[list[str]] = []
    original_register_tasks = TaskStateStore.register_tasks

    def wrapped_register_tasks(self: TaskStateStore, specs: Any, *, now: float | None = None) -> None:
        spec_list = list(specs)
        recorded_batches.append([spec.logical_id for spec in spec_list])
        original_register_tasks(self, spec_list, now=now)

    def fail_register_task(self: TaskStateStore, spec: TaskSpec, *, now: float | None = None) -> None:
        raise AssertionError(f"不应逐条注册: {spec.logical_id}")

    monkeypatch.setattr(TaskStateStore, "register_tasks", wrapped_register_tasks)
    monkeypatch.setattr(TaskStateStore, "register_task", fail_register_task)

    first_exit = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--non-applicable-index-jsonl",
            str(no_call_path),
            "--state-db",
            str(state_db),
            "--report-json",
            str(report_json),
        ]
    )
    assert first_exit == 0
    first_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert first_report["counts"]["newly_registered_task_count"] == 2
    assert first_report["counts"]["newly_marked_non_applicable_count"] == 1
    assert first_report["counts"]["already_non_applicable_count"] == 0
    assert first_report["distributions"]["final_state"] == {"non_applicable": 1, "pending": 1}

    second_exit = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--non-applicable-index-jsonl",
            str(no_call_path),
            "--state-db",
            str(state_db),
            "--report-json",
            str(report_json),
        ]
    )
    assert second_exit == 0
    second_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert second_report["counts"]["newly_registered_task_count"] == 0
    assert second_report["counts"]["newly_marked_non_applicable_count"] == 0
    assert second_report["counts"]["already_non_applicable_count"] == 1
    assert recorded_batches == [
        ["equivalence::call", "equivalence::skip"],
        ["equivalence::call", "equivalence::skip"],
    ]

    store = TaskStateStore(state_db)
    assert store.task_state(call_row["evaluation_key"]) == "pending"
    assert store.task_state(no_call_row["evaluation_key"]) == "non_applicable"
    assert store.non_applicable_result(no_call_row["evaluation_key"]) == {
        "reason": "upstream_pred_unavailable",
        "evidence_path": str(Path(no_call_row["evidence_path"]).resolve()),
        "evidence_sha256": str(no_call_row["evidence_sha256"]),
    }
    assert _table_count(state_db, "tasks") == 2
    assert _table_count(state_db, "non_applicable_results") == 1


def test_register_symbolic_plan_rejects_cross_plan_overlap_without_state_write(tmp_path: Path) -> None:
    overlap_logical_id = "equivalence::dup"
    call_row = _build_task_row(tmp_path, overlap_logical_id)
    no_call_row = _build_non_applicable_row(tmp_path, overlap_logical_id)
    plan_path = tmp_path / "plan.jsonl"
    no_call_path = tmp_path / "no_call.jsonl"
    state_db = tmp_path / "control" / "state.sqlite3"
    _write_jsonl(plan_path, [call_row])
    _write_jsonl(no_call_path, [no_call_row])

    with pytest.raises(RegisterSymbolicPlanError, match="重叠"):
        register_symbolic_plan(
            plan_jsonl=plan_path,
            non_applicable_index_jsonl=no_call_path,
            state_db=state_db,
        )

    assert not state_db.exists()


def test_register_symbolic_plan_reuses_existing_state_limits(tmp_path: Path) -> None:
    call_row = _build_task_row(tmp_path, "equivalence::extended-retry-state")
    plan_path = tmp_path / "plan.jsonl"
    no_call_path = tmp_path / "no_call.jsonl"
    state_db = tmp_path / "control" / "state.sqlite3"
    _write_jsonl(plan_path, [call_row])
    _write_jsonl(no_call_path, [])
    TaskStateStore(
        state_db,
        attempt_cap=101,
        logical_task_cap=17,
        max_attempts_per_task=5,
    )

    report = register_symbolic_plan(
        plan_jsonl=plan_path,
        non_applicable_index_jsonl=no_call_path,
        state_db=state_db,
    )

    assert report["counts"]["newly_registered_task_count"] == 1
    with sqlite3.connect(state_db) as connection:
        meta = dict(connection.execute("SELECT key, value FROM meta"))
    assert meta["attempt_cap"] == "101"
    assert meta["logical_task_cap"] == "17"
    assert meta["max_attempts_per_task"] == "5"


def test_register_symbolic_plan_rejects_duplicate_non_applicable_identity_without_state_write(
    tmp_path: Path,
) -> None:
    row = _build_non_applicable_row(tmp_path, "equivalence::dup")
    duplicated = json.loads(canonical_json(row))
    plan_path = tmp_path / "plan.jsonl"
    no_call_path = tmp_path / "no_call.jsonl"
    state_db = tmp_path / "control" / "state.sqlite3"
    _write_jsonl(plan_path, [])
    _write_jsonl(no_call_path, [row, duplicated])

    with pytest.raises(RegisterSymbolicPlanError, match="重复 evaluation_key"):
        register_symbolic_plan(
            plan_jsonl=plan_path,
            non_applicable_index_jsonl=no_call_path,
            state_db=state_db,
        )

    assert not state_db.exists()


def test_register_symbolic_plan_rejects_missing_evidence_and_writes_failure_report(
    tmp_path: Path,
) -> None:
    call_row = _build_task_row(tmp_path, "equivalence::call")
    no_call_row = _build_non_applicable_row(
        tmp_path,
        "equivalence::missing-evidence",
        skip_write_evidence=True,
    )
    plan_path = tmp_path / "plan.jsonl"
    no_call_path = tmp_path / "no_call.jsonl"
    state_db = tmp_path / "control" / "state.sqlite3"
    report_json = tmp_path / "reports" / "failure.json"
    _write_jsonl(plan_path, [call_row])
    _write_jsonl(no_call_path, [no_call_row])

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--non-applicable-index-jsonl",
            str(no_call_path),
            "--state-db",
            str(state_db),
            "--report-json",
            str(report_json),
        ]
    )

    assert exit_code == 2
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "registration_contract_error"
    assert report["model_invoked"] is False
    assert "证据文件不存在" in report["error"]
    assert not state_db.exists()


def test_register_symbolic_plan_preflights_state_conflict_before_bulk_registration(
    tmp_path: Path,
) -> None:
    call_row = _build_task_row(tmp_path, "equivalence::new-call")
    no_call_row = _build_non_applicable_row(tmp_path, "equivalence::already-frozen")
    plan_path = tmp_path / "plan.jsonl"
    no_call_path = tmp_path / "no_call.jsonl"
    state_db = tmp_path / "control" / "state.sqlite3"
    _write_jsonl(plan_path, [call_row])
    _write_jsonl(no_call_path, [no_call_row])

    store = TaskStateStore(state_db)
    frozen_spec = TaskSpec(**json.loads(canonical_json(no_call_row["task_spec"])))
    store.register_task(frozen_spec, now=1.0)
    lease = store.reserve_attempt(frozen_spec.evaluation_key, now=2.0, lease_seconds=60.0)
    frozen_path = tmp_path / "frozen" / f"{frozen_spec.evaluation_key}.json"
    _write_json(
        frozen_path,
        {
            "attempt_id": lease.attempt_id,
            "evaluation_key": frozen_spec.evaluation_key,
            "logical_id": frozen_spec.logical_id,
            "task_type": frozen_spec.task_type,
            "task_kind": "equivalence",
            "structured_output": {
                "decision": "equivalent",
                "evidence_basis": "symbolic_proof",
                "assumptions": [],
                "confidence": 1.0,
                "brief_reason": "fixture",
            },
        },
    )
    store.freeze_result(
        lease.attempt_id,
        result_path=str(frozen_path),
        result_sha256=_sha256_file(frozen_path),
        now=3.0,
    )

    with pytest.raises(StateContractError, match="不可标记 non_applicable"):
        register_symbolic_plan(
            plan_jsonl=plan_path,
            non_applicable_index_jsonl=no_call_path,
            state_db=state_db,
        )

    assert _table_count(state_db, "tasks") == 1
    with sqlite3.connect(state_db) as connection:
        rows = connection.execute("SELECT logical_id FROM tasks ORDER BY logical_id").fetchall()
    assert [row[0] for row in rows] == [frozen_spec.logical_id]
