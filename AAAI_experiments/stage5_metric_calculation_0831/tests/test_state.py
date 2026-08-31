from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskSpec,
    TaskStateStore,
)


def task(key: str, *, condition: str = "clean", priority: int = 10) -> TaskSpec:
    return TaskSpec(
        evaluation_key=key,
        logical_id=f"logical::{key}",
        task_type="pred_simplify",
        condition=condition,
        priority=priority,
        input_hash=f"input::{key}",
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )


def predecessor_manifest(tmp_path: Path, *, attempt_count: int) -> PredecessorAttemptManifest:
    manifest_path = tmp_path / f"predecessor-{attempt_count}.json"
    payload = {
        "attempt_count": attempt_count,
        "attempt_ids": [f"legacy::{index:05d}" for index in range(attempt_count)],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(raw)
    return PredecessorAttemptManifest(
        path=str(manifest_path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        attempt_count=attempt_count,
    )


def _table_count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_register_is_idempotent_but_rejects_contract_drift(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    store.register_task(task("a"))
    with pytest.raises(StateContractError):
        store.register_task(task("a", condition="noise001"))


def test_register_tasks_registers_once_and_is_restart_idempotent(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    specs = (task("a"), task("a"), task("b"))

    store.register_tasks(specs, now=12.0)
    store.register_tasks(specs, now=34.0)

    assert _table_count(state_db, "tasks") == 2
    assert _table_count(state_db, "events") == 2
    assert store.task_state("a") == "pending"
    assert store.task_state("b") == "pending"


def test_register_tasks_rejects_in_batch_identity_conflict_without_partial_write(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    conflicting = TaskSpec(
        evaluation_key="b",
        logical_id="logical::a",
        task_type="pred_simplify",
        condition="clean",
        priority=10,
        input_hash="input::b",
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )

    with pytest.raises(StateContractError, match="唯一标识冲突"):
        store.register_tasks((task("a"), conflicting), now=12.0)

    assert _table_count(state_db, "tasks") == 0
    assert _table_count(state_db, "events") == 0


def test_register_tasks_rolls_back_when_logical_task_cap_is_exceeded(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(
        state_db,
        attempt_cap=10,
        logical_task_cap=1,
    )

    with pytest.raises(StateContractError, match="逻辑任务预算"):
        store.register_tasks((task("a"), task("b")), now=12.0)

    assert _table_count(state_db, "tasks") == 0
    assert _table_count(state_db, "events") == 0


def test_register_tasks_rolls_back_when_existing_spec_drift_is_detected(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_task(task("a"), now=1.0)

    with pytest.raises(StateContractError, match="契约内容发生漂移"):
        store.register_tasks(
            (
                task("b"),
                task("a", condition="noise001"),
            ),
            now=2.0,
        )

    assert _table_count(state_db, "tasks") == 1
    assert _table_count(state_db, "events") == 1
    assert store.task_state("a") == "pending"


def test_retry_limit_is_initial_plus_two_retries(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10, max_attempts_per_task=3)
    store.register_task(task("a"))
    for expected in (1, 2, 3):
        lease = store.reserve_attempt("a", now=float(expected), lease_seconds=10)
        assert lease.attempt_number == expected
        store.finish_failure(
            lease.attempt_id,
            error_class="timeout",
            retryable=True,
            now=float(expected) + 0.5,
        )
    assert store.task_state("a") == "exhausted"
    with pytest.raises(StateContractError):
        store.reserve_attempt("a", now=10.0, lease_seconds=10)


def test_global_budget_is_reserved_before_process_start(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=2)
    store.register_task(task("a"))
    store.register_task(task("b"))
    first = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.finish_failure(first.attempt_id, error_class="cli_exit", retryable=True, now=2.0)
    second = store.reserve_attempt("b", now=3.0, lease_seconds=10)
    store.finish_failure(second.attempt_id, error_class="timeout", retryable=True, now=4.0)
    assert store.attempts_reserved() == 2
    with pytest.raises(StateContractError, match="全局尝试预算"):
        store.reserve_attempt("a", now=5.0, lease_seconds=10)


def test_predecessor_manifest_preserves_budget_across_revised_state_db(tmp_path: Path) -> None:
    manifest = predecessor_manifest(tmp_path, attempt_count=11)
    store = TaskStateStore(
        tmp_path / "state_v2.sqlite3",
        attempt_cap=12,
        predecessor_attempt_manifest=manifest,
    )
    store.register_task(task("a"))
    store.register_task(task("b"))
    lease = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.finish_failure(
        lease.attempt_id,
        error_class="cli_exit",
        retryable=True,
        now=2.0,
    )
    assert store.attempts_reserved() == 12
    with pytest.raises(StateContractError, match="全局尝试预算"):
        store.reserve_attempt("b", now=3.0, lease_seconds=10)

    reopened = TaskStateStore(
        tmp_path / "state_v2.sqlite3",
        attempt_cap=12,
        predecessor_attempt_manifest=manifest,
    )
    assert reopened.attempts_reserved() == 12

    with pytest.raises(StateContractError, match="继续传入同一 manifest"):
        TaskStateStore(
            tmp_path / "state_v2.sqlite3",
            attempt_cap=12,
        )

    with pytest.raises(StateContractError, match="attempt_count 不一致"):
        TaskStateStore(
            tmp_path / "state_v2.sqlite3",
            attempt_cap=12,
            attempt_offset=10,
            predecessor_attempt_manifest=manifest,
        )


def test_nonzero_legacy_offset_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(StateContractError, match="predecessor_attempt_manifest"):
        TaskStateStore(
            tmp_path / "state_v2.sqlite3",
            attempt_cap=12,
            attempt_offset=11,
        )


def test_logical_task_cap_is_enforced_at_registration(tmp_path: Path) -> None:
    store = TaskStateStore(
        tmp_path / "state.sqlite3",
        attempt_cap=10,
        logical_task_cap=1,
    )
    store.register_task(task("a"))
    with pytest.raises(StateContractError, match="逻辑任务预算"):
        store.register_task(task("b"))


def test_first_valid_result_is_frozen_and_never_retried(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    lease = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.freeze_result(
        lease.attempt_id,
        result_path="llm/frozen/a.json",
        result_sha256="abc",
        now=2.0,
    )
    assert store.task_state("a") == "frozen"
    assert store.frozen_result("a") == {
        "result_path": "llm/frozen/a.json",
        "result_sha256": "abc",
        "attempt_id": lease.attempt_id,
    }
    with pytest.raises(StateContractError):
        store.reserve_attempt("a", now=3.0, lease_seconds=10)


def test_expired_lease_consumes_attempt_and_can_resume(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    first = store.reserve_attempt("a", now=1.0, lease_seconds=2)
    assert store.recover_expired_leases(now=4.0) == [first.attempt_id]
    assert store.task_state("a") == "retry_wait"
    second = store.reserve_attempt("a", now=5.0, lease_seconds=2)
    assert second.attempt_number == 2


def test_clean_first_order_is_enforced_by_ready_queue(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("noise005", condition="noise005", priority=1))
    store.register_task(task("noise001", condition="noise001", priority=1))
    store.register_task(task("clean", condition="clean", priority=99))
    assert store.next_ready_key(allowed_conditions=("clean",)) == "clean"
    assert store.next_ready_key(allowed_conditions=("noise001",)) is None
    assert store.next_ready_key(allowed_conditions=("noise005",)) is None

    store.mark_non_applicable(
        "clean",
        reason="source_formula_missing",
        evidence_path="audit/clean.json",
        evidence_sha256="clean-sha",
        now=1.0,
    )
    assert store.next_ready_key(allowed_conditions=("noise001",)) == "noise001"
    assert store.next_ready_key(allowed_conditions=("noise005",)) is None

    store.mark_non_applicable(
        "noise001",
        reason="dependency_non_applicable",
        evidence_path="audit/noise001.json",
        evidence_sha256="noise001-sha",
        now=2.0,
    )
    assert store.next_ready_key(allowed_conditions=("noise005",)) == "noise005"


def test_clean_first_order_cannot_be_bypassed_by_direct_reservation(
    tmp_path: Path,
) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("clean", condition="clean"))
    store.register_task(task("noise001", condition="noise001"))
    store.register_task(task("noise005", condition="noise005"))

    with pytest.raises(StateContractError, match="条件尚未解锁"):
        store.reserve_attempt("noise001", now=1.0, lease_seconds=10)
    with pytest.raises(StateContractError, match="条件尚未解锁"):
        store.reserve_attempt("noise005", now=1.0, lease_seconds=10)
    assert store.attempts_reserved() == 0

    store.mark_non_applicable(
        "clean",
        reason="source_formula_missing",
        evidence_path="audit/clean.json",
        evidence_sha256="clean-sha",
        now=2.0,
    )
    noise001 = store.reserve_attempt("noise001", now=3.0, lease_seconds=10)
    with pytest.raises(StateContractError, match="条件尚未解锁"):
        store.reserve_attempt("noise005", now=3.0, lease_seconds=10)
    store.freeze_result(
        noise001.attempt_id,
        result_path="llm/frozen/noise001.json",
        result_sha256="noise001-sha",
        now=4.0,
    )

    noise005 = store.reserve_attempt("noise005", now=5.0, lease_seconds=10)
    assert noise005.attempt_number == 1


def test_lower_priority_phase_must_finish_before_direct_reservation(
    tmp_path: Path,
) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("gt", condition="clean", priority=10))
    store.register_task(task("pred", condition="clean", priority=20))

    with pytest.raises(StateContractError, match="优先级阶段尚未完成"):
        store.reserve_attempt("pred", now=1.0, lease_seconds=10)
    assert store.attempts_reserved() == 0

    store.mark_non_applicable(
        "gt",
        reason="source_formula_missing",
        evidence_path="audit/gt.json",
        evidence_sha256="gt-sha",
        now=2.0,
    )
    pred = store.reserve_attempt("pred", now=3.0, lease_seconds=10)
    assert pred.attempt_number == 1


def test_non_applicable_is_auditable_idempotent_and_costs_no_attempt(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    store.mark_non_applicable(
        "a",
        reason="invalid_seed_pair",
        evidence_path="audit/a.json",
        evidence_sha256="sha-a",
        now=1.0,
    )
    store.mark_non_applicable(
        "a",
        reason="invalid_seed_pair",
        evidence_path="audit/a.json",
        evidence_sha256="sha-a",
        now=2.0,
    )
    assert store.task_state("a") == "non_applicable"
    assert store.attempts_reserved() == 0
    assert store.non_applicable_result("a") == {
        "reason": "invalid_seed_pair",
        "evidence_path": "audit/a.json",
        "evidence_sha256": "sha-a",
    }


def test_promote_failed_attempt_freezes_whitelisted_archived_output(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    lease = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.finish_failure(
        lease.attempt_id,
        error_class="outer_json_invalid",
        retryable=True,
        now=2.0,
    )

    store.promote_failed_attempt(
        lease.attempt_id,
        result_path="llm/frozen/a.json",
        result_sha256="sha-a",
        allowed_error_classes=("outer_json_invalid",),
        audit_reason="validator_fix_rechecked",
        now=3.0,
    )

    assert store.task_state("a") == "frozen"
    assert store.frozen_result("a") == {
        "result_path": "llm/frozen/a.json",
        "result_sha256": "sha-a",
        "attempt_id": lease.attempt_id,
    }


def test_promote_failed_attempt_rejects_non_whitelisted_or_wrong_state(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    lease = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.finish_failure(
        lease.attempt_id,
        error_class="timeout",
        retryable=True,
        now=2.0,
    )
    with pytest.raises(StateContractError, match="不在补冻白名单内"):
        store.promote_failed_attempt(
            lease.attempt_id,
            result_path="llm/frozen/a.json",
            result_sha256="sha-a",
            allowed_error_classes=("outer_json_invalid",),
            audit_reason="validator_fix_rechecked",
            now=3.0,
        )

    accepted = TaskStateStore(tmp_path / "state_2.sqlite3", attempt_cap=10)
    accepted.register_task(task("b"))
    accepted_lease = accepted.reserve_attempt("b", now=1.0, lease_seconds=10)
    accepted.freeze_result(
        accepted_lease.attempt_id,
        result_path="llm/frozen/b.json",
        result_sha256="sha-b",
        now=2.0,
    )
    with pytest.raises(StateContractError, match="不是 failed"):
        accepted.promote_failed_attempt(
            accepted_lease.attempt_id,
            result_path="llm/frozen/b-2.json",
            result_sha256="sha-b2",
            allowed_error_classes=("timeout",),
            audit_reason="validator_fix_rechecked",
            now=3.0,
        )


def test_reopen_exhausted_failed_attempt_preserves_attempt_and_audits_reclassification(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10, max_attempts_per_task=3)
    store.register_task(task("a"))
    lease = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.finish_failure(
        lease.attempt_id,
        error_class="contract_drift",
        retryable=False,
        now=2.0,
    )
    assert store.task_state("a") == "exhausted"

    store.reopen_exhausted_failed_attempt(
        lease.attempt_id,
        allowed_error_classes=("contract_drift",),
        reclassified_error_class="turn_contract_violation",
        audit_reason="旧分类器把隔离的多轮运行时异常误判为永久契约漂移",
        now=3.0,
    )

    assert store.task_state("a") == "retry_wait"
    assert store.attempts_reserved() == 1
    retry = store.reserve_attempt("a", now=4.0, lease_seconds=10)
    assert retry.attempt_number == 2
    with sqlite3.connect(state_db) as connection:
        connection.row_factory = sqlite3.Row
        event = connection.execute(
            "SELECT event_type, details_json FROM events WHERE event_type='failed_attempt_reopened'"
        ).fetchone()
    assert event is not None
    assert event["event_type"] == "failed_attempt_reopened"
    details = json.loads(str(event["details_json"]))
    assert details["original_error_class"] == "contract_drift"
    assert details["reclassified_error_class"] == "turn_contract_violation"


def test_reopen_exhausted_failed_attempt_rejects_wrong_class_or_spent_budget(
    tmp_path: Path,
) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10, max_attempts_per_task=1)
    store.register_task(task("a"))
    lease = store.reserve_attempt("a", now=1.0, lease_seconds=10)
    store.finish_failure(
        lease.attempt_id,
        error_class="contract_drift",
        retryable=False,
        now=2.0,
    )
    with pytest.raises(StateContractError, match="不在重开白名单内"):
        store.reopen_exhausted_failed_attempt(
            lease.attempt_id,
            allowed_error_classes=("timeout",),
            reclassified_error_class="turn_contract_violation",
            audit_reason="test",
            now=3.0,
        )
    with pytest.raises(StateContractError, match="任务级尝试预算"):
        store.reopen_exhausted_failed_attempt(
            lease.attempt_id,
            allowed_error_classes=("contract_drift",),
            reclassified_error_class="turn_contract_violation",
            audit_reason="test",
            now=3.0,
        )
