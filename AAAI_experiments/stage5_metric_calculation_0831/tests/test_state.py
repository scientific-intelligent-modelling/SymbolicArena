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
    TaskSupersession,
    TaskStateStore,
)


def task(
    key: str,
    *,
    condition: str = "clean",
    priority: int = 10,
    dependencies: tuple[str, ...] = (),
    prompt_version: str = "simplify.v1",
    schema_version: str = "simplify.v1",
) -> TaskSpec:
    return TaskSpec(
        evaluation_key=key,
        logical_id=f"logical::{key}",
        task_type="pred_simplify",
        condition=condition,
        priority=priority,
        input_hash=f"input::{key}",
        prompt_version=prompt_version,
        schema_version=schema_version,
        dependencies=dependencies,
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


def supersession(predecessor_key: str, successor: TaskSpec) -> TaskSupersession:
    return TaskSupersession(
        predecessor_evaluation_key=predecessor_key,
        successor=successor,
        identity=f"dataset::{predecessor_key}",
        reason="rerun_with_revised_prompt",
        predecessor_plan_sha256=f"plan-prev::{predecessor_key}",
        successor_plan_sha256=f"plan-next::{successor.evaluation_key}",
    )


def _table_count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _write_legacy_state_v2_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE tasks (
                evaluation_key TEXT PRIMARY KEY,
                logical_id TEXT NOT NULL UNIQUE,
                task_type TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                priority INTEGER NOT NULL,
                input_hash TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_expires_at REAL,
                last_error_class TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX idx_tasks_ready
                ON tasks(state, condition_name, priority, logical_id);
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                evaluation_key TEXT NOT NULL REFERENCES tasks(evaluation_key),
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                reserved_at REAL NOT NULL,
                lease_expires_at REAL NOT NULL,
                finished_at REAL,
                error_class TEXT,
                retryable INTEGER,
                UNIQUE(evaluation_key, attempt_number)
            );
            CREATE TABLE frozen_results (
                evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
                result_path TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                frozen_at REAL NOT NULL
            );
            CREATE TABLE non_applicable_results (
                evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                reason TEXT NOT NULL,
                evidence_path TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                marked_at REAL NOT NULL
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_key TEXT,
                attempt_id TEXT,
                event_type TEXT NOT NULL,
                event_at REAL NOT NULL,
                details_json TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            (
                ("attempt_cap", "10"),
                ("logical_task_cap", "10"),
                ("max_attempts_per_task", "3"),
                ("attempt_offset", "0"),
                ("predecessor_attempt_manifest_path", ""),
                ("predecessor_attempt_manifest_sha256", ""),
                ("predecessor_attempt_count", "0"),
                ("schema_version", "state.v2"),
            ),
        )
        connection.commit()


def _write_state_v3_without_identity_unique_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE tasks (
                evaluation_key TEXT PRIMARY KEY,
                logical_id TEXT NOT NULL UNIQUE,
                task_type TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                priority INTEGER NOT NULL,
                input_hash TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_expires_at REAL,
                last_error_class TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX idx_tasks_ready
                ON tasks(state, condition_name, priority, logical_id);
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                evaluation_key TEXT NOT NULL REFERENCES tasks(evaluation_key),
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                reserved_at REAL NOT NULL,
                lease_expires_at REAL NOT NULL,
                finished_at REAL,
                error_class TEXT,
                retryable INTEGER,
                UNIQUE(evaluation_key, attempt_number)
            );
            CREATE TABLE frozen_results (
                evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
                result_path TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                frozen_at REAL NOT NULL
            );
            CREATE TABLE non_applicable_results (
                evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                reason TEXT NOT NULL,
                evidence_path TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                marked_at REAL NOT NULL
            );
            CREATE TABLE task_supersessions (
                predecessor_evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                successor_evaluation_key TEXT NOT NULL UNIQUE REFERENCES tasks(evaluation_key),
                predecessor_logical_id TEXT NOT NULL UNIQUE,
                successor_logical_id TEXT NOT NULL UNIQUE,
                identity TEXT NOT NULL,
                reason TEXT NOT NULL,
                predecessor_plan_sha256 TEXT NOT NULL,
                successor_plan_sha256 TEXT NOT NULL,
                superseded_at REAL NOT NULL
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_key TEXT,
                attempt_id TEXT,
                event_type TEXT NOT NULL,
                event_at REAL NOT NULL,
                details_json TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            (
                ("attempt_cap", "10"),
                ("logical_task_cap", "10"),
                ("max_attempts_per_task", "3"),
                ("attempt_offset", "0"),
                ("predecessor_attempt_manifest_path", ""),
                ("predecessor_attempt_manifest_sha256", ""),
                ("predecessor_attempt_count", "0"),
                ("schema_version", "state.v3"),
            ),
        )
        connection.commit()


def test_register_is_idempotent_but_rejects_contract_drift(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    store.register_task(task("a"))
    with pytest.raises(StateContractError):
        store.register_task(task("a", condition="noise001"))


def test_existing_state_v2_requires_explicit_schema_upgrade(tmp_path: Path) -> None:
    state_db = tmp_path / "state_v2.sqlite3"
    _write_legacy_state_v2_db(state_db)

    with pytest.raises(StateContractError, match="allow_schema_upgrade=True"):
        TaskStateStore(state_db, attempt_cap=10)

    with sqlite3.connect(state_db) as connection:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert meta["schema_version"] == "state.v2"
    assert "task_supersessions" not in tables

    upgraded = TaskStateStore(
        state_db,
        attempt_cap=10,
        logical_task_cap=10,
        max_attempts_per_task=3,
        allow_schema_upgrade=True,
    )
    upgraded.register_task(task("a"), now=1.0)
    with sqlite3.connect(state_db) as connection:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert meta["schema_version"] == "state.v3"
    assert "task_supersessions" in tables


def test_schema_upgrade_rejects_running_legacy_state_without_side_effects(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state_v2_running.sqlite3"
    _write_legacy_state_v2_db(state_db)
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            """INSERT INTO tasks(
                   evaluation_key, logical_id, task_type, condition_name, priority,
                   input_hash, prompt_version, schema_version, dependencies_json,
                   spec_json, state, attempt_count, lease_expires_at, last_error_class,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "legacy",
                "logical::legacy",
                "pred_simplify",
                "clean",
                10,
                "input::legacy",
                "simplify.v1",
                "simplify.v1",
                "[]",
                task("legacy").canonical_json(),
                "running",
                1,
                20.0,
                None,
                1.0,
                1.0,
            ),
        )
        connection.execute(
            """INSERT INTO attempts(
                   attempt_id, evaluation_key, attempt_number, status,
                   reserved_at, lease_expires_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("legacy.a01", "legacy", 1, "running", 1.0, 20.0),
        )
        connection.commit()

    with pytest.raises(StateContractError, match="running tasks/attempts"):
        TaskStateStore(
            state_db,
            attempt_cap=10,
            logical_task_cap=10,
            max_attempts_per_task=3,
            allow_schema_upgrade=True,
        )

    with sqlite3.connect(state_db) as connection:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert meta["schema_version"] == "state.v2"
    assert "task_supersessions" not in tables


def test_state_v3_identity_unique_schema_requires_explicit_upgrade(tmp_path: Path) -> None:
    state_db = tmp_path / "state_v3.sqlite3"
    _write_state_v3_without_identity_unique_db(state_db)

    with pytest.raises(StateContractError, match="全局唯一 supersession identity"):
        TaskStateStore(state_db, attempt_cap=10)

    upgraded = TaskStateStore(
        state_db,
        attempt_cap=10,
        logical_task_cap=10,
        max_attempts_per_task=3,
        allow_schema_upgrade=True,
    )
    upgraded.register_task(task("a"), now=1.0)

    with sqlite3.connect(state_db) as connection:
        connection.row_factory = sqlite3.Row
        indexes = connection.execute("PRAGMA index_list('task_supersessions')").fetchall()
    assert any(
        int(row["unique"]) == 1 and str(row["name"]) == "idx_task_supersessions_identity_unique"
        for row in indexes
    )


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


def test_supersession_batch_is_atomic_idempotent_and_tracks_summary(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10, logical_task_cap=2)
    predecessor = task("gt_v1")
    store.register_task(predecessor, now=1.0)
    lease = store.reserve_attempt("gt_v1", now=2.0, lease_seconds=10)
    store.freeze_result(
        lease.attempt_id,
        result_path="llm/frozen/gt_v1.json",
        result_sha256="sha-gt-v1",
        now=3.0,
    )
    successor = TaskSpec(
        evaluation_key="gt_v2",
        logical_id="logical::gt_v2",
        task_type=predecessor.task_type,
        condition=predecessor.condition,
        priority=predecessor.priority,
        input_hash=predecessor.input_hash,
        prompt_version="simplify.v2",
        schema_version="simplify.v2",
        dependencies=predecessor.dependencies,
    )

    store.register_supersession_batch(
        (
            TaskSupersession(
                predecessor_evaluation_key="gt_v1",
                successor=successor,
                identity="dataset::gt",
                reason="rerun_with_revised_prompt",
                predecessor_plan_sha256="plan-prev::gt_v1",
                successor_plan_sha256="plan-next::gt_v2",
            ),
        ),
        now=4.0,
    )
    store.register_supersession_batch(
        (
            TaskSupersession(
                predecessor_evaluation_key="gt_v1",
                successor=successor,
                identity="dataset::gt",
                reason="rerun_with_revised_prompt",
                predecessor_plan_sha256="plan-prev::gt_v1",
                successor_plan_sha256="plan-next::gt_v2",
            ),
        ),
        now=5.0,
    )

    assert store.task_state("gt_v1") == "superseded"
    assert store.task_state("gt_v2") == "pending"
    summary = store.state_summary()
    assert summary["logical_tasks"] == {
        "active": 1,
        "historical": 2,
        "superseded": 1,
        "state_counts": {"pending": 1, "superseded": 1},
    }
    assert summary["attempts"] == {
        "physical": 1,
        "reserved": 1,
        "offset": 0,
    }
    with sqlite3.connect(state_db) as connection:
        connection.row_factory = sqlite3.Row
        supersession_rows = connection.execute(
            "SELECT * FROM task_supersessions"
        ).fetchall()
        superseded_events = connection.execute(
            "SELECT COUNT(*) AS count FROM events WHERE event_type='task_superseded'"
        ).fetchone()
    assert len(supersession_rows) == 1
    assert dict(supersession_rows[0])["predecessor_evaluation_key"] == "gt_v1"
    assert dict(supersession_rows[0])["successor_evaluation_key"] == "gt_v2"
    assert dict(supersession_rows[0])["identity"] == "dataset::gt"
    assert superseded_events is not None
    assert int(superseded_events["count"]) == 1


def test_supersession_batch_rejects_drift_and_rolls_back_entire_batch(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    for key in ("a_v1", "b_v1"):
        spec = task(key)
        store.register_task(spec, now=1.0)
        lease = store.reserve_attempt(key, now=2.0, lease_seconds=10)
        store.freeze_result(
            lease.attempt_id,
            result_path=f"llm/frozen/{key}.json",
            result_sha256=f"sha-{key}",
            now=3.0,
        )

    with pytest.raises(StateContractError, match="identity 不一致"):
        store.register_supersession_batch(
            (
                TaskSupersession(
                    predecessor_evaluation_key="a_v1",
                    successor=TaskSpec(
                        evaluation_key="a_v2",
                        logical_id="logical::a_v2",
                        task_type="pred_simplify",
                        condition="clean",
                        priority=10,
                        input_hash="input::a_v1",
                        prompt_version="simplify.v2",
                        schema_version="simplify.v2",
                        dependencies=(),
                    ),
                    identity="dataset::a_v1",
                    reason="rerun_with_revised_prompt",
                    predecessor_plan_sha256="plan-prev::a_v1",
                    successor_plan_sha256="plan-next::a_v2",
                ),
                TaskSupersession(
                    predecessor_evaluation_key="b_v1",
                    successor=TaskSpec(
                        evaluation_key="b_v2",
                        logical_id="logical::b_v2",
                        task_type="pred_simplify",
                        condition="noise001",
                        priority=10,
                        input_hash="input::b_v1",
                        prompt_version="simplify.v2",
                        schema_version="simplify.v2",
                        dependencies=(),
                    ),
                    identity="dataset::b_v1",
                    reason="rerun_with_revised_prompt",
                    predecessor_plan_sha256="plan-prev::b_v1",
                    successor_plan_sha256="plan-next::b_v2",
                ),
            ),
            now=4.0,
        )

    assert store.task_state("a_v1") == "frozen"
    assert store.task_state("b_v1") == "frozen"
    assert _table_count(state_db, "tasks") == 2
    assert _table_count(state_db, "task_supersessions") == 0
    assert _table_count(state_db, "events") == 6


def test_supersession_allows_input_hash_change_but_rejects_duplicate_batch_identity(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    for key in ("x_v1", "y_v1"):
        store.register_task(task(key), now=1.0)
        lease = store.reserve_attempt(key, now=2.0, lease_seconds=10)
        store.freeze_result(
            lease.attempt_id,
            result_path=f"llm/frozen/{key}.json",
            result_sha256=f"sha-{key}",
            now=3.0,
        )

    store.register_supersession(
        "x_v1",
        TaskSpec(
            evaluation_key="x_v2",
            logical_id="logical::x_v2",
            task_type="pred_simplify",
            condition="clean",
            priority=10,
            input_hash="input::x_v2_changed",
            prompt_version="simplify.v2",
            schema_version="simplify.v2",
            dependencies=(),
        ),
        identity="dataset::shared-x",
        reason="rerun_with_revised_prompt",
        predecessor_plan_sha256="plan-prev::x_v1",
        successor_plan_sha256="plan-next::x_v2",
        now=4.0,
    )
    assert store.task_state("x_v1") == "superseded"
    assert store.task_state("x_v2") == "pending"

    with pytest.raises(StateContractError, match="identity 重复"):
        store.register_supersession_batch(
            (
                TaskSupersession(
                    predecessor_evaluation_key="y_v1",
                    successor=TaskSpec(
                        evaluation_key="y_v2",
                        logical_id="logical::y_v2",
                        task_type="pred_simplify",
                        condition="clean",
                        priority=10,
                        input_hash="input::y_v2_changed",
                        prompt_version="simplify.v2",
                        schema_version="simplify.v2",
                        dependencies=(),
                    ),
                    identity="dataset::dup",
                    reason="rerun_with_revised_prompt",
                    predecessor_plan_sha256="plan-prev::y_v1",
                    successor_plan_sha256="plan-next::y_v2",
                ),
                TaskSupersession(
                    predecessor_evaluation_key="x_v1",
                    successor=TaskSpec(
                        evaluation_key="x_v3",
                        logical_id="logical::x_v3",
                        task_type="pred_simplify",
                        condition="clean",
                        priority=10,
                        input_hash="input::x_v3_changed",
                        prompt_version="simplify.v3",
                        schema_version="simplify.v3",
                        dependencies=(),
                    ),
                    identity="dataset::dup",
                    reason="rerun_with_revised_prompt",
                    predecessor_plan_sha256="plan-prev::x_v1",
                    successor_plan_sha256="plan-next::x_v3",
                ),
            ),
            now=5.0,
        )


def test_supersession_rejects_cross_batch_duplicate_identity(tmp_path: Path) -> None:
    state_db = tmp_path / "cross_batch_identity.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    for key in ("x_v1", "y_v1"):
        store.register_task(task(key), now=1.0)
        lease = store.reserve_attempt(key, now=2.0, lease_seconds=10)
        store.freeze_result(
            lease.attempt_id,
            result_path=f"llm/frozen/{key}.json",
            result_sha256=f"sha-{key}",
            now=3.0,
        )

    store.register_supersession(
        "x_v1",
        TaskSpec(
            evaluation_key="x_v2",
            logical_id="logical::x_v2",
            task_type="pred_simplify",
            condition="clean",
            priority=10,
            input_hash="input::x_v2_changed",
            prompt_version="simplify.v2",
            schema_version="simplify.v2",
            dependencies=(),
        ),
        identity="dataset::shared",
        reason="rerun_with_revised_prompt",
        predecessor_plan_sha256="plan-prev::x_v1",
        successor_plan_sha256="plan-next::x_v2",
        now=4.0,
    )

    with pytest.raises(StateContractError, match="identity 重复"):
        store.register_supersession(
            "y_v1",
            TaskSpec(
                evaluation_key="y_v2",
                logical_id="logical::y_v2",
                task_type="pred_simplify",
                condition="clean",
                priority=10,
                input_hash="input::y_v2_changed",
                prompt_version="simplify.v2",
                schema_version="simplify.v2",
                dependencies=(),
            ),
            identity="dataset::shared",
            reason="rerun_with_revised_prompt",
            predecessor_plan_sha256="plan-prev::y_v1",
            successor_plan_sha256="plan-next::y_v2",
            now=5.0,
        )

    assert store.task_state("y_v1") == "frozen"
    assert _table_count(state_db, "task_supersessions") == 1


def test_supersession_requires_frozen_binding_no_running_attempt_and_no_active_dependents(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    predecessor = task("base")
    dependent = task("downstream", dependencies=("base",), priority=20)
    store.register_tasks((predecessor, dependent), now=1.0)
    lease = store.reserve_attempt("base", now=2.0, lease_seconds=10)
    store.freeze_result(
        lease.attempt_id,
        result_path="llm/frozen/base.json",
        result_sha256="sha-base",
        now=3.0,
    )

    with pytest.raises(StateContractError, match="active dependents"):
        store.register_supersession(
            "base",
            TaskSpec(
                evaluation_key="base_v2",
                logical_id="logical::base_v2",
                task_type=predecessor.task_type,
                condition=predecessor.condition,
                priority=predecessor.priority,
                input_hash=predecessor.input_hash,
                prompt_version="simplify.v2",
                schema_version="simplify.v2",
                dependencies=predecessor.dependencies,
            ),
            identity="dataset::base",
            reason="rerun_with_revised_prompt",
            predecessor_plan_sha256="plan-prev::base",
            successor_plan_sha256="plan-next::base_v2",
            now=4.0,
        )

    store.mark_non_applicable(
        "downstream",
        reason="legacy_branch_retired",
        evidence_path="audit/downstream.json",
        evidence_sha256="sha-downstream",
        now=5.0,
    )
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            """INSERT INTO attempts(
                   attempt_id, evaluation_key, attempt_number, status,
                   reserved_at, lease_expires_at
               ) VALUES (?, ?, ?, 'running', ?, ?)""",
            ("base.manual.a99", "base", 99, 6.0, 16.0),
        )
        connection.commit()
    with pytest.raises(StateContractError, match="running attempt"):
        store.register_supersession(
            "base",
            TaskSpec(
                evaluation_key="base_v2",
                logical_id="logical::base_v2",
                task_type=predecessor.task_type,
                condition=predecessor.condition,
                priority=predecessor.priority,
                input_hash=predecessor.input_hash,
                prompt_version="simplify.v2",
                schema_version="simplify.v2",
                dependencies=predecessor.dependencies,
            ),
            identity="dataset::base",
            reason="rerun_with_revised_prompt",
            predecessor_plan_sha256="plan-prev::base",
            successor_plan_sha256="plan-next::base_v2",
            now=7.0,
        )

    pending = TaskStateStore(tmp_path / "pending.sqlite3", attempt_cap=10)
    pending.register_task(task("pending"), now=1.0)
    with pytest.raises(StateContractError, match="不可 supersede"):
        pending.register_supersession(
            "pending",
            TaskSpec(
                evaluation_key="pending_v2",
                logical_id="logical::pending_v2",
                task_type="pred_simplify",
                condition="clean",
                priority=10,
                input_hash="input::pending",
                prompt_version="simplify.v2",
                schema_version="simplify.v2",
                dependencies=(),
            ),
            identity="dataset::pending",
            reason="rerun_with_revised_prompt",
            predecessor_plan_sha256="plan-prev::pending",
            successor_plan_sha256="plan-next::pending_v2",
            now=2.0,
        )


def test_superseded_tasks_no_longer_block_cap_or_phase_but_do_not_satisfy_dependencies(
    tmp_path: Path,
) -> None:
    cap_store = TaskStateStore(tmp_path / "cap.sqlite3", attempt_cap=10, logical_task_cap=1)
    cap_predecessor = task("cap_v1", priority=10)
    cap_store.register_task(cap_predecessor, now=1.0)
    cap_lease = cap_store.reserve_attempt("cap_v1", now=2.0, lease_seconds=10)
    cap_store.freeze_result(
        cap_lease.attempt_id,
        result_path="llm/frozen/cap_v1.json",
        result_sha256="sha-cap-v1",
        now=3.0,
    )
    cap_store.register_supersession(
        "cap_v1",
        TaskSpec(
            evaluation_key="cap_v2",
            logical_id="logical::cap_v2",
            task_type=cap_predecessor.task_type,
            condition=cap_predecessor.condition,
            priority=cap_predecessor.priority,
            input_hash=cap_predecessor.input_hash,
            prompt_version="simplify.v2",
            schema_version="simplify.v2",
            dependencies=cap_predecessor.dependencies,
        ),
        identity="dataset::cap",
        reason="rerun_with_revised_prompt",
        predecessor_plan_sha256="plan-prev::cap_v1",
        successor_plan_sha256="plan-next::cap_v2",
        now=4.0,
    )
    assert cap_store.task_state("cap_v1") == "superseded"
    assert cap_store.task_state("cap_v2") == "pending"

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10, logical_task_cap=2)
    predecessor = task("clean_v1", priority=10)
    store.register_task(predecessor, now=1.0)
    lease = store.reserve_attempt("clean_v1", now=2.0, lease_seconds=10)
    store.freeze_result(
        lease.attempt_id,
        result_path="llm/frozen/clean_v1.json",
        result_sha256="sha-clean-v1",
        now=3.0,
    )
    store.register_supersession(
        "clean_v1",
        TaskSpec(
            evaluation_key="clean_v2",
            logical_id="logical::clean_v2",
            task_type=predecessor.task_type,
            condition=predecessor.condition,
            priority=predecessor.priority,
            input_hash=predecessor.input_hash,
            prompt_version="simplify.v2",
            schema_version="simplify.v2",
            dependencies=predecessor.dependencies,
        ),
        identity="dataset::clean",
        reason="rerun_with_revised_prompt",
        predecessor_plan_sha256="plan-prev::clean_v1",
        successor_plan_sha256="plan-next::clean_v2",
        now=4.0,
    )

    store.register_task(task("after", priority=20), now=5.0)
    assert store.next_ready_key(allowed_conditions=("clean",)) == "clean_v2"
    successor_lease = store.reserve_attempt("clean_v2", now=6.0, lease_seconds=10)
    store.freeze_result(
        successor_lease.attempt_id,
        result_path="llm/frozen/clean_v2.json",
        result_sha256="sha-clean-v2",
        now=7.0,
    )
    assert store.next_ready_key(allowed_conditions=("clean",)) == "after"
    after_lease = store.reserve_attempt("after", now=8.0, lease_seconds=10)
    assert after_lease.attempt_number == 1

    dependent_store = TaskStateStore(tmp_path / "dependencies.sqlite3", attempt_cap=10)
    dependent_store.register_task(task("upstream_v1"), now=1.0)
    upstream_lease = dependent_store.reserve_attempt("upstream_v1", now=2.0, lease_seconds=10)
    dependent_store.freeze_result(
        upstream_lease.attempt_id,
        result_path="llm/frozen/upstream_v1.json",
        result_sha256="sha-upstream-v1",
        now=3.0,
    )
    dependent_store.register_supersession(
        "upstream_v1",
        TaskSpec(
            evaluation_key="upstream_v2",
            logical_id="logical::upstream_v2",
            task_type="pred_simplify",
            condition="clean",
            priority=10,
            input_hash="input::upstream_v1",
            prompt_version="simplify.v2",
            schema_version="simplify.v2",
            dependencies=(),
        ),
        identity="dataset::upstream",
        reason="rerun_with_revised_prompt",
        predecessor_plan_sha256="plan-prev::upstream_v1",
        successor_plan_sha256="plan-next::upstream_v2",
        now=4.0,
    )
    dependent_store.register_task(
        task("dependent", dependencies=("upstream_v1",), priority=20),
        now=5.0,
    )
    assert dependent_store.next_ready_key(allowed_conditions=("clean",)) == "upstream_v2"
    upstream_v2_lease = dependent_store.reserve_attempt("upstream_v2", now=6.0, lease_seconds=10)
    dependent_store.freeze_result(
        upstream_v2_lease.attempt_id,
        result_path="llm/frozen/upstream_v2.json",
        result_sha256="sha-upstream-v2",
        now=7.0,
    )
    assert dependent_store.next_ready_key(allowed_conditions=("clean",)) is None


def test_invalid_superseded_dependent_still_blocks_predecessor_supersession(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "invalid_dependent.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    store.register_tasks((task("base"), task("dependent", dependencies=("base",), priority=20)), now=1.0)
    lease = store.reserve_attempt("base", now=2.0, lease_seconds=10)
    store.freeze_result(
        lease.attempt_id,
        result_path="llm/frozen/base.json",
        result_sha256="sha-base",
        now=3.0,
    )
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            "UPDATE tasks SET state='superseded', updated_at=? WHERE evaluation_key='dependent'",
            (4.0,),
        )
        connection.commit()

    with pytest.raises(StateContractError, match="active dependents"):
        store.register_supersession(
            "base",
            TaskSpec(
                evaluation_key="base_v2",
                logical_id="logical::base_v2",
                task_type="pred_simplify",
                condition="clean",
                priority=10,
                input_hash="input::base_v2",
                prompt_version="simplify.v2",
                schema_version="simplify.v2",
                dependencies=(),
            ),
            identity="dataset::base",
            reason="rerun_with_revised_prompt",
            predecessor_plan_sha256="plan-prev::base",
            successor_plan_sha256="plan-next::base_v2",
            now=5.0,
        )


def test_invalid_superseded_without_binding_remains_active_for_cap_and_summary(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "broken.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10, logical_task_cap=3)
    store.register_tasks((task("a"), task("a_v2"), task("b", priority=20)), now=1.0)
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            "UPDATE tasks SET state='superseded', updated_at=? WHERE evaluation_key='a'",
            (2.0,),
        )
        connection.execute(
            """INSERT INTO task_supersessions(
                   predecessor_evaluation_key, successor_evaluation_key,
                   predecessor_logical_id, successor_logical_id, identity, reason,
                   predecessor_plan_sha256, successor_plan_sha256, superseded_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "a",
                "a_v2",
                "logical::a",
                "logical::a_v2",
                "dataset::a",
                "rerun_with_revised_prompt",
                "plan-prev::a",
                "plan-next::a_v2",
                2.0,
            ),
        )
        connection.commit()

    summary = store.state_summary()
    assert summary["logical_tasks"]["historical"] == 3
    assert summary["logical_tasks"]["superseded"] == 0
    assert summary["logical_tasks"]["active"] == 3
    with pytest.raises(StateContractError, match="逻辑任务预算"):
        store.register_task(task("c"), now=3.0)
    assert store.next_ready_key(allowed_conditions=("clean",)) == "a_v2"
