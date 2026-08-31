from __future__ import annotations

from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
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


def test_register_is_idempotent_but_rejects_contract_drift(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3", attempt_cap=10)
    store.register_task(task("a"))
    store.register_task(task("a"))
    with pytest.raises(StateContractError):
        store.register_task(task("a", condition="noise001"))


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
    assert store.next_ready_key(allowed_conditions=("noise001",)) == "noise001"
    assert store.next_ready_key(allowed_conditions=("noise005",)) == "noise005"

