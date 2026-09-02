from __future__ import annotations

import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.reopen_retryable_api_channel_failures import (
    reopen_retryable_api_channel_failures,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


def _task(key: str) -> TaskSpec:
    return TaskSpec(
        evaluation_key=key,
        logical_id=f"pred_simplify::demo::g0001::s520::noise001::{key[-1]}",
        task_type="pred_simplify",
        condition="noise001",
        priority=20,
        input_hash=key,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )


def _write_attempt(
    attempts_dir: Path,
    *,
    task: TaskSpec,
    attempt_id: str,
    error_class: str,
    channel: str,
    http_status: int,
    error_message: str,
) -> None:
    attempts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "attempt_id": attempt_id,
        "evaluation_key": task.evaluation_key,
        "metadata": {
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "evaluation_key": task.evaluation_key,
            "logical_id": task.logical_id,
            "task_type": task.task_type,
            "api_channel": channel,
            "http_status": http_status,
            "error_class": error_class,
            "retryable": False,
        },
        "validation": {
            "ok": False,
            "error_class": error_class,
            "error_message": error_message,
        },
    }
    (attempts_dir / f"{attempt_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_reopens_only_audited_channel_and_deployment_failures(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite3"
    attempts_dir = tmp_path / "attempts"
    store = TaskStateStore(state_db, attempt_cap=20)
    auth_task = _task("a" * 64)
    deployment_task = _task("b" * 64)
    ordinary_bad_request_task = _task("c" * 64)
    store.register_tasks([auth_task, deployment_task, ordinary_bad_request_task])

    auth_lease = store.reserve_attempt(auth_task.evaluation_key, now=1.0, lease_seconds=10)
    store.finish_failure(
        auth_lease.attempt_id,
        error_class="api_auth_error",
        retryable=False,
        now=2.0,
    )
    deployment_lease = store.reserve_attempt(
        deployment_task.evaluation_key,
        now=3.0,
        lease_seconds=10,
    )
    store.finish_failure(
        deployment_lease.attempt_id,
        error_class="api_http_error",
        retryable=False,
        now=4.0,
    )
    ordinary_lease = store.reserve_attempt(
        ordinary_bad_request_task.evaluation_key,
        now=5.0,
        lease_seconds=10,
    )
    store.finish_failure(
        ordinary_lease.attempt_id,
        error_class="api_http_error",
        retryable=False,
        now=6.0,
    )
    _write_attempt(
        attempts_dir,
        task=auth_task,
        attempt_id=auth_lease.attempt_id,
        error_class="api_auth_error",
        channel="yapi",
        http_status=403,
        error_message="User has been banned",
    )
    _write_attempt(
        attempts_dir,
        task=deployment_task,
        attempt_id=deployment_lease.attempt_id,
        error_class="api_http_error",
        channel="routify",
        http_status=400,
        error_message="AllModelsFailed: 555420 deployment request could not be completed",
    )
    _write_attempt(
        attempts_dir,
        task=ordinary_bad_request_task,
        attempt_id=ordinary_lease.attempt_id,
        error_class="api_http_error",
        channel="routify",
        http_status=400,
        error_message="invalid request schema",
    )

    report = reopen_retryable_api_channel_failures(
        state_db=state_db,
        attempts_dir=attempts_dir,
        condition="noise001",
        task_type="pred_simplify",
        failed_channel="yapi",
        audit_reason="test_reclassification",
        apply=True,
    )

    assert report["counts"] == {"candidate": 2, "reopened": 2}
    assert store.task_state(auth_task.evaluation_key) == "retry_wait"
    assert store.task_state(deployment_task.evaluation_key) == "retry_wait"
    assert store.task_state(ordinary_bad_request_task.evaluation_key) == "exhausted"
    assert store.attempts_reserved() == 3
