from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.retire_stale_exhausted_tasks import (
    retire_stale_exhausted_tasks,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.revise_exhausted_symbolic_plan import (
    ReviseExhaustedSymbolicPlanError,
    revise_exhausted_symbolic_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    _load_predecessor_attempt_manifest,
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plan_row(
    *,
    task_type: str,
    logical_id: str,
    condition: str = "clean",
) -> dict[str, object]:
    task_kind = "equivalence" if task_type == "equivalence" else "structure"
    prompt_path = STAGE_ROOT / f"config/prompts/{task_kind}.v1.txt"
    schema_path = STAGE_ROOT / f"config/schemas/{task_kind}.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256_bytes(prompt_path.read_bytes())
    schema_sha256 = _sha256_bytes(schema_path.read_bytes())
    request = {
        "lhs": "x0 + x1",
        "rhs": "x1 + x0",
        "evidence_hash": hashlib.sha256(logical_id.encode()).hexdigest(),
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_bytes(canonical_json(normalized_input).encode())
    priority = 30 if task_type == "equivalence" else 40
    spec = TaskSpec(
        evaluation_key=evaluation_key(
            task_type=task_type,
            logical_id=logical_id,
            prompt_version=prompt_path.stem,
            schema_version=schema_path.stem,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            normalized_input=normalized_input,
            evidence_hash=request["evidence_hash"],
        ),
        logical_id=logical_id,
        task_type=task_type,
        condition=condition,
        priority=priority,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=(),
    )
    return {
        "evaluation_key": spec.evaluation_key,
        "logical_id": logical_id,
        "task_type": task_type,
        "task_kind": task_kind,
        "condition": condition,
        "priority": priority,
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
        "rendered_prompt": render_prompt(prompt_template, request, schema),
    }


def _write_plan(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _close_frozen(store: TaskStateStore, row: dict[str, object], tmp_path: Path) -> None:
    lease = store.reserve_attempt(str(row["evaluation_key"]), now=1.0)
    result_path = tmp_path / f"{lease.attempt_id}.json"
    result_path.write_text("{}\n", encoding="utf-8")
    store.freeze_result(
        lease.attempt_id,
        result_path=str(result_path),
        result_sha256=_sha256_bytes(result_path.read_bytes()),
        now=1.5,
    )


def _close_exhausted(store: TaskStateStore, row: dict[str, object]) -> None:
    for index in range(3):
        lease = store.reserve_attempt(str(row["evaluation_key"]), now=2.0 + index)
        store.finish_failure(
            lease.attempt_id,
            error_class="structured_output_invalid",
            retryable=True,
            now=2.5 + index,
        )


@pytest.mark.parametrize(
    ("task_type", "frozen_id", "exhausted_id", "successor_suffix"),
    [
        (
            "equivalence",
            "equivalence::demo::g0001::s520::clean",
            "equivalence::demo::g0002::s521::clean",
            "v2",
        ),
        (
            "stab_structure",
            "stab_structure::demo::g0001::s520-s521",
            "stab_structure::demo::g0002::s520-s522",
            "v2",
        ),
        (
            "equivalence",
            "equivalence::demo::g0001::s520::clean",
            "equivalence::demo::g0002::s521::clean::v2",
            "v3",
        ),
    ],
)
def test_revises_only_exhausted_symbolic_tasks(
    tmp_path: Path,
    task_type: str,
    frozen_id: str,
    exhausted_id: str,
    successor_suffix: str,
) -> None:
    frozen_row = _plan_row(task_type=task_type, logical_id=frozen_id)
    exhausted_row = _plan_row(task_type=task_type, logical_id=exhausted_id)
    predecessor_plan = tmp_path / "predecessor.jsonl"
    _write_plan(predecessor_plan, [frozen_row, exhausted_row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    for row in (frozen_row, exhausted_row):
        store.register_task(TaskSpec(**row["task_spec"]))
    _close_frozen(store, frozen_row, tmp_path)
    _close_exhausted(store, exhausted_row)

    output_plan = tmp_path / "successor.jsonl"
    report = revise_exhausted_symbolic_plan(
        predecessor_plan_jsonl=predecessor_plan,
        state_db=state_db,
        output_jsonl=output_plan,
        report_json=tmp_path / "report.json",
        task_type=task_type,
        condition="clean",
        logical_id_suffix=successor_suffix,
        expected_task_count=2,
        expected_exhausted_count=1,
    )

    output_rows = [json.loads(line) for line in output_plan.read_text().splitlines()]
    by_logical_id = {row["logical_id"]: row for row in output_rows}
    assert by_logical_id[frozen_id] == frozen_row
    successor = by_logical_id[
        f"{exhausted_id.removesuffix('::v2')}::{successor_suffix}"
    ]
    unchanged_fields = set(exhausted_row) - {
        "evaluation_key",
        "logical_id",
        "task_spec",
    }
    assert all(successor[field] == exhausted_row[field] for field in unchanged_fields)
    assert successor["evaluation_key"] != exhausted_row["evaluation_key"]
    assert successor["task_spec"]["logical_id"] == (
        f"{exhausted_id.removesuffix('::v2')}::{successor_suffix}"
    )
    assert load_plan_jsonl(output_plan).plan_sha256 == report["output_sha256"]
    assert report["preserved_frozen_count"] == 1
    assert report["successor_task_count"] == 1
    assert report["state_db_mutated"] is False


def test_rejects_exhausted_symbolic_task_without_three_failed_attempts(
    tmp_path: Path,
) -> None:
    row = _plan_row(
        task_type="equivalence",
        logical_id="equivalence::demo::g0001::s520::clean",
    )
    predecessor_plan = tmp_path / "predecessor.jsonl"
    _write_plan(predecessor_plan, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(TaskSpec(**row["task_spec"]))
    _close_exhausted(store, row)
    with sqlite3.connect(state_db) as connection:
        connection.execute(
            "UPDATE attempts SET status='accepted' WHERE attempt_number=1"
        )

    with pytest.raises(ReviseExhaustedSymbolicPlanError, match="三次 failed"):
        revise_exhausted_symbolic_plan(
            predecessor_plan_jsonl=predecessor_plan,
            state_db=state_db,
            output_jsonl=tmp_path / "successor.jsonl",
            report_json=tmp_path / "report.json",
            task_type="equivalence",
            condition="clean",
            expected_task_count=1,
            expected_exhausted_count=1,
        )


def _empty_predecessor_manifest(tmp_path: Path) -> tuple[Path, PredecessorAttemptManifest]:
    source_db = tmp_path / "source.sqlite3"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE attempts(attempt_id TEXT)")
    source_attempts = tmp_path / "source_attempts"
    source_attempts.mkdir()
    manifest_path = tmp_path / "predecessor_attempts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "predecessor_attempts.v1",
                "attempt_ids": [],
                "attempt_count": 0,
                "attempt_file_sha256": {},
                "source_state_db": str(source_db),
                "source_attempts_dir": str(source_attempts),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = _load_predecessor_attempt_manifest(manifest_path)
    return manifest_path, PredecessorAttemptManifest(
        path=str(loaded.path),
        sha256=loaded.sha256,
        attempt_count=loaded.attempt_count,
    )


def test_retirement_accepts_explicit_symbolic_scope(tmp_path: Path) -> None:
    old_row = _plan_row(
        task_type="stab_structure",
        logical_id="stab_structure::demo::g0001::s520-s521",
    )
    successor_row = _plan_row(
        task_type="stab_structure",
        logical_id="stab_structure::demo::g0001::s520-s521::v2",
    )
    historical_plan = tmp_path / "historical.jsonl"
    active_plan = tmp_path / "active.jsonl"
    _write_plan(historical_plan, [old_row])
    _write_plan(active_plan, [successor_row])
    manifest_path, predecessor = _empty_predecessor_manifest(tmp_path)
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, predecessor_attempt_manifest=predecessor)
    for row in (old_row, successor_row):
        store.register_task(TaskSpec(**row["task_spec"]))
    _close_exhausted(store, old_row)
    _close_frozen(store, successor_row, tmp_path)

    report = retire_stale_exhausted_tasks(
        state_db=state_db,
        predecessor_attempt_manifest=manifest_path,
        active_plan_jsonl=active_plan,
        historical_plan_jsonls=[historical_plan],
        backup_sqlite=tmp_path / "state.before_retirement.sqlite3",
        manifest_jsonl=tmp_path / "retirement_manifest.jsonl",
        report_json=tmp_path / "retirement_report.json",
        expected_count=1,
        task_type="stab_structure",
        condition="clean",
    )

    assert report["task_type"] == "stab_structure"
    assert report["condition"] == "clean"
    assert report["retired_count"] == 1
    assert (tmp_path / "state.before_retirement.sqlite3").is_file()
    with sqlite3.connect(state_db) as connection:
        assert connection.execute(
            "SELECT state FROM tasks WHERE evaluation_key=?",
            (old_row["evaluation_key"],),
        ).fetchone()[0] == "superseded"


def test_retirement_keeps_pred_clean_defaults() -> None:
    parameters = inspect.signature(retire_stale_exhausted_tasks).parameters
    assert parameters["task_type"].default == "pred_simplify"
    assert parameters["condition"].default == "clean"
