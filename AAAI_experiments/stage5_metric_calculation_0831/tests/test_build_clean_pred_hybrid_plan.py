from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_clean_pred_hybrid_plan import (
    CleanPredHybridPlanError,
    build_clean_pred_hybrid_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_row(logical_id: str, *, expression: str) -> dict[str, object]:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha = _sha(prompt_path)
    schema_sha = _sha(schema_path)
    request = {
        "dataset_id": logical_id,
        "expression": expression,
        "original_expression": expression,
        "variables": ["x0"],
        "allowed_functions": [],
        "evidence_hash": hashlib.sha256(f"{logical_id}:{expression}".encode()).hexdigest(),
    }
    normalized = {
        "request": request,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    input_hash = hashlib.sha256(canonical_json(normalized).encode()).hexdigest()
    key = evaluation_key(
        task_type="pred_simplify",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        prompt_sha256=prompt_sha,
        schema_sha256=schema_sha,
        normalized_input=normalized,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type="pred_simplify",
        condition="clean",
        priority=20,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=(),
    )
    return {
        "condition": "clean",
        "dependencies": [],
        "evaluation_key": key,
        "input_hash": input_hash,
        "logical_id": logical_id,
        "normalized_input": normalized,
        "priority": 20,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_template": prompt_template,
        "prompt_version": prompt_path.stem,
        "rendered_prompt": render_prompt(prompt_template, request, schema),
        "request": request,
        "schema_content": schema,
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha,
        "schema_version": schema_path.stem,
        "task_kind": "simplify",
        "task_spec": json.loads(spec.canonical_json()),
        "task_type": "pred_simplify",
    }


def _write_plan(path: Path, rows: list[dict[str, object]]) -> list[str]:
    lines = [canonical_json(row) + "\n" for row in rows]
    path.write_text("".join(lines), encoding="utf-8")
    return lines


def _fixture(
    tmp_path: Path,
    *,
    conflict_version: bool = False,
    required_count: int = 2,
    total_count: int = 4,
) -> dict[str, object]:
    bases = [
        "pred_simplify::jaxsr::g0001::s520::clean",
        "pred_simplify::imcts::g0002::s521::clean",
        "pred_simplify::gplearn::g0001::s520::clean",
        "pred_simplify::dso::g0002::s521::clean",
    ]
    bases.extend(
        f"pred_simplify::jaxsr::g{index:04d}::s522::clean"
        for index in range(1000, 1000 + total_count - len(bases))
    )
    bases = bases[:total_count]
    old_ids = list(bases)
    if conflict_version:
        old_ids[0] = f"{bases[0]}::v2"
    old_rows = [_plan_row(logical_id, expression="x0 + 1") for logical_id in old_ids]
    fresh_rows = [_plan_row(logical_id, expression="x0 + 2") for logical_id in bases]
    old_plan = tmp_path / "old.jsonl"
    fresh_plan = tmp_path / "fresh.jsonl"
    old_lines = _write_plan(old_plan, old_rows)
    _write_plan(fresh_plan, fresh_rows)
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "aggregation_readiness": {
                    "required_pred_simplify_logical_ids": bases[:required_count],
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "bases": bases,
        "old_rows": old_rows,
        "old_lines": old_lines,
        "old_plan": old_plan,
        "fresh_plan": fresh_plan,
        "binding": binding,
        "required_count": required_count,
        "total_count": total_count,
    }


def _build(paths: dict[str, object], tmp_path: Path, *, state_db: Path | None = None) -> dict[str, object]:
    return build_clean_pred_hybrid_plan(
        predecessor_plan_jsonl=paths["old_plan"],
        fresh_plan_jsonl=paths["fresh_plan"],
        binding_manifest_json=paths["binding"],
        output_plan_jsonl=tmp_path / "hybrid.jsonl",
        supersession_manifest_json=tmp_path / "supersessions.json",
        report_json=tmp_path / "report.json",
        expected_total_count=paths["total_count"],
        expected_required_count=paths["required_count"],
        register_state_db=state_db,
    )


def test_builds_hybrid_plan_and_preserves_nonrequired_rows_byte_for_byte(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = _build(paths, tmp_path)

    assert report["counts"] == {
        "total": 4,
        "required": 2,
        "changed": 2,
        "preserved": 2,
    }
    assert report["state_registration"]["requested"] is False
    output_lines = (tmp_path / "hybrid.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    assert output_lines[2:] == paths["old_lines"][2:]
    rows = [json.loads(line) for line in output_lines]
    assert [row["logical_id"] for row in rows[:2]] == [
        f"{paths['bases'][0]}::v2",
        f"{paths['bases'][1]}::v2",
    ]
    assert [row["request"]["expression"] for row in rows[:2]] == ["x0 + 2", "x0 + 2"]
    assert len({row["logical_id"] for row in rows}) == 4
    assert len({row["evaluation_key"] for row in rows}) == 4
    assert len(load_plan_jsonl(tmp_path / "hybrid.jsonl").entries) == 4
    manifest = json.loads((tmp_path / "supersessions.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"supersessions": 2}
    assert manifest["hybrid_plan_sha256"] == _sha(tmp_path / "hybrid.jsonl")


def test_rejects_required_v2_conflict_without_outputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, conflict_version=True)
    with pytest.raises(CleanPredHybridPlanError, match="版本冲突"):
        _build(paths, tmp_path)
    assert not (tmp_path / "hybrid.jsonl").exists()


def _register_old_plan(state_db: Path, rows: list[dict[str, object]], *, pending_index: int | None) -> None:
    capacity = max(20, len(rows) * 2)
    store = TaskStateStore(state_db, attempt_cap=capacity, logical_task_cap=capacity)
    for index, row in enumerate(rows):
        spec = TaskSpec(**row["task_spec"])
        store.register_task(spec, now=float(index + 1))
        if pending_index == index:
            continue
        lease = store.reserve_attempt(spec.evaluation_key, now=float(index + 10))
        store.freeze_result(
            lease.attempt_id,
            result_path=f"frozen/{index}.json",
            result_sha256=f"sha-{index}",
            now=float(index + 20),
        )


def test_optional_state_registration_supersedes_exact_required_set(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, required_count=75, total_count=77)
    state_db = tmp_path / "copy.sqlite3"
    _register_old_plan(state_db, paths["old_rows"], pending_index=None)

    report = _build(paths, tmp_path, state_db=state_db)

    assert report["state_registration"]["mutated"] is True
    hybrid = [json.loads(line) for line in (tmp_path / "hybrid.jsonl").read_text().splitlines()]
    with sqlite3.connect(state_db) as connection:
        states = dict(connection.execute("SELECT evaluation_key, state FROM tasks"))
        mapping_count = connection.execute("SELECT COUNT(*) FROM task_supersessions").fetchone()[0]
    for old, new in zip(paths["old_rows"][:75], hybrid[:75], strict=True):
        assert states[old["evaluation_key"]] == "superseded"
        assert states[new["evaluation_key"]] == "pending"
    assert mapping_count == 75


def test_state_validation_failure_leaves_database_unchanged(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    state_db = tmp_path / "copy.sqlite3"
    _register_old_plan(state_db, paths["old_rows"], pending_index=1)
    with sqlite3.connect(state_db) as connection:
        before_tasks = list(connection.execute("SELECT evaluation_key, state FROM tasks ORDER BY 1"))
        before_supersessions = connection.execute("SELECT COUNT(*) FROM task_supersessions").fetchone()[0]

    with pytest.raises(CleanPredHybridPlanError, match="必须为 frozen"):
        _build(paths, tmp_path, state_db=state_db)

    with sqlite3.connect(state_db) as connection:
        after_tasks = list(connection.execute("SELECT evaluation_key, state FROM tasks ORDER BY 1"))
        after_supersessions = connection.execute("SELECT COUNT(*) FROM task_supersessions").fetchone()[0]
    assert after_tasks == before_tasks
    assert after_supersessions == before_supersessions == 0


def test_state_batch_failure_rolls_back_all_supersessions(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    state_db = tmp_path / "copy.sqlite3"
    _register_old_plan(state_db, paths["old_rows"], pending_index=None)
    blocked_predecessor = paths["old_rows"][1]["evaluation_key"]
    dependent = TaskSpec(
        evaluation_key="dependent-evaluation-key",
        logical_id="pred_simplify::dependent::g0001::s520::clean",
        task_type="pred_simplify",
        condition="clean",
        priority=20,
        input_hash="dependent-input-hash",
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(blocked_predecessor,),
    )
    TaskStateStore(state_db, attempt_cap=20, logical_task_cap=20).register_task(dependent)
    with sqlite3.connect(state_db) as connection:
        before_tasks = list(connection.execute("SELECT evaluation_key, state FROM tasks ORDER BY 1"))

    with pytest.raises(CleanPredHybridPlanError, match="active dependents"):
        _build(paths, tmp_path, state_db=state_db)

    hybrid = [json.loads(line) for line in (tmp_path / "hybrid.jsonl").read_text().splitlines()]
    with sqlite3.connect(state_db) as connection:
        after_tasks = list(connection.execute("SELECT evaluation_key, state FROM tasks ORDER BY 1"))
        mapping_count = connection.execute("SELECT COUNT(*) FROM task_supersessions").fetchone()[0]
        generated_successors = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE evaluation_key IN (?, ?)",
            (hybrid[0]["evaluation_key"], hybrid[1]["evaluation_key"]),
        ).fetchone()[0]
    assert after_tasks == before_tasks
    assert mapping_count == generated_successors == 0
