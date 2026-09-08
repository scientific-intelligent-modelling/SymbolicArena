from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.register_noise_pred_supersessions import (
    NoisePredSupersessionError,
    register_noise_pred_supersessions,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_row(logical_id: str, *, marker: str, condition: str) -> dict[str, object]:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha = _sha_file(prompt_path)
    schema_sha = _sha_file(schema_path)
    request = {
        "original_expression": marker,
        "evidence_hash": _sha_text(f"{logical_id}:{marker}"),
    }
    normalized = {
        "request": request,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    input_hash = _sha_text(canonical_json(normalized))
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
        condition=condition,
        priority=20,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=(),
    )
    return {
        "condition": condition,
        "dependencies": [],
        "evaluation_key": key,
        "input_hash": input_hash,
        "logical_id": logical_id,
        "normalized_input": normalized,
        "priority": 20,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_template": prompt,
        "prompt_version": prompt_path.stem,
        "rendered_prompt": render_prompt(prompt, request, schema),
        "request": request,
        "schema_content": schema,
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha,
        "schema_version": schema_path.stem,
        "task_kind": "simplify",
        "task_spec": json.loads(spec.canonical_json()),
        "task_type": "pred_simplify",
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    condition = "noise005"
    predecessor_rows = []
    successor_rows = []
    bindings = []
    for index, dataset in enumerate(("g0001", "g0002")):
        base = f"pred_simplify::alg00::{dataset}::s520::{condition}"
        predecessor = _plan_row(base, marker=f"old-{index}", condition=condition)
        successor = _plan_row(f"{base}::v2", marker=f"new-{index}", condition=condition)
        predecessor_rows.append(predecessor)
        successor_rows.append(successor)
        bindings.append(
            {
                "base_logical_id": base,
                "old_source_result_sha256": _sha_text(f"old-source-{index}"),
                "new_source_result_sha256": _sha_text(f"new-source-{index}"),
                "predecessor_evaluation_key": predecessor["evaluation_key"],
                "predecessor_logical_id": predecessor["logical_id"],
                "successor_evaluation_key": successor["evaluation_key"],
                "successor_logical_id": successor["logical_id"],
            }
        )

    plan_path = tmp_path / "hybrid.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in successor_rows),
        encoding="utf-8",
    )
    predecessor_plan_path = tmp_path / "predecessor.jsonl"
    predecessor_plan_path.write_text("predecessor-plan", encoding="utf-8")
    predecessor_plan_sha = _sha_file(predecessor_plan_path)
    fresh_plan_path = tmp_path / "fresh.jsonl"
    fresh_plan_path.write_text("fresh-plan", encoding="utf-8")
    refresh_plan_path = tmp_path / "refresh.jsonl"
    refresh_plan_path.write_text("refresh-plan", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "noise_pred_hybrid.v1",
                "status": "ok",
                "conditions": {
                    condition: {
                        "bindings": bindings,
                        "predecessor_plan": {
                            "path": str(predecessor_plan_path),
                            "rows": 2,
                            "sha256": predecessor_plan_sha,
                        },
                        "fresh_plan": {
                            "path": str(fresh_plan_path),
                            "rows": 2,
                            "sha256": _sha_file(fresh_plan_path),
                        },
                        "hybrid_plan": {
                            "path": str(plan_path),
                            "rows": 2,
                            "sha256": _sha_file(plan_path),
                        },
                        "refresh_plan": {
                            "path": str(refresh_plan_path),
                            "rows": 2,
                            "sha256": _sha_file(refresh_plan_path),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "predecessor_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "plan_jsonl": str(predecessor_plan_path),
                "plan_sha256": predecessor_plan_sha,
                "row_count": 2,
                "state_counts": {"frozen": 1, "non_applicable": 1},
            }
        ),
        encoding="utf-8",
    )

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(
        state_db,
        attempt_cap=100,
        logical_task_cap=100,
        max_attempts_per_task=4,
    )
    specs = [TaskSpec(**row["task_spec"]) for row in predecessor_rows]
    store.register_tasks(specs)
    lease = store.reserve_attempt(specs[0].evaluation_key)
    store.freeze_result(
        lease.attempt_id,
        result_path="frozen/0.json",
        result_sha256=_sha_text("frozen-result"),
    )
    store.mark_non_applicable(
        specs[1].evaluation_key,
        reason="audited missing expression",
        evidence_path="evidence/1.json",
        evidence_sha256=_sha_text("nonapp-evidence"),
    )
    return {
        "condition": condition,
        "manifest_path": manifest_path,
        "plan_path": plan_path,
        "summary_path": summary_path,
        "state_db": state_db,
        "predecessor_rows": predecessor_rows,
        "successor_rows": successor_rows,
    }


def _register(paths: dict[str, object], tmp_path: Path) -> dict[str, object]:
    return register_noise_pred_supersessions(
        condition=str(paths["condition"]),
        manifest_json=paths["manifest_path"],
        hybrid_plan_jsonl=paths["plan_path"],
        predecessor_frozen_summary_json=paths["summary_path"],
        state_db=paths["state_db"],
        report_json=tmp_path / "report.json",
        expected_count=2,
        expected_total_count=2,
    )


def test_registers_frozen_and_audited_nonapp_predecessors_atomically_and_idempotently(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    first = _register(paths, tmp_path)
    second = _register(paths, tmp_path)

    assert first["state_registration"]["mutated"] is True
    assert second["state_registration"]["mutated"] is False
    assert first["counts"]["predecessor_state_before"] == {
        "frozen": 1,
        "non_applicable": 1,
    }
    assert second["counts"]["mapping_after"] == 2
    with sqlite3.connect(paths["state_db"]) as connection:
        task_states = dict(connection.execute("SELECT logical_id, state FROM tasks"))
        mappings = connection.execute(
            """SELECT predecessor_plan_sha256, successor_plan_sha256
               FROM task_supersessions ORDER BY predecessor_logical_id"""
        ).fetchall()
    for row in paths["predecessor_rows"]:
        assert task_states[row["logical_id"]] == "superseded"
    for row in paths["successor_rows"]:
        assert task_states[row["logical_id"]] == "pending"
    assert mappings == [
        (_sha_text("predecessor-plan"), _sha_file(paths["plan_path"])),
        (_sha_text("predecessor-plan"), _sha_file(paths["plan_path"])),
    ]


def test_rejects_forbidden_source_without_mutating_state(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(Path(paths["manifest_path"]).read_text(encoding="utf-8"))
    manifest["source"] = "all_15alg_fullcpu_v1"
    Path(paths["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NoisePredSupersessionError, match="禁止来源"):
        _register(paths, tmp_path)

    with sqlite3.connect(paths["state_db"]) as connection:
        states = dict(connection.execute("SELECT logical_id, state FROM tasks"))
        mapping_count = connection.execute("SELECT COUNT(*) FROM task_supersessions").fetchone()[0]
    assert set(states.values()) == {"frozen", "non_applicable"}
    assert mapping_count == 0
