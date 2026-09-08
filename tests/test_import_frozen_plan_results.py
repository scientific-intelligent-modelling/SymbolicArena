from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    AnthropicApiChannel,
    AnthropicApiResponse,
    AnthropicApiRunner,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.import_frozen_plan_results import (
    FrozenPlanImportError,
    import_frozen_plan_results,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plan_row(tmp_path: Path, *, dependency: str | None = None) -> dict[str, Any]:
    prompt_path = tmp_path / "contract/equivalence.v1.txt"
    schema_path = tmp_path / "contract/equivalence.v1.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = "Judge equivalence.\n{{REQUEST_JSON}}\n"
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["equivalent", "not_equivalent", "undetermined"],
            },
            "evidence_basis": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "brief_reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "decision",
            "evidence_basis",
            "assumptions",
            "confidence",
            "brief_reason",
        ],
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(schema_path, schema)
    prompt_sha = _sha256_bytes(prompt_path.read_bytes())
    schema_sha = _sha256_bytes(schema_path.read_bytes())
    logical_id = "equivalence::demo::g0001::s520::clean"
    request = {"lhs": "x0 + 0", "rhs": "x0", "evidence_hash": "fixture-evidence"}
    normalized = {
        "request": request,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    key = evaluation_key(
        task_type="equivalence",
        logical_id=logical_id,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        prompt_sha256=prompt_sha,
        schema_sha256=schema_sha,
        normalized_input=normalized,
        evidence_hash="fixture-evidence",
    )
    dependencies = (dependency,) if dependency else ()
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type="equivalence",
        condition="clean",
        priority=30,
        input_hash=_sha256_bytes(canonical_json(normalized).encode("utf-8")),
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=dependencies,
    )
    return {
        "condition": "clean",
        "dependencies": list(dependencies),
        "evaluation_key": key,
        "input_hash": spec.input_hash,
        "logical_id": logical_id,
        "normalized_input": normalized,
        "priority": 30,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_template": prompt,
        "prompt_version": "equivalence.v1",
        "rendered_prompt": render_prompt(prompt, request, schema),
        "request": request,
        "schema_content": schema,
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha,
        "schema_version": "equivalence.v1",
        "task_kind": "equivalence",
        "task_spec": json.loads(spec.canonical_json()),
        "task_type": "equivalence",
    }


class _FakeTransport:
    def __call__(
        self,
        channel: AnthropicApiChannel,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> AnthropicApiResponse:
        output = {
            "decision": "equivalent",
            "evidence_basis": "symbolic_proof",
            "assumptions": [],
            "confidence": 1.0,
            "brief_reason": "Identical after removing additive zero.",
        }
        return AnthropicApiResponse(
            status_code=200,
            body={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": json.dumps(output)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
            text="fixture-response",
            headers={"request-id": "fixture"},
        )


def _fixture(tmp_path: Path, *, blocked_dependency: bool = False) -> dict[str, object]:
    dependency_key = "dep-" + "0" * 60 if blocked_dependency else None
    row = _plan_row(tmp_path, dependency=dependency_key)
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    spec = TaskSpec(**row["task_spec"])

    source_db = tmp_path / "source.sqlite3"
    source_store = TaskStateStore(source_db)
    if dependency_key:
        dependency = TaskSpec(
            evaluation_key=dependency_key,
            logical_id="pred_simplify::demo::g0001::s520::clean",
            task_type="pred_simplify",
            condition="clean",
            priority=20,
            input_hash="dependency",
            prompt_version="simplify.v1",
            schema_version="simplify.v1",
            dependencies=(),
        )
        source_store.register_task(dependency)
        dependency_lease = source_store.reserve_attempt(dependency_key)
        source_store.freeze_result(
            dependency_lease.attempt_id,
            result_path=str(tmp_path / "dependency.json"),
            result_sha256="1" * 64,
        )
    attempts_dir = tmp_path / "attempts"
    frozen_dir = tmp_path / "frozen"
    runner = AnthropicApiRunner(
        source_store,
        attempts_dir=attempts_dir,
        frozen_dir=frozen_dir,
        channels=[AnthropicApiChannel("routify", "https://fixture.invalid", "secret")],
        transport=_FakeTransport(),
        allow_single_channel=True,
    )
    result = runner.execute(
        __import__(
            "AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan",
            fromlist=["load_plan_jsonl"],
        ).load_plan_jsonl(plan_path).entries[0].definition
    )
    assert result.state == "frozen"

    target_db = tmp_path / "target.sqlite3"
    target_store = TaskStateStore(target_db)
    if dependency_key:
        target_store.register_task(dependency)
    target_store.register_task(spec)
    return {
        "attempts_dir": attempts_dir,
        "frozen_dir": frozen_dir,
        "plan_path": plan_path,
        "report_path": tmp_path / "report.json",
        "row": row,
        "target_db": target_db,
    }


def _run(paths: Mapping[str, object]) -> dict[str, object]:
    return import_frozen_plan_results(
        plan_jsonl=paths["plan_path"],
        state_db=paths["target_db"],
        frozen_dir=paths["frozen_dir"],
        attempts_dir=paths["attempts_dir"],
        report_json=paths["report_path"],
    )


def test_imports_valid_api_frozen_result_with_honest_local_audit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_path = Path(paths["frozen_dir"]) / f"{paths['row']['evaluation_key']}.json"
    source_sha = _sha256_bytes(source_path.read_bytes())

    report = _run(paths)

    assert report["status"] == "completed"
    assert report["counts"] == {
        "planned": 1,
        "imported": 1,
        "cached": 0,
        "failed": 0,
        "missing": 0,
    }
    assert TaskStateStore(paths["target_db"]).task_state(paths["row"]["evaluation_key"]) == "frozen"
    assert _sha256_bytes(source_path.read_bytes()) == source_sha
    audit_path = Path(paths["attempts_dir"]) / f"{paths['row']['evaluation_key']}.a01.cache_import.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["provider"] == "cache_import"
    assert audit["transport"] == "local_artifact_rebinding"
    assert audit["network_request"] is False
    assert audit["source_frozen_sha256"] == source_sha
    with sqlite3.connect(paths["target_db"]) as connection:
        assert connection.execute("SELECT status FROM attempts").fetchone()[0] == "accepted"


def test_rejects_tampered_frozen_before_mutating_state(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    frozen_path = Path(paths["frozen_dir"]) / f"{paths['row']['evaluation_key']}.json"
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["metadata"]["response_model"] = "not-opus"
    _write_json(frozen_path, payload)

    with pytest.raises(FrozenPlanImportError, match="response_model"):
        _run(paths)

    assert TaskStateStore(paths["target_db"]).task_state(paths["row"]["evaluation_key"]) == "pending"


def test_second_import_is_idempotent_and_validates_cached_binding(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    _run(paths)
    second = _run(paths)

    assert second["counts"]["imported"] == 0
    assert second["counts"]["cached"] == 1
    with sqlite3.connect(paths["target_db"]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_dependency_not_frozen_is_reported_without_running_attempt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, blocked_dependency=True)

    report = _run(paths)

    assert report["status"] == "incomplete"
    assert report["counts"]["failed"] == 1
    assert report["entries"][0]["failure_class"] == "state_not_ready"
    with sqlite3.connect(paths["target_db"]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_rejects_aborted_fullcpu_source_path(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    forbidden_dir = tmp_path / "all_15alg_fullcpu_v1"
    forbidden_dir.mkdir()
    forbidden_plan = forbidden_dir / "plan.jsonl"
    forbidden_plan.write_bytes(Path(paths["plan_path"]).read_bytes())
    paths["plan_path"] = forbidden_plan

    with pytest.raises(FrozenPlanImportError, match="all_15alg_fullcpu_v1"):
        _run(paths)
