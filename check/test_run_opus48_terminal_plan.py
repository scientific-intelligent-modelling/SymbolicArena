"""Offline contract tests for the Opus 4.8 terminal runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

from check import run_opus48_terminal_plan as runner
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_exhausted_simplifications import (
    run_isolated_simplify_semantic_validator,
)


PLAN = Path("AAAI_experiments/stage5_metric_calculation_0831/work/"
            "symbolfit_clean_terminal_refresh_20260916/pred_simplify_plan.jsonl")
DOWNSTREAM = Path("AAAI_experiments/stage5_metric_calculation_0831/work/"
                  "final_release_20260913/release_v2/downstream")
EQUIVALENCE_PLAN = DOWNSTREAM / "clean_equivalence_refresh_plan.jsonl"
STRUCTURE_PLAN = DOWNSTREAM / "clean_structure_refresh_plan.jsonl"


def test_source_plan_uses_distinct_model_bound_keys() -> None:
    rows, _ = runner.load_plan(PLAN)
    assert len(rows) == 150
    assert len({r["opus48_evaluation_key"] for r in rows}) == 150
    assert all(r["opus48_evaluation_key"] != r["evaluation_key"] for r in rows)


def test_sliding_window_waits_at_limit() -> None:
    now = [0.0]
    sleeps: list[float] = []
    persisted: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = runner.SlidingWindowLimiter(2, clock=lambda: now[0], sleeper=sleep)

    async def reserve_three() -> None:
        await limiter.reserve(persisted.append)
        await limiter.reserve(persisted.append)
        await limiter.reserve(persisted.append)

    asyncio.run(reserve_three())
    assert persisted[0] == 0.0
    assert persisted[1] >= 30.0
    assert persisted[2] >= 60.0
    assert len(sleeps) == 2


def test_invalid_json_is_fail_closed() -> None:
    row = runner.load_plan(PLAN)[0][0]
    with pytest.raises(runner.StructuredOutputViolation):
        runner.validate_message({"type": "message", "role": "assistant", "model": runner.MODEL,
                                 "stop_reason": "end_turn", "content": [{"type": "text", "text": "not json"}],
                                 "usage": {"input_tokens": 1, "output_tokens": 2}}, row)


@pytest.mark.parametrize("plan,kind", [
    (EQUIVALENCE_PLAN, "equivalence"),
    (STRUCTURE_PLAN, "stab_structure"),
])
def test_downstream_plan_has_independent_model_key_and_bound_pair(plan, kind) -> None:
    rows, _ = runner.load_plan(plan)
    assert rows
    assert all(row["task_type"] == kind for row in rows)
    assert all(row["opus48_evaluation_key"] != row["evaluation_key"] for row in rows)
    assert len({row["opus48_evaluation_key"] for row in rows}) == len(rows)
    first = rows[0]
    assert first["request"]["evidence_hash"] == first["request"]["deterministic_evidence"]["evidence_sha256"]
    assert first["dependencies"] == [
        first["request"]["deterministic_evidence"]["lhs_binding"]["frozen_evaluation_key"],
        first["request"]["deterministic_evidence"]["rhs_binding"]["frozen_evaluation_key"],
    ]


@pytest.mark.parametrize("plan,good", [
    (EQUIVALENCE_PLAN, {"decision": "not_equivalent", "evidence_basis": "mixed",
                        "assumptions": [], "confidence": 0.8, "brief_reason": "Different polynomial."}),
    (STRUCTURE_PLAN, {"decision": "different_structure", "confidence": 0.8,
                      "brief_reason": "Different operators."}),
])
def test_downstream_schema_and_contract_accept_valid_decision(plan, good) -> None:
    row = runner.load_plan(plan)[0][0]
    output, _, recovery = runner.validate_message(_Response(json.dumps(good)).json(), row)
    assert output == good
    assert recovery is None


@pytest.mark.parametrize("plan,bad", [
    (EQUIVALENCE_PLAN, {"decision": "equivalent", "evidence_basis": "insufficient",
                        "assumptions": [], "confidence": 0.8, "brief_reason": "No proof."}),
    (STRUCTURE_PLAN, {"decision": "equivalent", "confidence": 0.8,
                      "brief_reason": "Unsupported label."}),
])
def test_downstream_invalid_decision_is_not_accepted(plan, bad) -> None:
    row = runner.load_plan(plan)[0][0]
    with pytest.raises(runner.StructuredOutputViolation):
        runner.validate_message(_Response(json.dumps(bad)).json(), row)


def test_nonbare_response_model_is_rejected() -> None:
    row = runner.load_plan(EQUIVALENCE_PLAN)[0][0]
    body = _Response('{"decision":"undetermined","evidence_basis":"insufficient",'
                     '"assumptions":[],"confidence":0.5,"brief_reason":"Unclear."}').json()
    body["model"] = runner.KEY_MODEL
    with pytest.raises(runner.ContractViolation, match="response model drift"):
        runner.validate_message(body, row)


def test_invalid_structure_pair_is_rejected_before_any_http(tmp_path, monkeypatch) -> None:
    row = json.loads(next(STRUCTURE_PLAN.open(encoding="utf-8")))
    row["request"]["prediction_b_valid_output"] = False
    plan = tmp_path / "invalid_pair.jsonl"
    plan.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner.httpx, "AsyncClient", _Client)
    _Client.calls = 0
    with pytest.raises(ValueError, match="invalid seed pair"):
        runner.validate_pair_binding(row, line_number=1)
    with pytest.raises(ValueError):
        runner.load_plan(plan)
    assert _Client.calls == 0


def test_pair_dependency_drift_and_mixed_plan_fail_closed(tmp_path) -> None:
    pair = json.loads(next(STRUCTURE_PLAN.open(encoding="utf-8")))
    pair["dependencies"][0] = "0" * 64
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency keys"):
        runner.load_plan(bad)

    pred = json.loads(next(PLAN.open(encoding="utf-8")))
    pair = json.loads(next(STRUCTURE_PLAN.open(encoding="utf-8")))
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(pred) + "\n" + json.dumps(pair) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task types cannot share"):
        runner.load_plan(mixed)


class _Response:
    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = text

    def json(self) -> dict[str, object]:
        return {"type": "message", "role": "assistant", "model": runner.MODEL,
                "stop_reason": "end_turn", "content": [{"type": "text", "text": self.text}],
                "usage": {"input_tokens": 10, "output_tokens": 10}}


class _Client:
    responses: list[_Response] = []
    calls = 0
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, *args, **kwargs) -> _Response:
        type(self).calls += 1
        return type(self).responses.pop(0)


@pytest.mark.parametrize("plan,good", [
    (EQUIVALENCE_PLAN, {"decision": "equivalent", "evidence_basis": "symbolic_proof",
                        "assumptions": [], "confidence": 0.9, "brief_reason": "Identity."}),
    (STRUCTURE_PLAN, {"decision": "same_canonical_structure", "confidence": 0.9,
                      "brief_reason": "Matching canonical trees."}),
])
def test_downstream_freeze_preserves_pair_evidence_and_resumes(tmp_path, monkeypatch, plan, good) -> None:
    row = runner.load_plan(plan)[0][0]
    monkeypatch.setattr(runner.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(runner, "run_isolated_simplify_semantic_validator",
                        lambda **kwargs: pytest.fail("pair judgment must not run simplify validation"))
    _Client.responses = [_Response(json.dumps(good))]
    _Client.calls = 0
    first = asyncio.run(runner.run([row], runner.Ledger(tmp_path, "pair-plan", 1),
                                   "dummy", concurrency=1, rpm=500, timeout=1))
    assert first["frozen"] == 1
    frozen = json.loads((tmp_path / "frozen" / f"{row['opus48_evaluation_key']}.json").read_text())
    assert frozen["task_type"] == row["task_type"]
    assert frozen["request"] == row["request"]
    assert frozen["plan_dependencies"] == row["dependencies"]
    assert frozen["requested_model"] == frozen["response_model"] == runner.MODEL
    assert frozen["response"]["content"][0]["text"] == json.dumps(good)
    assert frozen["semantic_validation"] is None
    assert frozen["semantic_worker_rlimit_as_bytes"] is None
    resumed = asyncio.run(runner.run([row], runner.Ledger(tmp_path, "pair-plan", 1),
                                     "dummy", concurrency=1, rpm=500, timeout=1))
    assert resumed["frozen"] == 1
    assert _Client.calls == 1
    with pytest.raises(ValueError, match="different task type"):
        runner.Ledger(tmp_path, "pair-plan", 1).bind_task_type("pred_simplify")


def test_two_attempt_cap_and_resume_without_repeating_success(tmp_path, monkeypatch) -> None:
    row = runner.load_plan(PLAN)[0][0]
    monkeypatch.setattr(runner.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(runner, "run_isolated_simplify_semantic_validator",
                        lambda **kwargs: {"status": "promotable", "semantic_evidence": {"decision": "equivalent"}})
    # First pass: both model responses violate JSON, so no result is frozen.
    _Client.responses = [_Response("invalid"), _Response("invalid")]
    _Client.calls = 0
    ledger = runner.Ledger(tmp_path, "plan-a", 1)
    report = asyncio.run(runner.run([row], ledger, "dummy", concurrency=1, rpm=500, timeout=1))
    assert report["frozen"] == 0
    assert report["physical_attempts"] == 2
    assert _Client.calls == 2
    assert json.loads((tmp_path / "unresolved.json").read_text())["items"][0]["attempts"] == 2
    assert all(json.loads(path.read_text())["structured_output"] is None
               for path in (tmp_path / "attempts").glob("*.json"))

    # A separate plan/directory with a schema-valid successful response is resumed without a request.
    fresh = tmp_path / "fresh"
    good = {"outcome": "unchanged", "simplified_expression": row["request"]["expression"],
            "equivalence_assessment": "preserved", "assumptions": [], "confidence": 0.9,
            "brief_reason": "Identity."}
    _Client.responses = [_Response(json.dumps(good))]
    success_ledger = runner.Ledger(fresh, "plan-b", 1)
    first = asyncio.run(runner.run([row], success_ledger, "dummy", concurrency=1, rpm=500, timeout=1))
    assert first["frozen"] == 1
    assert _Client.last_kwargs["trust_env"] is False
    before = _Client.calls
    resumed_ledger = runner.Ledger(fresh, "plan-b", 1)
    second = asyncio.run(runner.run([row], resumed_ledger, "dummy", concurrency=1, rpm=500, timeout=1))
    assert second["frozen"] == 1
    assert _Client.calls == before


def test_batch_limit_keeps_full_plan_ledger_and_advances(tmp_path, monkeypatch) -> None:
    rows = runner.load_plan(PLAN)[0][:2]
    monkeypatch.setattr(runner.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(runner, "run_isolated_simplify_semantic_validator",
                        lambda **kwargs: {"status": "promotable", "semantic_evidence": {"decision": "equivalent"}})
    good = {"outcome": "unchanged", "simplified_expression": "x1",
            "equivalence_assessment": "preserved", "assumptions": [], "confidence": 0.9,
            "brief_reason": "Identity."}
    _Client.responses = [_Response(json.dumps(good)), _Response(json.dumps(good))]
    _Client.calls = 0
    first = asyncio.run(runner.run(rows, runner.Ledger(tmp_path, "full-plan", 2), "dummy",
                                   concurrency=2, rpm=500, timeout=1, max_new_tasks=1))
    assert first["scheduled_this_invocation"] == 1
    assert first["frozen"] == 1
    second = asyncio.run(runner.run(rows, runner.Ledger(tmp_path, "full-plan", 2), "dummy",
                                    concurrency=2, rpm=500, timeout=1, max_new_tasks=1))
    assert second["scheduled_this_invocation"] == 1
    assert second["frozen"] == 2
    assert _Client.calls == 2


def test_stage5_promotable_semantics_is_the_only_freeze_status(tmp_path, monkeypatch) -> None:
    row = runner.load_plan(PLAN)[0][0]
    monkeypatch.setattr(runner.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(runner, "run_isolated_simplify_semantic_validator",
                        lambda **kwargs: {"status": "promotable", "semantic_evidence": {"decision": "equivalent"}})
    good = {"outcome": "unchanged", "simplified_expression": row["request"]["expression"],
            "equivalence_assessment": "preserved", "assumptions": [], "confidence": 0.9,
            "brief_reason": "Identity."}
    _Client.responses = [_Response(json.dumps(good))]
    report = asyncio.run(runner.run([row], runner.Ledger(tmp_path, "promotable", 1),
                                    "dummy", concurrency=1, rpm=500, timeout=1))
    assert report["frozen"] == 1
    frozen = json.loads((tmp_path / "frozen" / f"{row['opus48_evaluation_key']}.json").read_text())
    assert frozen["semantic_validation"]["status"] == "promotable"


@pytest.mark.parametrize("status", ["semantic_rejected", "semantic_validator_timeout"])
def test_rejected_or_timed_out_semantics_never_freezes(tmp_path, monkeypatch, status) -> None:
    row = runner.load_plan(PLAN)[0][0]
    monkeypatch.setattr(runner.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(runner, "run_isolated_simplify_semantic_validator",
                        lambda **kwargs: {"status": status, "error": "not verified"})
    good = {"outcome": "unchanged", "simplified_expression": row["request"]["expression"],
            "equivalence_assessment": "preserved", "assumptions": [], "confidence": 0.9,
            "brief_reason": "Identity."}
    _Client.responses = [_Response(json.dumps(good)), _Response(json.dumps(good))]
    report = asyncio.run(runner.run([row], runner.Ledger(tmp_path, status, 1),
                                    "dummy", concurrency=1, rpm=500, timeout=1))
    assert report["frozen"] == 0
    assert report["physical_attempts"] == 2


@pytest.mark.parametrize("surround,recovery", [
    ("result:\n```json\n{}\n```", "single_valid_json_fence"),
    ("result:\n{}\nend", "single_valid_embedded_json_object"),
])
def test_unique_embedded_json_can_be_recovered(surround, recovery) -> None:
    row = runner.load_plan(PLAN)[0][0]
    good = {"outcome": "unchanged", "simplified_expression": row["request"]["expression"],
            "equivalence_assessment": "preserved", "assumptions": [], "confidence": 0.9,
            "brief_reason": "Identity."}
    text = surround.format(json.dumps(good))
    output, _, actual_recovery = runner.validate_message(_Response(text).json(), row)
    assert output == good
    assert actual_recovery == recovery


def test_isolated_helper_accepts_bounded_worker_command() -> None:
    command = [sys.executable, "-c", "import json; print(json.dumps({"
               "'status':'ok','semantic_evidence':{'decision':'equivalent'}}))"]
    result = run_isolated_simplify_semantic_validator(
        evaluation_key="test", request={}, structured_output={},
        timeout_seconds=2, worker_command=command)
    assert result["status"] == "promotable"


def test_existing_frozen_response_replays_under_two_gib_limit() -> None:
    frozen_path = Path(
        "AAAI_experiments/stage5_metric_calculation_0831/work/"
        "core50_terminal_collection_20260916_v3/opus48_prediction_run_20260916/frozen/"
        "2a9c2b039aea1d054db74d917476bf5de31ca4cd92b408a22fb46489546f5e85.json"
    )
    if not frozen_path.exists():
        pytest.skip("local frozen response not available")
    frozen = json.loads(frozen_path.read_text())
    result = run_isolated_simplify_semantic_validator(
        evaluation_key=frozen["evaluation_key"], request=frozen["request"],
        structured_output=frozen["structured_output"], timeout_seconds=15,
        worker_command=runner.limited_semantic_worker_command(2 * 1024**3))
    assert result["status"] == "promotable"


def test_memory_limited_worker_failure_is_not_promotable() -> None:
    result = run_isolated_simplify_semantic_validator(
        evaluation_key="test", request={}, structured_output={}, timeout_seconds=5,
        worker_command=runner.limited_semantic_worker_command(16 * 1024**2))
    assert result["status"] != "promotable"
