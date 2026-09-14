from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check import run_three_model_simplification_review as review  # noqa: E402


def _record(
    logical_id: str,
    original: str,
    effective: str,
    *,
    outcome: str = "simplified",
) -> dict[str, Any]:
    return {
        "logical_id": logical_id,
        "input_expression": original,
        "effective_expression": effective,
        "structured_output": {
            "outcome": outcome,
            "simplified_expression": effective,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _release_fixture(tmp_path: Path, *, n_per_stratum: int = 6) -> Path:
    root = tmp_path / "Core50_final_20260914"
    _write_jsonl(
        root / "ground_truth/opus5_ground_truth.jsonl",
        [
            _record(f"gt_simplify::g{i:04d}::v2", f"x + {i} - {i}", "x")
            for i in range(n_per_stratum)
        ]
        + [
            _record(
                f"gt_simplify::unchanged{i:04d}::v2",
                f"x + {i}",
                f"x + {i}",
                outcome="unchanged",
            )
            for i in range(n_per_stratum)
        ],
    )
    for condition in review.CONDITIONS:
        rows: list[dict[str, Any]] = []
        for algorithm in ("algo_a", "algo_b"):
            for index in range(n_per_stratum):
                rows.append(
                    _record(
                        f"pred_simplify::{algorithm}::g{index:04d}::s520::{condition}",
                        f"({algorithm}_{condition}_{index}) + 0",
                        f"{algorithm}_{condition}_{index}",
                    )
                )
        for algorithm in ("algo_a", "algo_b"):
            for index in range(2):
                rows.append(
                    _record(
                        f"pred_simplify::{algorithm}::unchanged{index}::s520::{condition}",
                        f"{algorithm}_{condition}_unchanged_{index}",
                        f"{algorithm}_{condition}_unchanged_{index}",
                        outcome="unchanged",
                    )
                )
        # 重复 pair 只能在 universe 中出现一次。
        rows.append(_record("pred_simplify::algo_b::duplicate::s521::" + condition, "dup + 0", "dup"))
        rows.append(_record("pred_simplify::algo_a::duplicate::s522::" + condition, "dup + 0", "dup"))
        _write_jsonl(root / f"results/{condition}/opus5_prediction.jsonl", rows)
    return root


def _valid_verdict(verdict: str = "pass") -> dict[str, Any]:
    is_pass = verdict == "pass"
    return {
        "equivalence": "pass" if is_pass else verdict,
        "domain_preserved": is_pass,
        "variable_mapping_preserved": is_pass,
        "operator_semantics_preserved": is_pass,
        "simplicity": "simpler",
        "verdict": verdict,
        "confidence": 0.98,
        "brief_reason": "The two expressions are exactly equivalent.",
        "counterexample": None,
    }


def test_universe_preserves_task_identity_and_excludes_unable(tmp_path: Path) -> None:
    root = _release_fixture(tmp_path, n_per_stratum=2)

    universe = review.build_review_universe(root)

    assert {item["outcome"] for item in universe} == {"simplified", "unchanged"}
    assert sum(item["original_expression"] == "dup + 0" for item in universe) == 6
    assert len({item["review_key"] for item in universe}) == len(universe)
    assert len({item["pair_sha256"] for item in universe}) < len(universe)
    assert {
        occurrence["source_kind"]
        for row in universe
        for occurrence in row["occurrences"]
    } == {"ground_truth", "prediction"}


def test_stratified_sample_is_deterministic_and_covers_algorithm_conditions(tmp_path: Path) -> None:
    universe = review.build_review_universe(_release_fixture(tmp_path, n_per_stratum=8))

    first = review.stratified_sample(universe, sample_size=20, seed=20260914)
    second = review.stratified_sample(list(reversed(universe)), sample_size=20, seed=20260914)

    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len({row["review_key"] for row in first}) == 20
    prediction_strata = {
        (occurrence["algorithm"], occurrence["condition"])
        for item in first
        for occurrence in item["occurrences"]
        if occurrence["source_kind"] == "prediction"
    }
    assert prediction_strata == {
        (algorithm, condition)
        for algorithm in ("algo_a", "algo_b")
        for condition in review.CONDITIONS
    }


def test_request_caps_are_frozen_for_one_hundred_by_three() -> None:
    assert review.logical_task_count(100) == 300
    assert review.physical_request_cap(100) == 450
    assert review.MAX_ATTEMPTS_PER_TASK == 2
    with pytest.raises(review.ReviewError, match="sample_size"):
        review.logical_task_count(0)


def test_profiles_route_models_to_the_correct_protocol_without_leaking_tokens(tmp_path: Path) -> None:
    anthropic_profile = tmp_path / "anthropic.json"
    anthropic_profile.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": review.ANTHROPIC_BASE_URL,
                    "ANTHROPIC_AUTH_TOKEN": "sk-anthropic-secret",
                }
            }
        ),
        encoding="utf-8",
    )
    openai_profile = tmp_path / "openai.toml"
    openai_profile.write_text(
        "\n".join(
            [
                "[model_providers.custom]",
                'base_url = "https://routify-pub.alibaba-inc.com/protocol/openai/v1"',
                'wire_api = "responses"',
                'http_headers = {Authorization = "Bearer sk-openai-secret"}',
            ]
        ),
        encoding="utf-8",
    )

    anthropic = review.load_anthropic_profile(anthropic_profile)
    openai = review.load_openai_profile(openai_profile, provider="custom")
    derived_openai = review.derive_openai_profile_from_anthropic(anthropic)
    kimi = review.build_http_request(review.MODELS["kimi-k3"], anthropic, "fixture")
    glm = review.build_http_request(review.MODELS["glm-5.2"], anthropic, "fixture")
    gpt = review.build_http_request(review.MODELS["gpt-5.6-sol"], openai, "fixture")

    assert kimi.url.endswith("/protocol/anthropic/v1/messages")
    assert glm.url == kimi.url
    assert json.loads(kimi.body)["stream"] is False
    assert json.loads(glm.body)["model"] == "glm-5.2"
    assert gpt.url.endswith("/protocol/openai/v1/responses")
    gpt_body = json.loads(gpt.body)
    assert gpt_body["stream"] is False
    assert gpt_body["store"] is False
    assert gpt_body["text"]["format"]["strict"] is True
    assert derived_openai.base_url == review.OPENAI_BASE_URL
    assert derived_openai.headers["Authorization"] == "Bearer sk-anthropic-secret"
    safe = json.dumps([
        anthropic.safe_metadata,
        openai.safe_metadata,
        derived_openai.safe_metadata,
        kimi.safe_metadata,
        gpt.safe_metadata,
    ])
    assert "sk-anthropic-secret" not in safe
    assert "sk-openai-secret" not in safe


def test_strict_parser_accepts_fenced_json_and_rejects_extra_or_inconsistent_fields() -> None:
    parsed = review.parse_review_json("```json\n" + json.dumps(_valid_verdict()) + "\n```")
    assert parsed["verdict"] == "pass"

    extra = _valid_verdict()
    extra["unexpected"] = None
    with pytest.raises(review.ReviewError, match="字段"):
        review.parse_review_json(json.dumps(extra))

    inconsistent = _valid_verdict()
    inconsistent["domain_preserved"] = False
    with pytest.raises(review.ReviewError, match="pass"):
        review.parse_review_json(json.dumps(inconsistent))


def test_common_domain_policy_accepts_domain_only_exp_log_power_rewrite() -> None:
    verdict = _valid_verdict()
    verdict.update(
        {
            "domain_preserved": False,
            "brief_reason": (
                "exp(x*log(x)/d) and x**(x/d) agree wherever both real-valued; "
                "the power form merely has a larger real domain."
            ),
        }
    )

    parsed = review.parse_review_json(
        json.dumps(verdict), review_policy="common_domain_approx"
    )

    assert parsed["verdict"] == "pass"
    assert parsed["domain_preserved"] is False
    with pytest.raises(review.ReviewError, match="pass"):
        review.parse_review_json(json.dumps(verdict), review_policy="strict")


def test_common_domain_policy_accepts_fitted_constant_rounding_within_tolerance() -> None:
    verdict = _valid_verdict()
    verdict["brief_reason"] = (
        "The fitted coefficients differ only within abs_tol=1e-9 and rel_tol=1e-6."
    )

    parsed = review.parse_review_json(
        json.dumps(verdict), review_policy="common_domain_approx"
    )

    assert parsed["verdict"] == "pass"


@pytest.mark.parametrize(
    ("expression_pair", "failed_field"),
    [
        (("sqrt(x**2)", "x"), "equivalence"),
        (("x_0 + x_1", "x_0 + x_2"), "variable_mapping_preserved"),
    ],
)
def test_common_domain_policy_still_rejects_value_or_variable_changes(
    expression_pair: tuple[str, str], failed_field: str
) -> None:
    verdict = _valid_verdict("fail")
    verdict.update(
        {
            "domain_preserved": True,
            "variable_mapping_preserved": failed_field != "variable_mapping_preserved",
            "operator_semantics_preserved": True,
            "brief_reason": f"Counterexample for {expression_pair!r}.",
            "counterexample": "x=-1" if expression_pair[0] == "sqrt(x**2)" else "x_1 != x_2",
        }
    )

    parsed = review.parse_review_json(
        json.dumps(verdict), review_policy="common_domain_approx"
    )

    assert parsed["verdict"] == "fail"
    assert parsed[failed_field] in {False, "fail"}


def test_review_policy_changes_prompt_version_key_and_contract() -> None:
    sample = {
        "sample_id": "sample_0001",
        "review_key": "b" * 64,
        "pair_sha256": "a" * 64,
        "original_expression": "exp(x*log(x))",
        "effective_expression": "x**x",
        "occurrences": [],
    }

    strict = review.apply_review_policy(sample, "strict")
    relaxed = review.apply_review_policy(sample, "common_domain_approx")
    prompt = review.render_review_prompt(relaxed, review_policy="common_domain_approx")

    assert strict["review_key"] != relaxed["review_key"]
    assert strict["prompt_version"] != relaxed["prompt_version"]
    assert "strict" in strict["prompt_version"]
    assert "common_domain_approx" in relaxed["prompt_version"]
    assert "abs_tol=1e-9" in prompt
    assert "rel_tol=1e-6" in prompt
    assert "common real-valued domain" in prompt
    assert "domain expansion or contraction alone" in prompt


def test_sample_jsonl_reuses_same_items_but_rekeys_for_v2(tmp_path: Path) -> None:
    source = tmp_path / "sample-v1.jsonl"
    rows = [
        {
            "sample_id": f"sample_{index:04d}",
            "review_key": str(index) * 64,
            "pair_sha256": "a" * 64,
            "original_expression": f"x + {index} - {index}",
            "effective_expression": "x",
            "occurrences": [],
        }
        for index in range(1, 4)
    ]
    _write_jsonl(source, rows)

    reused = review.prepare_sample(
        release_root=tmp_path / "unused",
        output_root=tmp_path / "audit-v2",
        sample_size=3,
        seed=20260914,
        review_policy="common_domain_approx",
        sample_jsonl=source,
    )

    assert [row["sample_id"] for row in reused] == [row["sample_id"] for row in rows]
    assert [row["original_expression"] for row in reused] == [
        row["original_expression"] for row in rows
    ]
    assert all(row["review_key"] != source_row["review_key"] for row, source_row in zip(reused, rows))


def test_cli_defaults_to_v2_and_keeps_policy_output_roots_separate() -> None:
    parser = review.build_parser()
    relaxed = parser.parse_args(["preflight"])
    strict = parser.parse_args(["preflight", "--review-policy", "strict"])

    assert relaxed.review_policy == "common_domain_approx"
    assert review._resolve_output_root(relaxed) == review.DEFAULT_OUTPUT_ROOT_V2.resolve()
    assert review._resolve_output_root(strict) == review.DEFAULT_OUTPUT_ROOT.resolve()
    assert review._resolve_output_root(relaxed) != review._resolve_output_root(strict)


def test_task_record_is_redacted_and_completed_task_resumes_without_request(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "audit"
    sample = {
        "sample_id": "sample_0001",
        "review_key": "b" * 64,
        "pair_sha256": "a" * 64,
        "original_expression": "x + 0",
        "effective_expression": "x",
        "occurrences": [],
    }
    profile = review.ApiProfile(
        protocol="anthropic",
        base_url=review.ANTHROPIC_BASE_URL,
        headers={"x-api-key": "sk-never-write-this"},
        safe_metadata={"api_host": "routify-pub.alibaba-inc.com"},
    )
    calls: list[str] = []

    def transport(request: review.HttpRequest, timeout: float) -> review.HttpResponse:
        calls.append(request.url)
        return review.HttpResponse(
            status=200,
            payload={
                "id": "msg_fixture",
                "model": "kimi-k3",
                "content": [{"type": "text", "text": json.dumps(_valid_verdict())}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

    budget = review.RequestBudget(output_root / "request_budget.json", cap=450)
    first = review.review_one_task(
        sample=sample,
        model=review.MODELS["kimi-k3"],
        profile=profile,
        output_root=output_root,
        budget=budget,
        transport=transport,
        timeout=1,
    )
    second = review.review_one_task(
        sample=sample,
        model=review.MODELS["kimi-k3"],
        profile=profile,
        output_root=output_root,
        budget=budget,
        transport=transport,
        timeout=1,
    )

    assert first["status"] == second["status"] == "completed"
    assert len(calls) == 1
    persisted = next((output_root / "per_model/kimi-k3/tasks").glob("*.json")).read_text()
    assert "sk-never-write-this" not in persisted
    assert json.loads(persisted)["review"]["verdict"] == "pass"


def test_request_budget_is_persistent_and_hard_capped(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    budget = review.RequestBudget(path, cap=2)
    assert budget.reserve("task-a", 1) == 1
    assert budget.reserve("task-a", 2) == 2
    with pytest.raises(review.RequestCapExceeded):
        budget.reserve("task-b", 1)
    assert review.RequestBudget(path, cap=2).used == 2


def test_consensus_outputs_majority_and_disagreement_files(tmp_path: Path) -> None:
    sample = {
        "sample_id": "sample_0001",
        "review_key": "b" * 64,
        "pair_sha256": "a" * 64,
        "original_expression": "x + 0",
        "effective_expression": "x",
        "occurrences": [],
    }
    records = {
        "kimi-k3": {"status": "completed", "review": _valid_verdict("pass")},
        "glm-5.2": {"status": "completed", "review": _valid_verdict("pass")},
        "gpt-5.6-sol": {"status": "completed", "review": _valid_verdict("fail")},
    }

    rows = review.write_consensus_artifacts(tmp_path, [sample], {"sample_0001": records})

    assert rows[0]["consensus"] == "accepted_with_dissent"
    with (tmp_path / "consensus.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["pass_count"] == "2"
    with (tmp_path / "disagreements.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1
