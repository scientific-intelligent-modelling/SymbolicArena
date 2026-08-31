from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence as symbolic_evidence
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    SimplificationContractError,
    SymbolicEvidenceError,
    build_symbolic_artifact,
    build_pair_evidence,
    normalized_tree_edit_distance,
    operator_f1,
    tree_similarity,
    validate_simplification,
    variable_f1,
)


STAGE = Path("AAAI_experiments/stage5_metric_calculation_0831")
GT_TASKS = STAGE / "reports/clean_gt_simplify_tasks.jsonl"
FREEZE_GLOB = "source_snapshot/trajectory_freeze/clean_freeze_*.jsonl.gz"


def _iter_gt_requests() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with GT_TASKS.open() as handle:
        for line in handle:
            task = json.loads(line)
            rows.append(task["normalized_input"]["request"])
    return rows


def _preferred_prediction_source(payload: dict[str, object]) -> str:
    canonical_artifact = payload.get("canonical_artifact")
    if isinstance(canonical_artifact, dict):
        for key in (
            "instantiated_expression",
            "return_expression_source",
            "normalized_expression",
            "raw_equation",
        ):
            value = canonical_artifact.get(key)
            if isinstance(value, str) and value.strip():
                return value
    equation = payload.get("equation")
    if isinstance(equation, str) and equation.strip():
        return equation
    raise AssertionError(f"payload 缺少可解析表达式: {payload.get('tool')}::{payload.get('dataset')}")


def _iter_clean_prediction_payloads() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle_path in sorted(STAGE.glob(FREEZE_GLOB)):
        with gzip.open(bundle_path, "rt") as handle:
            for line in handle:
                outer = json.loads(line)
                raw_text = outer["result"]["raw_text"]
                rows.append(json.loads(raw_text))
    return rows


@pytest.mark.parametrize(
    ("expression", "allowed_variables", "allowed_functions"),
    [
        ("__import__('os').system('id')", {"x0"}, set()),
        ("(lambda x: x)(x0)", {"x0"}, set()),
        ("obj.attr", {"x0"}, set()),
        ("items[0]", {"items"}, set()),
        ("[x for x in data]", {"data"}, set()),
    ],
)
def test_build_symbolic_artifact_rejects_malicious_inputs(
    expression: str,
    allowed_variables: set[str],
    allowed_functions: set[str],
) -> None:
    with pytest.raises(SymbolicEvidenceError):
        build_symbolic_artifact(
            expression,
            allowed_variables=allowed_variables,
            allowed_functions=allowed_functions,
        )


def test_function_definition_cannot_bypass_allowed_variables() -> None:
    with pytest.raises(SymbolicEvidenceError, match="未授权变量"):
        build_symbolic_artifact(
            "def equation(x0, injected):\n    return x0 + injected",
            allowed_variables={"x0"},
            allowed_functions=set(),
        )


def test_real_ground_truth_tasks_all_parse() -> None:
    requests = _iter_gt_requests()
    artifacts = [
        build_symbolic_artifact(
            request["expression"],
            allowed_variables=set(request["variables"]),
            allowed_functions=set(request["allowed_functions"]),
        )
        for request in requests
    ]
    assert len(artifacts) == 50
    assert {artifact["artifact_sha256"] for artifact in artifacts}


def test_real_clean_prediction_formulas_all_parse() -> None:
    payloads = _iter_clean_prediction_payloads()
    artifacts = [build_symbolic_artifact(_preferred_prediction_source(payload)) for payload in payloads]
    assert len(artifacts) == 2250
    assert any("piecewise" in artifact["operator_set"] for artifact in artifacts)
    assert any("gradient" in artifact["function_set"] for artifact in artifacts)
    assert any("params__neg_1" in artifact["variables"] for artifact in artifacts)


def test_artifact_hash_is_stable() -> None:
    expression = "np.where(np.abs(x0) > 0.001, np.divide(x1 + x2, x0), 1.0)"
    kwargs = {
        "allowed_variables": {"x0", "x1", "x2"},
        "allowed_functions": {"where", "abs", "divide"},
    }
    first = build_symbolic_artifact(expression, **kwargs)
    second = build_symbolic_artifact(expression, **kwargs)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["canonical_expression"] == second["canonical_expression"]
    assert first["canonical_tree"] == second["canonical_tree"]
    assert first["constants_abstracted_tree_fingerprint"] == second["constants_abstracted_tree_fingerprint"]


def test_tree_distance_similarity_and_f1_are_deterministic() -> None:
    lhs = build_symbolic_artifact("x0 + x1")
    rhs = build_symbolic_artifact("x0 - x1")
    distance = normalized_tree_edit_distance(lhs, rhs)
    similarity = tree_similarity(lhs, rhs)
    assert distance == normalized_tree_edit_distance(rhs, lhs)
    assert similarity == tree_similarity(rhs, lhs)
    assert 0.0 <= distance <= 1.0
    assert 0.0 <= similarity <= 1.0
    assert pytest.approx(distance + similarity, rel=0, abs=1e-12) == 1.0
    assert variable_f1(lhs, rhs) == 1.0
    assert operator_f1(lhs, rhs) == pytest.approx(2.0 / 3.0, rel=0, abs=1e-12)


def test_tree_distance_uses_ordered_tree_edit_distance_not_token_levenshtein() -> None:
    lhs = build_symbolic_artifact("x0")
    rhs = build_symbolic_artifact("x0 + x1")
    # 插入 Add 根节点并插入第二个 Symbol，共 2 次操作；总节点数为 1 + 3。
    assert normalized_tree_edit_distance(lhs, rhs) == pytest.approx(0.5, rel=0, abs=1e-12)


def test_constants_abstracted_fingerprint_supports_structure_check() -> None:
    lhs = build_symbolic_artifact("2 * x0 + 3")
    rhs = build_symbolic_artifact("7 * x0 + 9")
    assert lhs["artifact_sha256"] != rhs["artifact_sha256"]
    assert lhs["constants_abstracted_tree_fingerprint"] == rhs["constants_abstracted_tree_fingerprint"]
    assert lhs["constants_abstracted_canonical_tree"] == rhs["constants_abstracted_canonical_tree"]


def test_operator_f1_uses_operator_set_semantics() -> None:
    lhs = build_symbolic_artifact("x0 + x1 + x2")
    rhs = build_symbolic_artifact("x0 + x1")
    assert operator_f1(lhs, rhs) == 1.0


def test_build_pair_evidence_is_serializable_and_deterministic() -> None:
    kwargs = {
        "lhs": "x0 * (x1 + x2)",
        "rhs": "x0 * x1 + x0 * x2",
        "allowed_variables": {"x0", "x1", "x2"},
        "allowed_functions": set(),
        "seed": 23,
    }
    first = build_pair_evidence(**kwargs)
    second = build_pair_evidence(**kwargs)
    assert first["decision"] == "equivalent"
    assert first["symbolic_difference"]["decision"] == "equivalent"
    assert first["tree"]["normalized_edit_distance"] >= 0.0
    assert first["tree"]["tree_similarity"] <= 1.0
    assert first["numeric_probes"] == first["symbolic_difference"]["numeric_probes"]
    assert first["probe_hash"] == first["symbolic_difference"]["probe_hash"]
    assert "max_abs_error" in first
    assert "max_rel_error" in first
    assert "max_tolerance" in first
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["lhs_artifact"]["artifact_sha256"] == second["lhs_artifact"]["artifact_sha256"]
    json.dumps(first, ensure_ascii=False, sort_keys=True)


def test_build_pair_evidence_reports_counterexample_and_variable_gap() -> None:
    evidence = build_pair_evidence(
        lhs="x0 + x1",
        rhs="x0",
        allowed_variables={"x0", "x1"},
        allowed_functions=set(),
        seed=29,
    )
    assert evidence["decision"] == "not_equivalent"
    assert evidence["symbolic_difference"]["counterexample"]
    assert evidence["variable"]["f1"] == pytest.approx(2.0 / 3.0, rel=0, abs=1e-12)


def test_build_pair_evidence_never_upgrades_probe_only_match_to_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_NODE_LIMIT", 1)
    evidence = build_pair_evidence(
        lhs="x0 * (x1 + x2)",
        rhs="x0 * x1 + x0 * x2",
        allowed_variables={"x0", "x1", "x2"},
        allowed_functions=set(),
        seed=23,
    )
    assert evidence["decision"] == "undetermined"
    assert evidence["symbolic_difference"]["decision"] == "undetermined"
    assert evidence["probe_count"] > 0
    assert evidence["numeric_probes"]
    assert any("复杂度超过符号证明阈值" in item for item in evidence["symbolic_difference"]["assumptions"])


def test_build_pair_evidence_external_probes_preserve_undetermined_and_record_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_NODE_LIMIT", 1)
    evidence = build_pair_evidence(
        lhs="(x0 + x1) / x1",
        rhs="x0 / x1 + 1",
        allowed_variables={"x0", "x1"},
        allowed_functions=set(),
        seed=101,
        probe_points=[
            {"split": "id_test", "row_index": 3, "values": {"x0": 1.0, "x1": 2.0}},
            {"split": "ood_test", "row_index": 9, "values": {"x0": 1.0, "x1": 0.0}},
        ],
        probe_source="dataset_probes_v1",
        probe_sample_sha256="a" * 64,
    )
    assert evidence["decision"] == "undetermined"
    assert evidence["symbolic_difference"]["decision"] == "undetermined"
    assert evidence["probe_source"] == "dataset_probes_v1"
    assert evidence["probe_sample_sha256"] == "a" * 64
    assert evidence["probe_count"] == 1
    assert evidence["skipped_probe_count"] == 1
    assert evidence["skipped_probe_reasons"] == {"both_nonfinite": 1}
    assert evidence["numeric_probes"][0]["split"] == "id_test"
    assert evidence["numeric_probes"][0]["row_index"] == 3
    assert evidence["symbolic_difference"]["skipped_probes"][0]["split"] == "ood_test"


def test_build_pair_evidence_external_probes_report_counterexample() -> None:
    evidence = build_pair_evidence(
        lhs="x0 + x1",
        rhs="x0 - x1",
        allowed_variables={"x0", "x1"},
        allowed_functions=set(),
        seed=31,
        probe_points=[
            {"split": "id_test", "row_index": 1, "values": {"x0": 1.0, "x1": 2.0}},
        ],
        probe_source="dataset_probes_v1",
    )
    assert evidence["decision"] == "not_equivalent"
    assert evidence["probe_source"] == "dataset_probes_v1"
    assert evidence["symbolic_difference"]["counterexample"]["split"] == "id_test"
    assert evidence["symbolic_difference"]["counterexample"]["row_index"] == 1


def test_same_artifact_short_circuit_is_deterministic_symbolic_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_NODE_LIMIT", 1)
    evidence = build_pair_evidence(
        lhs="gradient(x0)",
        rhs="gradient(x0)",
        allowed_variables={"x0"},
        allowed_functions={"gradient"},
        seed=19,
    )
    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_difference"]["proof_basis"] == "artifact_identity"
    assert evidence["probe_count"] == 0
    assert evidence["numeric_probes"] == []


def test_relative_tolerance_is_used_for_numeric_counterexample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_NODE_LIMIT", 1)
    monkeypatch.setattr(
        symbolic_evidence,
        "_probe_points",
        lambda variable_names, seed, target_count: [{"x0": 1.0}],
    )
    evidence = build_pair_evidence(
        lhs="1000000000 + x0",
        rhs="1000000000.5 + x0",
        allowed_variables={"x0"},
        allowed_functions=set(),
        seed=7,
        tolerance=0.1,
        rel_tolerance=1e-9,
    )
    assert evidence["decision"] == "undetermined"
    assert evidence["probe_count"] == 1
    probe = evidence["numeric_probes"][0]
    assert probe["abs_error"] == "0.5"
    assert float(probe["rel_error"]) == pytest.approx(0.5 / 1000000001.5, rel=0, abs=1e-18)
    assert float(probe["tolerance"]) == pytest.approx(0.1 + 1e-9 * 1000000001.5, rel=0, abs=1e-12)


def test_validate_simplification_accepts_equivalent_result() -> None:
    evidence = validate_simplification(
        original="np.where(np.abs(x0) > 0.001, np.divide(x1 + x2, x0), 1.0)",
        simplified="np.where(np.abs(x0) > 0.001, (x1 + x2) / x0, 1.0)",
        allowed_variables={"x0", "x1", "x2"},
        allowed_functions={"where", "abs", "divide"},
        seed=7,
    )
    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_decision"] == "equivalent"
    assert evidence["probe_count"] > 0


def test_validate_simplification_rejects_counterexample() -> None:
    with pytest.raises(SimplificationContractError) as exc_info:
        validate_simplification(
            original="x0 + x1",
            simplified="x0 - x1",
            allowed_variables={"x0", "x1"},
            allowed_functions=set(),
            seed=11,
        )
    assert exc_info.value.evidence["decision"] == "contract_error"
    assert exc_info.value.evidence["counterexample"]
    assert "rel_error" in exc_info.value.evidence["counterexample"]
    assert "tolerance" in exc_info.value.evidence["counterexample"]


def test_validate_simplification_external_probe_counterexample_is_preserved() -> None:
    with pytest.raises(SimplificationContractError) as exc_info:
        validate_simplification(
            original="x0 + x1",
            simplified="x0 - x1",
            allowed_variables={"x0", "x1"},
            allowed_functions=set(),
            seed=11,
            probe_points=[
                {"split": "ood_test", "row_index": 7, "values": {"x0": 1.0, "x1": 2.0}},
            ],
            probe_source="dataset_probes_v1",
            probe_sample_sha256="b" * 64,
        )
    assert exc_info.value.evidence["probe_source"] == "dataset_probes_v1"
    assert exc_info.value.evidence["probe_sample_sha256"] == "b" * 64
    assert exc_info.value.evidence["counterexample"]["split"] == "ood_test"
    assert exc_info.value.evidence["counterexample"]["row_index"] == 7


def test_validate_simplification_rejects_new_function_and_variable() -> None:
    with pytest.raises(SymbolicEvidenceError):
        validate_simplification(
            original="x0 + x1",
            simplified="sqrt(x0) + x2",
            allowed_variables={"x0", "x1"},
            allowed_functions=set(),
            seed=5,
        )


def test_piecewise_abs_aliases_preserve_literal_where_semantics() -> None:
    original = (
        "0.23 + 14.2*np.where(np.abs(3*x5) > 0.001, "
        "np.divide(x2 + x4, 3*x5), 1.0)"
    )
    simplified = (
        "0.23 + 14.2*Piecewise(((x2 + x4)/(3*x5), "
        "Abs(3*x5) > 0.001), (1.0, True))"
    )

    evidence = validate_simplification(
        original=original,
        simplified=simplified,
        allowed_variables={"x2", "x4", "x5"},
        allowed_functions={"where", "abs", "divide"},
        seed=520,
        probe_points=[
            {
                "split": "id_test",
                "row_index": 0,
                "values": {"x2": 2.0, "x4": 1.0, "x5": 0.5},
            },
            {
                "split": "ood_test",
                "row_index": 1,
                "values": {"x2": -1.0, "x4": 4.0, "x5": 0.0},
            },
        ],
        probe_source="unit_test",
        probe_sample_sha256="a" * 64,
    )

    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_decision"] == "equivalent"


def test_special_function_identity_is_equivalent_via_artifact_short_circuit() -> None:
    evidence = validate_simplification(
        original="gradient(x0)",
        simplified="gradient(x0)",
        allowed_variables={"x0"},
        allowed_functions={"gradient"},
        seed=19,
    )
    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_decision"] == "equivalent"
    assert evidence["proof_basis"] == "artifact_identity"
    assert evidence["probe_count"] == 0
