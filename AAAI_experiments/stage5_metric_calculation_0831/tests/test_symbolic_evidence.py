from __future__ import annotations

import gzip
import json
from pathlib import Path
import time

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
        ("items[x0]", {"items", "x0"}, set()),
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


def test_build_symbolic_artifact_supports_numpy_log1p_exactly() -> None:
    artifact = build_symbolic_artifact(
        "np.log1p(x0)",
        allowed_variables={"x0"},
        allowed_functions={"log1p"},
    )

    assert artifact["canonical_expression"] == "log(x0 + 1)"
    assert artifact["function_set"] == ("log1p",)


def test_build_symbolic_artifact_supports_numpy_cosh() -> None:
    artifact = build_symbolic_artifact(
        "np.cosh(x0)",
        allowed_variables={"x0"},
        allowed_functions={"cosh"},
    )

    assert artifact["canonical_expression"] == "cosh(x0)"
    assert artifact["function_set"] == ("cosh",)


def test_build_symbolic_artifact_preserves_numpy_mean_as_opaque() -> None:
    artifact = build_symbolic_artifact(
        "np.mean(x0)",
        allowed_variables={"x0"},
        allowed_functions={"mean"},
    )

    assert artifact["canonical_expression"] == "mean(x0)"
    assert artifact["function_set"] == ("mean",)


def test_build_symbolic_artifact_preserves_numpy_linalg_norm_as_opaque() -> None:
    artifact = build_symbolic_artifact(
        "np.linalg.norm(x0)",
        allowed_variables={"x0"},
        allowed_functions={"norm"},
    )

    assert artifact["canonical_expression"] == "norm(x0)"
    assert artifact["function_set"] == ("norm",)


def test_build_symbolic_artifact_preserves_static_input_index_as_opaque() -> None:
    artifact = build_symbolic_artifact(
        "x2[1] - x3[0]",
        allowed_variables={"x2", "x3"},
        allowed_functions=set(),
    )

    assert artifact["canonical_expression"] == "index(x2, 1) - index(x3, 0)"
    assert artifact["function_set"] == ("index",)


def test_build_symbolic_artifact_supports_scalar_numpy_clip_exactly() -> None:
    artifact = build_symbolic_artifact(
        "np.clip(x0, -1, 1)",
        allowed_variables={"x0"},
        allowed_functions={"clip"},
    )

    assert artifact["canonical_expression"] == "Min(1, Max(-1, x0))"
    assert artifact["function_set"] == ("clip",)

    lower_bounded = build_symbolic_artifact(
        "np.clip(x0, a_min=0, a_max=None)",
        allowed_variables={"x0"},
        allowed_functions={"clip"},
    )
    assert lower_bounded["canonical_expression"] == "Max(0, x0)"
    assert lower_bounded["function_set"] == ("clip",)


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


def test_build_pair_evidence_times_out_expensive_symbolic_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_simplify = symbolic_evidence.sp.simplify

    def slow_simplify(expression):
        time.sleep(0.2)
        return original_simplify(expression)

    monkeypatch.setattr(symbolic_evidence.sp, "simplify", slow_simplify)
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_TIMEOUT_SECONDS", 0.01)

    evidence = build_pair_evidence(
        lhs="x0 * (x1 + x2)",
        rhs="x0 * x1 + x0 * x2",
        allowed_variables={"x0", "x1", "x2"},
        allowed_functions=set(),
        seed=23,
    )

    assert evidence["decision"] == "undetermined"
    assert any(
        "符号证明超时" in item
        for item in evidence["symbolic_difference"]["assumptions"]
    )


def test_build_pair_evidence_times_out_expensive_numeric_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_real_to_float = symbolic_evidence._sympy_real_to_float

    def slow_real_to_float(value, *, digits=50):
        time.sleep(0.2)
        return original_real_to_float(value, digits=digits)

    monkeypatch.setattr(symbolic_evidence, "_sympy_real_to_float", slow_real_to_float)
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_NODE_LIMIT", 1)
    monkeypatch.setattr(symbolic_evidence, "NUMERIC_PROBE_TIMEOUT_SECONDS", 0.01)

    started_at = time.monotonic()
    evidence = build_pair_evidence(
        lhs="x0 + 1",
        rhs="x0 + 2",
        allowed_variables={"x0"},
        allowed_functions=set(),
        seed=23,
    )

    assert time.monotonic() - started_at < 0.5
    assert evidence["decision"] == "undetermined"
    assert evidence["symbolic_difference"]["skipped_probe_reasons"] == {
        "numeric_probe_timeout": 1
    }
    assert any(
        "数值探针超时" in item
        for item in evidence["symbolic_difference"]["assumptions"]
    )


def test_build_pair_evidence_limits_total_numeric_probe_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_real_to_float = symbolic_evidence._sympy_real_to_float

    def moderately_slow_real_to_float(value, *, digits=50):
        time.sleep(0.006)
        return original_real_to_float(value, digits=digits)

    monkeypatch.setattr(
        symbolic_evidence,
        "_sympy_real_to_float",
        moderately_slow_real_to_float,
    )
    monkeypatch.setattr(symbolic_evidence, "SYMBOLIC_PROOF_NODE_LIMIT", 1)
    monkeypatch.setattr(symbolic_evidence, "NUMERIC_PROBE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(symbolic_evidence, "NUMERIC_PROBE_TOTAL_TIMEOUT_SECONDS", 0.03)

    started_at = time.monotonic()
    evidence = build_pair_evidence(
        lhs="x0 * (x1 + x2)",
        rhs="x0 * x1 + x0 * x2",
        allowed_variables={"x0", "x1", "x2"},
        allowed_functions=set(),
        seed=23,
    )

    assert time.monotonic() - started_at < 0.1
    assert evidence["decision"] == "undetermined"
    assert evidence["symbolic_difference"]["skipped_probe_reasons"] == {
        "numeric_probe_timeout": 1
    }


def test_build_pair_evidence_can_skip_tree_distance_for_structure_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("结构裁决不应计算高复杂度树编辑距离")

    monkeypatch.setattr(symbolic_evidence, "_ordered_tree_edit_distance", fail_if_called)

    evidence = build_pair_evidence(
        lhs="x0 * (x1 + x2)",
        rhs="x0 * x1 + x0 * x2",
        allowed_variables={"x0", "x1", "x2"},
        allowed_functions=set(),
        seed=23,
        include_tree_distance=False,
    )

    assert evidence["tree"] == {
        "computed": False,
        "normalized_edit_distance": None,
        "tree_similarity": None,
    }


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


def test_validate_simplification_short_circuits_exact_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression = (
        "x4/x5 + (x4/x5 + (-x2/(x5*(x1 + x2/x5 + 2*x3 + log(sqrt(x5)))) "
        "- x2/(x5*(x1 + 2*x3)) + 2*x4/x5))*sin(sqrt(log(x3)))"
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("精确源码一致时不应进入高成本等价验证")

    monkeypatch.setattr(symbolic_evidence, "_equivalence_core", fail_if_called)
    evidence = validate_simplification(
        original=expression,
        simplified=expression,
        allowed_variables={"x1", "x2", "x3", "x4", "x5"},
        allowed_functions={"log", "sin", "sqrt"},
        seed=520,
        probe_points=[
            {
                "split": "id_test",
                "row_index": 0,
                "values": {"x1": 1.0, "x2": 2.0, "x3": 3.0, "x4": 4.0, "x5": 5.0},
            }
        ],
        probe_source="dataset_probes_v1",
        probe_sample_sha256="a" * 64,
    )

    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_decision"] == "equivalent"
    assert evidence["proof_basis"] == "artifact_identity"
    assert evidence["probe_count"] == 0
    assert evidence["original_sha256"] == evidence["simplified_sha256"]


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


def test_validate_simplification_accepts_exact_decimal_rebuild_for_drsr_g0031() -> None:
    evidence = validate_simplification(
        original=(
            "-0.2934868971342668 + -0.12741047181401519*(1 - cos(x)**2) + "
            "-0.08204681377994838*(2*x - sin(2*x)) + -0.3910717874828409*y + "
            "0.0947660838403801*y*(1 - cos(x)**2) + "
            "0.03951161654272465*y*(2*x - sin(2*x)) + "
            "-0.9964401599100835*cos(x) - -0.356291962406277*sin(x) + "
            "-2.8406151724099575*(1 - sin(x)**2)*cos(y)"
        ),
        simplified=(
            "(0.03951161654272465*y - 0.08204681377994838)*(2*x - sin(2*x)) + "
            "(0.0947660838403801*y - 0.12741047181401519)*sin(x)**2 - "
            "2.8406151724099575*cos(x)**2*cos(y) - 0.9964401599100835*cos(x) + "
            "0.356291962406277*sin(x) - 0.3910717874828409*y - 0.2934868971342668"
        ),
        allowed_variables={"x", "y"},
        allowed_functions={"cos", "sin"},
        seed=522,
        probe_points=[
            {"split": "id_test", "row_index": 2, "values": {"x": -1.29601738954305, "y": -2.02716923268692}},
            {"split": "id_test", "row_index": 7, "values": {"x": -1.52151659891278, "y": -1.20684142398226}},
            {"split": "id_test", "row_index": 12, "values": {"x": -1.41853781260912, "y": -1.58209982848495}},
            {"split": "id_test", "row_index": 15, "values": {"x": -4.32048932008737, "y": 1.44999454562668}},
        ],
        probe_source="dataset_probes_v1",
        probe_sample_sha256="c" * 64,
    )

    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_decision"] == "equivalent"
    assert evidence["proof_basis"] == "symbolic_difference_zero"
    assert evidence["probe_source"] == "dataset_probes_v1"


def test_pair_evidence_skips_expensive_exact_decimal_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_evidence

    def reject_exact_decimal(*args: object, **kwargs: object) -> None:
        raise AssertionError("通用 pair evidence 不应执行高风险精确小数重建")

    monkeypatch.setattr(symbolic_evidence, "_build_exact_decimal_pair", reject_exact_decimal)
    evidence = symbolic_evidence.build_pair_evidence(
        "x1*x2 + sin((x1 - 1)*(x2 - 1))",
        (
            "-0.4938247615860779*x1**2 + 2.082683469582711*x1*x2 "
            "- 0.6380172051626938*x1 + 0.03572694110233234*x2**2 "
            "- 0.2119782419543728*x2 + 1.2333719555333358*sin(x1) "
            "+ 1.5117991637749062*cos(x2) - 1.0065012927817525 "
            "+ 0.014447767606811525*exp(-5.892973355471428*x1*x2)"
        ),
        allowed_variables={"x1", "x2"},
        allowed_functions={"sin", "cos", "exp"},
        seed=520,
        probe_points=[
            {"split": "id_test", "row_index": 0, "values": {"x1": 0.5, "x2": 1.5}},
        ],
        probe_source="dataset_probes_v1",
        probe_sample_sha256="e" * 64,
    )

    assert evidence["decision"] in {"not_equivalent", "undetermined"}


def test_validate_simplification_accepts_high_precision_probe_recheck_for_dso_g0036() -> None:
    evidence = validate_simplification(
        original="x0*(-x1 + exp(x1 - x1/sin(x0 + x1 - (-x0 + x1*(x0 + (x0 - exp(2*x0))/x0))/x0**2)))",
        simplified="x0*(exp(x1 - x1/sin(x0 + x1 + 1/x0 - x1*(x0 + 1)/x0**2 + x1*exp(2*x0)/x0**3)) - x1)",
        allowed_variables={"x0", "x1"},
        allowed_functions={"exp", "sin"},
        seed=520,
        probe_points=[
            {
                "split": "ood_test",
                "row_index": 816,
                "values": {
                    "x0": 5.3863613867106334e-09,
                    "x1": -0.10034954717906915,
                },
            }
        ],
        probe_source="dataset_probes_v1",
        probe_sample_sha256="d" * 64,
    )

    assert evidence["decision"] == "equivalent"
    assert evidence["symbolic_decision"] == "equivalent"
    assert evidence["proof_basis"] == "symbolic_difference_zero"
    assert evidence["probe_count"] == 1
    assert evidence["max_abs_error"] == "0"


def test_validate_simplification_keeps_exact_decimal_rounding_counterexample_for_drsr_g0041() -> None:
    with pytest.raises(SimplificationContractError) as exc_info:
        validate_simplification(
            original=(
                "-0.46670925507939076*0.18330990085757692*x0/"
                "(0.905859281929194*x1*x2 + -0.7430329173164343*x3 + "
                "-0.5894043940212699*x4 + -0.8595230589106997*x5)"
            ),
            simplified=(
                "-0.085552427277916698*x0/"
                "(0.905859281929194*x1*x2 - 0.7430329173164343*x3 - "
                "0.5894043940212699*x4 - 0.8595230589106997*x5)"
            ),
            allowed_variables={"x0", "x1", "x2", "x3", "x4", "x5"},
            allowed_functions=set(),
            seed=520,
        )

    assert exc_info.value.evidence["symbolic_decision"] == "not_equivalent"
    assert exc_info.value.evidence["proof_basis"] == "symbolic_nonzero_exact_difference"
    assert exc_info.value.evidence["counterexample"] is None


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


def test_where_accepts_bitwise_conjunction_of_comparisons() -> None:
    artifact = build_symbolic_artifact(
        "where((x0 >= -2) & (x0 <= -1), 1, 0)",
        allowed_variables={"x0"},
        allowed_functions={"where"},
    )

    assert artifact["variables"] == ("x0",)
    assert "Piecewise" in artifact["canonical_expression"]


def test_nan_literal_is_preserved_as_nonfinite_symbolic_constant() -> None:
    artifact = build_symbolic_artifact(
        "nan",
        allowed_variables={"x0"},
        allowed_functions=set(),
    )

    assert artifact["canonical_expression"] == "nan"
    assert artifact["variables"] == ()


def test_inverse_sine_and_cosine_functions_are_supported() -> None:
    artifact = build_symbolic_artifact(
        "arccos(x0) + arcsin(x0)",
        allowed_variables={"x0"},
        allowed_functions={"arccos", "arcsin"},
    )

    assert artifact["function_set"] == ("arccos", "arcsin")


def test_comparison_indicator_allows_equivalent_explicit_piecewise_notation() -> None:
    evidence = validate_simplification(
        original="2*x0 - 0.5*(x0 >= 1e-6)",
        simplified="2*x0 - 0.5*Piecewise((1, x0 >= 1e-6), (0, True))",
        allowed_variables={"x0"},
        allowed_functions=set(),
        seed=521,
        probe_points=[
            {"split": "id_test", "row_index": 0, "values": {"x0": 0.0}},
            {"split": "ood_test", "row_index": 1, "values": {"x0": 2.0}},
        ],
        probe_source="unit_test",
        probe_sample_sha256="b" * 64,
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


def test_large_nested_radical_avoids_sympy_assumption_explosion() -> None:
    expression = (
        "0.000979 - 2.783098*tanh(tanh((0.923697*x0 + 0.238334*x2)"
        "/sqrt((1.989044*x0 - 0.760356*x2)**2 + 1)))/sqrt(tanh((0.923697*x0 "
        "+ 0.238334*x2)/(sqrt((1.772454*x0 - (exp(0.895912*x2) - "
        "cos(1.772454*x0))/sqrt((1.772454*x0 - 0.238334*x2)**2 + 1))**2 + 1)"
        "*((0.238334*x2 + exp(0.895912*x2) - cos(1.772454*x0))**2 + 1)))**2 + 1)"
    )

    artifact = build_symbolic_artifact(
        expression,
        allowed_variables={"x0", "x2"},
        allowed_functions={"cos", "exp", "sqrt", "tanh"},
    )

    assert artifact["variables"] == ("x0", "x2")
    assert artifact["function_set"] == ("cos", "exp", "sqrt", "tanh")
    assert artifact["construction_mode"] == "unevaluated_large_ast"


def test_legacy_evaluated_artifact_hash_can_be_revalidated() -> None:
    expression = " + ".join(
        f"{index + 1}*x0**{index % 3 + 1}" for index in range(32)
    )
    automatic = build_symbolic_artifact(expression, allowed_variables={"x0"})
    legacy = build_symbolic_artifact(
        expression,
        allowed_variables={"x0"},
        evaluate_expressions=True,
    )

    evidence = validate_simplification(
        original=expression,
        simplified=expression,
        allowed_variables={"x0"},
        allowed_functions=set(),
        seed=520,
        original_construction_mode="evaluated",
    )

    assert automatic["construction_mode"] == "unevaluated_large_ast"
    assert legacy["construction_mode"] == "evaluated"
    assert automatic["artifact_sha256"] != legacy["artifact_sha256"]
    assert evidence["original_sha256"] == legacy["artifact_sha256"]
