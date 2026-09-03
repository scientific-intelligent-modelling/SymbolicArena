from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _base_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in (
        "gt_plan",
        "gt_index",
        "pred_plan",
        "pred_index",
        "equivalence_plan",
        "equivalence_index",
        "structure_plan",
        "structure_index",
        "evidence",
    ):
        paths[name] = tmp_path / f"{name}.jsonl"
    return paths


def _simplify_request(
    *, dataset_id: str, expression: str, algorithm: str | None = None, seed: int | None = None
) -> dict[str, object]:
    request: dict[str, object] = {
        "dataset_id": dataset_id,
        "original_expression": expression,
        "expression": expression,
        "allowed_functions": [],
        "variables": ["x"],
        "dataset_probe_evidence": {
            "schema_version": "dataset_probes_v1",
            "sample_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
            "variables": ["x"],
            "points": [
                {"split": "id_test", "row_index": 0, "values": {"x": 1.0}},
                {"split": "ood_test", "row_index": 0, "values": {"x": 2.0}},
            ],
        },
    }
    if algorithm is not None:
        request.update({"algorithm": algorithm, "algorithm_slug": algorithm, "seed": seed})
    return request


def _make_fixture(tmp_path: Path, *, include_six_pairs: bool = True) -> tuple[Path, dict[str, Path]]:
    paths = _base_paths(tmp_path)
    gt_expr = "x + 1"
    gt_req = _simplify_request(dataset_id="D", expression=gt_expr)
    gt_logical = "gt_simplify::D::v2"
    gt_eval = "g" * 64
    gt_result_sha = "r" * 64
    gt_plan_row = {
        "logical_id": gt_logical,
        "evaluation_key": gt_eval,
        "task_type": "gt_simplify",
        "condition": "clean",
        "priority": 10,
        "request": gt_req,
        "normalized_input": {"request": gt_req},
    }
    gt_index_row = {
        "logical_id": gt_logical,
        "evaluation_key": gt_eval,
        "task_type": "gt_simplify",
        "state": "frozen",
        "effective_expression": "x + 2",
        "expression_resolution": "llm_simplified_expression",
        "structured_output": {"simplified_expression": "x + 2", "outcome": "simplified"},
        "result_sha256": gt_result_sha,
    }
    _write_jsonl(paths["gt_plan"], [gt_plan_row])
    _write_jsonl(paths["gt_index"], [gt_index_row])

    pred_plan_rows: list[dict[str, object]] = []
    pred_index_rows: list[dict[str, object]] = []
    groups = [("imcts", "g0005"), ("imcts", "g0009"), ("udsr", "g0026")]
    if include_six_pairs:
        groups.extend(
            [("gplearn", "g0014"), ("gplearn", "g0024"), ("gplearn", "g0029"), ("gplearn", "g0032")]
        )
    for algorithm, dataset_index in groups:
        if not include_six_pairs and (algorithm, dataset_index) != ("imcts", "g0005"):
            continue
        for seed in (520, 521, 522):
            logical = f"pred_simplify::{algorithm}::{dataset_index}::s{seed}::clean"
            evaluation = hashlib.sha256(logical.encode()).hexdigest()
            result_sha = hashlib.sha256((logical + "result").encode()).hexdigest()
            expression = f"x + {seed}"
            request = _simplify_request(
                dataset_id="D", expression=expression, algorithm=algorithm, seed=seed
            )
            request["dataset_index"] = dataset_index
            plan_row = {
                "logical_id": logical,
                "evaluation_key": evaluation,
                "task_type": "pred_simplify",
                "condition": "clean",
                "priority": 20,
                "request": request,
                "normalized_input": {"request": request},
            }
            pred_plan_rows.append(plan_row)
            pred_index_rows.append(
                {
                    "logical_id": logical,
                    "evaluation_key": evaluation,
                    "task_type": "pred_simplify",
                    "state": "frozen",
                    "effective_expression": expression,
                    "expression_resolution": "llm_simplified_expression",
                    "structured_output": {
                        "simplified_expression": expression,
                        "outcome": "unchanged",
                    },
                    "result_sha256": result_sha,
                }
            )
    _write_jsonl(paths["pred_plan"], pred_plan_rows)
    _write_jsonl(paths["pred_index"], pred_index_rows)

    # Optional files are valid empty JSONL inputs for a draft focused on formula identity.
    for name in ("equivalence_plan", "equivalence_index", "structure_plan", "structure_index", "evidence"):
        _write_jsonl(paths[name], [])

    audit_rows: list[dict[str, object]] = [
        {
            "audit_logical_id": "formula_audit::ground_truth::D",
            "audit_scope": "ground_truth_simplification",
            "condition": "clean",
            "dataset_id": "D",
            "algorithm": None,
            "seed": None,
            "source_identity": {
                "source_formula_logical_id": gt_logical,
                "source_formula_evaluation_key": gt_eval,
                "source_formula_result_sha256": gt_result_sha,
            },
            "expressions": {
                "original": gt_expr,
                "candidate_simplified": "x + 2",
                "reference_simplified": None,
            },
            "stored_decisions": {"simplification_decision": "preserved", "reference_equivalence": "not_applicable"},
            "final_simplification_decision": "not_preserved",
            "final_reference_decision": "not_applicable",
            "simplification_resolution_source": "round2",
            "reference_resolution_source": "scope_not_applicable",
            "offline_severity": "critical",
            "final_simplification_quality": "more_complex",
        }
    ]
    affected_seed = {
        ("imcts", "g0005"): {522},
        ("imcts", "g0009"): {522},
        ("udsr", "g0026"): {521},
        ("gplearn", "g0014"): {520},
        ("gplearn", "g0024"): {520},
        ("gplearn", "g0029"): {521},
        ("gplearn", "g0032"): {521},
    }
    for algorithm, dataset_index in groups:
        if not include_six_pairs and (algorithm, dataset_index) != ("imcts", "g0005"):
            continue
        for seed in (520, 521, 522):
            if seed not in affected_seed.get((algorithm, dataset_index), set()):
                continue
            logical = f"pred_simplify::{algorithm}::{dataset_index}::s{seed}::clean"
            pred_index = next(row for row in pred_index_rows if row["logical_id"] == logical)
            audit_rows.append(
                {
                    "audit_logical_id": f"formula_audit::prediction::{algorithm}::{dataset_index}::s{seed}::clean",
                    "audit_scope": "prediction_formula",
                    "condition": "clean",
                    "dataset_id": "D",
                    "algorithm": algorithm,
                    "seed": seed,
                    "source_identity": {
                        "source_formula_logical_id": logical,
                        "source_formula_evaluation_key": pred_index["evaluation_key"],
                        "source_formula_result_sha256": pred_index["result_sha256"],
                    },
                    "expressions": {
                        "original": f"x + {seed}",
                        "candidate_simplified": "x + 1",
                        "reference_simplified": "x + 1",
                    },
                    "stored_decisions": {"simplification_decision": "preserved", "reference_equivalence": "equivalent"},
                    "final_simplification_decision": (
                        "undetermined" if algorithm == "gplearn" else "not_preserved"
                    ),
                    "final_reference_decision": "not_equivalent",
                    "simplification_resolution_source": "round2",
                    "reference_resolution_source": "round2",
                    "offline_severity": "critical",
                    "final_simplification_quality": "more_complex",
                }
            )
    audit_path = tmp_path / "audit_final.jsonl"
    # The production finalizer stores a row hash over the row without this field.
    for row in audit_rows:
        row["final_record_sha256"] = _sha256_json(
            {key: value for key, value in row.items() if key != "final_record_sha256"}
        )
    _write_jsonl(audit_path, audit_rows)
    return audit_path, paths


def test_draft_applies_identity_fallback_and_never_model_suggestion(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_corrections import (
        build_draft,
    )

    audit_path, paths = _make_fixture(tmp_path, include_six_pairs=False)
    report = build_draft(
        audit_final_jsonl=audit_path,
        base_inputs=paths,
        output_root=tmp_path / "overlay",
        undetermined_policy="retain",
    )
    assert report["counts"]["expression_fallbacks"] == 2
    pred_rows = [
        json.loads(line)
        for line in (tmp_path / "overlay" / "views" / "pred_effective.jsonl").read_text().splitlines()
    ]
    assert all(row["effective_expression"] != row.get("suggested_simplified_expression") for row in pred_rows)


def test_six_structure_tasks_have_fresh_overlay_evaluation_keys(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_corrections import (
        build_draft,
    )
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
        load_plan_jsonl,
    )

    audit_path, paths = _make_fixture(tmp_path)
    report = build_draft(
        audit_final_jsonl=audit_path,
        base_inputs=paths,
        output_root=tmp_path / "overlay",
        undetermined_policy="retain",
    )
    assert report["counts"]["stale_structure"] == 14
    rows = _read_jsonl(tmp_path / "overlay" / "plans" / "structure_overlay_plan.jsonl")
    assert len(rows) == 14
    assert all(row["dependencies"] == [] for row in rows)
    assert len(load_plan_jsonl(tmp_path / "overlay" / "plans" / "structure_overlay_plan.jsonl").entries) == 14
    assert all("overlay" in row["evaluation_key"] or row["evaluation_key"] != row["logical_id"] for row in rows)
    assert {row["logical_id"] for row in rows} == {
        "stab_structure::imcts::g0005::s520-s522",
        "stab_structure::imcts::g0005::s521-s522",
        "stab_structure::imcts::g0009::s520-s522",
        "stab_structure::imcts::g0009::s521-s522",
        "stab_structure::udsr::g0026::s520-s521",
        "stab_structure::udsr::g0026::s521-s522",
        "stab_structure::gplearn::g0014::s520-s521",
        "stab_structure::gplearn::g0014::s520-s522",
        "stab_structure::gplearn::g0024::s520-s521",
        "stab_structure::gplearn::g0024::s520-s522",
        "stab_structure::gplearn::g0029::s520-s521",
        "stab_structure::gplearn::g0029::s521-s522",
        "stab_structure::gplearn::g0032::s520-s521",
        "stab_structure::gplearn::g0032::s521-s522",
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_hash_mismatch_fails_before_writing_overlay(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_corrections import (
        CorrectionOverlayError,
        build_draft,
    )

    audit_path, paths = _make_fixture(tmp_path, include_six_pairs=False)
    bad = _read_jsonl(paths["gt_index"])[0]
    bad["evaluation_key"] = "z" * 64
    _write_jsonl(paths["gt_index"], [bad])
    with pytest.raises(CorrectionOverlayError):
        build_draft(
            audit_final_jsonl=audit_path,
            base_inputs=paths,
            output_root=tmp_path / "overlay",
            undetermined_policy="retain",
        )
    assert not (tmp_path / "overlay").exists()


def test_final_binds_every_structure_result_before_publishing_manifest(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_corrections import (
        finalize_draft,
        build_draft,
    )

    audit_path, paths = _make_fixture(tmp_path)
    overlay = tmp_path / "overlay"
    build_draft(
        audit_final_jsonl=audit_path,
        base_inputs=paths,
        output_root=overlay,
        undetermined_policy="retain",
    )
    plan_rows = _read_jsonl(overlay / "plans" / "structure_overlay_plan.jsonl")
    result_path = tmp_path / "structure_results.jsonl"
    _write_jsonl(
        result_path,
        [
            {
                "logical_id": row["logical_id"],
                "evaluation_key": row["evaluation_key"],
                "structured_output": {
                    "decision": "different_structure",
                    "confidence": 0.9,
                    "brief_reason": "synthetic",
                },
            }
            for row in plan_rows
        ],
    )
    final = finalize_draft(
        draft_manifest_json=overlay / "manifest" / "draft_manifest.json",
        structure_results=[result_path],
    )
    assert final["status"] == "ok"
    assert final["counts"]["structure_replacements"] == 14
    assert (overlay / "manifest" / "corrections_manifest.json").is_file()


def test_final_is_blocked_when_a_structure_result_is_missing(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_corrections import (
        finalize_draft,
        build_draft,
    )

    audit_path, paths = _make_fixture(tmp_path)
    overlay = tmp_path / "overlay"
    build_draft(
        audit_final_jsonl=audit_path,
        base_inputs=paths,
        output_root=overlay,
        undetermined_policy="retain",
    )
    plan_rows = _read_jsonl(overlay / "plans" / "structure_overlay_plan.jsonl")
    _write_jsonl(
        tmp_path / "partial_results.jsonl",
        [
            {
                "logical_id": plan_rows[0]["logical_id"],
                "evaluation_key": plan_rows[0]["evaluation_key"],
                "structured_output": {"decision": "undetermined"},
            }
        ],
    )
    final = finalize_draft(
        draft_manifest_json=overlay / "manifest" / "draft_manifest.json",
        structure_results=[tmp_path / "partial_results.jsonl"],
    )
    assert final["status"] == "blocked"
    assert len(final["missing_structure_ids"]) == 13
