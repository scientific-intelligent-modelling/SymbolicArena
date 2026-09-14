from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.publish_core50_final_release import (
    ArtifactSpec,
    PublicationError,
    aggregate_six_axis,
    bind_raw_and_semantic,
    compact_llm_evidence,
    config_template,
    expression_fingerprint,
    normalize_trajectory,
    require_dependencies,
    row_generation,
    run_key,
    validate_raw_record,
    validate_eff_curve_primary,
    validate_eff_status_row,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_spec_checks_sha_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["value"])
        writer.writeheader()
        writer.writerows([{"value": "a"}, {"value": "b"}])
    spec = ArtifactSpec.from_json(
        {"path": str(path), "sha256": _sha(path), "rows": 2, "format": "csv"},
        repo_root=tmp_path,
        context="fixture",
    )
    assert [row["value"] for row in spec.read_rows()] == ["a", "b"]
    bad = ArtifactSpec(path, "0" * 64, 2, "csv")
    with pytest.raises(PublicationError, match="SHA 漂移"):
        bad.verify()


def test_expression_fingerprint_keeps_variable_identity() -> None:
    assert expression_fingerprint("np.sin(x1) + x2") == expression_fingerprint(
        "sin(x1)+x2"
    )
    assert expression_fingerprint("sin(x1)+x2") != expression_fingerprint(
        "sin(x2)+x1"
    )


def test_raw_and_semantic_binding_checks_formula_not_only_result_sha() -> None:
    payload = {
        "tool": "SymbolFit",
        "dataset": "demo",
        "seed": 520,
        "condition": "clean",
        "equation": "x0 + 1",
    }
    text = json.dumps(payload)
    sha = hashlib.sha256(text.encode()).hexdigest()
    record = {
        "source": {
            "algorithm": "SymbolFit",
            "dataset_id": "demo",
            "seed": 520,
            "noise_tag": "clean",
            "task_id": "symbolfit_s520_clean_g0001",
        },
        "result": {"raw_text": text, "sha256": sha},
    }
    key, parsed = validate_raw_record(record)
    semantic = {
        key: {
            "source_result_sha256": sha,
            "raw_equation": "x0 + 2",
        }
    }
    with pytest.raises(PublicationError, match="raw equation"):
        bind_raw_and_semantic(
            {key: {"record": record, "payload": parsed}},
            semantic,
        )


def test_dependencies_require_current_keys_and_expressions() -> None:
    plan = {
        "dependencies": ["pred-key", "gt-key"],
        "request": {
            "deterministic_evidence": {
                "lhs_binding": {
                    "frozen_evaluation_key": "pred-key",
                    "frozen_effective_expression": "x0 + 1",
                },
                "rhs_binding": {
                    "frozen_evaluation_key": "gt-key",
                    "frozen_effective_expression": "x0",
                },
            },
        },
    }
    require_dependencies(
        plan,
        {"pred-key": "x0+1", "gt-key": "x0"},
        context="fixture",
    )
    with pytest.raises(PublicationError, match="表达式绑定不一致"):
        require_dependencies(
            plan,
            {"pred-key": "x1+1", "gt-key": "x0"},
            context="fixture",
        )


def test_normalize_trajectory_preserves_legacy_formal_flag(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    source.write_text("placeholder", encoding="utf-8")
    spec = ArtifactSpec(
        source,
        _sha(source),
        1,
        "csv",
        defaults={"formal_ready_source_tiers": ["current_canonical"]},
    )
    row = {
        "algorithm": "Demo",
        "dataset_id": "task",
        "seed": "520",
        "condition": "clean",
        "source_tier": "legacy_fallback",
        "trajectory_basis": "internal_with_explicit_legacy_fallback.v1",
    }
    for minute in range(1, 181):
        row[f"q_{minute:04d}"] = "0.5"
    normalized = normalize_trajectory(
        row,
        key=run_key("Demo", "task", 520, "clean"),
        source_spec=spec,
    )
    assert normalized["formal_ready"] == "false"
    assert normalized["trajectory_basis"] == "internal_with_explicit_legacy_fallback.v1"
    assert normalized["m_eff"] == pytest.approx(1.0)


def test_config_template_declares_all_frozen_inputs() -> None:
    template = config_template()
    assert template["schema_version"] == "core50_final_release_publish_config_v1"
    assert set(template["artifacts"]["conditions"]) == {"clean", "noise001", "noise005"}
    assert template["llm_transport_policy"]["new_evidence_required"] == "routify"
    assert template["artifacts"]["ground_truth"]["index"]["generation"] == "mixed"
    clean = template["artifacts"]["conditions"]["clean"]
    assert clean["trajectory"]["mode"] == "eff_revision_v3"
    assert clean["equivalence"]["audit_overrides"][0]["rows"] == 28
    assert clean["structure"]["audit_overlays"][0]["index"]["rows"] == 14
    eff = template["artifacts"]["eff_revision"]
    assert eff["conditions"]["clean"]["run_status"]["rows"] == 2250
    assert eff["conditions"]["noise005"]["minute_metrics"]["rows"] == 405000


def test_mixed_index_requires_per_row_generation(tmp_path: Path) -> None:
    spec = ArtifactSpec(tmp_path / "unused.jsonl", "0" * 64, 1, "jsonl", generation="mixed")
    assert row_generation({"evidence_generation": "new"}, spec) == "new"
    with pytest.raises(PublicationError, match="evidence_generation"):
        row_generation({"logical_id": "missing"}, spec)


def test_new_llm_evidence_requires_routify(tmp_path: Path) -> None:
    response = {
        "logical_id": "pred::one",
        "evaluation_key": "eval",
        "metadata": {
            "api_channel": "routify",
            "requested_model": "claude-opus-5",
            "response_model": "claude-opus-5",
        },
    }
    path = tmp_path / "response.json"
    path.write_text(json.dumps(response), encoding="utf-8")
    plan = {
        "logical_id": "pred::one",
        "evaluation_key": "eval",
        "condition": "clean",
        "dependencies": [],
        "request": {"expression": "x0"},
    }
    index = {
        "logical_id": "pred::one",
        "evaluation_key": "eval",
        "state": "frozen",
        "task_type": "pred_simplify",
        "result_path": str(path),
        "result_sha256": _sha(path),
        "effective_expression": "x0",
        "structured_output": {"outcome": "unchanged"},
    }
    evidence = compact_llm_evidence(
        plan,
        index,
        repo_root=tmp_path,
        generation="new",
        new_transport="routify",
    )
    assert evidence["api_channel"] == "routify"
    response["metadata"]["api_channel"] = "yapi"
    path.write_text(json.dumps(response), encoding="utf-8")
    index["result_sha256"] = _sha(path)
    with pytest.raises(PublicationError, match="不是 Routify"):
        compact_llm_evidence(
            plan,
            index,
            repo_root=tmp_path,
            generation="new",
            new_transport="routify",
        )


def test_unavailable_eff_must_remain_empty_not_zero() -> None:
    key = run_key("demo", "task", 520, "clean")
    row = {
        "availability_status": "unavailable",
        "q_star": "",
        "m_eff": "",
        "eff_score": "",
        "unavailable_reason": "missing evidence",
        "required_action": "recollect",
    }
    assert validate_eff_status_row(row, key=key) == "unavailable"
    row["m_eff"] = "0"
    with pytest.raises(PublicationError, match="不得填0"):
        validate_eff_status_row(row, key=key)


def test_incomplete_algorithm_curve_cannot_publish_partial_mean() -> None:
    row = {
        "mean_id_quality": "",
        "mean_ood_quality": "",
        "mean_quality": "",
        "mean_relative_progress": "",
        "cumulative_eff_score": "",
        "diagnostic_available_mean_quality": "0.5",
    }
    validate_eff_curve_primary(row, formal=False, context="demo.minute1")
    row["mean_quality"] = "0"
    with pytest.raises(PublicationError, match="不得发布部分主均值"):
        validate_eff_curve_primary(row, formal=False, context="demo.minute1")


def test_six_axis_keeps_eff_empty_without_150_runs() -> None:
    run_rows = []
    task_rows = []
    for algorithm_index in range(15):
        slug = f"algorithm{algorithm_index:02d}"
        for run_index in range(150):
            run_rows.append(
                {
                    "algorithm": slug,
                    "algorithm_slug": slug,
                    "id_quality": 0.5,
                    "ood_quality": 0.4,
                    "m_sym": 0.3,
                    "m_min": 0.2,
                    "m_eff": "" if algorithm_index == 0 and run_index == 149 else 0.6,
                    "trajectory_formal_ready": "false"
                    if algorithm_index == 0 and run_index == 149
                    else "true",
                    "trajectory_basis": "native.v1",
                }
            )
        for task_index in range(50):
            task_rows.append(
                {
                    "algorithm_slug": slug,
                    "m_stab": ""
                    if algorithm_index == 0 and task_index == 49
                    else 0.7,
                }
            )
    output = aggregate_six_axis(run_rows, task_rows, condition="clean")
    assert len(run_rows) == 2250
    assert len(output) == 15
    incomplete = next(row for row in output if row["algorithm_slug"] == "algorithm00")
    complete = next(row for row in output if row["algorithm_slug"] == "algorithm01")
    assert incomplete["eff_available_run_count"] == 149
    assert incomplete["EFF"] == ""
    assert incomplete["STAB"] == ""
    assert incomplete["stab_available_task_count"] == 49
    assert incomplete["assembly_complete"] == "false"
    assert incomplete["formal_ready"] == "false"
    assert complete["EFF"] == pytest.approx(60.0)


def test_non_applicable_structure_is_preserved_without_fake_api_response(tmp_path):
    plan = {
        "logical_id": "stab_structure::a::g0001::s520-s521",
        "evaluation_key": "na",
        "task_type": "stab_structure",
        "condition": "clean",
        "dependencies": ["left", "right"],
        "request": {"prediction_a_valid_output": False, "prediction_b_valid_output": True},
    }
    index = {
        "logical_id": plan["logical_id"],
        "evaluation_key": "na",
        "state": "non_applicable",
        "task_type": "stab_structure",
        "condition": "clean",
        "non_applicable": {"reason": "invalid_seed_or_expression"},
    }
    evidence = compact_llm_evidence(
        plan, index, repo_root=tmp_path, generation="new", new_transport="routify"
    )
    assert evidence["state"] == "non_applicable"
    assert evidence["network_request"] is False
    assert evidence["response_path"] is None


def test_unresolved_structure_preserves_attempts_without_decision(tmp_path):
    plan = {
        "logical_id": "stab_structure::gplearn::g0029::s520-s521::v2",
        "evaluation_key": "current",
        "task_type": "stab_structure",
        "condition": "clean",
        "dependencies": ["left", "right"],
        "request": {},
    }
    index = {
        "logical_id": plan["logical_id"],
        "evaluation_key": "current",
        "state": "unresolved",
        "task_type": "stab_structure",
        "condition": "clean",
        "structured_output": None,
        "unresolved": {"attempts": [{"sha256": "a" * 64}] * 6},
    }
    evidence = compact_llm_evidence(
        plan, index, repo_root=tmp_path, generation="new", new_transport="routify"
    )
    assert evidence["state"] == "unresolved"
    assert evidence["structured_output"] is None
    assert len(evidence["unresolved"]["attempts"]) == 6
