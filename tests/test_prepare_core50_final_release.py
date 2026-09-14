from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_core50_final_release import (
    FinalReleasePreparationError,
    _artifact_expression,
    _corrected_freeze_row,
    _load_old_opus,
    _unwrap_frozen_row,
    build_identity_tasks,
    build_release_identity_recovery_plan,
    canonical_pred_logical_id,
    expression_fingerprint,
    merge_final_rows,
)


def _frozen(task_id: str, equation: str = "x0") -> dict:
    raw = json.dumps({"equation": equation, "feature_names": ["t"]}, sort_keys=True)
    return {
        "result": {"raw_text": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()},
        "source": {"task_id": task_id},
    }


def test_expression_fingerprint_ignores_format_and_numpy_prefix_not_variable_binding() -> None:
    assert expression_fingerprint("np.sin(x0) + x1") == expression_fingerprint(
        "sin( x0 )+x1"
    )
    assert expression_fingerprint("sin(x0) + x1") != expression_fingerprint(
        "sin(x1) + x0"
    )


def test_pred_logical_id_version_suffix_is_not_part_of_task_identity() -> None:
    canonical = "pred_simplify::drsr::g0002::s522::clean"
    assert canonical_pred_logical_id(canonical) == canonical
    assert canonical_pred_logical_id(f"{canonical}::v2") == canonical


def test_old_opus_index_matches_versioned_record_by_canonical_identity(tmp_path: Path) -> None:
    for index, condition in enumerate(("clean", "noise001", "noise005"), start=1):
        path = tmp_path / "results" / condition / "opus5_simplification.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = "::v2" if condition == "clean" else ""
        row = {
            "logical_id": f"pred_simplify::tool::g000{index}::s520::{condition}{suffix}",
            "evaluation_key": str(index) * 64,
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    index = _load_old_opus(tmp_path)
    canonical = "pred_simplify::tool::g0001::s520::clean"
    assert index[canonical]["logical_id"] == f"{canonical}::v2"


def test_artifact_expression_maps_slots_to_feature_names() -> None:
    artifact = {"instantiated_expression": "x1 + 2*x0"}
    assert _artifact_expression(artifact, ["time", "mass"]) == "mass + 2*time"


def test_unwrap_rejects_reverse_binding_sha_mismatch() -> None:
    row = _frozen("tool_s520_clean_g0001")
    row["result"]["raw_text"] = "{}"
    with pytest.raises(FinalReleasePreparationError, match="SHA-256"):
        _unwrap_frozen_row(row)


def test_corrected_freeze_keeps_reverse_binding_to_immutable_raw() -> None:
    row = _frozen("tool_s520_clean_g0001")
    _, original_sha = _unwrap_frozen_row(row)
    corrected = _corrected_freeze_row(
        row,
        {
            "canonical_artifact": {"instantiated_expression": "x0 + 1"},
            "canonical_artifact_sha256": "a" * 64,
            "effective_raw_semantic_expression_sha256": "b" * 64,
        },
    )
    payload, derived_sha = _unwrap_frozen_row(corrected)
    assert derived_sha != original_sha
    assert payload["final_release_source_binding"]["original_result_sha256"] == original_sha
    assert payload["final_release_source_binding"]["corrected_canonical_artifact_sha256"] == "a" * 64


def test_merge_replaces_only_matching_task_and_rejects_missing_scope() -> None:
    base = [_frozen(f"symbolfit_s520_noise001_g{i:04d}") for i in range(1, 2251)]
    replacements = {
        f"symbolfit_s520_noise001_g{i:04d}": _frozen(
            f"symbolfit_s520_noise001_g{i:04d}", "x0 + 1"
        )
        for i in range(1, 151)
    }
    merged, count = merge_final_rows(
        base, replacements,
        condition="noise001",
    )
    assert count == 150
    assert merged[0] == replacements["symbolfit_s520_noise001_g0001"]


def test_merge_rejects_reverse_condition_replacement() -> None:
    base = [_frozen(f"tool_s520_clean_g{i:04d}") for i in range(1, 2251)]
    with pytest.raises(FinalReleasePreparationError, match="replacement 未命中"):
        merge_final_rows(
            base,
            {"symbolfit_s520_clean_g9999": _frozen("symbolfit_s520_clean_g9999")},
            condition="clean",
        )


def _recovery_fixture(tmp_path: Path, *, state: str) -> tuple[Path, Path, dict]:
    row = build_identity_tasks(REPO_ROOT)[1].to_json_record()
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    db = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE tasks (
          evaluation_key TEXT PRIMARY KEY, logical_id TEXT, task_type TEXT,
          condition_name TEXT, state TEXT, attempt_count INTEGER,
          last_error_class TEXT
        );
        CREATE TABLE frozen_results (evaluation_key TEXT PRIMARY KEY);
        CREATE TABLE attempts (
          evaluation_key TEXT, attempt_number INTEGER, status TEXT,
          error_class TEXT, retryable INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",
        (
            row["evaluation_key"], row["logical_id"], row["task_type"],
            row["condition"], state, 1, "structured_output_invalid",
        ),
    )
    if state == "frozen":
        connection.execute("INSERT INTO frozen_results VALUES (?)", (row["evaluation_key"],))
        connection.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?)",
            (row["evaluation_key"], 1, "accepted", None, None),
        )
    else:
        connection.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?)",
            (row["evaluation_key"], 1, "failed", "structured_output_invalid", 0),
        )
    connection.commit()
    connection.close()
    return plan, db, row


def test_release_recovery_does_not_retry_frozen_legal_unable(tmp_path: Path) -> None:
    plan, db, _ = _recovery_fixture(tmp_path, state="frozen")
    output = tmp_path / "recovery.jsonl"
    manifest_path = tmp_path / "supersession.json"
    report = build_release_identity_recovery_plan(
        predecessor_plan_jsonl=plan,
        state_db=db,
        recovery_prompt_path=REPO_ROOT
        / "AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify.identity.v1.txt",
        output_jsonl=output,
        supersession_manifest=manifest_path,
    )
    assert report["successor_count"] == 0
    assert output.read_text(encoding="utf-8") == ""


def test_release_recovery_preserves_request_and_supersedes_only_exhausted(tmp_path: Path) -> None:
    plan, db, predecessor = _recovery_fixture(tmp_path, state="exhausted")
    output = tmp_path / "recovery.jsonl"
    manifest_path = tmp_path / "supersession.json"
    report = build_release_identity_recovery_plan(
        predecessor_plan_jsonl=plan,
        state_db=db,
        recovery_prompt_path=REPO_ROOT
        / "AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify.identity.v1.txt",
        output_jsonl=output,
        supersession_manifest=manifest_path,
    )
    successor = json.loads(output.read_text(encoding="utf-8"))
    assert report["successor_count"] == 1
    assert successor["evaluation_key"] != predecessor["evaluation_key"]
    assert successor["request"] == predecessor["request"]
    assert successor["request"]["original_expression"] == successor["request"]["expression"]
    assert report["recovery_max_attempts"] == 1
