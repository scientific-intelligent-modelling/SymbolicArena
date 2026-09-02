from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_plan import (  # noqa: E402
    build_formula_audit_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (  # noqa: E402
    load_plan_jsonl,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_formula_audit_plan_is_deterministic_blind_and_complete(tmp_path: Path) -> None:
    first = build_formula_audit_plan(audit_root=tmp_path / "first")
    second = build_formula_audit_plan(audit_root=tmp_path / "second")

    assert first["counts"]["ground_truth_audit_count"] == 50
    assert first["counts"]["prediction_universe_count"] == 6749
    assert first["counts"]["prediction_excluded_count"] == 1
    assert first["counts"]["prediction_sample_count"] == 1000
    assert first["counts"]["initial_api_task_count"] == 1050
    assert first["budget"]["maximum_total_api_attempts"] == 1575
    assert first["outputs"]["plan_sha256"] == second["outputs"]["plan_sha256"]
    assert first["outputs"]["manifest_sha256"] == second["outputs"]["manifest_sha256"]

    plan_path = Path(first["outputs"]["plan_jsonl"])
    loaded = load_plan_jsonl(plan_path)
    assert len(loaded.entries) == 1050
    rows = _read_jsonl(plan_path)
    assert {row["condition"] for row in rows} == {"clean"}
    assert {row["task_kind"] for row in rows} == {"formula_audit"}

    forbidden = {
        "algorithm",
        "algorithm_slug",
        "condition",
        "dataset_id",
        "dataset_index",
        "seed",
        "source_gt_logical_id",
        "source_pred_logical_id",
        "stored_simplification_outcome",
        "stored_equivalence_decision",
    }
    for row in rows:
        request = row["request"]
        assert isinstance(request, dict)
        assert not forbidden.intersection(request)
        assert len(request["audit_binding_sha256"]) == 64

    excluded = _read_jsonl(Path(first["outputs"]["excluded_jsonl"]))
    assert len(excluded) == 1
    assert excluded[0]["state"] == "non_applicable"
    assert excluded[0]["reason"] == "unresolved_parameter_values"
