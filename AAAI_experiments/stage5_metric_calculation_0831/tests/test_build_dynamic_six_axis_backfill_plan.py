from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_dynamic_six_axis_backfill_plan import (
    CleanEvidence,
    DynamicBackfillPlanError,
    SourceBundle,
    build_dynamic_six_axis_backfill_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import phi_nmse


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_row(
    *,
    condition: str,
    seed: int,
    expressions: tuple[str | None, ...] = ("x0", "x0 + 1", "x0 + 1"),
    nmses: tuple[float, ...] = (10.0, 0.1, 0.2),
    source_path: str = "/formal/result.json",
) -> dict[str, object]:
    return {
        "source": {
            "algorithm": "algo",
            "dataset_id": "d1",
            "seed": seed,
            "noise_tag": condition,
            "task_id": f"algo_s{seed}_{condition}_g0001",
            "path": source_path,
        },
        "snapshots": [
            {
                "minute": minute,
                "expression": expression,
                "id_nmse": nmse,
                "ood_nmse": nmse,
                "status": "ok",
                "selected_path": f"/formal/minute_{minute:04d}.json",
                "selected_sha256": f"{minute:064x}",
            }
            for minute, (expression, nmse) in enumerate(zip(expressions, nmses), start=1)
        ],
    }


def _numeric_csv(path: Path, *, condition: str, seeds: tuple[int, ...], clean: bool) -> None:
    first = phi_nmse(10.0)
    second = phi_nmse(0.1)
    fieldnames = [
        "logical_key",
        "algorithm",
        "dataset_id",
        "seed",
        "condition",
        "source_tier",
        "q_0001",
        "q_0002",
        "q_0003",
    ]
    if not clean:
        fieldnames.extend(
            field
            for minute in range(1, 4)
            for field in (f"id_q_{minute:04d}", f"ood_q_{minute:04d}")
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in seeds:
            row = {
                "logical_key": f"algo::d1::s{seed}::{condition}",
                "algorithm": "algo",
                "dataset_id": "d1",
                "seed": seed,
                "condition": condition,
                "source_tier": "canonical" if clean else "base",
                "q_0001": first,
                "q_0002": second,
                "q_0003": second,
            }
            if not clean:
                for minute, value in enumerate((first, second, second), start=1):
                    row[f"id_q_{minute:04d}"] = value
                    row[f"ood_q_{minute:04d}"] = value
            writer.writerow(row)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_noise_plan_reconstructs_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "noise.jsonl"
    seeds = (520, 521, 522)
    _jsonl(source, [_source_row(condition="noise001", seed=seed) for seed in seeds])
    numeric = tmp_path / "noise.csv"
    _numeric_csv(numeric, condition="noise001", seeds=seeds, clean=False)

    output = tmp_path / "output"
    manifest = build_dynamic_six_axis_backfill_plan(
        numeric_paths={"noise001": numeric},
        source_bundles=[SourceBundle("noise001", "base", 0, source)],
        clean_evidence=[],
        output_dir=output,
        horizon=3,
        expected_runs_by_condition={"noise001": 3},
    )

    assert manifest["status"] == "ready"
    assert manifest["targets"] == {
        "expected": {
            "run_count": 3,
            "run_minute_count": 9,
            "task_minute_group_count": 3,
            "seed_pair_count_per_group": 3,
            "task_minute_pair_count": 9,
        },
        "actual": {
            "run_count": 3,
            "run_minute_count": 9,
            "task_minute_group_count": 3,
            "seed_pair_count_per_group": 3,
            "task_minute_pair_count": 9,
        },
        "contract_satisfied": True,
    }
    assert manifest["deduplication"] == {
        "pred_simplify_task_count": 2,
        "equivalence_task_count": 2,
    }
    points = _read_jsonl(output / "run_minute_plan.jsonl")
    assert all(point["expression_status"] == "resolved" for point in points)
    assert all(
        {"id_quality", "ood_quality", "sym_score", "min_score", "cumulative_eff", "stab_score"}
        <= set(point)
        for point in points
    )
    assert points[2]["expression"] == "x0 + 1"
    assert points[2]["relative_progress"] == pytest.approx(1.0)
    assert points[2]["cumulative_eff"] == pytest.approx(
        (phi_nmse(10.0) / phi_nmse(0.1) + 2.0) / 3.0
    )
    assert len(_read_jsonl(output / "symbolic_task_plan.jsonl")) == 4
    pairs = _read_jsonl(output / "task_minute_stab_pair_plan.jsonl")
    assert len(pairs) == 9
    assert all(pair["numerical_disagreement"] == pytest.approx(0.0) for pair in pairs)
    assert not list(output.glob(".*.tmp"))


def test_clean_uses_persisted_canonical_components_instead_of_raw_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clean.jsonl"
    seeds = (520, 521, 522)
    _jsonl(source, [_source_row(condition="clean", seed=seed) for seed in seeds])
    numeric = tmp_path / "clean.csv"
    _numeric_csv(numeric, condition="clean", seeds=seeds, clean=True)
    evidence_path = tmp_path / "evidence.jsonl"
    first = phi_nmse(10.0)
    second = phi_nmse(0.1)
    _jsonl(
        evidence_path,
        [
            {
                "logical_key": f"algo::d1::s{seed}::clean",
                "evaluation_path": "canonical_replay.v1",
                "trajectory_evidence_schema_version": "clean_selected_trajectory_evidence.v1",
                "quality_trajectory": [first, second, second],
                "id_quality_trajectory": [first, second, second],
                "ood_quality_trajectory": [first, second, second],
                "selected_expression_trajectory": [
                    "x0",
                    "canonical(x0 + 1)",
                    "canonical(x0 + 1)",
                ],
                "valid_output_trajectory": [True, True, True],
                # minute 3 的 raw NMSE 是 0.2，故意不同于 canonical replay 的 0.1。
                "trajectory_sources": ["snapshot:1", "snapshot:3", "carry_forward:3"],
            }
            for seed in seeds
        ],
    )

    output = tmp_path / "output"
    manifest = build_dynamic_six_axis_backfill_plan(
        numeric_paths={"clean": numeric},
        source_bundles=[SourceBundle("clean", "base", 0, source)],
        clean_evidence=[CleanEvidence("canonical", evidence_path)],
        output_dir=output,
        horizon=3,
        expected_runs_by_condition={"clean": 3},
    )

    assert manifest["status"] == "ready"
    assert manifest["conditions"]["clean"] == {
        "run_count": 3,
        "run_minute_count": 9,
        "resolved": 9,
    }
    points = _read_jsonl(output / "run_minute_plan.jsonl")
    canonical = [point for point in points if point["minute"] > 1]
    assert all(point["expression"] == "canonical(x0 + 1)" for point in canonical)
    assert all(point["id_quality"] == pytest.approx(second) for point in canonical)
    assert all(
        point["source_evidence"]["canonical_evidence"]["evaluation_path"]
        == "canonical_replay.v1"
        for point in canonical
    )
    assert manifest["unresolved"]["record_count"] == 0


def test_clean_component_mismatch_remains_unresolved(tmp_path: Path) -> None:
    source = tmp_path / "clean.jsonl"
    seeds = (520, 521, 522)
    _jsonl(source, [_source_row(condition="clean", seed=seed) for seed in seeds])
    numeric = tmp_path / "clean.csv"
    _numeric_csv(numeric, condition="clean", seeds=seeds, clean=True)
    first = phi_nmse(10.0)
    second = phi_nmse(0.1)
    evidence_path = tmp_path / "evidence.jsonl"
    _jsonl(
        evidence_path,
        [
            {
                "logical_key": f"algo::d1::s{seed}::clean",
                "evaluation_path": "canonical_replay.v1",
                "trajectory_evidence_schema_version": "clean_selected_trajectory_evidence.v1",
                "quality_trajectory": [first, second, second],
                "id_quality_trajectory": [first, second, second],
                "ood_quality_trajectory": [first, 0.0, second],
                "selected_expression_trajectory": ["x0", "x0 + 1", "x0 + 1"],
                "valid_output_trajectory": [True, True, True],
                "trajectory_sources": ["snapshot:1", "snapshot:2", "carry_forward:2"],
            }
            for seed in seeds
        ],
    )

    output = tmp_path / "output"
    manifest = build_dynamic_six_axis_backfill_plan(
        numeric_paths={"clean": numeric},
        source_bundles=[SourceBundle("clean", "base", 0, source)],
        clean_evidence=[CleanEvidence("canonical", evidence_path)],
        output_dir=output,
        horizon=3,
        expected_runs_by_condition={"clean": 3},
    )

    assert manifest["status"] == "complete_with_unresolved"
    assert manifest["unresolved"]["record_count"] == 3
    points = _read_jsonl(output / "run_minute_plan.jsonl")
    minute_two = [point for point in points if point["minute"] == 2]
    assert all(point["expression_status"] == "unresolved" for point in minute_two)
    assert all("ID/OOD" in point["unresolved_reason"] for point in minute_two)


def test_rejects_forbidden_fullcpu_source(tmp_path: Path) -> None:
    source = tmp_path / "noise.jsonl"
    _jsonl(
        source,
        [
            _source_row(
                condition="noise001",
                seed=520,
                source_path="/runs/all_15alg_fullcpu_v1/result.json",
            )
        ],
    )
    numeric = tmp_path / "noise.csv"
    _numeric_csv(numeric, condition="noise001", seeds=(520,), clean=False)

    with pytest.raises(DynamicBackfillPlanError, match="禁止来源"):
        build_dynamic_six_axis_backfill_plan(
            numeric_paths={"noise001": numeric},
            source_bundles=[SourceBundle("noise001", "base", 0, source)],
            clean_evidence=[],
            output_dir=tmp_path / "output",
            horizon=3,
            expected_runs_by_condition={"noise001": 1},
            expected_seeds=(520,),
        )
