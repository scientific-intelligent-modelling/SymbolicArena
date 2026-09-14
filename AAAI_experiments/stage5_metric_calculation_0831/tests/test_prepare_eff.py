"""prepare_eff 真实仓库手工校验命令：

dry:
  python -m AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff \
    --limit-runs 32 \
    --output-jsonl /tmp/eff_prepare_dry.jsonl \
    --output-csv /tmp/eff_prepare_dry.csv \
    --output-report /tmp/eff_prepare_dry.json \
    --print-summary

full:
  python -m AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff \
    --output-jsonl /tmp/eff_prepare_full.jsonl \
    --output-csv /tmp/eff_prepare_full.csv \
    --output-report /tmp/eff_prepare_full.json \
    --print-summary
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    efficiency_from_qualities,
    phi_nmse,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff import (
    EffPreparationContractError,
    build_eff_preparation,
    reconstruct_native_incumbent_trajectory,
)


STAGE5_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _native_snapshot(
    minute: int,
    *,
    expression: str | None,
    id_nmse: float | None,
    ood_nmse: float | None,
    record_type: str = "periodic_best",
    **objective: float,
) -> dict[str, object]:
    payload = _snapshot_payload(
        minute,
        record_type=record_type,
        expression=expression,
        id_nmse=id_nmse,
        ood_nmse=ood_nmse,
    )
    payload.update(objective)
    return payload


def test_native_incumbent_can_improve_while_test_quality_declines() -> None:
    trajectory = reconstruct_native_incumbent_trajectory(
        {
            1: _native_snapshot(
                1,
                expression="x0",
                id_nmse=1.0e-8,
                ood_nmse=1.0e-8,
                source_loss=2.0,
            ),
            2: _native_snapshot(
                2,
                expression="x0 + 1",
                id_nmse=1.0,
                ood_nmse=1.0,
                source_loss=1.0,
            ),
        },
        algorithm="gplearn",
        horizon=2,
    )

    assert [point.expression for point in trajectory.points] == ["x0", "x0 + 1"]
    assert trajectory.points[1].quality < trajectory.points[0].quality
    assert trajectory.objective_values == (2.0, 1.0)


def test_native_incumbent_never_uses_future_backfill() -> None:
    future = _native_snapshot(
        1,
        expression="future_formula",
        id_nmse=1.0e-12,
        ood_nmse=1.0e-12,
        record_type="periodic_backfill",
        source_loss=1.0,
    )
    future["backfilled_from_minute"] = 2
    trajectory = reconstruct_native_incumbent_trajectory(
        {
            1: future,
            2: _native_snapshot(
                2,
                expression="observed_formula",
                id_nmse=1.0,
                ood_nmse=1.0,
                source_loss=1.0,
            ),
        },
        algorithm="gplearn",
        horizon=2,
    )

    assert trajectory.points[0].expression == ""
    assert trajectory.points[0].quality == 0.0
    assert trajectory.points[0].source.startswith("future_backfill_ignored:2")
    assert trajectory.points[1].expression == "observed_formula"


def test_native_incumbent_carries_between_updates_and_ignores_final_formula() -> None:
    trajectory = reconstruct_native_incumbent_trajectory(
        {
            1: _native_snapshot(
                1,
                expression="incumbent",
                id_nmse=0.1,
                ood_nmse=0.2,
                source_score=3.0,
            ),
            2: _native_snapshot(
                2,
                expression=None,
                id_nmse=None,
                ood_nmse=None,
                record_type="periodic_heartbeat",
            ),
            3: _native_snapshot(
                3,
                expression="future_final",
                id_nmse=1.0e-12,
                ood_nmse=1.0e-12,
                record_type="final_best",
            ),
        },
        algorithm="dso",
        horizon=3,
    )

    assert [point.expression for point in trajectory.points] == ["incumbent"] * 3
    assert trajectory.incumbent_source_minutes == (1, 1, 1)
    assert trajectory.points[1].source == "native_carry_forward:1"
    assert trajectory.points[2].source == "native_endpoint_carry_forward:1"


def test_native_incumbent_missing_auditable_objective_is_unavailable() -> None:
    with pytest.raises(EffPreparationContractError, match="缺少可审计原生目标"):
        reconstruct_native_incumbent_trajectory(
            {
                1: _native_snapshot(
                    1,
                    expression="x0",
                    id_nmse=0.1,
                    ood_nmse=0.1,
                )
            },
            algorithm="e2esr",
            horizon=1,
        )


@pytest.mark.parametrize(
    ("algorithm", "field", "values", "expected"),
    [
        ("fepysr", "source_score", (2.0, 1.0, 1.0), ["a", "b", "b"]),
        ("dso", "source_score", (1.0, 2.0, 2.0), ["a", "b", "b"]),
    ],
)
def test_native_incumbent_respects_direction_and_keeps_earliest_tie(
    algorithm: str,
    field: str,
    values: tuple[float, float, float],
    expected: list[str],
) -> None:
    snapshots = {
        minute: _native_snapshot(
            minute,
            expression=expression,
            id_nmse=0.1 * minute,
            ood_nmse=0.1 * minute,
            **{field: values[minute - 1]},
        )
        for minute, expression in enumerate(("a", "b", "tie"), start=1)
    }
    trajectory = reconstruct_native_incumbent_trajectory(
        snapshots,
        algorithm=algorithm,
        horizon=3,
    )

    assert [point.expression for point in trajectory.points] == expected
    assert trajectory.incumbent_source_minutes == (1, 2, 2)


def test_native_eff_q_star_is_only_posterior_normalization() -> None:
    trajectory = reconstruct_native_incumbent_trajectory(
        {
            1: _native_snapshot(
                1,
                expression="high_test_quality",
                id_nmse=1.0e-8,
                ood_nmse=1.0e-8,
                source_loss=2.0,
            ),
            2: _native_snapshot(
                2,
                expression="better_native_loss",
                id_nmse=1.0,
                ood_nmse=1.0,
                source_loss=1.0,
            ),
        },
        algorithm="gplearn",
        horizon=2,
    )
    qualities = [point.quality for point in trajectory.points]

    assert trajectory.q_star == max(qualities)
    assert trajectory.m_eff == pytest.approx(
        sum(value / trajectory.q_star for value in qualities) / 2
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_payload(
    minute: int,
    *,
    record_type: str,
    expression: str | None,
    id_nmse: float | None,
    ood_nmse: float | None,
    backfilled_from_minute: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": "fixture",
        "record_type": record_type,
        "checkpoint_index": minute,
        "elapsed_seconds": minute * 60,
        "elapsed_minutes": minute,
        "equation": expression,
        "status": "ok" if expression is not None and id_nmse is not None and ood_nmse is not None else "running",
        "id_test": {"nmse": id_nmse} if id_nmse is not None else None,
        "ood_test": {"nmse": ood_nmse} if ood_nmse is not None else None,
    }
    if backfilled_from_minute is not None:
        payload["backfilled_from_minute"] = backfilled_from_minute
    if expression is not None:
        payload["canonical_artifact"] = {"instantiated_expression": expression}
    return payload


def _frozen_snapshot_from_payload(
    minute: int,
    payload: dict[str, object],
    *,
    outer_path: str,
) -> dict[str, object]:
    raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return {
        "status": "ok",
        "minute": minute,
        "checkpoint_index": minute,
        "record_type": payload["record_type"],
        "payload_status": payload.get("status"),
        "backfilled_from_minute": payload.get("backfilled_from_minute"),
        "has_expression": payload.get("equation") is not None,
        "expression": payload.get("equation"),
        "id_nmse": payload["id_test"]["nmse"] if payload.get("id_test") else None,
        "ood_nmse": payload["ood_test"]["nmse"] if payload.get("ood_test") else None,
        "outer_path": outer_path,
        "outer_status": "ok",
        "outer_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "selected_path": outer_path,
        "selected_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_text": raw_text,
    }


def _heartbeat_snapshot(minute: int, *, outer_path: str) -> dict[str, object]:
    payload = _snapshot_payload(
        minute,
        record_type="periodic_heartbeat",
        expression=None,
        id_nmse=None,
        ood_nmse=None,
    )
    return _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)


def _missing_snapshot(minute: int, *, outer_path: str) -> dict[str, object]:
    return {
        "status": "missing",
        "minute": minute,
        "outer_path": outer_path,
        "outer_status": "missing",
        "outer_sha256": None,
        "selected_path": None,
        "selected_sha256": None,
        "raw_text": None,
        "conflict": False,
        "duplicate_semantically_equal": False,
        "inner_path": None,
        "inner_sha256": None,
        "inner_status": None,
    }


def _write_bundle(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    stage5_root = tmp_path / "AAAI_experiments/stage5_metric_calculation_0831"
    freeze_dir = stage5_root / "source_snapshot/trajectory_freeze"
    reports_dir = stage5_root / "reports"
    manifests_dir = stage5_root / "manifests"
    evidence_dir = stage5_root / "source_snapshot/recovery_evidence"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    current_best_path = evidence_dir / "fepysr_s520_clean_g0021.current_best.json"
    current_best_payload = {"equation": "repair_eq"}
    current_best_path.write_text(
        json.dumps(current_best_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    wrapper_path = tmp_path / "scientific_intelligent_modelling/algorithms/fepysr_wrapper/wrapper.py"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text("# fixture wrapper\n", encoding="utf-8")

    host = "fixturehost"
    bundle_path = freeze_dir / f"clean_freeze_{host}.jsonl.gz"
    bundle_report_path = freeze_dir / f"clean_freeze_{host}.report.json"

    repair_record = {
        "source": {
            "algorithm": "fepysr",
            "dataset_id": "Nguyen-12",
            "seed": "520",
            "noise_tag": "clean",
            "task_id": "fepysr_s520_clean_g0021",
            "host": host,
        },
        "snapshots": [],
    }
    for minute in range(1, 181):
        outer_path = f"/bundle/fepysr/minute_{minute:04d}.json"
        if minute == 164:
            repair_payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="repair_eq",
                id_nmse=1.0e-3,
                ood_nmse=2.0e-3,
            )
            repair_payload["source_score"] = 1.0
            repair_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, repair_payload, outer_path=outer_path)
            )
        elif 165 <= minute <= 179:
            repair_record["snapshots"].append(_missing_snapshot(minute, outer_path=outer_path))
        elif minute == 180:
            final_payload = _snapshot_payload(
                minute,
                record_type="recovered_final",
                expression="future_eq",
                id_nmse=1.0e-8,
                ood_nmse=1.0e-8,
            )
            repair_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, final_payload, outer_path=outer_path)
            )
        else:
            repair_record["snapshots"].append(_heartbeat_snapshot(minute, outer_path=outer_path))

    future_record = {
        "source": {
            "algorithm": "tpsr",
            "dataset_id": "CRK11",
            "seed": "520",
            "noise_tag": "clean",
            "task_id": "tpsr_s520_clean_g0001",
            "host": host,
        },
        "snapshots": [],
    }
    for minute in range(1, 181):
        outer_path = f"/bundle/tpsr/minute_{minute:04d}.json"
        if minute == 53:
            payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="pre_future_eq",
                id_nmse=1.0e-2,
                ood_nmse=1.0e-2,
            )
            payload["source_score"] = 1.0
            future_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        elif minute in (54, 55):
            payload = _snapshot_payload(
                minute,
                record_type="periodic_backfill",
                expression="leaked_eq",
                id_nmse=1.0e-9,
                ood_nmse=1.0e-9,
                backfilled_from_minute=56,
            )
            payload["source_score"] = 2.0
            future_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        elif minute == 56:
            payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="future_anchor_eq",
                id_nmse=1.0e-4,
                ood_nmse=1.0e-4,
            )
            payload["source_score"] = 2.0
            payload["checkpoint_index"] = 54
            future_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        else:
            future_record["snapshots"].append(_heartbeat_snapshot(minute, outer_path=outer_path))

    plain_record = {
        "source": {
            "algorithm": "dso",
            "dataset_id": "plain",
            "seed": "521",
            "noise_tag": "clean",
            "task_id": "dso_s521_clean_g0002",
            "host": host,
        },
        "snapshots": [],
    }
    for minute in range(1, 181):
        outer_path = f"/bundle/dso/minute_{minute:04d}.json"
        if minute == 1:
            payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="plain_eq",
                id_nmse=1.0,
                ood_nmse=1.0,
            )
            payload["source_score"] = 1.0
            plain_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        else:
            plain_record["snapshots"].append(_heartbeat_snapshot(minute, outer_path=outer_path))

    _write_bundle(bundle_path, [repair_record, future_record, plain_record])
    bundle_report_payload = {
        "tasks": 3,
        "expected_snapshots": 540,
        "available_snapshots": 525,
        "missing_snapshots": 15,
        "conflicting_snapshots": 0,
        "parse_errors": 0,
        "result_missing_or_invalid": 0,
        "freeze_raw": True,
        "verify_inner": False,
    }
    bundle_report_path.write_text(
        json.dumps(bundle_report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    repair_manifest_path = manifests_dir / "trajectory_repairs.v1.json"
    minute_164_snapshot = repair_record["snapshots"][163]
    minute_180_snapshot = repair_record["snapshots"][179]
    repair_manifest_payload = {
        "schema_version": "trajectory_repairs.v1",
        "condition": "clean",
        "horizon": 180,
        "repairs": [
            {
                "logical_key": "fepysr::Nguyen-12::s520::clean",
                "task_id": "fepysr_s520_clean_g0021",
                "host": host,
                "rule": "carry_forward_last_observed_best",
                "missing_minutes": list(range(165, 180)),
                "source_minute": 164,
                "source_snapshot_sha256": minute_164_snapshot["selected_sha256"],
                "source_record_type": "periodic_best",
                "source_elapsed_seconds": 164 * 60,
                "excluded_future_minute": 180,
                "excluded_future_snapshot_sha256": minute_180_snapshot["selected_sha256"],
                "supporting_evidence": {
                    "current_best_path": str(current_best_path.relative_to(tmp_path)),
                    "current_best_sha256": _sha256_file(current_best_path),
                    "wrapper_path": str(wrapper_path.relative_to(tmp_path)),
                    "wrapper_sha256": _sha256_file(wrapper_path),
                },
            }
        ],
    }
    repair_manifest_path.write_text(
        json.dumps(repair_manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    formula_recovery_path = manifests_dir / "formula_recovery.v1.json"
    formula_recovery_path.write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": "clean",
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    binding_report_path = reports_dir / "freeze_binding.json"
    binding_payload = {
        "noise_tag": "clean",
        "horizon": 180,
        "inventory_counts": {
            "hosts": 1,
            "tasks": 3,
            "expected_points": 540,
            "existing_points": 525,
            "missing_points": 15,
        },
        "freeze_counts": {
            "hosts": 1,
            "tasks": 3,
            "expected_points": 540,
            "existing_points": 525,
            "missing_points": 15,
        },
        "missing_point_details": [
            {
                "logical_key": "fepysr::Nguyen-12::s520::clean",
                "minute": minute,
                "host": host,
                "task_id": "fepysr_s520_clean_g0021",
            }
            for minute in range(165, 180)
        ],
        "input_files": {
            "freeze_records": [
                {
                    "host": host,
                    "path": str(bundle_path.resolve()),
                    "sha256": _sha256_file(bundle_path),
                    "size_bytes": bundle_path.stat().st_size,
                }
            ],
            "freeze_reports": [
                {
                    "host": host,
                    "path": str(bundle_report_path.resolve()),
                    "sha256": _sha256_file(bundle_report_path),
                    "size_bytes": bundle_report_path.stat().st_size,
                }
            ],
            "inventory_records": [],
            "inventory_reports": [],
        },
        "per_host": {
            host: {
                "freeze": {
                    "counts": {
                        "tasks": 3,
                        "expected_points": 540,
                        "existing_points": 525,
                        "missing_points": 15,
                        "result_ok": 3,
                        "result_missing_or_invalid": 0,
                    },
                    "records_file": {
                        "host": host,
                        "path": str(bundle_path.resolve()),
                        "sha256": _sha256_file(bundle_path),
                        "size_bytes": bundle_path.stat().st_size,
                    },
                    "report_file": {
                        "host": host,
                        "path": str(bundle_report_path.resolve()),
                        "sha256": _sha256_file(bundle_report_path),
                        "size_bytes": bundle_report_path.stat().st_size,
                    },
                    "report": bundle_report_payload,
                }
            }
        },
        "binding": {
            "drift_count": 0,
            "drift_details": [],
            "truncated_drift_details": 0,
        },
        "contract_ok": True,
    }
    binding_report_path.write_text(
        json.dumps(binding_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "stage5_root": stage5_root,
        "binding_report": binding_report_path,
        "repair_manifest": repair_manifest_path,
        "formula_recovery_manifest": formula_recovery_path,
    }


def _write_rerun_overlay(
    fixture: dict[str, Path],
    *,
    damage: str | None = None,
) -> Path:
    overlay_dir = fixture["stage5_root"] / "work/clean_rerun_overlay_v1"
    bundle_path = overlay_dir / "clean_eff_overlay.jsonl.gz"
    logical_key = "tpsr::CRK11::s520::clean"
    snapshots = []
    for minute in range(1, 181):
        payload = _snapshot_payload(
            minute,
            record_type=(
                "budget_end_internal_best" if minute == 180 else "periodic_best"
            ),
            expression="overlay_eq",
            id_nmse=1.0e-6,
            ood_nmse=2.0e-6,
        )
        payload.update(tool="tpsr", dataset="CRK11", seed=520)
        payload["source_score"] = 1.0
        snapshots.append(
            _frozen_snapshot_from_payload(
                minute,
                payload,
                outer_path=f"/rerun/tpsr/progress/minute_{minute:04d}.json",
            )
        )
    if damage == "missing_minute":
        snapshots.pop(17)
    elif damage == "bad_endpoint":
        endpoint_payload = json.loads(snapshots[-1]["raw_text"])
        endpoint_payload["record_type"] = "periodic_best"
        snapshots[-1] = _frozen_snapshot_from_payload(
            180,
            endpoint_payload,
            outer_path="/rerun/tpsr/progress/minute_0180.json",
        )
    result_payload = {
        "tool": "tpsr",
        "dataset": "CRK11",
        "seed": 520,
        "status": "ok",
        "equation": "overlay_eq",
    }
    result_raw = json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
    record = {
        "source": {
            "algorithm": "tpsr",
            "dataset_id": "CRK11",
            "seed": 520,
            "noise_tag": "clean",
            "task_id": "tpsr_s520_clean_g0001",
            "host": "rerun-host",
            "batch": "rerun",
            "path": "/rerun/tpsr/result.json",
        },
        "result": {
            "status": "ok",
            "sha256": hashlib.sha256(result_raw.encode()).hexdigest(),
            "raw_text": result_raw,
        },
        "snapshots": snapshots,
        "overlay": {
            "schema_version": "clean_rerun_eff_overlay_v1",
            "scope": "mixed_clean_rerun_overlay",
            "replacement_scope": "final_and_eff",
            "logical_key": logical_key,
            "origin": "test_rerun",
        },
    }
    records = [record, record] if damage == "duplicate" else [record]
    _write_bundle(bundle_path, records)
    bundle_sha = _sha256_file(bundle_path)
    if damage == "bad_bundle_sha":
        bundle_sha = "0" * 64
    manifest_path = overlay_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "clean_rerun_eff_overlay_v1",
                "status": "passed",
                "scope": "mixed_clean_rerun_overlay",
                "horizon_minutes": 180,
                "overlay_unique_keys": 1,
                "replacement_count": 1,
                "eff_replacement_count": 1,
                "eff_replacement_keys": [
                    "tpsr::wrong::s520::clean"
                    if damage == "wrong_key"
                    else logical_key
                ],
                "algorithm_counts": {"tpsr": 1},
                "checkpoint_identity": {
                    "expected_per_run": 180,
                    "verified_runs": 1,
                    "verified_points": 180,
                    "all_verified": True,
                },
                "outputs": {
                    "overlay_bundle": {
                        "path": str(bundle_path.resolve()),
                        "sha256": bundle_sha,
                        "size_bytes": bundle_path.stat().st_size,
                        "rows": 1,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_build_eff_preparation_applies_audited_repairs_and_future_backfill_counts(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    rows, report = build_eff_preparation(
        freeze_binding_report=fixture["binding_report"],
        repair_manifest=fixture["repair_manifest"],
        formula_recovery_manifest=fixture["formula_recovery_manifest"],
        repo_root=tmp_path,
        expected_hosts=1,
        expected_tasks=3,
        expected_points=540,
        expected_existing_points=525,
        expected_missing_points=15,
        expected_audited_repair_points=15,
        expected_future_backfill_ignored_points=2,
        expected_checkpoint_normalization_points=1,
        replay_performance=False,
    )

    assert len(rows) == 3
    assert report["summary"]["success_count"] == 3
    assert report["summary"]["unresolved_run_count"] == 0
    assert report["summary"]["audited_repair_points"] == 15
    assert report["summary"]["future_backfill_ignored_points"] == 2
    assert report["summary"]["checkpoint_normalization_points"] == 1
    assert report["summary"]["missing_points_after_repairs"] == 0
    assert report["summary"]["formal_eff_ready"] is True
    assert report["inputs"]["formula_recovery_manifest"] == {
        "path": str(fixture["formula_recovery_manifest"].resolve()),
        "sha256": _sha256_file(fixture["formula_recovery_manifest"]),
        "condition": "clean",
        "entry_count": 0,
    }

    by_key = {row["logical_key"]: row for row in rows}
    repaired = by_key["fepysr::Nguyen-12::s520::clean"]
    assert repaired["audited_repair_points"] == 15
    assert repaired["future_backfill_ignored_points"] == 0
    assert repaired["trajectory_sources"][164] == "native_carry_forward:164"
    assert repaired["trajectory_sources"][178] == "native_carry_forward:164"
    assert repaired["trajectory_sources"][179] == "native_endpoint_carry_forward:164"
    assert repaired["quality_trajectory"][163] == pytest.approx((phi_nmse(1.0e-3) + phi_nmse(2.0e-3)) / 2.0)
    assert repaired["selected_expression_trajectory"][163] == "repair_eq"
    assert repaired["selected_expression_trajectory"][178] == "repair_eq"
    assert repaired["selected_expression_trajectory"][179] == "repair_eq"
    assert repaired["id_quality_trajectory"][163] == pytest.approx(phi_nmse(1.0e-3))
    assert repaired["ood_quality_trajectory"][163] == pytest.approx(phi_nmse(2.0e-3))
    assert repaired["valid_output_trajectory"][163] is True
    assert all(
        quality == pytest.approx((id_quality + ood_quality) / 2.0)
        for quality, id_quality, ood_quality in zip(
            repaired["quality_trajectory"],
            repaired["id_quality_trajectory"],
            repaired["ood_quality_trajectory"],
        )
    )
    assert repaired["m_eff"] == pytest.approx(
        efficiency_from_qualities(repaired["quality_trajectory"], horizon=180)
    )

    future = by_key["tpsr::CRK11::s520::clean"]
    assert future["future_backfill_ignored_points"] == 2
    assert future["checkpoint_normalizations"] == [
        {
            "minute": 56,
            "original_checkpoint_index": 54,
            "normalized_checkpoint_index": 56,
            "future_backfill_minutes": [54, 55],
        }
    ]
    assert future["trajectory_sources"][53] == "future_backfill_ignored:56;native_carry_forward:53"
    assert future["trajectory_sources"][54] == "future_backfill_ignored:56;native_carry_forward:53"
    assert future["quality_trajectory"][55] == pytest.approx((phi_nmse(1.0e-4) + phi_nmse(1.0e-4)) / 2.0)

    plain = by_key["dso::plain::s521::clean"]
    assert plain["audited_repair_points"] == 0
    assert all(0.0 <= value <= 1.0 for value in plain["quality_trajectory"])
    assert plain["selected_expression_trajectory"] == ["plain_eq"] * 180
    assert plain["valid_output_trajectory"] == [True] * 180


def test_eff_preparation_replaces_exact_overlay_trajectory(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    overlay_manifest = _write_rerun_overlay(fixture)
    rows, report = build_eff_preparation(
        freeze_binding_report=fixture["binding_report"],
        repair_manifest=fixture["repair_manifest"],
        formula_recovery_manifest=fixture["formula_recovery_manifest"],
        rerun_overlay_manifest=overlay_manifest,
        repo_root=tmp_path,
        expected_hosts=1,
        expected_tasks=3,
        expected_points=540,
        expected_existing_points=525,
        expected_missing_points=15,
        expected_audited_repair_points=15,
        expected_future_backfill_ignored_points=None,
        expected_checkpoint_normalization_points=None,
        expected_overlay_replacements=1,
        replay_performance=False,
    )

    by_key = {row["logical_key"]: row for row in rows}
    replaced = by_key["tpsr::CRK11::s520::clean"]
    assert replaced["host"] == "rerun-host"
    assert replaced["bundle_path"].endswith("clean_eff_overlay.jsonl.gz")
    assert replaced["bundle_report_path"] == str(overlay_manifest.resolve())
    assert replaced["repair_applied"] is False
    assert len(replaced["quality_trajectory"]) == 180
    assert replaced["trajectory_sources"][-1] == "native_carry_forward:1"
    assert report["summary"]["overlay_replacement_count"] == 1
    assert report["summary"]["formal_eff_ready"] is True
    assert report["inputs"]["rerun_overlay_manifest"]["sha256"] == _sha256_file(
        overlay_manifest
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("bad_bundle_sha", "overlay bundle SHA"),
        ("duplicate", "overlay logical_key 重复"),
        ("wrong_key", "eff_replacement_keys"),
        ("missing_minute", "恰有 180"),
        ("bad_endpoint", "minute_0180"),
    ],
)
def test_eff_preparation_rejects_invalid_overlay(
    tmp_path: Path, damage: str, message: str
) -> None:
    fixture = _build_fixture(tmp_path)
    overlay_manifest = _write_rerun_overlay(fixture, damage=damage)
    with pytest.raises(EffPreparationContractError, match=message):
        build_eff_preparation(
            freeze_binding_report=fixture["binding_report"],
            repair_manifest=fixture["repair_manifest"],
            formula_recovery_manifest=fixture["formula_recovery_manifest"],
            rerun_overlay_manifest=overlay_manifest,
            repo_root=tmp_path,
            expected_hosts=1,
            expected_tasks=3,
            expected_points=540,
            expected_existing_points=525,
            expected_missing_points=15,
            expected_overlay_replacements=1,
            replay_performance=False,
        )


def test_limited_eff_preparation_is_not_formal_ready(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    rows, report = build_eff_preparation(
        freeze_binding_report=fixture["binding_report"],
        repair_manifest=fixture["repair_manifest"],
        formula_recovery_manifest=fixture["formula_recovery_manifest"],
        repo_root=tmp_path,
        expected_hosts=1,
        expected_tasks=3,
        expected_points=540,
        expected_existing_points=525,
        expected_missing_points=15,
        expected_audited_repair_points=15,
        expected_future_backfill_ignored_points=2,
        expected_checkpoint_normalization_points=1,
        limit_runs=1,
        replay_performance=False,
    )

    assert len(rows) == 1
    assert report["summary"]["full_contract_checked"] is False
    assert report["summary"]["formal_eff_ready"] is False


def test_replay_invalid_candidate_succeeds_with_zero_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import prepare_eff as module

    snapshots = {
        1: {
            "record_type": "periodic_best",
            "checkpoint_index": 1,
            "status": "ok",
            "equation": "1 / (x0 - x0)",
            "canonical_artifact": {"instantiated_expression": "1 / (x0 - x0)"},
        }
    }

    def fake_replay(payload, *, algorithm, repo_root, cache, **kwargs):
        return {
            "evaluation_path": "canonical_replay.v1",
            "canonical_artifact": payload["canonical_artifact"],
            "canonical_artifact_sha256": "b" * 64,
            "artifact_rebuilt": False,
            "id_test": None,
            "ood_test": None,
            "id_quality": 0.0,
            "ood_quality": 0.0,
            "valid_output": False,
            "invalid_reason": "canonical prediction 含 NaN/Inf",
            "error": None,
        }

    monkeypatch.setattr(module, "replay_payload_performance", fake_replay)
    replayed, counts = module._replay_trajectory_payloads(
        snapshots,
        algorithm="demo",
        repo_root=tmp_path,
        cache=module.PerformanceReplayCache(),
    )

    assert counts == {
        "attempted": 1,
        "succeeded": 1,
        "failed": 0,
        "invalid_output": 1,
        "artifact_rebuilt": 0,
    }
    assert replayed[1]["status"] == "invalid"
    assert replayed[1]["canonical_replay_invalid_reason"]
    assert "canonical_replay_error" not in replayed[1]
    trajectory = module.reconstruct_trajectory(replayed, horizon=1)
    evidence = module._trajectory_evidence(
        trajectory, logical_key="demo::case::s520::clean"
    )
    assert evidence["quality_trajectory"] == [0.0]
    assert evidence["id_quality_trajectory"] == [0.0]
    assert evidence["ood_quality_trajectory"] == [0.0]
    assert evidence["selected_expression_trajectory"] == [None]
    assert evidence["valid_output_trajectory"] == [False]


def test_eff_final_without_artifact_uses_logical_task_recovery_params(
    tmp_path: Path,
) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import prepare_eff as module
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import (
        load_formula_recovery_manifest,
    )

    dataset_dir = tmp_path / "sim-datasets-data/ssr50/datasets/demo/case"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.yaml").write_text(
        "dataset:\n  name: case\n  target:\n    name: y\n  features:\n    - name: a\n    - name: b\n",
        encoding="utf-8",
    )
    for split in ("train", "valid", "id_test", "ood_test"):
        (dataset_dir / f"{split}.csv").write_text(
            "a,b,y\n10,1,2\n20,2,4\n30,3,6\n", encoding="utf-8"
        )
    equation = "def equation(a, b, params):\n    return params[0] * b\n"
    result_sha = "a" * 64
    recovery_path = tmp_path / "formula_recovery.v1.json"
    recovery_path.write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": "clean",
                "entries": [
                    {
                        "task_id": "drsr_s520_clean_g0001",
                        "resolution": "recovered_params",
                        "frozen_result_sha256": result_sha,
                        "equation_sha256": hashlib.sha256(equation.encode()).hexdigest(),
                        "params": [2.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshots = {
        180: {
            "record_type": "recovered_final",
            "checkpoint_index": "final",
            "status": "ok",
            "equation": equation,
            "dataset_dir": str(dataset_dir),
        }
    }
    replayed, counts = module._replay_trajectory_payloads(
        snapshots,
        algorithm="drsr",
        repo_root=tmp_path,
        cache=module.PerformanceReplayCache(),
        recovery_manifest=load_formula_recovery_manifest(
            recovery_path, expected_condition="clean"
        ),
        task_id="drsr_s520_clean_g0001",
        condition="clean",
        result_sha256=result_sha,
    )

    assert counts == {
        "attempted": 1,
        "succeeded": 1,
        "failed": 0,
        "invalid_output": 0,
        "artifact_rebuilt": 1,
    }
    assert replayed[180]["canonical_artifact"]["parameter_values"] == [2.0]
    assert replayed[180]["id_test"]["nmse"] == pytest.approx(0.0)


def test_cli_writes_jsonl_csv_and_report(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    output_jsonl = tmp_path / "eff_preparation.jsonl"
    output_csv = tmp_path / "eff_preparation.csv"
    output_report = tmp_path / "eff_preparation.json"

    cmd = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff",
        "--freeze-binding-report",
        str(fixture["binding_report"]),
        "--repair-manifest",
        str(fixture["repair_manifest"]),
        "--formula-recovery-manifest",
        str(fixture["formula_recovery_manifest"]),
        "--repo-root",
        str(tmp_path),
        "--expected-hosts",
        "1",
        "--expected-tasks",
        "3",
        "--expected-points",
        "540",
        "--expected-existing-points",
        "525",
        "--expected-missing-points",
        "15",
        "--expected-audited-repair-points",
        "15",
        "--expected-future-backfill-ignored-points",
        "2",
        "--expected-checkpoint-normalization-points",
        "1",
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
        "--output-report",
        str(output_report),
        "--skip-canonical-replay",
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(completed.stdout)
    assert report["status"] == "ok"
    assert report["contract_ok"] is True
    assert report["summary"]["success_count"] == 3
    assert report["outputs"]["eff_jsonl"] == str(output_jsonl.resolve())
    assert report["outputs"]["eff_jsonl_row_count"] == 3
    assert report["outputs"]["eff_csv"] == str(output_csv.resolve())
    assert report["outputs"]["eff_csv_row_count"] == 3
    assert output_jsonl.exists()
    assert output_csv.exists()
    assert output_report.exists()

    with output_jsonl.open("r", encoding="utf-8") as handle:
        jsonl_rows = [json.loads(line) for line in handle if line.strip()]
    assert len(jsonl_rows) == 3
    assert all(len(row["quality_trajectory"]) == 180 for row in jsonl_rows)
    assert all(len(row["selected_expression_trajectory"]) == 180 for row in jsonl_rows)
    assert all(len(row["id_quality_trajectory"]) == 180 for row in jsonl_rows)
    assert all(len(row["ood_quality_trajectory"]) == 180 for row in jsonl_rows)
    assert all(len(row["valid_output_trajectory"]) == 180 for row in jsonl_rows)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3
    assert "q_0180" in csv_rows[0]
    assert "id_q_0180" in csv_rows[0]
    assert "ood_q_0180" in csv_rows[0]
    assert "expression_0180" in csv_rows[0]
    assert "valid_output_0180" in csv_rows[0]
    assert "trajectory_source_0180" in csv_rows[0]
    assert float(csv_rows[0]["m_eff"]) >= 0.0


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE5_EFF_DRY") != "1",
    reason="仅手工触发真实仓库 dry CLI 校验",
)
def test_real_repo_cli_dry_run(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "real_dry.jsonl"
    output_csv = tmp_path / "real_dry.csv"
    output_report = tmp_path / "real_dry.json"
    cmd = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff",
        "--limit-runs",
        "32",
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
        "--output-report",
        str(output_report),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert payload["summary"]["success_count"] == 32
    assert payload["summary"]["full_contract_checked"] is False


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE5_EFF_FULL") != "1",
    reason="仅手工触发真实仓库 full CLI 校验",
)
def test_real_repo_cli_full_run(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "real_full.jsonl"
    output_csv = tmp_path / "real_full.csv"
    output_report = tmp_path / "real_full.json"
    cmd = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff",
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
        "--output-report",
        str(output_report),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert payload["summary"]["success_count"] == 2250
    assert payload["summary"]["audited_repair_points"] == 15
    assert payload["summary"]["future_backfill_ignored_points"] == 34
    assert payload["summary"]["formal_eff_ready"] is True
