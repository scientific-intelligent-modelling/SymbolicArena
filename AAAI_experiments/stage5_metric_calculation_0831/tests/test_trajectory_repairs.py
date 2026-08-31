from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.trajectories import (
    reconstruct_trajectory,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.trajectory_repairs import (
    TrajectoryRepairContractError,
    apply_repair_manifest,
    load_repair_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE5_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _load_gap_record() -> dict[str, object]:
    path = (
        STAGE5_ROOT
        / "source_snapshot/trajectory_freeze/clean_freeze_iaaccn23.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source = record["source"]
            if (
                source["algorithm"] == "fepysr"
                and source["dataset_id"] == "Nguyen-12"
                and int(source["seed"]) == 520
            ):
                return record
    raise AssertionError("未找到 gap record")


def test_real_gap_is_repaired_only_from_minute_164() -> None:
    manifest = load_repair_manifest(
        STAGE5_ROOT / "manifests/trajectory_repairs.v1.json",
        repo_root=REPO_ROOT,
    )
    snapshots, audit = apply_repair_manifest(
        _load_gap_record(),
        manifest=manifest,
        repo_root=REPO_ROOT,
    )

    assert audit["applied_minutes"] == list(range(165, 180))
    assert audit["source_minute"] == 164
    assert snapshots[164]["record_type"] == "periodic_best"
    assert snapshots[165]["record_type"] == "audited_carry_forward"
    assert snapshots[165]["recovery_provenance"]["source_minute"] == 164
    assert snapshots[179]["checkpoint_index"] == 179
    assert snapshots[180]["record_type"] == "recovered_final"
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[164].source == "audited_repair:164"
    assert trajectory[178].source == "audited_repair:164"


def test_repair_rejects_source_snapshot_hash_drift() -> None:
    manifest = load_repair_manifest(
        STAGE5_ROOT / "manifests/trajectory_repairs.v1.json",
        repo_root=REPO_ROOT,
    )
    record = copy.deepcopy(_load_gap_record())
    record["snapshots"][163]["selected_sha256"] = "0" * 64
    with pytest.raises(TrajectoryRepairContractError, match="SHA"):
        apply_repair_manifest(record, manifest=manifest, repo_root=REPO_ROOT)


def test_repair_rejects_unlisted_missing_minute() -> None:
    manifest = load_repair_manifest(
        STAGE5_ROOT / "manifests/trajectory_repairs.v1.json",
        repo_root=REPO_ROOT,
    )
    record = copy.deepcopy(_load_gap_record())
    record["snapshots"][99]["status"] = "missing"
    record["snapshots"][99]["raw_text"] = None
    with pytest.raises(TrajectoryRepairContractError, match="缺失分钟集合"):
        apply_repair_manifest(record, manifest=manifest, repo_root=REPO_ROOT)
