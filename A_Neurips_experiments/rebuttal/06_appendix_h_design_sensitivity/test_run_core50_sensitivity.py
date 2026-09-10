from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_core50_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("core50_h_sensitivity", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path, selected_prefix: int = 50) -> tuple[Path, Path, Path]:
    rows = []
    difficulties = ("easy", "medium", "hard", "extreme")
    for index in range(64):
        family = "family_a" if index < 32 else "family_b"
        rows.append(
            {
                "dataset_id": f"g{index:03d}",
                "dataset_name": f"dataset_{index:03d}",
                "dataset_rel": f"sim-datasets-data/family/{index:03d}",
                "family": family,
                "subgroup": f"{family}_subgroup_{index % 8}",
                "basename": f"basename_{index:03d}",
                "semantic_duplicate_group": f"semantic_{index:03d}",
                "metadata_class": "non-dummy",
                "operator_group": f"operator_{index % 4}",
                "feature_count_bin": f"feature_{index % 3}",
                "sample_count_bin": f"sample_{index % 3}",
                "complexity_bin": f"complexity_{index % 3}",
                "difficulty_bin": difficulties[index % 4],
                "failure_mode": f"failure_{index % 4}",
                "winner_probe": MODULE.selector.PROBE4[index % len(MODULE.selector.PROBE4)],
                "info_score": float(index) / 63.0,
            }
        )
    dataset_level = pd.DataFrame(rows)
    dataset_algorithm = pd.DataFrame(
        [
            {
                "dataset_id": row["dataset_id"],
                "method_norm": method,
                "n_valid_total_seeds": 3,
            }
            for row in rows
            for method in MODULE.PROBE4
        ]
    )
    dataset_level_path = tmp_path / "dataset_level.csv"
    dataset_algorithm_path = tmp_path / "dataset_algorithm.csv"
    dataset_level.to_csv(dataset_level_path, index=False)
    dataset_algorithm.to_csv(dataset_algorithm_path, index=False)

    selected_positions = list(range(25)) + list(range(32, 32 + selected_prefix - 25))
    manifest = dataset_level.iloc[selected_positions][["dataset_name", "dataset_rel"]].copy()
    manifest.insert(0, "core50_index", np.arange(1, len(manifest) + 1))
    manifest = manifest.rename(columns={"dataset_rel": "dataset_dir"})
    manifest_path = tmp_path / "core_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return dataset_level_path, dataset_algorithm_path, manifest_path


def _args(
    tmp_path: Path,
    dataset_level: Path,
    dataset_algorithm: Path,
    manifest: Path,
    **overrides,
) -> Namespace:
    values = {
        "dataset_level": dataset_level,
        "dataset_algorithm": dataset_algorithm,
        "core_manifest": manifest,
        "output": tmp_path / "output",
        "seed": 20260910,
        "restarts": 1,
        "grid_restarts": 1,
        "n_configs": 8,
        "delta": 0.05,
        "expected_reservoir_size": 64,
        "subset_size": 50,
        "resume": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_joint_sampling_is_reproducible_bounded_and_normalized() -> None:
    first = MODULE.sample_joint_weights(5000, seed=17)
    second = MODULE.sample_joint_weights(5000, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5000
    assert first["config_id"].is_unique
    for column, default in zip(
        MODULE.RAW_WEIGHT_COLUMNS, MODULE.DEFAULT_WEIGHTS, strict=True
    ):
        assert (first[column] >= default - 0.05 - 1e-12).all()
        assert (first[column] <= default + 0.05 + 1e-12).all()
    normalized = first[list(MODULE.NORMALIZED_WEIGHT_COLUMNS)].to_numpy()
    assert np.all(normalized > 0)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert first[list(MODULE.RAW_WEIGHT_COLUMNS)].nunique().gt(1).all()


def test_run_fails_closed_when_default_selection_does_not_match_frozen(tmp_path: Path) -> None:
    dataset_level, dataset_algorithm, manifest = _fixture(tmp_path)
    args = _args(tmp_path, dataset_level, dataset_algorithm, manifest)

    audit = MODULE.run(args)

    assert audit["status"] == "blocked"
    assert audit["baseline_gate"] == "fail_exact_reproduction"
    assert audit["baseline_frozen_overlap"] < 50
    output_files = {path.name for path in args.output.iterdir()}
    assert output_files == {"audit.json"}


def test_repair_checkpoint_keeps_only_complete_configurations(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    config_results = pd.DataFrame(
        [
            {"config_id": "joint_0000", "status": "ok", "objective": 1.0},
            {"config_id": "joint_0001", "status": "ok", "objective": 0.9},
        ]
    )
    memberships = pd.DataFrame(
        [
            {"config_id": "joint_0000", "dataset_id": f"g{i:03d}"}
            for i in range(50)
        ]
        + [
            {"config_id": "joint_0001", "dataset_id": f"g{i:03d}"}
            for i in range(49)
        ]
    )
    config_results.to_csv(output / MODULE.CONFIG_RESULTS_FILE, index=False)
    memberships.to_csv(output / MODULE.MEMBERSHIPS_FILE, index=False)

    completed = MODULE.repair_checkpoint(output, subset_size=50)

    assert completed == {"joint_0000"}
    repaired_results = pd.read_csv(output / MODULE.CONFIG_RESULTS_FILE)
    repaired_memberships = pd.read_csv(output / MODULE.MEMBERSHIPS_FILE)
    assert repaired_results["config_id"].tolist() == ["joint_0000"]
    assert len(repaired_memberships) == 50
    assert repaired_memberships["config_id"].unique().tolist() == ["joint_0000"]


def test_success_contract_writes_config_membership_frequency_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_level, dataset_algorithm, manifest = _fixture(tmp_path)
    args = _args(tmp_path, dataset_level, dataset_algorithm, manifest, n_configs=3)

    prepared = MODULE.load_prepared_inputs(
        args.dataset_level, args.dataset_algorithm, args.core_manifest
    )
    frozen_ids = prepared.frozen_ids

    def fake_select(*_args, **_kwargs):
        return prepared.problem.indices(frozen_ids), [], 0

    monkeypatch.setattr(MODULE, "select_for_objective", fake_select)
    audit = MODULE.run(args)

    assert audit["status"] == "complete"
    assert audit["configurations"] == 3
    assert audit["membership_rows"] == 150
    assert len(pd.read_csv(args.output / MODULE.CONFIG_RESULTS_FILE)) == 3
    assert len(pd.read_csv(args.output / MODULE.MEMBERSHIPS_FILE)) == 150
    assert len(pd.read_csv(args.output / MODULE.FREQUENCY_FILE)) == 64
    assert len(pd.read_csv(args.output / MODULE.MANIFEST_FILE)) == 50
