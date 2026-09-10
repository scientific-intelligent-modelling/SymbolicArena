from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("plot_appendix_h.py")
SPEC = importlib.util.spec_from_file_location("plot_appendix_h", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def complete_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("appendix_h_plot")
    probe_path = root / "probe4_perturbation_results.csv"
    config_path = root / "joint_config_results.csv"
    membership_path = root / "selected_memberships.csv"
    frequency_path = root / "membership_frequency.csv"

    probe_ids = np.arange(1, 5001)
    gaps = np.linspace(0.0, 2.0, 5000)
    best_scores = np.full(5000, 0.9)
    pd.DataFrame(
        {
            "configuration_id": probe_ids,
            "selected_panel": "a;b;c;d",
            "selected_score": best_scores * (1.0 - gaps / 100.0),
            "selected_rank": np.where(gaps == 0, 1, 2),
            "best_panel": np.where(gaps == 0, "a;b;c;d", "a;b;c;e"),
            "best_score": best_scores,
            "relative_gap_pct": gaps,
            "selected_is_best": gaps == 0,
        }
    ).to_csv(probe_path, index=False)

    config_ids = np.asarray([f"joint_{index:04d}" for index in range(5000)])
    overlap = np.where(np.arange(5000) % 2 == 0, 50, 49)
    jaccard = overlap / (100.0 - overlap)
    pd.DataFrame(
        {
            "config_id": config_ids,
            "status": "ok",
            "selected_count": 50,
            "feasible": True,
            "overlap_with_frozen": overlap,
            "jaccard_with_frozen": jaccard,
        }
    ).to_csv(config_path, index=False)

    memberships: list[pd.DataFrame] = []
    base_tasks = np.asarray([f"g{index:04d}" for index in range(50)])
    for start in range(0, 5000, 250):
        frames = []
        for index in range(start, min(start + 250, 5000)):
            tasks = base_tasks.copy()
            if index % 2 == 1:
                tasks[-1] = "g0050"
            frames.append(pd.DataFrame({"config_id": config_ids[index], "dataset_id": tasks}))
        memberships.append(pd.concat(frames, ignore_index=True))
    membership_frame = pd.concat(memberships, ignore_index=True)
    membership_frame.to_csv(membership_path, index=False)
    observed = membership_frame.groupby("dataset_id")["config_id"].nunique()
    frequency = pd.DataFrame({"dataset_id": [f"g{index:04d}" for index in range(664)]})
    frequency["selection_count"] = frequency["dataset_id"].map(observed).fillna(0).astype(int)
    frequency["selection_frequency"] = frequency["selection_count"] / 5000
    frequency.to_csv(frequency_path, index=False)
    return probe_path, config_path, membership_path, frequency_path


def test_plot_appendix_h_writes_double_panel_and_summary(
    tmp_path: Path,
    complete_inputs: tuple[Path, Path, Path, Path],
) -> None:
    probe_path, config_path, membership_path, frequency_path = complete_inputs
    output = tmp_path / "figures"

    result = MODULE.build_figure(
        probe_results_path=probe_path,
        core_config_path=config_path,
        core_membership_path=membership_path,
        core_frequency_path=frequency_path,
        output_dir=output,
    )

    assert result["status"] == "complete"
    assert (output / "figure_h1_design_sensitivity.png").stat().st_size > 1000
    assert (output / "figure_h1_design_sensitivity.pdf").stat().st_size > 1000
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation"]["probe_configurations"] == 5000
    assert summary["validation"]["core_configurations"] == 5000
    assert summary["validation"]["core_membership_rows"] == 250000
    assert summary["probe4"]["within_1pct_fraction"] == pytest.approx(0.5)
    assert summary["core50"]["tasks_selected_every_configuration"] == 49
    assert summary["core50"]["tasks_selected_at_least_75pct"] == 49
    assert summary["core50"]["minimum_overlap_with_frozen"] == 49
    assert set(summary["inputs"]) == {
        "probe4_perturbation_results",
        "core50_configuration_results",
        "core50_memberships",
        "core50_membership_frequency",
    }
    assert all(len(item["sha256"]) == 64 for item in summary["inputs"].values())


def test_missing_probe_configuration_fails_without_outputs(
    tmp_path: Path,
    complete_inputs: tuple[Path, Path, Path, Path],
) -> None:
    probe_path, config_path, membership_path, frequency_path = complete_inputs
    incomplete_probe = tmp_path / "probe_incomplete.csv"
    pd.read_csv(probe_path).iloc[:-1].to_csv(incomplete_probe, index=False)
    output = tmp_path / "must_not_exist"

    with pytest.raises(MODULE.PlotContractError, match="5000"):
        MODULE.build_figure(
            probe_results_path=incomplete_probe,
            core_config_path=config_path,
            core_membership_path=membership_path,
            core_frequency_path=frequency_path,
            output_dir=output,
        )
    assert not output.exists()


def test_duplicate_task_within_configuration_fails_closed(
    tmp_path: Path,
    complete_inputs: tuple[Path, Path, Path, Path],
) -> None:
    probe_path, config_path, membership_path, frequency_path = complete_inputs
    invalid_membership = tmp_path / "invalid_membership.csv"
    memberships = pd.read_csv(membership_path)
    memberships.loc[1, "dataset_id"] = memberships.loc[0, "dataset_id"]
    memberships.to_csv(invalid_membership, index=False)
    output = tmp_path / "must_not_exist"

    with pytest.raises(MODULE.PlotContractError, match="50 个唯一任务"):
        MODULE.build_figure(
            probe_results_path=probe_path,
            core_config_path=config_path,
            core_membership_path=invalid_membership,
            core_frequency_path=frequency_path,
            output_dir=output,
        )
    assert not output.exists()
