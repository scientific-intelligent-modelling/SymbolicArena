from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("run_probe4_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("run_probe4_sensitivity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_candidates(path: Path) -> None:
    rows = [
        {
            "panel": "a;b;c;d",
            "health": 0.8,
            "panel_fidelity": 0.8,
            "complementarity": 0.8,
            "coverage": 0.8,
        },
        {
            "panel": "a;b;c;e",
            "health": 0.9,
            "panel_fidelity": 0.9,
            "complementarity": 0.9,
            "coverage": 0.9,
        },
        {
            "panel": "a;b;d;e",
            "health": 0.7,
            "panel_fidelity": 0.7,
            "complementarity": 0.7,
            "coverage": 0.7,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _contract() -> object:
    return MODULE.BaselineContract(
        selected_panel="a;b;c;d",
        expected_best_panel="a;b;c;e",
        expected_selected_rank=2,
        expected_best_score=0.9,
        expected_selected_score=0.8,
        score_tolerance=1e-12,
    )


def test_sample_weights_is_deterministic_bounded_and_normalized() -> None:
    first = MODULE.sample_weights(n_draws=100, seed=314159)
    second = MODULE.sample_weights(n_draws=100, seed=314159)

    np.testing.assert_array_equal(first.raw, second.raw)
    np.testing.assert_array_equal(first.normalized, second.normalized)
    assert first.raw.shape == (100, 4)
    assert np.all(first.raw >= np.asarray(MODULE.DEFAULT_WEIGHTS) - 0.05)
    assert np.all(first.raw <= np.asarray(MODULE.DEFAULT_WEIGHTS) + 0.05)
    np.testing.assert_allclose(first.normalized.sum(axis=1), 1.0)


def test_load_candidates_rejects_duplicate_or_invalid_rows(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    _write_candidates(duplicate)
    text = duplicate.read_text(encoding="utf-8")
    duplicate.write_text(text + text.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(MODULE.ContractError, match="重复"):
        MODULE.load_candidates(duplicate)

    invalid = tmp_path / "invalid.csv"
    _write_candidates(invalid)
    invalid.write_text(
        invalid.read_text(encoding="utf-8").replace("0.8,0.8", "nan,0.8", 1),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.ContractError, match="有限"):
        MODULE.load_candidates(invalid)


def test_baseline_audit_fails_closed_on_wrong_historical_result(tmp_path: Path) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source)
    candidates = MODULE.load_candidates(source)
    wrong = MODULE.BaselineContract(
        selected_panel="a;b;c;d",
        expected_best_panel="a;b;d;e",
        expected_selected_rank=2,
        expected_best_score=None,
        expected_selected_score=None,
        score_tolerance=1e-12,
    )

    with pytest.raises(MODULE.ContractError, match="基线最优组合"):
        MODULE.audit_default_baseline(candidates, wrong)


def test_run_experiment_writes_complete_reproducible_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source)
    output = tmp_path / "output"

    result = MODULE.run_experiment(
        input_path=source,
        output_dir=output,
        n_draws=7,
        seed=20260831,
        contract=_contract(),
    )

    assert result["configuration_count"] == 7
    assert result["candidate_count"] == 3
    with (output / "probe4_weight_configurations.csv").open(encoding="utf-8") as handle:
        configurations = list(csv.DictReader(handle))
    with (output / "probe4_candidate_scores.csv").open(encoding="utf-8") as handle:
        candidate_scores = list(csv.DictReader(handle))
    with (output / "probe4_perturbation_results.csv").open(encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "probe4_baseline_audit.json").read_text(encoding="utf-8"))

    assert len(configurations) == 7
    assert len(candidate_scores) == 7 * 3
    assert len(summaries) == 7
    assert all(abs(sum(float(row[key]) for key in ("wH", "wF", "wC", "wV")) - 1) < 1e-12 for row in configurations)
    assert manifest["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["contract"]["perturbation"] == "independent_uniform_plus_minus_0.05_then_renormalize"
    assert audit["passed"] is True
    assert audit["selected_rank"] == 2

    second = tmp_path / "output_second"
    MODULE.run_experiment(
        input_path=source,
        output_dir=second,
        n_draws=7,
        seed=20260831,
        contract=_contract(),
    )
    assert (output / "probe4_weight_configurations.csv").read_bytes() == (
        second / "probe4_weight_configurations.csv"
    ).read_bytes()
    assert (output / "probe4_candidate_scores.csv").read_bytes() == (
        second / "probe4_candidate_scores.csv"
    ).read_bytes()


def test_run_experiment_does_not_write_when_baseline_audit_fails(tmp_path: Path) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source)
    output = tmp_path / "must_not_exist"
    bad_contract = MODULE.BaselineContract(
        selected_panel="a;b;c;d",
        expected_best_panel="a;b;d;e",
        expected_selected_rank=2,
        expected_best_score=None,
        expected_selected_score=None,
        score_tolerance=1e-12,
    )

    with pytest.raises(MODULE.ContractError):
        MODULE.run_experiment(
            input_path=source,
            output_dir=output,
            n_draws=3,
            seed=1,
            contract=bad_contract,
        )
    assert not output.exists()
