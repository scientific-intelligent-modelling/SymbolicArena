from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.formula_parameter_recovery import (  # noqa: E402
    FormulaParameterRecoveryError,
    build_formula_recovery_manifest,
)


def _write_index(path: Path, *, result_path: Path, task_id: str) -> None:
    fields = [
        "logical_key", "batch", "noise_tag", "algorithm", "dataset_id", "seed",
        "task_id", "host", "remote_result_path", "local_result_path", "local_sha256",
        "status", "equation_present", "canonical_artifact_present", "id_nmse_source",
        "id_nmse_result", "ood_nmse_source", "ood_nmse_result", "metrics_match",
    ]
    row = {
        "logical_key": "drsr::D1::s520::noise001",
        "batch": "formal",
        "noise_tag": "noise001",
        "algorithm": "drsr",
        "dataset_id": "D1",
        "seed": "520",
        "task_id": task_id,
        "host": "iaaccn22",
        "remote_result_path": "/remote/result.json",
        "local_result_path": str(result_path),
        "local_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "status": "ok",
        "equation_present": "True",
        "canonical_artifact_present": "False",
        "id_nmse_source": "0.1",
        "id_nmse_result": "0.1",
        "ood_nmse_source": "0.2",
        "ood_nmse_result": "0.2",
        "metrics_match": "True",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _synthetic_fixture(
    tmp_path: Path,
    candidate_functions: list[str],
    *,
    result_function: str = "def equation(col0, params):\n    return params[0] + params[1] * col0\n",
) -> tuple[Path, Path]:
    task_id = "drsr_s520_noise001_g0001"
    result_path = tmp_path / "results/noise001/drsr" / task_id / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "equation": result_function,
                "canonical_artifact": None,
                "id_test": {"nmse": 0.1},
                "ood_test": {"nmse": 0.2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "result_index_noise001.csv"
    _write_index(index_path, result_path=result_path, task_id=task_id)
    history_dir = tmp_path / "history/noise001" / task_id
    history_dir.mkdir(parents=True)
    for number, function in enumerate(candidate_functions):
        (history_dir / f"best_sample_{number}.json").write_text(
            json.dumps(
                {
                    "iteration": number + 1,
                    "sample_order": number,
                    "score": -float(number),
                    "function": function,
                    "params": [1.0 + number, 2.0 + number],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return index_path, tmp_path / "history"


def test_real_noise001_recovery_reproduces_reference_substantive_content(tmp_path: Path) -> None:
    reference = json.loads(
        (STAGE_ROOT / "manifests/noise001_formula_recovery.v1.json").read_text(encoding="utf-8")
    )
    output = tmp_path / "noise001_formula_recovery.v1.json"
    report = build_formula_recovery_manifest(
        result_index_csv=STAGE_ROOT / "source_snapshot/result_index_noise001.csv",
        recovery_history_root=STAGE_ROOT / "source_snapshot/recovery_history",
        condition="noise001",
        output_json=output,
        expected_missing_count=7,
    )
    actual = json.loads(output.read_text(encoding="utf-8"))

    assert report["resolution_counts"] == {"recovered_params": 7}
    assert actual["schema_version"] == "formula_recovery.v1"
    assert actual["condition"] == "noise001"
    expected_by_id = {entry["task_id"]: entry for entry in reference["entries"]}
    actual_by_id = {entry["task_id"]: entry for entry in actual["entries"]}
    assert set(actual_by_id) == set(expected_by_id)
    for task_id, expected in expected_by_id.items():
        recovered = actual_by_id[task_id]
        assert recovered["resolution"] == "recovered_params"
        assert recovered["frozen_result_sha256"] == expected["frozen_result_sha256"]
        assert recovered["equation_sha256"] == expected["equation_sha256"]
        assert recovered["params"] == expected["params"]
        assert recovered["source_evidence"]["sample_order"] == expected["source_evidence"]["sample_order"]
        assert (
            recovered["source_evidence"]["candidate_sha256"]
            == expected["source_evidence"]["candidate_sha256"]
        )
        assert Path(recovered["source_evidence"]["candidate_path"]).is_file()


@pytest.mark.parametrize(
    ("candidate_functions", "reason"),
    [
        (["def equation(x0, params):\n    return params[0] * x0\n"], "no_matching_history_candidate"),
        (
            [
                "def equation(x0, params):\n    return params[0] + params[1] * x0\n",
                "def equation(x0, params):\n    return params[0] + params[1] * x0\n",
            ],
            "multiple_matching_history_candidates",
        ),
    ],
)
def test_zero_or_multiple_skeleton_matches_fail_closed(
    tmp_path: Path, candidate_functions: list[str], reason: str
) -> None:
    index_path, history_root = _synthetic_fixture(tmp_path, candidate_functions)
    output = tmp_path / "manifest.json"
    report = build_formula_recovery_manifest(
        result_index_csv=index_path,
        recovery_history_root=history_root,
        condition="noise001",
        output_json=output,
        expected_missing_count=1,
    )

    entry = json.loads(output.read_text())["entries"][0]
    assert entry["resolution"] == "unavailable"
    assert entry["reason"] == reason
    assert "params" not in entry
    assert report["resolution_counts"] == {"unavailable": 1}


def test_output_refuses_drift(tmp_path: Path) -> None:
    index_path, history_root = _synthetic_fixture(
        tmp_path,
        ["def equation(x0, params):\n    return params[0] + params[1] * x0\n"],
    )
    output = tmp_path / "manifest.json"
    output.write_text("user-drift\n", encoding="utf-8")
    with pytest.raises(FormulaParameterRecoveryError, match="漂移"):
        build_formula_recovery_manifest(
            result_index_csv=index_path,
            recovery_history_root=history_root,
            condition="noise001",
            output_json=output,
            expected_missing_count=1,
        )
    assert output.read_text() == "user-drift\n"


def test_recovery_discovers_candidates_in_nested_experiment_directory(
    tmp_path: Path,
) -> None:
    index_path, history_root = _synthetic_fixture(
        tmp_path,
        ["def equation(x0, params):\n    return params[0] + params[1] * x0\n"],
    )
    task_dir = history_root / "noise001/drsr_s520_noise001_g0001"
    candidate = task_dir / "best_sample_0.json"
    nested_dir = task_dir / "experiments/run-1/best_history"
    nested_dir.mkdir(parents=True)
    candidate.rename(nested_dir / candidate.name)

    report = build_formula_recovery_manifest(
        result_index_csv=index_path,
        recovery_history_root=history_root,
        condition="noise001",
        output_json=tmp_path / "manifest.json",
        expected_missing_count=1,
    )

    assert report["resolution_counts"] == {"recovered_params": 1}


def test_recovery_accepts_python_negative_parameter_index(tmp_path: Path) -> None:
    function = "def equation(x0, params):\n    return params[0] * x0 + params[-1]\n"
    index_path, history_root = _synthetic_fixture(
        tmp_path,
        [function],
        result_function=function,
    )

    report = build_formula_recovery_manifest(
        result_index_csv=index_path,
        recovery_history_root=history_root,
        condition="noise001",
        output_json=tmp_path / "manifest.json",
        expected_missing_count=1,
    )

    assert report["resolution_counts"] == {"recovered_params": 1}


def test_recovery_allows_candidate_without_unused_score(tmp_path: Path) -> None:
    function = "def equation(x0, params):\n    return params[0] + params[1] * x0\n"
    index_path, history_root = _synthetic_fixture(tmp_path, [function])
    candidate_path = (
        history_root
        / "noise001/drsr_s520_noise001_g0001/best_sample_0.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate.pop("score")
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")

    report = build_formula_recovery_manifest(
        result_index_csv=index_path,
        recovery_history_root=history_root,
        condition="noise001",
        output_json=tmp_path / "manifest.json",
        expected_missing_count=1,
    )

    assert report["resolution_counts"] == {"recovered_params": 1}


def test_recovery_audits_noncanonical_history_filename_without_blocking(
    tmp_path: Path,
) -> None:
    function = "def equation(x0, params):\n    return params[0] + params[1] * x0\n"
    index_path, history_root = _synthetic_fixture(tmp_path, [function])
    task_dir = history_root / "noise001/drsr_s520_noise001_g0001"
    ignored_path = task_dir / "best_sample_None.json"
    ignored_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "manifest.json"

    report = build_formula_recovery_manifest(
        result_index_csv=index_path,
        recovery_history_root=history_root,
        condition="noise001",
        output_json=output,
        expected_missing_count=1,
    )

    entry = json.loads(output.read_text(encoding="utf-8"))["entries"][0]
    assert report["resolution_counts"] == {"recovered_params": 1}
    assert entry["ignored_history_files"] == [
        {
            "path": str(ignored_path.resolve()),
            "reason": "noncanonical_sample_filename",
            "sha256": hashlib.sha256(ignored_path.read_bytes()).hexdigest(),
        }
    ]
