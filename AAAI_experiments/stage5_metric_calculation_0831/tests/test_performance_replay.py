from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import (
    FormulaRecoveryManifest,
    PerformanceReplayError,
    load_formula_recovery_manifest,
    replay_payload_performance,
    resolve_local_dataset_dir,
)


def _write_recovery_manifest(
    path: Path,
    *,
    condition: str,
    task_id: str,
    result_sha256: str,
    equation: str,
    resolution: str = "recovered_params",
) -> Path:
    entry: dict[str, object] = {
        "task_id": task_id,
        "resolution": resolution,
        "frozen_result_sha256": result_sha256,
        "equation_sha256": hashlib.sha256(equation.encode("utf-8")).hexdigest(),
    }
    if resolution == "recovered_params":
        entry["params"] = [2.0]
    else:
        entry["reason"] = "no_matching_history_candidate"
    path.write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": condition,
                "entries": [entry],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_dataset(repo_root: Path) -> Path:
    dataset_dir = repo_root / "sim-datasets-data/ssr50/datasets/demo/case"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.yaml").write_text(
        """dataset:
  name: case
  target:
    name: y
  features:
    - name: a
    - name: b
""",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "a": [100.0, 200.0, 300.0],
            "b": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
        }
    )
    for split in ("train", "valid", "id_test", "ood_test"):
        frame.to_csv(dataset_dir / f"{split}.csv", index=False)
    return dataset_dir


def _qlattice_payload(*, record_type: str) -> dict[str, object]:
    return {
        "tool": "QLattice",
        "dataset": "case",
        "dataset_dir": "/home/remote/sim-datasets-data/ssr50/datasets/demo/case",
        "expected_dataset_rel": "sim-datasets-data/ssr50/datasets/demo/case",
        "feature_names": ["a", "b"],
        "status": "ok",
        "record_type": record_type,
        "equation": "2*x1",
        # 旧工件错误地把 QLattice 的 x1 平移为了 x0。
        "canonical_artifact": {
            "tool_name": "QLattice",
            "raw_equation": "2*x1",
            "normalized_expression": "2*x0",
            "instantiated_expression": "2*x0",
        },
        "id_test": {"nmse": 1.0e6},
        "ood_test": {"nmse": 1.0e6},
    }


def test_resolve_local_dataset_dir_maps_remote_sim_data_path(tmp_path: Path) -> None:
    expected = _write_dataset(tmp_path)
    resolved = resolve_local_dataset_dir(_qlattice_payload(record_type="final_best"), repo_root=tmp_path)
    assert resolved == expected.resolve()


@pytest.mark.parametrize("record_type", ["periodic_best", "final_best"])
def test_minute_and_final_use_identical_corrected_canonical_replay(
    tmp_path: Path,
    record_type: str,
) -> None:
    _write_dataset(tmp_path)
    replay = replay_payload_performance(
        _qlattice_payload(record_type=record_type),
        algorithm="QLattice",
        repo_root=tmp_path,
    )

    assert replay["evaluation_path"] == "canonical_replay.v1"
    assert replay["artifact_rebuilt"] is True
    assert replay["canonical_artifact"]["instantiated_expression"] == "2*x1"
    assert replay["id_test"]["nmse"] == pytest.approx(0.0)
    assert replay["ood_test"]["nmse"] == pytest.approx(0.0)
    assert replay["native_id_nmse"] == pytest.approx(1.0e6)
    assert replay["valid_output"] is True


def test_replay_rejects_unresolvable_dataset_path(tmp_path: Path) -> None:
    payload = _qlattice_payload(record_type="final_best")
    payload.pop("expected_dataset_rel")
    payload["dataset_dir"] = "/remote/without/a/local/mapping"
    with pytest.raises(PerformanceReplayError, match="数据集目录"):
        replay_payload_performance(payload, algorithm="QLattice", repo_root=tmp_path)


def test_replay_classifies_nonfinite_prediction_as_invalid_output(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    payload = _qlattice_payload(record_type="final_best")
    payload["equation"] = "1 / (x1 - x1)"
    payload["canonical_artifact"]["raw_equation"] = "1 / (x1 - x1)"

    replay = replay_payload_performance(payload, algorithm="QLattice", repo_root=tmp_path)

    assert replay["evaluation_path"] == "canonical_replay.v1"
    assert replay["valid_output"] is False
    assert replay["id_quality"] == 0.0
    assert replay["ood_quality"] == 0.0
    assert replay["id_test"] is None
    assert replay["ood_test"] is None
    assert replay["invalid_reason"]
    assert replay["error"] is None


def test_replay_treats_missing_executable_artifact_as_evidence_error(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path)
    payload = {
        "dataset_dir": str(dataset_dir),
        "canonical_artifact": {"tool_name": "demo"},
    }

    with pytest.raises(PerformanceReplayError, match="可执行表达式"):
        replay_payload_performance(payload, algorithm="demo", repo_root=tmp_path)


def test_pyoperon_replay_uses_ast_execution_hint_instead_of_sympy_eager_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyOperon 浮点树不得因 SymPy 自动求值而让单条 replay 无限膨胀。"""

    dataset_dir = _write_dataset(tmp_path)
    expression = "2.0*x1"
    observed_artifacts: list[dict[str, object]] = []

    def _predict(artifact: dict[str, object], X: np.ndarray) -> np.ndarray:
        observed_artifacts.append(dict(artifact))
        assert artifact["executable_expression"] == expression
        return 2.0 * np.asarray(X, dtype=float)[:, 1]

    monkeypatch.setattr(
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay."
        "_predict_from_canonical_artifact",
        _predict,
    )
    payload = {
        "dataset_dir": str(dataset_dir),
        "equation": expression,
        "canonical_artifact": {
            "tool_name": "pyoperon",
            "raw_equation": expression,
            "normalized_expression": expression,
            "instantiated_expression": expression,
        },
    }

    replay = replay_payload_performance(payload, algorithm="pyoperon", repo_root=tmp_path)

    assert replay["artifact_rebuilt"] is False
    assert replay["valid_output"] is True
    assert replay["canonical_artifact"]["executable_expression"] == expression
    assert len(observed_artifacts) == 2


def test_replay_result_is_json_serializable(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    replay = replay_payload_performance(
        _qlattice_payload(record_type="final_best"),
        algorithm="QLattice",
        repo_root=tmp_path,
    )
    json.dumps(replay, ensure_ascii=False, sort_keys=True)


def test_formula_recovery_manifest_binds_condition_result_and_equation_sha(
    tmp_path: Path,
) -> None:
    equation = "def f(x, params):\n    return params[0] * x\n"
    result_sha = "a" * 64
    path = _write_recovery_manifest(
        tmp_path / "formula_recovery.v1.json",
        condition="clean",
        task_id="drsr_s520_clean_g0001",
        result_sha256=result_sha,
        equation=equation,
    )
    manifest = load_formula_recovery_manifest(path, expected_condition="clean")

    assert isinstance(manifest, FormulaRecoveryManifest)
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.params_for(
        task_id="drsr_s520_clean_g0001",
        condition="clean",
        result_sha256=result_sha,
        equation=equation,
    ) == [2.0]

    with pytest.raises(PerformanceReplayError, match="condition"):
        manifest.params_for(
            task_id="drsr_s520_clean_g0001",
            condition="noise001",
            result_sha256=result_sha,
            equation=equation,
        )
    with pytest.raises(PerformanceReplayError, match="result SHA"):
        manifest.params_for(
            task_id="drsr_s520_clean_g0001",
            condition="clean",
            result_sha256="b" * 64,
            equation=equation,
        )
    with pytest.raises(PerformanceReplayError, match="equation SHA"):
        manifest.params_for(
            task_id="drsr_s520_clean_g0001",
            condition="clean",
            result_sha256=result_sha,
            equation=equation + "# drift",
        )


def test_formula_recovery_manifest_rejects_unavailable_and_missing_entries(
    tmp_path: Path,
) -> None:
    equation = "def f(x, params):\n    return params[0] * x\n"
    result_sha = "a" * 64
    path = _write_recovery_manifest(
        tmp_path / "formula_recovery.v1.json",
        condition="noise005",
        task_id="drsr_s521_noise005_g0007",
        result_sha256=result_sha,
        equation=equation,
        resolution="unavailable",
    )
    manifest = load_formula_recovery_manifest(path, expected_condition="noise005")

    with pytest.raises(PerformanceReplayError, match="unavailable"):
        manifest.params_for(
            task_id="drsr_s521_noise005_g0007",
            condition="noise005",
            result_sha256=result_sha,
            equation=equation,
        )
    with pytest.raises(PerformanceReplayError, match="缺少恢复记录"):
        manifest.require_params_for(
            task_id="drsr_s520_noise005_g0006",
            condition="noise005",
            result_sha256=result_sha,
            equation=equation,
        )


@pytest.mark.parametrize("algorithm", ["QLattice", "iMCTS"])
def test_replay_forces_raw_equation_rebuild_for_audited_tools(
    tmp_path: Path,
    algorithm: str,
) -> None:
    _write_dataset(tmp_path)
    payload = _qlattice_payload(record_type="final_best")
    payload["canonical_artifact"] = {
        "tool_name": algorithm,
        "raw_equation": "x0",
        "normalized_expression": "x0",
        "instantiated_expression": "x0",
    }
    replay = replay_payload_performance(payload, algorithm=algorithm, repo_root=tmp_path)
    assert replay["artifact_rebuilt"] is True
    assert replay["canonical_artifact"]["raw_equation"] == "2*x1"


def test_replay_drsr_prefers_artifact_params_then_recovery_params(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path)
    equation = "def f(a, b, params):\n    return params[0] * b\n"
    raw_result = "frozen-result"
    result_sha = hashlib.sha256(raw_result.encode()).hexdigest()
    manifest_path = _write_recovery_manifest(
        tmp_path / "formula_recovery.v1.json",
        condition="clean",
        task_id="drsr_s520_clean_g0001",
        result_sha256=result_sha,
        equation=equation,
    )
    manifest = load_formula_recovery_manifest(manifest_path, expected_condition="clean")
    payload = {
        "dataset_dir": str(dataset_dir),
        "equation": equation,
        "canonical_artifact": {"tool_name": "drsr", "parameter_values": [3.0]},
    }
    replay = replay_payload_performance(
        payload,
        algorithm="drsr",
        repo_root=tmp_path,
        recovery_manifest=manifest,
        task_id="drsr_s520_clean_g0001",
        condition="clean",
        result_sha256=result_sha,
    )
    assert replay["artifact_rebuilt"] is True
    assert replay["canonical_artifact"]["parameter_values"] == [3.0]

    payload["canonical_artifact"] = {"tool_name": "drsr"}
    replay = replay_payload_performance(
        payload,
        algorithm="drsr",
        repo_root=tmp_path,
        recovery_manifest=manifest,
        task_id="drsr_s520_clean_g0001",
        condition="clean",
        result_sha256=result_sha,
    )
    assert replay["canonical_artifact"]["parameter_values"] == [2.0]


def test_real_formula_recovery_manifests_load_and_noise005_keeps_unavailable() -> None:
    stage_root = Path(__file__).resolve().parents[1]
    manifests = {
        "clean": stage_root / "manifests/formula_recovery.v1.json",
        "noise001": stage_root / "manifests/noise001_formula_recovery.v1.json",
        "noise005": stage_root / "manifests/noise005_formula_recovery.v1.json",
    }
    loaded = {
        condition: load_formula_recovery_manifest(path, expected_condition=condition)
        for condition, path in manifests.items()
    }
    assert all(item.sha256 == hashlib.sha256(item.path.read_bytes()).hexdigest() for item in loaded.values())
    unavailable = loaded["noise005"].entries["drsr_s521_noise005_g0007"]
    assert unavailable["resolution"] == "unavailable"
    assert "params" not in unavailable


def test_replay_drsr_missing_params_is_evidence_error(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path)
    payload = {
        "dataset_dir": str(dataset_dir),
        "equation": "def f(a, b, params):\n    return params[0] * b\n",
        "canonical_artifact": None,
    }
    with pytest.raises(PerformanceReplayError, match="恢复 manifest"):
        replay_payload_performance(
            payload,
            algorithm="drsr",
            repo_root=tmp_path,
            task_id="drsr_s520_clean_g0001",
            condition="clean",
            result_sha256="a" * 64,
        )
