"""A fast FePySR run still has a native final incumbent before minute one."""

import json

from scientific_intelligent_modelling.benchmarks import runner
from scientific_intelligent_modelling.benchmarks.runner import (
    _write_final_progress_payload_if_requested,
)


def test_fast_fepysr_final_snapshot_retains_native_score(tmp_path) -> None:
    native = tmp_path / "native"
    native.mkdir()
    (native / ".fepysr_current_best.json").write_text(
        json.dumps({"equation": "1.5", "score": 0.2, "source": "mean_constant_baseline"}),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    result = {
        "status": "ok", "tool": "fepysr", "equation": "1.5",
        "canonical_artifact": {"raw_equation": "1.5"}, "seconds": 14.6,
    }

    _write_final_progress_payload_if_requested(
        result=result, progress_snapshot_interval_seconds=60,
        output_dir=output, experiment_dir=native, tool_name="fepysr",
    )

    minute = json.loads((output / "progress/minute_0000.json").read_text(encoding="utf-8"))
    assert minute["record_type"] == "budget_end_internal_best"
    assert minute["candidate_available"] is True
    assert minute["algorithm_native_incumbent"] is True
    assert minute["source_score"] == 0.2
    assert minute["internal_objective_direction"] == "min"


def test_early_final_native_read_receives_minute_zero(monkeypatch, tmp_path) -> None:
    observed = []

    def build_native(**kwargs):
        observed.append(kwargs["checkpoint_index"])
        return {"equation": "1.5", "elapsed_seconds": 14.6}

    monkeypatch.setattr(runner, "_build_periodic_snapshot_payload", build_native)
    _write_final_progress_payload_if_requested(
        result={"status": "ok", "tool": "fepysr", "equation": "1.5",
                "canonical_artifact": {"raw_equation": "1.5"}, "seconds": 14.6},
        progress_snapshot_interval_seconds=60, output_dir=tmp_path / "run",
        experiment_dir=tmp_path / "native", tool_name="fepysr",
        dataset=object(), params={}, seed=520, started_at=0.0,
    )

    assert observed == [0]
