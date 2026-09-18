"""LLM-SR snapshots must retain the native metric used to choose a candidate."""

import json

from scientific_intelligent_modelling.benchmarks.runner import (
    _extract_llmsr_periodic_candidate,
)


def test_llmsr_nmse_is_exposed_as_native_minimization_loss(tmp_path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    source = "def equation(x0, params):\n    return x0\n"
    for index, nmse in ((1, 0.4), (2, 0.2)):
        (samples / f"top01_samples_{index}.json").write_text(
            json.dumps({"function": source, "nmse": nmse, "mse": 10 * nmse}),
            encoding="utf-8",
        )

    candidate = _extract_llmsr_periodic_candidate(tmp_path)

    assert candidate is not None
    assert candidate["nmse"] == 0.2
    assert candidate["loss"] == 0.2
    assert candidate["internal_loss"] == 0.2
    assert candidate["internal_objective"] == "native_nmse"
    assert candidate["objective_direction"] == "min"


def test_llmsr_score_fallback_preserves_maximization_direction(tmp_path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "top01_samples_1.json").write_text(
        json.dumps({"function": "def equation(x0, params):\n    return x0\n", "score": 0.8}),
        encoding="utf-8",
    )

    candidate = _extract_llmsr_periodic_candidate(tmp_path)

    assert candidate is not None
    assert candidate["score"] == 0.8
    assert candidate["internal_objective"] == "native_score"
    assert candidate["objective_direction"] == "max"
    assert "loss" not in candidate
