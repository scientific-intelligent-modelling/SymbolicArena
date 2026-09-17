"""A versioned retry completes a base plan without rewriting its ledger."""

from check.finalize_core50_current_gplearn_release import validate_phase_summaries


def test_base_and_retry_form_one_complete_phase() -> None:
    base = {"plan_tasks": 450, "frozen": 447, "remaining": 3,
            "physical_attempts": 456, "estimated_cost_cny_assumed_tariff": 19.79}
    retry = {"plan_tasks": 3, "frozen": 3, "remaining": 0,
             "physical_attempts": 3, "estimated_cost_cny_assumed_tariff": 0.04}
    result = validate_phase_summaries(base, retry)
    assert result["logical_tasks_complete"] == 450
    assert result["physical_attempts"] == 459
