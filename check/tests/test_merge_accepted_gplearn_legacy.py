"""Current numerical axes must not be replaced by historical gplearn values."""

from __future__ import annotations

from check.merge_accepted_gplearn_legacy import merge_row


def test_legacy_exception_supplies_only_symbolic_axes() -> None:
    current = {"condition": "clean", "algorithm": "gplearn", "ID": "30.7",
               "OOD": "24.8", "EFF": "95.5", "SYM": "", "MIN": "", "STAB": "",
               "formal_ready": "False"}
    old = {"condition": "clean", "algorithm": "gplearn", "ID": "30.2",
           "OOD": "24.3", "EFF": "95.5", "SYM": "26.3", "MIN": "62.8",
           "STAB": "23.6", "run_count": "150", "task_count": "50"}
    result = merge_row(current, old, old_sha="a" * 64)
    assert (result["ID"], result["OOD"], result["EFF"]) == ("30.7", "24.8", "95.5")
    assert (result["SYM"], result["MIN"], result["STAB"]) == ("26.3", "62.8", "23.6")
    assert result["formal_ready"] == "False"
    assert result["score_complete"] == "True"
    assert result["symbolic_source"] == "accepted_legacy_gplearn_20260914"
