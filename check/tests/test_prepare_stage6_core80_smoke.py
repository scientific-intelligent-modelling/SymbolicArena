"""Stage6 dispatch must be restricted to the additional thirty tasks."""

import pytest

from check.prepare_stage6_core80_smoke import new30_rows


def _rows(count: int) -> list[dict[str, str]]:
    return [{"dataset_id": f"g{i:04d}", "dataset_rel": f"dataset/{i}"}
            for i in range(1, count + 1)]


def test_new30_is_exact_core80_difference() -> None:
    result = new30_rows(_rows(50), _rows(80))
    assert len(result) == 30
    assert {row["dataset_id"] for row in result} == {
        f"g{i:04d}" for i in range(51, 81)
    }


def test_new30_rejects_a_different_core50() -> None:
    old = _rows(50)
    old[0]["dataset_rel"] = "unrelated/dataset"
    with pytest.raises(ValueError, match="does not contain"):
        new30_rows(old, _rows(80))
