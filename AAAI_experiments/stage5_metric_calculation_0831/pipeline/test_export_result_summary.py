from __future__ import annotations

import pytest

from .export_result_summary import (
    ResultSummaryError,
    merge_clean_eff_rows,
    postprocessed_best_so_far,
)


def test_merge_clean_eff_rows_prefers_current_and_tracks_fallback() -> None:
    legacy = [
        {"logical_key": f"alg::d{i}::s520::clean", "value": "legacy"}
        for i in range(2250)
    ]
    current = [dict(legacy[0], value="current")]

    merged, fallback = merge_clean_eff_rows(current, legacy)

    assert merged[legacy[0]["logical_key"]]["value"] == "current"
    assert len(merged) == 2250
    assert len(fallback) == 2249


def test_merge_clean_eff_rows_rejects_key_outside_frozen_grid() -> None:
    legacy = [
        {"logical_key": f"alg::d{i}::s520::clean"}
        for i in range(2250)
    ]

    with pytest.raises(ResultSummaryError, match="基底外"):
        merge_clean_eff_rows(
            [{"logical_key": "alg::outside::s520::clean"}], legacy
        )


def test_postprocessed_best_so_far_is_monotone_and_holds_missing_minutes() -> None:
    record = {
        "snapshots": [
            {"minute": 1, "id_nmse": 1.0, "ood_nmse": 1.0},
            {"minute": 2, "id_nmse": None, "ood_nmse": None},
            {"minute": 3, "id_nmse": 1e-6, "ood_nmse": 1e-6},
            {"minute": 4, "id_nmse": 1e-3, "ood_nmse": 1e-3},
        ]
    }

    ids, oods, quality = postprocessed_best_so_far(record)

    assert len(ids) == len(oods) == len(quality) == 180
    assert quality[1] == quality[0]
    assert quality[2] > quality[1]
    assert quality[3] == quality[2]
    assert all(left <= right for left, right in zip(quality, quality[1:]))
