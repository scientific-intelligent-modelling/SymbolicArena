"""Evidence-derived fields must not silently become historical labels."""

import pytest

from check.reconstruct_core50_selector_metadata import (
    declared_dummy_flag,
    ood_range_layout,
)


def test_ood_layout_describes_range_format_only() -> None:
    single = {"ood_range": [2.0, 3.0]}
    multiple = {"ood_range": [[0.0, 1.0], [3.0, 4.0]]}
    assert ood_range_layout([single]) == "single_interval"
    assert ood_range_layout([multiple]) == "multiple_intervals"
    assert ood_range_layout([single, multiple]) == "mixed_intervals"


def test_reversed_ood_bounds_remain_unresolved() -> None:
    assert ood_range_layout([{"ood_range": [3.0, 2.0]}]) == "invalid_bounds"
    assert ood_range_layout([{"ood_range": [[0.0, 1.0], [4.0, 2.0]]}]) == "invalid_bounds"


def test_malformed_ood_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported ood_range"):
        ood_range_layout([{"ood_range": "unknown"}])


def test_srsd_dummy_flag_requires_consistent_description() -> None:
    dummy = [{"description": "meaningless"}]
    meaningful = [{"description": "mass"}]
    assert declared_dummy_flag("srsd", "hard dummy", dummy)[0] == 1
    assert declared_dummy_flag("srsd", "hard non-dummy", meaningful)[0] == 0
    with pytest.raises(ValueError, match="conflicts"):
        declared_dummy_flag("srsd", "hard dummy", meaningful)
    assert declared_dummy_flag("nguyen", "nguyen", meaningful)[0] == 0
