from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor
from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor


def test_fepysr_uses_guarded_internal_pysr_timeout() -> None:
    reg = FePySRRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_symbolfit_uses_guarded_internal_pysr_timeout() -> None:
    reg = SymbolFitRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_explicit_timeout_guard_overrides_default_guard() -> None:
    reg = SymbolFitRegressor(timeout_in_seconds=3600, timeout_guard_seconds=120)

    assert reg.params["timeout_in_seconds"] == 3480
