"""Focused checks for SymbolFit terminal identity validation."""

from .prepare_symbolfit_clean_terminal_refresh import _terminal_identity_matches


def test_terminal_index_can_be_omitted_when_source_path_is_bound():
    assert _terminal_identity_matches({"seed": 520}, seed=520, index=1)


def test_explicit_terminal_index_must_match():
    assert not _terminal_identity_matches(
        {"seed": 520, "task_global_index": 2}, seed=520, index=1
    )
