"""Focused checks for conservative terminal-expression matching."""

import hashlib

import pytest

from .audit_current_terminal_symbolic_bindings import (
    expression_fingerprint,
    verify_response_artifact,
)


def test_indexed_variables_and_numpy_qualifier_match_named_expression():
    terminal = "x0 + np.sin(x1)"
    historical = "t + sin(P)"
    assert expression_fingerprint(terminal, ["t", "P"], map_indexed=True) == (
        expression_fingerprint(historical, ["t", "P"], map_indexed=False)
    )


def test_changed_operator_does_not_match():
    assert expression_fingerprint("x0 + x1", ["t", "P"], map_indexed=True) != (
        expression_fingerprint("t * P", ["t", "P"], map_indexed=False)
    )


def test_out_of_range_variable_fails_closed():
    with pytest.raises(ValueError, match="exceeds"):
        expression_fingerprint("x2 + 1", ["t", "P"], map_indexed=True)


def test_reuse_requires_existing_response_with_matching_sha(tmp_path):
    path = tmp_path / "response.json"
    path.write_bytes(b'{"ok":true}')
    row = {
        "response_path": "response.json",
        "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert verify_response_artifact(row, tmp_path)
    row["response_sha256"] = "0" * 64
    assert not verify_response_artifact(row, tmp_path)
