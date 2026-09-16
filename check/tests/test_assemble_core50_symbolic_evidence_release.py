"""Focused checks for selective provenance packaging."""

from __future__ import annotations

import gzip
import json

import pytest

from check.assemble_core50_symbolic_evidence_release import _gzip_plan, _source_path


def test_gzip_plan_keeps_only_requested_evaluation_keys(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    records = [{"evaluation_key": "a", "request": {"x": 1}},
               {"evaluation_key": "b", "request": {"x": 2}}]
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    packed = tmp_path / "selected.jsonl.gz"
    report = _gzip_plan(source, packed, {"b"})
    assert report["row_count"] == 1
    with gzip.open(packed, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == [records[1]]


def test_response_path_cannot_escape_repository() -> None:
    with pytest.raises(ValueError, match="outside repository"):
        _source_path("/etc/passwd")
