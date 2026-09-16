"""Offline regression tests for the conservative Opus 4.8 retry handoff."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from check import prepare_opus48_failed_simplification_retry as retry
from check.run_opus48_terminal_plan import load_plan


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp("opus48_retry") / "plan"
    manifest = retry.prepare_retry(
        source_plan=retry.DEFAULT_PLAN,
        unresolved=retry.DEFAULT_RUN / "unresolved.json",
        attempts_dir=retry.DEFAULT_RUN / "attempts",
        terminal_inputs=retry.DEFAULT_TERMINALS,
        prompt_path=retry.DEFAULT_PROMPT,
        output_dir=output,
    )
    return output, manifest


def test_all_70_failed_keys_are_replaced_and_terminal_hashes_match(prepared) -> None:
    output, manifest = prepared
    rows, _ = load_plan(output / "pred_simplify_retry_plan.jsonl")
    audit = list(csv.DictReader((output / "retry_audit.csv").open(encoding="utf-8", newline="")))
    assert manifest["retry_count"] == len(rows) == 70
    assert len(audit) == 70
    assert manifest["failure_categories"] == {
        "semantic_rejected": 63, "max_tokens": 6, "semantic_timeout": 1,
    }
    old = json.loads((retry.DEFAULT_RUN / "unresolved.json").read_text(encoding="utf-8"))["items"]
    assert {row["retry_parent_opus48_key"] for row in rows} == {
        item["evaluation_key"] for item in old
    }
    assert len({row["opus48_evaluation_key"] for row in rows}) == 70
    assert all(row["opus48_evaluation_key"] != row["retry_parent_opus48_key"] for row in rows)
    by_logical_id = {item["logical_id"]: item for item in audit}
    assert all(row["request"]["terminal_binding_evidence"]["terminal_expression_sha256"]
               == by_logical_id[row["logical_id"]]["terminal_expression_sha256"]
               for row in rows)


def test_successor_keeps_original_schema_and_semantic_contract(prepared) -> None:
    output, _ = prepared
    source_rows, _ = load_plan(retry.DEFAULT_PLAN)
    source = {row["logical_id"]: row for row in source_rows}
    rows, _ = load_plan(output / "pred_simplify_retry_plan.jsonl")
    for row in rows:
        previous = source[row["logical_id"]]
        assert row["schema_sha256"] == previous["schema_sha256"]
        assert row["schema_content"] == previous["schema_content"]
        assert row["request"]["expression"] == previous["request"]["expression"]
        assert row["request"]["ast_source_evidence"] == previous["request"]["ast_source_evidence"]
        assert row["request"]["terminal_binding_evidence"] == previous["request"]["terminal_binding_evidence"]
        assert row["prompt_version"] == "simplify_retry_conservative.v2"
        assert row["input_hash"] != previous["input_hash"]


def test_source_formula_sha_mismatch_is_rejected() -> None:
    row = load_plan(retry.DEFAULT_PLAN)[0][0]
    request = row["request"]
    key = "{}::{}::s{}::{}".format(
        request["algorithm_slug"], request["dataset_id"], request["seed"], request["noise_tag"]
    )
    terminal = dict(retry.load_terminals(retry.DEFAULT_TERMINALS)[key])
    terminal["terminal_expression_sha256"] = retry.sha256_text("wrong_formula")
    with pytest.raises(retry.RetryPlanError, match="terminal expression hash"):
        retry.validate_source_binding(row, terminal)


def test_missing_unresolved_row_is_not_silently_skipped(tmp_path: Path) -> None:
    payload = json.loads((retry.DEFAULT_RUN / "unresolved.json").read_text(encoding="utf-8"))
    payload["items"].pop()
    altered = tmp_path / "unresolved.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(retry.RetryPlanError, match="unresolved count drift"):
        retry.prepare_retry(
            source_plan=retry.DEFAULT_PLAN, unresolved=altered,
            attempts_dir=retry.DEFAULT_RUN / "attempts",
            terminal_inputs=retry.DEFAULT_TERMINALS, prompt_path=retry.DEFAULT_PROMPT,
            output_dir=tmp_path / "must_not_exist",
        )
    assert not (tmp_path / "must_not_exist").exists()
