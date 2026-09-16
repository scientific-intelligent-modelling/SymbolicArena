"""Prepare a fail-closed, offline Opus 4.8 retry plan for unresolved terminal formulas.

This command never sends requests. The source plan and failed attempt files are
read-only; each successor binds the same terminal snapshot to a new prompt key.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import load_plan_jsonl
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec
from check.run_opus48_terminal_plan import KEY_MODEL, EFFORT, load_plan, sha256_text


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
DEFAULT_PLAN = COLLECTION / "opus_prediction_plan_v2/pred_simplify_plan.jsonl"
DEFAULT_RUN = COLLECTION / "opus48_prediction_run_20260916"
DEFAULT_TERMINALS = COLLECTION / "terminal_inputs.jsonl.gz"
DEFAULT_PROMPT = STAGE5 / "config/prompts/simplify_retry_conservative.v2.txt"
FAILURE_CATEGORIES = {"semantic_rejected", "max_tokens", "semantic_timeout"}


class RetryPlanError(ValueError):
    """A source or successor binding is incomplete or inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_terminals(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = "{}::{}::s{}::{}".format(
                row.get("algorithm_slug"), row.get("dataset_id"),
                row.get("seed"), row.get("condition"),
            )
            if not isinstance(row.get("logical_key"), str) or key in rows:
                raise RetryPlanError(f"terminal_inputs line {number}: missing/duplicate identity")
            rows[key] = row
    return rows


def failure_category(attempt: Mapping[str, Any]) -> str:
    semantic = attempt.get("semantic_validation") or {}
    if semantic.get("status") == "semantic_rejected":
        return "semantic_rejected"
    if semantic.get("status") == "semantic_validator_timeout":
        return "semantic_timeout"
    response = attempt.get("response") or {}
    if response.get("stop_reason") == "max_tokens":
        return "max_tokens"
    raise RetryPlanError(f"unclassified final failure: {attempt.get('attempt_id')}")


def _limited_string(value: object, maximum: int = 240) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:maximum] + ("..." if len(value) > maximum else "")


def retry_context(attempt: Mapping[str, Any], category: str) -> dict[str, Any]:
    context: dict[str, Any] = {"previous_failure_category": category}
    if category == "semantic_rejected":
        output = attempt.get("structured_output") or {}
        candidate = output.get("simplified_expression")
        if isinstance(candidate, str):
            context["rejected_candidate_sha256"] = sha256_text(candidate)
            context["rejected_candidate_excerpt"] = _limited_string(candidate)
        semantic = attempt.get("semantic_validation") or {}
        context["validator_reason"] = _limited_string(semantic.get("error"))
        evidence = semantic.get("semantic_evidence") or {}
        context["validator_decision"] = evidence.get("decision")
        counterexample = evidence.get("counterexample")
        if counterexample is not None:
            context["counterexample_excerpt"] = _limited_string(canonical_json(counterexample), 360)
    elif category == "max_tokens":
        context["retry_guidance"] = "Use a brief conservative answer; do not increase token budget automatically."
    else:
        context["retry_guidance"] = "Prior local validator timed out; use unchanged if no exact reduction is obvious."
    return context


def validate_source_binding(row: Mapping[str, Any], terminal: Mapping[str, Any]) -> None:
    request = row["request"]
    binding = request.get("terminal_binding_evidence") or {}
    ast = request.get("ast_source_evidence") or {}
    expected_key = "{}::{}::s{}::{}".format(
        request["algorithm_slug"], request["dataset_id"], request["seed"], request["noise_tag"]
    )
    terminal_identity = "{}::{}::s{}::{}".format(
        terminal.get("algorithm_slug"), terminal.get("dataset_id"),
        terminal.get("seed"), terminal.get("condition"),
    )
    if terminal_identity != expected_key:
        raise RetryPlanError(f"terminal identity drift: {row['logical_id']}")
    terminal_expression = terminal.get("terminal_expression")
    if not isinstance(terminal_expression, str) or not terminal_expression:
        raise RetryPlanError(f"terminal expression missing: {row['logical_id']}")
    checks = {
        "terminal expression hash": (
            sha256_text(terminal_expression), terminal.get("terminal_expression_sha256")),
        "plan terminal expression hash": (
            terminal.get("terminal_expression_sha256"), binding.get("terminal_expression_sha256")),
        "snapshot hash": (
            terminal.get("terminal_source_sha256"), row.get("terminal_snapshot_raw_sha256")),
        "binding snapshot hash": (
            terminal.get("terminal_source_sha256"), binding.get("terminal_snapshot_sha256")),
        "selected result hash": (
            terminal.get("selected_result_sha256"), binding.get("selected_result_sha256")),
        "original terminal formula": (
            terminal_expression, ast.get("selected_expression_before_variable_mapping")),
        "mapped evaluation formula": (
            request.get("expression"), ast.get("semantic_expression_after_variable_mapping")),
    }
    for name, (left, right) in checks.items():
        if not left or left != right:
            raise RetryPlanError(f"{row['logical_id']}: {name} mismatch")


def prepare_retry(
    *, source_plan: Path, unresolved: Path, attempts_dir: Path,
    terminal_inputs: Path, prompt_path: Path, output_dir: Path,
    expected_source_count: int = 2336, expected_unresolved_count: int = 70,
) -> dict[str, Any]:
    rows, source_sha = load_plan(source_plan)
    if len(rows) != expected_source_count:
        raise RetryPlanError(f"source plan count {len(rows)} != {expected_source_count}")
    by_old_key = {row["opus48_evaluation_key"]: row for row in rows}
    terminals = load_terminals(terminal_inputs)
    unresolved_items = json.loads(unresolved.read_text(encoding="utf-8")).get("items")
    if not isinstance(unresolved_items, list) or len(unresolved_items) != expected_unresolved_count:
        raise RetryPlanError("unresolved count drift")
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha = file_sha256(prompt_path)
    audit: list[dict[str, Any]] = []
    successors: list[dict[str, Any]] = []
    seen_old: set[str] = set()
    seen_new: set[str] = set()

    for item in unresolved_items:
        old_key = item["evaluation_key"]
        if old_key in seen_old or old_key not in by_old_key:
            raise RetryPlanError(f"duplicate/unknown unresolved key: {old_key}")
        seen_old.add(old_key)
        source = by_old_key[old_key]
        if item.get("logical_id") != source["logical_id"] or item.get("source_evaluation_key") != source["evaluation_key"]:
            raise RetryPlanError(f"unresolved identity mismatch: {old_key}")
        attempts = sorted(attempts_dir.glob(f"{old_key}.a*.json"))
        if len(attempts) != item.get("attempts") or len(attempts) != 2:
            raise RetryPlanError(f"attempt count mismatch: {old_key}")
        attempt_records = [json.loads(path.read_text(encoding="utf-8")) for path in attempts]
        for attempt, attempt_path in zip(attempt_records, attempts):
            if (attempt.get("evaluation_key") != old_key
                    or attempt.get("source_evaluation_key") != source["evaluation_key"]
                    or attempt.get("logical_id") != source["logical_id"]
                    or attempt.get("source_input_hash") != source["input_hash"]
                    or attempt.get("model") != "claude-opus-4-8"
                    or attempt.get("terminal_expression") != source["request"]["expression"]
                    or attempt.get("attempt_id") != attempt_path.stem):
                raise RetryPlanError(f"attempt binding mismatch: {attempt_path}")
            if attempt.get("error") is None:
                raise RetryPlanError(f"unresolved task has successful attempt: {attempt_path}")
        final_attempt = attempt_records[-1]
        category = failure_category(final_attempt)
        if category not in FAILURE_CATEGORIES:
            raise RetryPlanError(f"unrecognized failure: {old_key}")
        request = dict(source["request"])
        logical_key = "{}::{}::s{}::{}".format(
            request["algorithm_slug"], request["dataset_id"], request["seed"], request["noise_tag"]
        )
        if logical_key not in terminals:
            raise RetryPlanError(f"terminal missing: {logical_key}")
        validate_source_binding(source, terminals[logical_key])
        request["retry_context"] = retry_context(final_attempt, category)
        normalized = {"request": request, "prompt_sha256": prompt_sha,
                      "schema_sha256": source["schema_sha256"]}
        input_hash = sha256_text(canonical_json(normalized))
        prompt_version = prompt_path.stem
        new_opus5_key = evaluation_key(
            task_type="pred_simplify", logical_id=source["logical_id"],
            prompt_version=prompt_version, schema_version=source["schema_version"],
            prompt_sha256=prompt_sha, schema_sha256=source["schema_sha256"],
            normalized_input=normalized, evidence_hash=request["evidence_hash"],
        )
        new_opus48_key = evaluation_key(
            task_type="pred_simplify", logical_id=source["logical_id"],
            prompt_version=prompt_version, schema_version=source["schema_version"],
            prompt_sha256=prompt_sha, schema_sha256=source["schema_sha256"],
            normalized_input=normalized, evidence_hash=request["evidence_hash"],
            model=KEY_MODEL, effort=EFFORT,
        )
        if new_opus48_key in seen_new or new_opus48_key in {old_key, source["evaluation_key"]}:
            raise RetryPlanError(f"retry key collision: {old_key}")
        seen_new.add(new_opus48_key)
        successor = dict(source)
        successor.pop("opus48_evaluation_key", None)
        successor.update({
            "evaluation_key": new_opus5_key,
            "input_hash": input_hash,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha,
            "prompt_path": str(prompt_path.resolve()),
            "prompt_template": prompt,
            "request": request,
            "normalized_input": normalized,
            "rendered_prompt": render_prompt(prompt, request, source["schema_content"]),
            "task_spec": json.loads(TaskSpec(
                evaluation_key=new_opus5_key, logical_id=source["logical_id"],
                task_type="pred_simplify", condition=source["condition"],
                priority=int(source["priority"]), input_hash=input_hash,
                prompt_version=prompt_version, schema_version=source["schema_version"],
                dependencies=tuple(source["dependencies"]),
            ).canonical_json()),
            "retry_parent_opus48_key": old_key,
            "retry_parent_source_key": source["evaluation_key"],
            "retry_failure_category": category,
        })
        successors.append(successor)
        audit.append({
            "logical_id": source["logical_id"], "condition": source["condition"],
            "algorithm_slug": request["algorithm_slug"], "dataset_id": request["dataset_id"],
            "seed": request["seed"], "failure_category": category,
            "old_opus48_key": old_key, "new_opus48_key": new_opus48_key,
            "new_source_evaluation_key": new_opus5_key,
            "terminal_expression_sha256": terminals[logical_key]["terminal_expression_sha256"],
            "terminal_snapshot_sha256": terminals[logical_key]["terminal_source_sha256"],
            "attempt_1_sha256": file_sha256(attempts[0]),
            "attempt_2_sha256": file_sha256(attempts[1]),
        })

    if len(seen_old) != expected_unresolved_count:
        raise RetryPlanError("unresolved rows were skipped")
    counts = Counter(item["failure_category"] for item in audit)
    successors.sort(key=lambda row: (row["priority"], row["logical_id"]))
    audit.sort(key=lambda row: row["logical_id"])
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_out = output_dir / "pred_simplify_retry_plan.jsonl"
    plan_out.write_text("".join(canonical_json(row) + "\n" for row in successors), encoding="utf-8")
    loaded = load_plan_jsonl(plan_out)
    if len(loaded.entries) != expected_unresolved_count:
        raise RetryPlanError("successor plan failed contract round-trip")
    rerun_rows, _ = load_plan(plan_out)
    if {row["opus48_evaluation_key"] for row in rerun_rows} != seen_new:
        raise RetryPlanError("Opus 4.8 successor key round-trip failed")
    audit_out = output_dir / "retry_audit.csv"
    with audit_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    manifest = {
        "status": "plan_only_no_api", "model": KEY_MODEL, "effort": EFFORT,
        "prompt_version": prompt_path.stem, "prompt_sha256": prompt_sha,
        "source_plan_sha256": source_sha, "unresolved_sha256": file_sha256(unresolved),
        "terminal_inputs_sha256": file_sha256(terminal_inputs),
        "source_count": len(rows), "retry_count": len(successors),
        "failure_categories": dict(sorted(counts.items())),
        "plan_sha256": file_sha256(plan_out), "audit_sha256": file_sha256(audit_out),
        "max_tokens_handling": "Six cases receive the conservative prompt; do not automatically increase 16k output tokens.",
        "semantic_timeout_handling": "Inspect the one validator timeout separately; do not automatically increase its 2 GiB worker limit.",
        "unresolved_policy": "No row may be omitted; failed semantic validation remains unresolved, never silently scored zero.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--unresolved", type=Path, default=DEFAULT_RUN / "unresolved.json")
    parser.add_argument("--attempts-dir", type=Path, default=DEFAULT_RUN / "attempts")
    parser.add_argument("--terminal-inputs", type=Path, default=DEFAULT_TERMINALS)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-count", type=int, default=2336)
    parser.add_argument("--expected-unresolved-count", type=int, default=70)
    args = parser.parse_args()
    manifest = prepare_retry(
        source_plan=args.source_plan, unresolved=args.unresolved,
        attempts_dir=args.attempts_dir, terminal_inputs=args.terminal_inputs,
        prompt_path=args.prompt, output_dir=args.output_dir,
        expected_source_count=args.expected_source_count,
        expected_unresolved_count=args.expected_unresolved_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
