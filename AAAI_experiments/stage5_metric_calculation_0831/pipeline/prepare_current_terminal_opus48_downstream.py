"""Build model-scoped downstream plans for the selected Core-50 terminal formulas.

This module never calls an API. It keeps historical judgments as reviewable reuse
candidates, rather than silently promoting them into the new model's cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
from typing import Any, Iterable, Mapping

from sympy.core.cache import clear_cache

from . import symbolic_task_builder as builder
from .audit_current_terminal_symbolic_bindings import verify_response_artifact


CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = (520, 521, 522)
PAIRS = ((520, 521), (520, 522), (521, 522))
MODEL = "claude-opus-4-8"
BASE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COLLECTION = BASE / "work/core50_terminal_collection_20260916_v3"
DEFAULT_RELEASE = REPO_ROOT / "AAAI_experiments/Core50_final_20260914/results"
DEFAULT_STAGE = BASE / "work/final_release_20260913/release_v2"


class DownstreamPlanError(ValueError):
    pass


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


class _JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.partial = path.with_name(path.name + ".partial")
        self.handle = self.partial.open("w", encoding="utf-8")
        self.count = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % 100 == 0:
            self.handle.flush()

    append = add

    def __len__(self) -> int:
        return self.count

    def finish(self) -> None:
        self.handle.close()
        self.partial.replace(self.path)


def _pair_worker(connection, arguments: dict[str, Any], memory_limit_bytes: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        evidence = builder._build_full_pair_evidence(**arguments)
        connection.send(("ok", evidence))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (OSError, MemoryError):
            pass
    finally:
        connection.close()


def _pair_evidence_isolated(*, timeout_seconds: float = 120.0,
                            memory_limit_bytes: int = 2 * 1024**3, **arguments: Any) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_pair_worker, args=(sender, arguments, memory_limit_bytes))
    child.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            raise DownstreamPlanError(f"deterministic_pair_evidence_timeout_after_{timeout_seconds:g}s")
        try:
            status, payload = receiver.recv()
        except EOFError as exc:
            raise DownstreamPlanError("deterministic_pair_evidence_worker_exited_without_result") from exc
        if status != "ok":
            raise DownstreamPlanError(f"deterministic_pair_evidence_worker_error:{payload}")
        return payload
    finally:
        receiver.close()
        if child.is_alive():
            child.terminate()
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def _run_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["condition"]), str(row["algorithm_slug"]).replace("-", "").replace("_", "").lower(),
            str(row["dataset_id"]), int(row["seed"]))


def _historical_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    request = row["request"]
    return (str(row["condition"]), str(request["algorithm_slug"]).replace("-", "").replace("_", "").lower(),
            str(request["dataset_id"]), int(request["seed"]))


def _unique_index(rows: Iterable[dict[str, Any]], key_fn) -> dict[Any, dict[str, Any]]:
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise DownstreamPlanError(f"duplicate key: {key}")
        result[key] = row
    return result


@dataclass(frozen=True)
class Prediction:
    current: dict[str, Any]
    plan: builder.SimplifyPlanRecord
    frozen: builder.FrozenSimplifyRecord
    source: str
    frozen_path: str
    prompt_version: str
    schema_version: str
    prompt_sha256: str
    schema_sha256: str


def _plan_record(row: Mapping[str, Any]) -> builder.SimplifyPlanRecord:
    return builder.SimplifyPlanRecord(
        logical_id=str(row["logical_id"]), task_type=str(row["task_type"]),
        evaluation_key=str(row["evaluation_key"]), priority=int(row["priority"]),
        request=dict(row["request"]),
    )


def _frozen_record(*, plan: Mapping[str, Any], outcome: Mapping[str, Any], plan_sha: str,
                   result_sha: str, result_key: str) -> builder.FrozenSimplifyRecord:
    state = str(outcome["outcome"])
    if state in {"simplified", "unchanged"}:
        effective = outcome.get("simplified_expression")
        resolution = builder.LLM_SIMPLIFIED_EXPRESSION
    elif state == "unable":
        effective = plan["request"].get("original_expression")
        resolution = builder.ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE
    else:
        raise DownstreamPlanError(f"invalid simplify outcome: {state}")
    row = builder.FrozenSimplifyRecord(
        evaluation_key=result_key, logical_id=str(plan["logical_id"]),
        task_type=str(plan["task_type"]), plan_sha256=plan_sha,
        state="frozen", simplified_status=state,
        simplified_expression=outcome.get("simplified_expression"),
        effective_expression=effective, expression_resolution=resolution,
        structured_output=dict(outcome), non_applicable=None, result_sha256=result_sha,
    )
    builder._validate_effective_expression_binding(
        row={"effective_expression": effective, "expression_resolution": resolution},
        plan_request=plan["request"], logical_id=str(plan["logical_id"]),
        simplified_status=state, simplified_expression=outcome.get("simplified_expression"),
    )
    return row


def _frozen_binding_issues(raw: Mapping[str, Any], plan: Mapping[str, Any],
                           current: Mapping[str, Any]) -> list[str]:
    binding = raw.get("terminal_binding_evidence") or raw.get("request", {}).get("terminal_binding_evidence") or {}
    checks = {
        "status": raw.get("status") == "frozen",
        "source_evaluation_key": raw.get("source_evaluation_key") == plan["evaluation_key"],
        "source_input_hash": raw.get("source_input_hash") == plan["input_hash"],
        "terminal_expression_sha256": binding.get("terminal_expression_sha256") == current.get("terminal_expression_sha256"),
        "terminal_snapshot_sha256": binding.get("terminal_snapshot_sha256") == current.get("terminal_source_sha256"),
        "terminal_expression": raw.get("terminal_expression") in {
            current.get("terminal_expression"), plan["request"].get("original_expression")},
        "semantic_validation": raw.get("semantic_validation", {}).get("status") == "promotable",
    }
    return [name for name, ok in checks.items() if not ok]


def _load_current(collection: Path) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], dict[Any, dict[str, str]]]:
    import gzip

    with gzip.open(collection / "terminal_inputs.jsonl.gz", "rt", encoding="utf-8") as handle:
        current = _unique_index((json.loads(line) for line in handle if line.strip()), _run_key)
    audit = _unique_index(_csv(collection / "symbolic_preflight/prediction_binding_audit.csv"),
                          lambda r: (r["condition"], r["algorithm"], r["dataset_id"], int(r["seed"])))
    if set(current) != set(audit):
        raise DownstreamPlanError("terminal/audit run grid differs")
    for key, row in current.items():
        if (row.get("terminal_expression_sha256") or "") != audit[key]["terminal_expression_sha256"]:
            raise DownstreamPlanError(f"terminal expression SHA drift: {key}")
        if row.get("terminal_source_sha256") != audit[key]["terminal_source_sha256"]:
            raise DownstreamPlanError(f"terminal source SHA drift: {key}")
    return current, audit


def _load_predictions(collection: Path, release: Path, stage: Path,
                      current: Mapping[Any, dict[str, Any]], audit: Mapping[Any, dict[str, str]]
                      ) -> tuple[dict[Any, Prediction], list[dict[str, Any]]]:
    plan_path = collection / "opus_prediction_plan_v2/pred_simplify_plan.jsonl"
    fresh_plan = _unique_index(_rows(plan_path), _historical_key)
    fresh_plan_sha = _sha_file(plan_path)
    retry_plan_path = collection / "opus48_prediction_retry_v2_plan/pred_simplify_retry_plan.jsonl"
    retry_plan = _unique_index(_rows(retry_plan_path), _historical_key) if retry_plan_path.is_file() else {}
    if not set(retry_plan).issubset(fresh_plan):
        raise DownstreamPlanError("retry plan contains runs outside the original prediction inventory")
    retry_plan_sha = _sha_file(retry_plan_path) if retry_plan else None
    fresh_plan.update(retry_plan)
    fresh_frozen: dict[str, Path] = {}
    for frozen_dir in (collection / "opus48_prediction_run_20260916/frozen",
                       collection / "opus48_prediction_retry_v2_run/frozen"):
        for frozen_path in sorted(frozen_dir.glob("*.json")):
            raw = json.loads(frozen_path.read_text(encoding="utf-8"))
            source_key = raw.get("source_evaluation_key")
            if source_key in fresh_frozen:
                raise DownstreamPlanError(f"duplicate Opus4.8 frozen source key: {source_key}")
            if frozen_path.stem != raw.get("evaluation_key"):
                raise DownstreamPlanError(f"Opus4.8 frozen filename/key mismatch: {frozen_path}")
            fresh_frozen[source_key] = frozen_path
    old_plans: dict[Any, dict[str, Any]] = {}
    old_indexes: dict[str, dict[str, Any]] = {}
    old_plan_sha: dict[str, str] = {}
    for condition in CONDITIONS:
        path = stage / f"inputs/pred_plan_full_{condition}_effective.jsonl"
        old_plan_sha[condition] = _sha_file(path)
        old_plans.update(_unique_index(_rows(path), _historical_key))
        path = release / condition / "opus5_prediction.jsonl"
        old_indexes.update(_unique_index(_rows(path), lambda r: r["evaluation_key"]))
    predictions: dict[Any, Prediction] = {}
    unresolved: list[dict[str, Any]] = []
    for key, row in current.items():
        status = audit[key]["binding_status"]
        if status in {"deferred_gplearn", "invalid_final_output"}:
            continue
        if status.startswith("candidate_reuse"):
            old = old_plans.get(key)
            source = old_indexes.get(old["evaluation_key"]) if old else None
            if (not old or not source or source.get("state") != "frozen"
                or audit[key].get("old_response_integrity_verified") != "True"):
                unresolved.append({"key": key, "reason": "historical_prediction_artifact_missing_or_invalid"})
                continue
            if old["evaluation_key"] != audit[key]["old_opus_evaluation_key"]:
                unresolved.append({"key": key, "reason": "historical_prediction_key_drift"})
                continue
            try:
                frozen = _frozen_record(plan=old, outcome=source["structured_output"],
                                        plan_sha=old_plan_sha[key[0]], result_sha=source["response_sha256"],
                                        result_key=old["evaluation_key"])
            except (KeyError, ValueError, builder.SymbolicTaskBuilderError) as exc:
                unresolved.append({"key": key, "reason": f"invalid_historical_prediction:{exc}"})
                continue
            predictions[key] = Prediction(row, _plan_record(old), frozen, status, source["response_path"],
                                          old["prompt_version"], old["schema_version"],
                                          old["prompt_sha256"], old["schema_sha256"])
            continue
        plan = fresh_plan.get(key)
        if not plan:
            unresolved.append({"key": key, "reason": "missing_current_prediction_plan"})
            continue
        path = fresh_frozen.get(plan["evaluation_key"])
        if path is None:
            unresolved.append({"key": key, "reason": "opus48_prediction_not_frozen"})
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        bad = _frozen_binding_issues(raw, plan, row)
        if bad:
            unresolved.append({"key": key, "reason": "opus48_frozen_binding_invalid:" + ",".join(bad)})
            continue
        try:
            frozen = _frozen_record(plan=plan, outcome=raw["structured_output"],
                                    plan_sha=retry_plan_sha if key in retry_plan else fresh_plan_sha,
                                    result_sha=_sha_file(path),
                                    result_key=raw["evaluation_key"])
        except (KeyError, ValueError, builder.SymbolicTaskBuilderError) as exc:
            unresolved.append({"key": key, "reason": f"invalid_opus48_prediction:{exc}"})
            continue
        provenance = ("opus48_retry_v2_frozen" if key in retry_plan else
                      "opus48_frozen" if raw.get("terminal_binding_evidence") else
                      "opus48_frozen_legacy_top_level_missing")
        predictions[key] = Prediction(row, _plan_record(plan), frozen, provenance, str(path),
                                      plan["prompt_version"], plan["schema_version"],
                                      plan["prompt_sha256"], plan["schema_sha256"])
    return predictions, unresolved


def _binding(prefix: str, pred: Prediction) -> dict[str, Any]:
    p, f, cur = pred.plan, pred.frozen, pred.current
    return {
        f"{prefix}_logical_id": p.logical_id,
        f"{prefix}_plan_evaluation_key": p.evaluation_key,
        f"{prefix}_frozen_evaluation_key": f.evaluation_key,
        f"{prefix}_frozen_plan_sha256": f.plan_sha256,
        f"{prefix}_simplify_status": f.simplified_status,
        f"{prefix}_expression_resolution": f.expression_resolution,
        f"simplified_{prefix}_expression": f.simplified_expression,
        f"effective_{prefix}_expression": f.effective_expression,
        f"{prefix}_result_sha256": builder._source_result_sha256(p.request, context=p.logical_id),
        f"{prefix}_current_selected_result_sha256": cur["selected_result_sha256"],
        f"{prefix}_terminal_expression": cur["terminal_expression"],
        f"{prefix}_terminal_expression_sha256": cur["terminal_expression_sha256"],
        f"{prefix}_terminal_source_sha256": cur["terminal_source_sha256"],
        f"{prefix}_terminal_archive_member": cur["terminal_archive_member"],
        f"{prefix}_variable_mapping": p.request.get("ast_source_evidence", {}).get("variable_mapping")
        or {f"x{i}": name for i, name in enumerate(cur["feature_names"])},
    }


def _gt_records(stage: Path) -> dict[str, tuple[builder.SimplifyPlanRecord, builder.FrozenSimplifyRecord]]:
    plan_path = stage / "inputs/gt_plan_full.jsonl"
    rows = _unique_index(_rows(plan_path), lambda r: r["request"]["dataset_id"])
    plan_sha = _sha_file(plan_path)
    indexes = _unique_index(_rows(stage / "simplify_full/gt_frozen_index.jsonl"), lambda r: r["evaluation_key"])
    result = {}
    for dataset, plan in rows.items():
        index = indexes[plan["evaluation_key"]]
        if index.get("state") != "frozen":
            raise DownstreamPlanError(f"GT not frozen: {dataset}")
        frozen = _frozen_record(plan=plan, outcome=index["structured_output"],
                                plan_sha=plan_sha, result_sha=index["result_sha256"],
                                result_key=plan["evaluation_key"])
        result[dataset] = (_plan_record(plan), frozen)
    if len(result) != 50:
        raise DownstreamPlanError(f"expected 50 GT references, found {len(result)}")
    return result


def _old_decisions(stage: Path, release: Path, phase: str) -> dict[Any, tuple[dict[str, Any], dict[str, Any]]]:
    old = {}
    suffix = "equivalence" if phase == "equivalence" else "structure"
    for condition in CONDITIONS:
        plan_path = stage / f"downstream/{condition}_{suffix}_full_plan.jsonl"
        indexes = _unique_index(_rows(release / condition / f"opus5_{suffix}.jsonl"),
                                lambda row: row["logical_id"])
        fields = ("algorithm_slug", "dataset_id", "variables", "effective_prediction_expression",
                  "effective_ground_truth_expression", "prediction_result_sha256",
                  "ground_truth_frozen_evaluation_key") if phase == "equivalence" else (
                  "algorithm_slug", "dataset_id", "effective_prediction_a_expression",
                  "effective_prediction_b_expression", "prediction_a_result_sha256", "prediction_b_result_sha256",
                  "prediction_a_valid_output", "prediction_b_valid_output")
        with plan_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                plan = json.loads(line)
                req = plan["request"]
                key = (condition, req["algorithm_slug"].replace("-", "").replace("_", "").lower(), req["dataset_id"],
                       int(req["seed"])) if phase == "equivalence" else (
                       condition, req["algorithm_slug"].replace("-", "").replace("_", "").lower(), req["dataset_id"],
                       int(req["seed_a"]), int(req["seed_b"]))
                if key in old:
                    raise DownstreamPlanError(f"duplicate old {phase}: {key}")
                index = indexes.get(plan["logical_id"])
                if index is not None:
                    old[key] = ({"evaluation_key": plan["evaluation_key"],
                                 "prompt_version": plan["prompt_version"],
                                 "schema_version": plan["schema_version"],
                                 "prompt_sha256": plan["prompt_sha256"],
                                 "schema_sha256": plan["schema_sha256"],
                                 "request": {name: req.get(name) for name in fields}}, index)
    return old


def _candidate(old: tuple[dict[str, Any], dict[str, Any]] | None, request: Mapping[str, Any],
               phase: str, contract: builder.PromptSchemaBundle | None = None) -> dict[str, Any] | None:
    if old is None:
        return None
    plan, index = old
    if contract is not None and any(plan.get(name) != getattr(contract, name) for name in (
            "prompt_version", "schema_version", "prompt_sha256", "schema_sha256")):
        return None
    old_req = plan["request"]
    fields = ("algorithm_slug", "dataset_id", "variables", "effective_prediction_expression",
              "effective_ground_truth_expression", "prediction_result_sha256",
              "ground_truth_frozen_evaluation_key") if phase == "equivalence" else (
              "algorithm_slug", "dataset_id", "effective_prediction_a_expression",
              "effective_prediction_b_expression", "prediction_a_result_sha256", "prediction_b_result_sha256",
              "prediction_a_valid_output", "prediction_b_valid_output")
    if any(old_req.get(name) != request.get(name) for name in fields):
        return None
    source_fields = ("prediction",) if phase == "equivalence" else ("prediction_a", "prediction_b")
    if any(request.get(f"{prefix}_current_selected_result_sha256") != old_req.get(f"{prefix}_result_sha256")
           for prefix in source_fields):
        return None
    if (index.get("state") != "frozen" or index.get("evaluation_key") != plan.get("evaluation_key")
        or not verify_response_artifact(index, REPO_ROOT)):
        return None
    return {"old_evaluation_key": index["evaluation_key"], "old_response_path": index["response_path"],
            "old_response_sha256": index["response_sha256"], "old_decision": index.get("effective_decision"),
            "reuse_status": "candidate_only_requires_review", "matched_fields": list(fields)}


def inventory(*, collection: Path, release: Path, stage: Path) -> tuple[dict[str, Any], dict[Any, Prediction],
                                                                    dict[Any, dict[str, Any]], list[dict[str, Any]]]:
    current, audit = _load_current(collection)
    predictions, unresolved = _load_predictions(collection, release, stage, current, audit)
    gt = _gt_records(stage)
    for pred in predictions.values():
        if pred.current["dataset_id"] not in gt:
            raise DownstreamPlanError(f"missing GT: {pred.current['dataset_id']}")
    counts = Counter(pred.source for pred in predictions.values())
    pairs = Counter()
    for condition, algorithm, dataset in {(k[0], k[1], k[2]) for k in current}:
        for a, b in PAIRS:
            ka, kb = (condition, algorithm, dataset, a), (condition, algorithm, dataset, b)
            if algorithm == "gplearn":
                pairs["deferred_gplearn"] += 1
            elif current[ka]["minute180_valid_output"] is not True or current[kb]["minute180_valid_output"] is not True:
                pairs["non_applicable_invalid_seed"] += 1
            elif ka in predictions and kb in predictions:
                pairs["ready"] += 1
            else:
                pairs["waiting_for_prediction"] += 1
    report = {"schema_version": "current_terminal_opus48_downstream.v1", "model": MODEL,
              "run_count": len(current), "prediction_ready": len(predictions),
              "prediction_sources": dict(counts), "prediction_unresolved": len(unresolved),
              "prediction_unresolved_reasons": dict(Counter(item["reason"] for item in unresolved)),
              "pair_counts": dict(pairs), "api_requests_sent": 0,
              "note": "Inventory only; ready does not imply a downstream judgment is complete."}
    return report, predictions, current, unresolved


def _active_binding_rows(current: Mapping[Any, dict[str, Any]],
                         predictions: Mapping[Any, Prediction]) -> list[dict[str, Any]]:
    rows = []
    for key, terminal in sorted(current.items()):
        pred = predictions.get(key)
        row = {
            "condition": key[0], "algorithm": key[1], "dataset_id": key[2], "seed": key[3],
            "task_id": terminal["task_id"], "terminal_expression": terminal.get("terminal_expression"),
            "terminal_expression_sha256": terminal.get("terminal_expression_sha256"),
            "terminal_source_sha256": terminal.get("terminal_source_sha256"),
            "terminal_archive_member": terminal.get("terminal_archive_member"),
            "selected_result_sha256": terminal.get("selected_result_sha256"),
            "feature_names": terminal.get("feature_names"),
            "valid_output": terminal.get("minute180_valid_output"),
            "processing_status": "ready" if pred else (
                "deferred_gplearn" if key[1] == "gplearn" else
                "invalid_final_output" if terminal.get("minute180_valid_output") is not True else
                "unresolved_prediction"),
        }
        if pred:
            row.update({
                "source": pred.source,
                "prediction_model": MODEL if pred.source.startswith("opus48") else "claude-opus-5",
                "prediction_input_expression": pred.plan.request.get("original_expression"),
                "prediction_effective_expression": pred.frozen.effective_expression,
                "prediction_simplified_expression": pred.frozen.simplified_expression,
                "prediction_simplify_status": pred.frozen.simplified_status,
                "prediction_expression_resolution": pred.frozen.expression_resolution,
                "prediction_structured_output": pred.frozen.structured_output,
                "prediction_plan_evaluation_key": pred.plan.evaluation_key,
                "prediction_frozen_evaluation_key": pred.frozen.evaluation_key,
                "prediction_plan_sha256": pred.frozen.plan_sha256,
                "prediction_response_sha256": pred.frozen.result_sha256,
                "prediction_response_path": pred.frozen_path,
                "prediction_prompt_version": pred.prompt_version,
                "prediction_prompt_sha256": pred.prompt_sha256,
                "prediction_schema_version": pred.schema_version,
                "prediction_schema_sha256": pred.schema_sha256,
                "variable_mapping": pred.plan.request.get("ast_source_evidence", {}).get("variable_mapping")
                or {f"x{i}": name for i, name in enumerate(terminal["feature_names"])},
            })
        rows.append(row)
    return rows


def _gt_reference_rows(stage: Path) -> list[dict[str, Any]]:
    plan_path = stage / "inputs/gt_plan_full.jsonl"
    plan_sha = _sha_file(plan_path)
    plans = _unique_index(_rows(plan_path), lambda row: row["request"]["dataset_id"])
    indexes = _unique_index(_rows(stage / "simplify_full/gt_frozen_index.jsonl"),
                            lambda row: row["evaluation_key"])
    result = []
    for dataset, plan in sorted(plans.items()):
        frozen = indexes[plan["evaluation_key"]]
        result.append({
            "dataset_id": dataset, "input_expression": plan["request"].get("original_expression"),
            "fixed_reference_expression": frozen["effective_expression"],
            "gt_source_evidence_sha256": plan["request"].get("ast_source_evidence", {}).get(
                "ground_truth_source_evidence_sha256"),
            "gt_plan_evaluation_key": plan["evaluation_key"],
            "gt_frozen_evaluation_key": frozen["evaluation_key"],
            "gt_plan_sha256": plan_sha, "gt_result_sha256": frozen["result_sha256"],
            "gt_structured_output": frozen.get("structured_output"),
            "gt_prompt_version": plan["prompt_version"], "gt_schema_version": plan["schema_version"],
            "gt_prompt_sha256": plan["prompt_sha256"], "gt_schema_sha256": plan["schema_sha256"],
        })
    return result


def build(*, collection: Path, release: Path, stage: Path, output: Path,
          max_per_phase: int | None = None, pair_timeout_seconds: float = 120.0,
          pair_memory_gib: float = 2.0) -> dict[str, Any]:
    report, predictions, current, unresolved = inventory(collection=collection, release=release, stage=stage)
    gt = _gt_records(stage)
    old_eq, old_struct = _old_decisions(stage, release, "equivalence"), _old_decisions(stage, release, "structure")
    contracts = {phase: builder._load_prompt_schema(REPO_ROOT, task_kind=phase)
                 for phase in ("equivalence", "structure")}
    plans = {phase: _JsonlSink(output / f"{phase}_plan.jsonl") for phase in ("equivalence", "structure")}
    candidates = {phase: _JsonlSink(output / f"{phase}_reuse_candidates.jsonl")
                  for phase in ("equivalence", "structure")}
    non_applicable = _JsonlSink(output / "structure_non_applicable.jsonl")
    memory_limit = int(pair_memory_gib * 1024**3)
    if pair_timeout_seconds <= 0 or memory_limit <= 0:
        raise DownstreamPlanError("pair timeout and memory limit must be positive")
    eq_attempts = 0
    for key, pred in sorted(predictions.items()):
        if max_per_phase is not None and len(plans["equivalence"]) >= max_per_phase:
            break
        condition, algorithm, dataset, seed = key
        gtp, gtf = gt[dataset]
        base = {"algorithm": pred.plan.request["algorithm"], "algorithm_slug": pred.plan.request["algorithm_slug"],
                "dataset_id": dataset, "dataset_index": pred.plan.request["dataset_index"],
                "noise_tag": condition, "seed": seed, "prediction_task_id": pred.current["task_id"],
                "prediction_valid_output": True, "variables": pred.plan.request["variables"],
                "allowed_functions": sorted(set(pred.plan.request["allowed_functions"]) |
                                            set(gtp.request["allowed_functions"])),
                "ground_truth_logical_id": gtp.logical_id,
                "ground_truth_plan_evaluation_key": gtp.evaluation_key,
                "ground_truth_frozen_evaluation_key": gtf.evaluation_key,
                "ground_truth_frozen_plan_sha256": gtf.plan_sha256,
                "ground_truth_simplify_status": gtf.simplified_status,
                "ground_truth_expression_resolution": gtf.expression_resolution,
                "simplified_ground_truth_expression": gtf.simplified_expression,
                "effective_ground_truth_expression": gtf.effective_expression,
                "model_binding": MODEL,
                **_binding("prediction", pred)}
        logical_id = f"equivalence::{pred.plan.request['algorithm_slug']}::{pred.plan.request['dataset_index']}::s{seed}::{condition}"
        reuse = _candidate(old_eq.get(key), base, "equivalence", contracts["equivalence"])
        if reuse:
            candidates["equivalence"].add({"logical_id": logical_id,
                                           "terminal_expression_sha256": pred.current["terminal_expression_sha256"],
                                           "terminal_source_sha256": pred.current["terminal_source_sha256"],
                                           "effective_prediction_expression": pred.frozen.effective_expression,
                                           "effective_ground_truth_expression": gtf.effective_expression,
                                           **reuse})
            continue
        eq_attempts += 1
        try:
            evidence = _pair_evidence_isolated(
                timeout_seconds=pair_timeout_seconds, memory_limit_bytes=memory_limit,
                logical_id=logical_id, phase="equivalence", left_plan=gtp, left_frozen=gtf,
                right_plan=pred.plan, right_frozen=pred.frozen, pair_seed=seed)
            request = {**base, "deterministic_evidence": evidence, "evidence_hash": evidence["evidence_sha256"]}
            task = builder._task_from_request(
                logical_id=logical_id, task_type="equivalence", priority=30, request=request,
                evidence_hash=evidence["evidence_sha256"], contract=contracts["equivalence"],
                dependencies=(gtf.evaluation_key, pred.frozen.evaluation_key), condition=condition)
            row = task.to_json_record()
            plans["equivalence"].add(row)
        except (ValueError, KeyError, TimeoutError, builder.SymbolicTaskBuilderError) as exc:
            unresolved.append({"key": key, "phase": "equivalence", "reason": str(exc)})
        if eq_attempts % 25 == 0:
            clear_cache()
            gc.collect()
    structure_attempts = 0
    for condition, algorithm, dataset in sorted({k[:3] for k in current}):
        for a, b in PAIRS:
            if max_per_phase is not None and len(plans["structure"]) >= max_per_phase:
                break
            ka, kb = (condition, algorithm, dataset, a), (condition, algorithm, dataset, b)
            if algorithm == "gplearn":
                continue
            if current[ka]["minute180_valid_output"] is not True or current[kb]["minute180_valid_output"] is not True:
                non_applicable.add({"condition": condition, "algorithm": algorithm, "dataset_id": dataset,
                                       "seed_a": a, "seed_b": b, "decision": "non_applicable",
                                       "reason": "invalid_seed", "terminal_a_sha256": current[ka]["terminal_expression_sha256"],
                                       "terminal_b_sha256": current[kb]["terminal_expression_sha256"]})
                continue
            if ka not in predictions or kb not in predictions:
                unresolved.append({"key": (condition, algorithm, dataset, a, b), "phase": "structure",
                                   "reason": "waiting_for_prediction_simplification"})
                continue
            pa, pb = predictions[ka], predictions[kb]
            logical_id = f"stab_structure::{pa.plan.request['algorithm_slug']}::{pa.plan.request['dataset_index']}::s{a}-s{b}"
            structure_attempts += 1
            base = {"algorithm": pa.plan.request["algorithm"], "algorithm_slug": pa.plan.request["algorithm_slug"],
                    "dataset_id": dataset, "dataset_index": pa.plan.request["dataset_index"],
                    "noise_tag": condition, "seed_a": a, "seed_b": b,
                    "prediction_a_task_id": pa.current["task_id"], "prediction_b_task_id": pb.current["task_id"],
                    "prediction_a_valid_output": True, "prediction_b_valid_output": True,
                    "allowed_functions": sorted(set(pa.plan.request["allowed_functions"]) |
                                                set(pb.plan.request["allowed_functions"])),
                    "model_binding": MODEL, **_binding("prediction_a", pa), **_binding("prediction_b", pb)}
            reuse = _candidate(old_struct.get((condition, algorithm, dataset, a, b)), base, "structure",
                               contracts["structure"])
            if reuse:
                candidates["structure"].add({"logical_id": logical_id,
                                             "terminal_a_sha256": pa.current["terminal_expression_sha256"],
                                             "terminal_b_sha256": pb.current["terminal_expression_sha256"],
                                             "terminal_a_source_sha256": pa.current["terminal_source_sha256"],
                                             "terminal_b_source_sha256": pb.current["terminal_source_sha256"],
                                             "effective_prediction_a_expression": pa.frozen.effective_expression,
                                             "effective_prediction_b_expression": pb.frozen.effective_expression,
                                             **reuse})
                continue
            try:
                evidence = _pair_evidence_isolated(
                    timeout_seconds=pair_timeout_seconds, memory_limit_bytes=memory_limit,
                    logical_id=logical_id, phase="structure", left_plan=pa.plan, left_frozen=pa.frozen,
                    right_plan=pb.plan, right_frozen=pb.frozen, pair_seed=a * 1000 + b)
                request = {**base, "deterministic_evidence": evidence,
                           "deterministic_pair_evidence": evidence, "evidence_hash": evidence["evidence_sha256"]}
                task = builder._task_from_request(
                    logical_id=logical_id, task_type="stab_structure", priority=40, request=request,
                    evidence_hash=evidence["evidence_sha256"], contract=contracts["structure"],
                    dependencies=(pa.frozen.evaluation_key, pb.frozen.evaluation_key), condition=condition)
                plans["structure"].add(task.to_json_record())
            except (ValueError, KeyError, TimeoutError, builder.SymbolicTaskBuilderError) as exc:
                unresolved.append({"key": (condition, algorithm, dataset, a, b), "phase": "structure",
                                   "reason": str(exc)})
            if structure_attempts % 25 == 0:
                clear_cache()
                gc.collect()
    for sink in (*plans.values(), *candidates.values(), non_applicable):
        sink.finish()
    active_rows = _active_binding_rows(current, predictions)
    _write(output / "active_prediction_binding.jsonl", active_rows)
    _write(output / "gt_reference_binding.jsonl", _gt_reference_rows(stage))
    _write(output / "active_prediction_unavailable.jsonl", (
        row for row in active_rows if row["processing_status"] != "ready"))
    _write(output / "unresolved.jsonl", unresolved)
    report.update({"complete_plan": max_per_phase is None, "equivalence_planned": len(plans["equivalence"]),
                   "structure_planned": len(plans["structure"]),
                   "equivalence_reuse_candidates": len(candidates["equivalence"]),
                   "structure_reuse_candidates": len(candidates["structure"]),
                   "structure_non_applicable": len(non_applicable), "unresolved_rows": len(unresolved),
                   "pair_timeout_seconds": pair_timeout_seconds, "pair_memory_gib": pair_memory_gib,
                   "outputs": {path.name: _sha_file(path) for path in output.iterdir() if path.suffix == ".jsonl"}})
    report["prediction_provenance_validation"] = {
        "opus48_base_frozen": report["prediction_sources"].get("opus48_frozen", 0)
        + report["prediction_sources"].get("opus48_frozen_legacy_top_level_missing", 0),
        "opus48_retry_v2_frozen": report["prediction_sources"].get("opus48_retry_v2_frozen", 0),
        "binding_check": "source plan key/input hash, terminal formula/snapshot SHA, promotable semantic status",
        "legacy_binding_note": "First canary stores identical terminal binding in frozen.request, not top level.",
    }
    (output / "binding_manifest.json").write_text(json.dumps({
        "schema_version": "current_terminal_binding_manifest.v1",
        "run_count": len(active_rows), "ready_count": len(predictions),
        "source_counts": report["prediction_sources"],
        "validation": report["prediction_provenance_validation"],
        "active_prediction_binding_sha256": report["outputs"]["active_prediction_binding.jsonl"],
        "gt_reference_binding_sha256": report["outputs"]["gt_reference_binding.jsonl"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--max-per-phase", type=int)
    parser.add_argument("--pair-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--pair-memory-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.inventory_only:
        report, _, _, _ = inventory(collection=args.collection, release=args.release, stage=args.stage)
    else:
        if args.output is None:
            parser.error("--output is required without --inventory-only")
        report = build(collection=args.collection, release=args.release, stage=args.stage,
                       output=args.output, max_per_phase=args.max_per_phase,
                       pair_timeout_seconds=args.pair_timeout_seconds,
                       pair_memory_gib=args.pair_memory_gib)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
