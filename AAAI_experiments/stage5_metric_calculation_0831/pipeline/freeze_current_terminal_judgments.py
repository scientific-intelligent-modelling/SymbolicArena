"""Bind frozen Opus judgments to the selected Core-50 terminal expressions.

Reuse candidates are promoted only after their original plan, response file, and
current expression dependencies all agree. Missing decisions remain unresolved.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from . import prepare_current_terminal_opus48_downstream as downstream
from .audit_current_terminal_symbolic_bindings import verify_response_artifact


class JudgmentBindingError(ValueError):
    pass


DECISIONS = {
    "equivalence": {"equivalent", "not_equivalent"},
    "structure": {"mathematically_equivalent", "same_canonical_structure", "different_structure"},
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise JudgmentBindingError(reason)


def _logical_id_matches(current: str, historical: str, condition: str) -> bool:
    if historical in {current, current + "::" + condition}:
        return True
    for prefix in (current + "::v", current + "::" + condition + "::v"):
        if historical.startswith(prefix) and historical[len(prefix):].isdigit():
            return True
    return False


def _run_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["condition"]), str(row["algorithm"]).casefold(),
            str(row["dataset_id"]), int(row["seed"]))


def _plan_key(row: Mapping[str, Any], phase: str) -> tuple[Any, ...]:
    request = row["request"]
    prefix = (str(row["condition"]), str(request["algorithm_slug"]).casefold(),
              str(request["dataset_id"]))
    return prefix + ((int(request["seed"]),) if phase == "equivalence" else
                     (int(request["seed_a"]), int(request["seed_b"])))


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (str(row["condition"]), str(row["algorithm"]).casefold(),
            str(row["dataset_id"]), int(row["seed_left"]), int(row["seed_right"]))


def _index(rows: list[dict[str, Any]], key_fn, name: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        _require(key not in result, f"{name} duplicate: {key}")
        result[key] = row
    return result


def _check_current(
    phase: str, key: tuple[Any, ...], request: Mapping[str, Any],
    bindings: Mapping[tuple[str, str, str, int], dict[str, Any]],
    gt: Mapping[str, dict[str, Any]],
) -> list[str]:
    """Check source formulas, feature mapping, and frozen simplification keys."""
    condition, algorithm, dataset = key[:3]
    _require(request.get("noise_tag") in (None, condition), f"noise condition drift: {key}")
    _require(dataset in gt, f"GT missing: {key}")
    if phase == "equivalence":
        seed = key[3]
        pred = bindings[(condition, algorithm, dataset, seed)]
        _require(pred["processing_status"] == "ready" and pred["valid_output"] is True,
                 f"prediction not ready: {key}")
        checks = {
            "terminal_expression_sha256": (request.get("prediction_terminal_expression_sha256"),
                                           pred["terminal_expression_sha256"]),
            "terminal_source_sha256": (request.get("prediction_terminal_source_sha256"),
                                       pred["terminal_source_sha256"]),
            "selected_result_sha256": (request.get("prediction_current_selected_result_sha256"),
                                       pred["selected_result_sha256"]),
            "prediction_expression": (request.get("effective_prediction_expression"),
                                      pred["prediction_effective_expression"]),
            "prediction_key": (request.get("prediction_frozen_evaluation_key"),
                               pred["prediction_frozen_evaluation_key"]),
            "gt_expression": (request.get("effective_ground_truth_expression"),
                              gt[dataset]["fixed_reference_expression"]),
            "gt_key": (request.get("ground_truth_frozen_evaluation_key"),
                       gt[dataset]["gt_frozen_evaluation_key"]),
            "variables": (request.get("variables"), pred["feature_names"]),
            "variable_mapping": (request.get("prediction_variable_mapping"), pred["variable_mapping"]),
        }
        for name, (actual, expected) in checks.items():
            _require(actual == expected, f"{name} drift: {key}")
        return [gt[dataset]["gt_frozen_evaluation_key"], pred["prediction_frozen_evaluation_key"]]
    a, b = key[3:]
    _require((a, b) in downstream.PAIRS, f"seed pair invalid: {key}")
    deps = []
    for side, seed in (("a", a), ("b", b)):
        pred = bindings[(condition, algorithm, dataset, seed)]
        _require(pred["processing_status"] == "ready" and pred["valid_output"] is True,
                 f"pair prediction not ready: {key}")
        checks = {
            "terminal_expression_sha256": (request.get(f"prediction_{side}_terminal_expression_sha256"),
                                           pred["terminal_expression_sha256"]),
            "terminal_source_sha256": (request.get(f"prediction_{side}_terminal_source_sha256"),
                                       pred["terminal_source_sha256"]),
            "selected_result_sha256": (request.get(f"prediction_{side}_current_selected_result_sha256"),
                                       pred["selected_result_sha256"]),
            "prediction_expression": (request.get(f"effective_prediction_{side}_expression"),
                                      pred["prediction_effective_expression"]),
            "prediction_key": (request.get(f"prediction_{side}_frozen_evaluation_key"),
                               pred["prediction_frozen_evaluation_key"]),
            "variable_mapping": (request.get(f"prediction_{side}_variable_mapping"),
                                 pred["variable_mapping"]),
        }
        for name, (actual, expected) in checks.items():
            _require(actual == expected, f"{side}_{name} drift: {key}")
        deps.append(pred["prediction_frozen_evaluation_key"])
    return deps


def _effective_row(
    phase: str, key: tuple[Any, ...], request: Mapping[str, Any],
    *, decision: str, dependencies: list[str], evaluation_key: str,
    response_path: str, response_sha256: str, structured_output: Mapping[str, Any],
    model: str, prompt_version: str, prompt_sha256: str, schema_version: str,
    schema_sha256: str, source: str,
) -> dict[str, Any]:
    _require(decision in DECISIONS[phase], f"undecided: {phase}/{key}/{decision}")
    common = {
        "condition": key[0], "algorithm": key[1], "dataset_id": key[2],
        "state": "frozen", "effective_decision": decision,
        "dependencies": dependencies, "evaluation_key": evaluation_key,
        "response_path": response_path, "response_sha256": response_sha256,
        "structured_output": structured_output, "source": source, "model": model,
        "prompt_version": prompt_version, "prompt_sha256": prompt_sha256,
        "schema_version": schema_version, "schema_sha256": schema_sha256,
    }
    if phase == "equivalence":
        return {**common, "seed": key[3],
                "terminal_expression": request["prediction_terminal_expression"],
                "terminal_expression_sha256": request["prediction_terminal_expression_sha256"],
                "terminal_source_sha256": request["prediction_terminal_source_sha256"],
                "selected_result_sha256": request["prediction_current_selected_result_sha256"],
                "prediction_effective_expression": request["effective_prediction_expression"],
                "gt_reference_expression": request["effective_ground_truth_expression"],
                "variable_mapping": request["prediction_variable_mapping"]}
    return {**common, "seed_left": key[3], "seed_right": key[4],
            "left_terminal_expression": request["prediction_a_terminal_expression"],
            "right_terminal_expression": request["prediction_b_terminal_expression"],
            "left_terminal_expression_sha256": request["prediction_a_terminal_expression_sha256"],
            "right_terminal_expression_sha256": request["prediction_b_terminal_expression_sha256"],
            "left_terminal_source_sha256": request["prediction_a_terminal_source_sha256"],
            "right_terminal_source_sha256": request["prediction_b_terminal_source_sha256"],
            "left_selected_result_sha256": request["prediction_a_current_selected_result_sha256"],
            "right_selected_result_sha256": request["prediction_b_current_selected_result_sha256"],
            "left_effective_expression": request["effective_prediction_a_expression"],
            "right_effective_expression": request["effective_prediction_b_expression"],
            "left_variable_mapping": request["prediction_a_variable_mapping"],
            "right_variable_mapping": request["prediction_b_variable_mapping"]}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def freeze(*, plan_dir: Path, collection: Path, output: Path) -> dict[str, Any]:
    bindings = _index(_rows(plan_dir / "active_prediction_binding.jsonl"), _run_key, "prediction")
    gt = _index(_rows(plan_dir / "gt_reference_binding.jsonl"), lambda row: row["dataset_id"], "GT")
    _require(len(bindings) == 6750 and len(gt) == 50, "prediction/GT inventory incomplete")
    results: dict[str, list[dict[str, Any]]] = {phase: [] for phase in DECISIONS}
    unresolved: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for phase in DECISIONS:
        old = downstream._old_decisions(downstream.DEFAULT_STAGE, downstream.DEFAULT_RELEASE, phase)
        old_by_evaluation_key = _index(
            [{"identity": key, "plan": pair[0], "index": pair[1]} for key, pair in old.items()],
            lambda row: row["plan"]["evaluation_key"], f"old {phase} evaluation key")
        candidates = _rows(plan_dir / f"{phase}_reuse_candidates.jsonl")
        planned = _rows(plan_dir / f"{phase}_plan.jsonl")
        planned += _rows(collection / f"opus48_undetermined_selected_v1/{phase}_plan.jsonl")
        # Repair only the exact unresolved source task, not another condition with the same logical ID.
        if phase == "structure":
            planned += _rows(collection / "opus48_structure_noise005_json_repair_v2_plan/structure_retry_plan.jsonl")
        new_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        for plan in planned:
            key = _plan_key(plan, phase)
            if key in new_by_key:
                _require(phase == "structure" and key ==
                         ("noise005", "qlattice", "feynman-i.15.3t", 520, 522),
                         f"unexpected duplicate plan: {key}")
            new_by_key[key] = plan
        candidate_keys: set[tuple[Any, ...]] = set()
        for candidate in candidates:
            match = old_by_evaluation_key.get(candidate["old_evaluation_key"])
            _require(match is not None, f"old candidate key missing: {candidate['logical_id']}")
            key, old_plan, old_index = match["identity"], match["plan"], match["index"]
            if candidate["old_decision"] == "undetermined":
                _require(key in new_by_key, f"undetermined without new plan: {key}")
                continue
            _require(key not in candidate_keys and key not in new_by_key,
                     f"duplicate candidate/plan: {key}")
            candidate_keys.add(key)
            old_req = old_plan["request"]
            _require(_logical_id_matches(candidate["logical_id"], old_index["logical_id"], key[0]),
                     f"old logical ID drift: {key}")
            _require(candidate["old_response_sha256"] == old_index["response_sha256"] and
                     candidate["old_response_path"] == old_index["response_path"] and
                     candidate["old_decision"] == old_index["effective_decision"] and
                     verify_response_artifact(old_index, downstream.REPO_ROOT),
                     f"old response integrity drift: {key}")
            _require(old_index["state"] == "frozen", f"old response not frozen: {key}")
            # Old plans omit the new terminal hashes. Bind them only after all original
            # expression/result dependencies and current terminal source agree.
            request = dict(old_req)
            if phase == "equivalence":
                pred = bindings[key]
                _require(candidate["terminal_expression_sha256"] == pred["terminal_expression_sha256"] and
                         candidate["terminal_source_sha256"] == pred["terminal_source_sha256"] and
                         candidate["effective_prediction_expression"] == pred["prediction_effective_expression"] and
                         candidate["effective_ground_truth_expression"] == gt[key[2]]["fixed_reference_expression"] and
                         old_req["prediction_result_sha256"] == pred["selected_result_sha256"],
                         f"old current-terminal drift: {key}")
                request.update({"prediction_terminal_expression": pred["terminal_expression"],
                                "prediction_terminal_expression_sha256": pred["terminal_expression_sha256"],
                                "prediction_terminal_source_sha256": pred["terminal_source_sha256"],
                                "prediction_current_selected_result_sha256": pred["selected_result_sha256"],
                                "prediction_frozen_evaluation_key": pred["prediction_frozen_evaluation_key"],
                                "prediction_variable_mapping": pred["variable_mapping"],
                                "ground_truth_frozen_evaluation_key": gt[key[2]]["gt_frozen_evaluation_key"]})
            else:
                for side, seed in (("a", key[3]), ("b", key[4])):
                    pred = bindings[(*key[:3], seed)]
                    _require(candidate[f"terminal_{side}_sha256"] == pred["terminal_expression_sha256"] and
                             candidate[f"terminal_{side}_source_sha256"] == pred["terminal_source_sha256"] and
                             candidate[f"effective_prediction_{side}_expression"] == pred["prediction_effective_expression"] and
                             old_req[f"prediction_{side}_result_sha256"] == pred["selected_result_sha256"],
                             f"old pair current-terminal drift: {key}")
                    request.update({f"prediction_{side}_terminal_expression": pred["terminal_expression"],
                                    f"prediction_{side}_terminal_expression_sha256": pred["terminal_expression_sha256"],
                                    f"prediction_{side}_terminal_source_sha256": pred["terminal_source_sha256"],
                                    f"prediction_{side}_current_selected_result_sha256": pred["selected_result_sha256"],
                                    f"prediction_{side}_frozen_evaluation_key": pred["prediction_frozen_evaluation_key"],
                                    f"prediction_{side}_variable_mapping": pred["variable_mapping"]})
            deps = _check_current(phase, key, request, bindings, gt)
            _require(old_index["dependencies"] == deps, f"old dependency drift: {key}")
            result = _effective_row(phase, key, request,
                                    decision=old_index["effective_decision"], dependencies=deps,
                                    evaluation_key=old_index["evaluation_key"],
                                    response_path=old_index["response_path"],
                                    response_sha256=old_index["response_sha256"],
                                    structured_output=old_index["structured_output"],
                                    model=old_index.get("response_model") or old_index.get("requested_model") or "claude-opus-5",
                                    prompt_version=old_plan["prompt_version"],
                                    prompt_sha256=old_plan["prompt_sha256"],
                                    schema_version=old_plan["schema_version"],
                                    schema_sha256=old_plan["schema_sha256"], source="verified_opus5_reuse")
            result["historical_dependency_plan_sha256"] = (
                [old_req.get("prediction_frozen_plan_sha256")] if phase == "equivalence" else
                [old_req.get("prediction_a_frozen_plan_sha256"), old_req.get("prediction_b_frozen_plan_sha256")])
            result["current_dependency_plan_sha256"] = (
                [bindings[key]["prediction_plan_sha256"]] if phase == "equivalence" else
                [bindings[(*key[:3], seed)]["prediction_plan_sha256"] for seed in key[3:]])
            result["audit_overlay_applied"] = old_index.get("audit_overlay_applied", False)
            result["audit_source"] = old_index.get("audit_source")
            result["underlying_structured_output"] = old_index.get("underlying_structured_output")
            result["evidence_generation"] = old_index.get("evidence_generation")
            results[phase].append(result)
            source_counts[f"{phase}:verified_opus5_reuse"] += 1
        frozen_paths: dict[str, Path] = {}
        names = ("opus48_equivalence_run_20260916", "opus48_undetermined_equivalence_run") if phase == "equivalence" else (
            "opus48_structure_run_20260916", "opus48_undetermined_structure_run",
            "opus48_structure_noise005_json_repair_v2_run")
        for name in names:
            for path in (collection / name / "frozen").glob("*.json"):
                raw = json.loads(path.read_text(encoding="utf-8"))
                source_key = raw.get("source_evaluation_key")
                _require(source_key not in frozen_paths, f"duplicate frozen source key: {source_key}")
                frozen_paths[source_key] = path
        for key, plan in sorted(new_by_key.items()):
            deps = _check_current(phase, key, plan["request"], bindings, gt)
            _require(plan["dependencies"] == deps, f"new plan dependencies drift: {key}")
            path = frozen_paths.get(plan["evaluation_key"])
            if path is None:
                unresolved.append({"phase": phase, "key": list(key), "reason": "new_judgment_not_frozen",
                                   "source_evaluation_key": plan["evaluation_key"]})
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            _require(path.stem == raw["evaluation_key"] and raw["status"] == "frozen" and
                     raw["source_evaluation_key"] == plan["evaluation_key"] and
                     raw["source_input_hash"] == plan["input_hash"] and
                     raw["request"] == plan["request"] and
                     raw["plan_dependencies"] == deps and
                     raw["prompt_sha256"] == plan["prompt_sha256"] and
                     raw["schema_sha256"] == plan["schema_sha256"] and
                     raw["model"] == "claude-opus-4-8" and
                     raw["response_model"] == "claude-opus-4-8",
                     f"new frozen artifact drift: {key}")
            decision = raw["structured_output"]["decision"]
            if decision not in DECISIONS[phase]:
                unresolved.append({"phase": phase, "key": list(key),
                                   "reason": f"new_judgment_{decision}"})
                continue
            result = _effective_row(phase, key, plan["request"], decision=decision,
                                    dependencies=deps, evaluation_key=raw["evaluation_key"],
                                    response_path=str(path), response_sha256=_sha(path),
                                    structured_output=raw["structured_output"],
                                    model=raw["response_model"],
                                    prompt_version=plan["prompt_version"],
                                    prompt_sha256=plan["prompt_sha256"],
                                    schema_version=plan["schema_version"],
                                    schema_sha256=plan["schema_sha256"],
                                    source="opus48_json_repair" if "json_repair" in str(path) else "opus48_frozen")
            results[phase].append(result)
            source_counts[f"{phase}:{result['source']}"] += 1
        expected: set[tuple[Any, ...]] = set()
        for key, binding in bindings.items():
            if key[1] == "gplearn" or binding["valid_output"] is not True:
                continue
            if phase == "equivalence":
                expected.add(key)
            else:
                for a, b in downstream.PAIRS:
                    ka, kb = (*key[:3], a), (*key[:3], b)
                    if bindings[ka]["valid_output"] is True and bindings[kb]["valid_output"] is True:
                        expected.add((*key[:3], a, b))
        actual = {_run_key(row) if phase == "equivalence" else _pair_key(row)
                  for row in results[phase]}
        _require(len(actual) == len(results[phase]), f"duplicate effective {phase} row")
        _require(actual <= expected, f"unexpected effective {phase} row")
        _require(expected == actual | set(new_by_key).difference(actual),
                 f"missing {phase} planning coverage")
        _write_jsonl(output / f"{phase}_effective_index.jsonl", sorted(results[phase],
                     key=lambda row: _run_key(row) if phase == "equivalence" else _pair_key(row)))
    _write_jsonl(output / "unresolved.jsonl", unresolved)
    manifest = {"schema_version": "current_terminal_judgments.v1",
                "non_gplearn_judgments_ready": not unresolved, "formal_ready": False,
                "gplearn_deferred": True,
                "counts": {phase: len(rows) for phase, rows in results.items()},
                "source_counts": dict(source_counts), "unresolved_count": len(unresolved),
                "input_sha256": {path.name: _sha(path) for path in (
                    plan_dir / "active_prediction_binding.jsonl", plan_dir / "gt_reference_binding.jsonl",
                    plan_dir / "equivalence_plan.jsonl", plan_dir / "structure_plan.jsonl",
                    plan_dir / "equivalence_reuse_candidates.jsonl",
                    plan_dir / "structure_reuse_candidates.jsonl")},
                "output_sha256": {path.name: _sha(path) for path in output.glob("*.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--collection", type=Path, default=downstream.DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(plan_dir=args.plan_dir, collection=args.collection, output=args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
