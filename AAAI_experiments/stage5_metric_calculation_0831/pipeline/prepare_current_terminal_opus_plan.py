"""Prepare simplify.v1 prediction requests for changed Stage4 trajectory endpoints.

The selected terminal snapshot is the formula source. The selected result payload
supplies identity and noise-protocol evidence, never a replacement expression.
No API call is made by this module.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sympy

from . import clean_task_builder as builder
from . import performance_replay


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE / "work/core50_terminal_collection_20260916_v3"
FORMAL = ROOT / "AAAI_experiments/Core50_final_20260914/results"
ARCHIVE = ROOT / "AAAI_experiments/Core50_raw_snapshots_20260914.tar.zst"
TRAJECTORIES = STAGE / "work/final_release_20260913/release_v2/eff_revision_v3"
CHANGED_STATUS = "needs_prediction_simplification_changed_expression"
CONDITIONS = ("clean", "noise001", "noise005")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def key_of(row: dict) -> tuple[str, str, str, int]:
    return (row["condition"], row.get("algorithm_slug", row.get("algorithm")).lower(), row["dataset_id"], int(row["seed"]))


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def index_jsonl(path: Path, label: str) -> dict[tuple[str, str, str, int], dict]:
    indexed = {}
    for row in read_jsonl(path):
        key = key_of(row)
        if key in indexed:
            raise ValueError(f"duplicate {label}: {key}")
        indexed[key] = row
    return indexed


def load_old_formal(formal: Path) -> dict[tuple[str, str, str, int], dict]:
    indexed = {}
    for condition in CONDITIONS:
        for item in read_jsonl(formal / condition / "raw_results.jsonl.gz"):
            source = item["source"]
            key = (condition, source["algorithm"].lower(), source["dataset_id"], int(source["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate old formal key: {key}")
            indexed[key] = item
    return indexed


def read_preflight(path: Path) -> dict[tuple[str, str, str, int], dict]:
    indexed = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["binding_status"] != CHANGED_STATUS:
                continue
            key = key_of(row)
            if key in indexed:
                raise ValueError(f"duplicate changed endpoint: {key}")
            indexed[key] = row
    return indexed


def archive_snapshot_index(archive: Path, members: set[str]) -> dict[str, dict[str, str]]:
    command = ["tar", "-I", "zstd", "-xOf", str(archive), "physical_records.csv.gz"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout and process.stderr
    selected: dict[str, dict[str, str]] = {}
    try:
        with io.TextIOWrapper(gzip.GzipFile(fileobj=process.stdout), encoding="utf-8", newline="") as text:
            for row in csv.DictReader(text):
                member = row["archive_member"]
                if member not in members:
                    continue
                if member in selected:
                    raise ValueError(f"duplicate archive member: {member}")
                selected[member] = row
    finally:
        stderr = process.stderr.read().decode("utf-8", "replace")
        code = process.wait()
        if code:
            raise RuntimeError(f"archive index read failed: {stderr.strip()}")
    return selected


def load_ground_truth_and_probes() -> tuple[dict[str, list[str]], dict[str, str], dict[str, dict], str]:
    gt_path = ROOT / builder.DEFAULT_GROUND_TRUTH_JSONL
    probes_path = ROOT / builder.DEFAULT_DATASET_PROBES_JSONL
    gt_rows = list(read_jsonl(gt_path))
    variables = {}
    targets = {}
    for row in gt_rows:
        dataset = row["dataset_id"]
        if dataset in variables:
            raise ValueError(f"duplicate Ground Truth dataset: {dataset}")
        variables[dataset] = row["ordered_variables"]
        targets[dataset] = row["target"]
    probes, probes_sha = builder.load_dataset_probes(probes_path)
    if len(variables) != 50 or len(probes) != 50 or set(variables) != set(probes):
        raise ValueError("Ground Truth/probe grid is not the frozen 50 tasks")
    return variables, targets, probes, probes_sha


def load_evaluation_paths(root: Path = TRAJECTORIES) -> dict[tuple[str, str, str, int], dict]:
    rows = {}
    for condition in CONDITIONS:
        for item in read_jsonl(root / condition / "native_trajectories.jsonl.gz"):
            key = key_of(item)
            if key in rows:
                raise ValueError(f"duplicate numeric trajectory: {key}")
            rows[key] = item
    return rows


def _payload_record(index: dict, key: tuple, expected_sha: str, label: str) -> tuple[str, dict]:
    row = index.get(key)
    if row is None:
        raise ValueError(f"missing {label} payload")
    raw = row.get("raw_text")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"missing {label} raw_text")
    if sha(raw.encode("utf-8")) != expected_sha:
        raise ValueError(f"{label} raw_text SHA mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} raw payload is not an object")
    return raw, payload


def _actual_host(path: str) -> str | None:
    match = re.search(r"(?:^|/)(iaaccn\d+)(?:/|$)", path)
    return match.group(1) if match else None


def _same_source_path(left: str, right: str) -> bool:
    def resolved(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()

    return resolved(left) == resolved(right)


def _verified_recovery_function(snapshot: dict, terminal_expression: str) -> str | None:
    function = snapshot.get("function")
    params = snapshot.get("params")
    if not isinstance(function, str) or not isinstance(params, list):
        return None
    try:
        tree = ast.parse(function)
        returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        if len(returns) != 1 or returns[0].value is None:
            return None
        expression_tree = returns[0].value
        # 恢复候选只允许数值运算与 params 索引；不执行原始 Python 函数。
        allowed = (
            ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Subscript,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
            ast.Load,
        )
        if any(not isinstance(node, allowed) for node in ast.walk(expression_tree)):
            return None
        if any(
            isinstance(node, ast.Name) and node.id != "params" and not re.fullmatch(r"x\d+", node.id)
            for node in ast.walk(expression_tree)
        ):
            return None
        instantiated = builder.instantiate_parameters(ast.unparse(expression_tree), params)
        variables = {name: sympy.Symbol(name) for name in set(re.findall(r"\bx\d+\b", instantiated + terminal_expression))}
        difference = sympy.simplify(
            sympy.sympify(instantiated, locals=variables)
            - sympy.sympify(terminal_expression, locals=variables)
        )
        return instantiated if difference == 0 else None
    except (SyntaxError, TypeError, ValueError, builder.CleanTaskBuilderError, sympy.SympifyError):
        return None


def audit_formula_source(snapshot: dict, terminal: dict, *, evaluation_path: str) -> dict:
    """Verify the expression actually evaluated, not a stale snapshot artifact."""

    expression = terminal["terminal_expression"]
    features = terminal["feature_names"]
    artifact = snapshot.get("canonical_artifact")
    original = None
    source = None
    parse_error = None
    try:
        original, source = builder.select_formula_with_source(snapshot)
        canonical_matches = builder._canonical_mapped_ast(original, features) == builder._canonical_mapped_ast(expression, features)
    except (builder.CleanTaskBuilderError, SyntaxError, TypeError, ValueError) as exc:
        canonical_matches = False
        parse_error = str(exc)
    evidence = {
        "evaluation_path": evaluation_path,
        "snapshot_equation": snapshot.get("equation"),
        "snapshot_canonical_expression": original,
        "snapshot_canonical_source": source,
        "snapshot_canonical_parse_error": parse_error,
        "terminal_expression": expression,
        "canonical_ast_matches_terminal": canonical_matches,
        "discrepancy": not canonical_matches,
        "resolution": "snapshot_canonical_ast_match" if canonical_matches else "unresolved",
        "corrected_artifact_sha256": None,
        "status": "verified" if canonical_matches else "unresolved",
    }
    if canonical_matches:
        evidence["corrected_artifact"] = dict(artifact) if isinstance(artifact, dict) else None
        return evidence

    if evaluation_path == "canonical_replay.v1" and snapshot.get("function") and snapshot.get("params"):
        instantiated = _verified_recovery_function(snapshot, expression)
        if instantiated is not None:
            evidence.update({
                "status": "verified", "resolution": "recovery_function_params_symbolic_equality",
                "recovery_instantiated_expression": instantiated,
                "corrected_artifact_sha256": sha(instantiated.encode("utf-8")),
                "corrected_artifact": {"instantiated_expression": expression, "tool_name": terminal["algorithm_slug"]},
            })
            return evidence

    algorithm = terminal["algorithm_slug"].lower()
    if algorithm == "symbolfit" and evaluation_path == "internal_loss_selected_execution_runner_canonical_capture.v1":
        # 专用数值路径从原始 equation 和同一快照的 ID/OOD capture 读数。
        id_nmse = (snapshot.get("id_test") or {}).get("nmse")
        ood_nmse = (snapshot.get("ood_test") or {}).get("nmse")
        if snapshot.get("equation") == expression and isinstance(id_nmse, (int, float)) and isinstance(ood_nmse, (int, float)):
            evidence.update({
                "status": "verified", "resolution": "symbolfit_execution_runner_equation_and_metrics",
                "captured_id_nmse": id_nmse, "captured_ood_nmse": ood_nmse,
                "corrected_artifact": dict(artifact) if isinstance(artifact, dict) else None,
            })
        return evidence

    if evaluation_path == "canonical_replay.v1" and algorithm in {"qlattice", "drsr", "imcts", "llmsr"}:
        try:
            corrected, rebuilt = performance_replay._corrected_artifact(
                snapshot, algorithm=algorithm, expected_n_features=len(features)
            )
            corrected_expression = corrected.get("instantiated_expression") or corrected.get("normalized_expression")
            if rebuilt and isinstance(corrected_expression, str) and builder._canonical_mapped_ast(
                corrected_expression, features
            ) == builder._canonical_mapped_ast(expression, features):
                evidence.update({
                    "status": "verified", "resolution": "canonical_replay_forced_raw_rebuild",
                    "corrected_artifact_sha256": sha(canonical_json(corrected).encode("utf-8")),
                    "corrected_artifact": corrected,
                })
        except (performance_replay.PerformanceReplayError, builder.CleanTaskBuilderError, ValueError) as exc:
            evidence["replay_error"] = str(exc)
    return evidence


def build_one(
    *,
    key: tuple[str, str, str, int],
    terminal: dict,
    preflight: dict,
    old: dict,
    snapshot_record: dict,
    result_record: dict,
    archive_record: dict,
    evaluation_path: str,
    contract: builder.PromptSchemaBundle,
    gt_variables: dict,
    gt_targets: dict,
    probes: dict,
) -> dict:
    condition, algorithm, dataset, seed = key
    if algorithm == "gplearn" or terminal["minute180_valid_output"] is not True:
        raise ValueError("gplearn or invalid endpoint cannot enter this plan")
    if preflight["logical_key"] != terminal["logical_key"] or preflight["terminal_archive_member"] != terminal["terminal_archive_member"]:
        raise ValueError("preflight does not bind to selected terminal")
    expression = terminal.get("terminal_expression")
    if not isinstance(expression, str) or sha(expression.encode("utf-8")) != terminal.get("terminal_expression_sha256"):
        raise ValueError("terminal expression SHA mismatch")
    if preflight["terminal_expression_sha256"] != terminal["terminal_expression_sha256"]:
        raise ValueError("preflight terminal expression SHA drift")
    if old["source"]["dataset_id"] != dataset or old["source"]["algorithm"].lower() != algorithm or int(old["source"]["seed"]) != seed:
        raise ValueError("old stable task identity maps to another run")
    if old["source"]["noise_tag"] != condition or old["result"]["sha256"] != terminal["old_formal_result_sha256"]:
        raise ValueError("old formal result identity/hash drift")
    if sha(old["result"]["raw_text"].encode("utf-8")) != old["result"]["sha256"]:
        raise ValueError("old formal result payload SHA mismatch")
    stable_task_id = old["source"]["task_id"]
    selected_task_id = terminal["task_id"]
    if stable_task_id != terminal["old_formal_task_id"] or preflight["task_id"] != selected_task_id:
        raise ValueError("stable or selected task ID drift")
    if snapshot_record.get("selected_task_id") != selected_task_id or result_record.get("selected_task_id") != selected_task_id:
        raise ValueError("snapshot/result selected task ID mismatch")
    if result_record.get("result_source_path") != terminal["selected_result_source_path"]:
        raise ValueError("selected result source path mismatch")
    if snapshot_record["archive_member"] != terminal["terminal_archive_member"]:
        raise ValueError("snapshot archive member drift")
    snapshot_sha = terminal["terminal_source_sha256"]
    if snapshot_record.get("terminal_source_sha256") != snapshot_sha:
        raise ValueError("snapshot collection SHA mismatch")
    if result_record.get("result_sha256") != terminal["selected_result_sha256"]:
        raise ValueError("selected result collection SHA mismatch")
    if archive_record.get("sha256") != snapshot_sha or not _same_source_path(
        archive_record.get("source_path", ""), terminal["terminal_source_path"]
    ):
        raise ValueError("snapshot SHA/path differs from archive physical index")
    snapshot_raw, snapshot = _payload_record({key: snapshot_record}, key, snapshot_sha, "snapshot")
    result_raw, selected_result = _payload_record(
        {key: result_record}, key, terminal["selected_result_sha256"], "selected result"
    )
    if selected_result.get("dataset") != dataset or int(selected_result.get("seed", -1)) != seed:
        raise ValueError("selected result dataset or seed mismatch")
    if selected_result.get("feature_names") != terminal["feature_names"]:
        raise ValueError("snapshot/result feature mapping mismatch")
    if selected_result.get("target_name") != terminal["target_name"]:
        raise ValueError("snapshot/result target mismatch")
    formula_evidence = audit_formula_source(snapshot, terminal, evaluation_path=evaluation_path)
    if formula_evidence["status"] != "verified":
        raise ValueError(f"unverified numerical formula binding: {formula_evidence.get('replay_error') or formula_evidence['resolution']}")

    recovery_source = formula_evidence["resolution"] == "recovery_function_params_symbolic_equality"
    if not recovery_source:
        if snapshot.get("dataset") != dataset or int(snapshot.get("seed", -1)) != seed:
            raise ValueError("snapshot dataset or seed mismatch")
        if snapshot.get("feature_names") != terminal["feature_names"] or snapshot.get("target_name") != terminal["target_name"]:
            raise ValueError("snapshot feature mapping or target mismatch")

    actual_noise = selected_result.get("train_label_noise")
    if condition != "clean" and not isinstance(actual_noise, dict):
        raise ValueError("selected noisy result lacks train_label_noise")
    if "train_label_noise" in snapshot and snapshot["train_label_noise"] != actual_noise:
        raise ValueError("snapshot/result train_label_noise mismatch")
    merged_snapshot = dict(selected_result if recovery_source else snapshot)
    if "train_label_noise" not in merged_snapshot and isinstance(actual_noise, dict):
        merged_snapshot["train_label_noise"] = actual_noise
    builder._validate_condition_payload({"task_id": stable_task_id}, merged_snapshot, condition=condition)
    corrected_artifact = dict(formula_evidence["corrected_artifact"] or {})
    # 提示词总用数值轨迹的终点式；源 snapshot/canonical 和纠正依据另存，不静默覆盖。
    corrected_artifact["instantiated_expression"] = expression
    corrected_artifact["variables"] = builder._expression_canonical_variables(expression)
    corrected_artifact["normalization_mode"] = "native_trajectory_terminal_binding"
    merged_snapshot["canonical_artifact"] = corrected_artifact
    if not isinstance(merged_snapshot.get("feature_names"), list):
        raise ValueError("snapshot feature names unavailable")

    provenance = {
        "condition": condition, "algorithm_slug": algorithm, "dataset_id": dataset, "seed": seed,
        "stable_evaluation_task_id": stable_task_id,
        "actual_selected_task_id": selected_task_id,
        "selected_result_path": terminal["selected_result_source_path"],
        "selected_result_sha256": terminal["selected_result_sha256"],
        "terminal_archive_member": terminal["terminal_archive_member"],
        "terminal_source_path": terminal["terminal_source_path"],
        "terminal_snapshot_sha256": snapshot_sha,
        "terminal_expression_sha256": terminal["terminal_expression_sha256"],
        "trajectory_bundle_sha256": terminal["trajectory_bundle_sha256"],
        "snapshot_selected_expression_source": formula_evidence["snapshot_canonical_source"],
        "snapshot_canonical_ast_matches_terminal": formula_evidence["canonical_ast_matches_terminal"],
        "numerical_formula_binding_resolution": formula_evidence["resolution"],
        "numerical_evaluation_path": evaluation_path,
        "selected_noise_contract_origin": "selected_result_payload" if isinstance(actual_noise, dict) else None,
        "metadata_origin": "selected_result_payload" if recovery_source else "terminal_snapshot",
    }
    source = {
        "algorithm": terminal["algorithm"],
        "batch": "core50_terminal_collection_20260916_v3",
        "dataset_id": dataset,
        "host": _actual_host(terminal["terminal_source_path"]),
        "noise_tag": condition,
        "path": terminal["terminal_source_path"],
        "seed": str(seed),
        "task_id": stable_task_id,
        "source_row_sha256": sha(canonical_json(provenance).encode("utf-8")),
    }
    # 噪声契约来自所选结果；原始 snapshot 及 SHA 仍单独完整保留。
    synthetic_raw = json.dumps(merged_snapshot, ensure_ascii=False, sort_keys=True)
    freeze = {"source": source, "result": {"raw_text": synthetic_raw, "sha256": sha(synthetic_raw.encode("utf-8"))}}
    planned, no_call = builder._build_pred_task(
        freeze,
        contract=contract,
        ground_truth_variables=gt_variables,
        ground_truth_targets=gt_targets,
        recovery_entries={},
        dataset_probes=probes,
        condition=condition,
    )
    if planned is None or no_call is not None:
        raise ValueError(f"builder returned no-call: {no_call and no_call.get('reason')}")
    request = dict(planned.request)
    request["terminal_binding_evidence"] = provenance
    evidence_hash = sha(canonical_json({"builder_evidence_hash": request["evidence_hash"], "terminal": provenance}).encode("utf-8"))
    request["evidence_hash"] = evidence_hash
    planned = builder._build_task_definition(
        logical_id=planned.logical_id, task_type=planned.task_type,
        priority=planned.priority, request=request, evidence_hash=evidence_hash,
        contract=contract, condition=condition,
    )
    record = planned.to_json_record()
    record["source_binding"] = provenance
    record["terminal_snapshot"] = snapshot
    record["formula_source_audit"] = {k: v for k, v in formula_evidence.items() if k != "corrected_artifact"}
    record["derived_terminal_canonical_artifact"] = corrected_artifact
    record["terminal_snapshot_raw_sha256"] = snapshot_sha
    record["selected_result_noise_contract"] = actual_noise
    record["selected_result_raw_sha256"] = sha(result_raw.encode("utf-8"))
    return record


def prepare(
    *, collection: Path = COLLECTION,
    preflight: Path | None = None,
    formal: Path = FORMAL,
    archive: Path = ARCHIVE,
    output: Path | None = None,
    expected_count: int = 2336,
) -> dict:
    preflight = preflight or collection / "symbolic_preflight/prediction_binding_audit.csv"
    output = output or collection / "opus_prediction_plan_v2"
    terminals = index_jsonl(collection / "terminal_inputs.jsonl.gz", "terminal")
    changed = read_preflight(preflight)
    if len(changed) != expected_count:
        raise ValueError(f"changed-expression inventory is {len(changed)}, expected {expected_count}")
    snapshots = index_jsonl(collection / "terminal_snapshot_payloads.jsonl.gz", "snapshot")
    results = index_jsonl(collection / "selected_result_payloads.jsonl.gz", "selected result")
    old = load_old_formal(formal)
    trajectories = load_evaluation_paths()
    if not set(changed) <= set(terminals) or not set(changed) <= set(snapshots) or not set(changed) <= set(results):
        raise ValueError("changed-expression input keys are incomplete")
    members = {terminals[key]["terminal_archive_member"] for key in changed}
    physical = archive_snapshot_index(archive, members)
    gt_variables, gt_targets, probes, probes_sha = load_ground_truth_and_probes()
    contract = builder._load_prompt_schema(ROOT)
    plans = []
    unresolved = []
    discrepancies = []
    for key in sorted(changed):
        terminal = terminals[key]
        member = terminal["terminal_archive_member"]
        try:
            if member not in physical:
                raise ValueError("terminal snapshot absent from archive physical index")
            numeric = trajectories[key]
            if numeric["expression"][-1] != terminal["terminal_expression"] or numeric["source_sha256"][-1] != terminal["terminal_source_sha256"]:
                raise ValueError("numeric trajectory terminal differs from collected terminal")
            _, snapshot = _payload_record(
                {key: snapshots[key]}, key, terminal["terminal_source_sha256"], "snapshot"
            )
            formula_evidence = audit_formula_source(
                snapshot, terminal, evaluation_path=numeric["evaluation_path"]
            )
            if formula_evidence["discrepancy"]:
                discrepancies.append({
                    "logical_key": terminal["logical_key"], "algorithm": key[1],
                    "condition": key[0], "dataset_id": key[2], "seed": key[3],
                    "evaluation_path": numeric["evaluation_path"],
                    "resolution": formula_evidence["resolution"],
                    "status": formula_evidence["status"],
                    "snapshot_equation": formula_evidence["snapshot_equation"],
                    "snapshot_canonical_expression": formula_evidence["snapshot_canonical_expression"],
                    "terminal_expression": terminal["terminal_expression"],
                    "terminal_source_sha256": terminal["terminal_source_sha256"],
                    "corrected_artifact_sha256": formula_evidence["corrected_artifact_sha256"],
                    "replay_error": formula_evidence.get("replay_error") or formula_evidence["snapshot_canonical_parse_error"],
                })
            plans.append(build_one(
                key=key, terminal=terminal, preflight=changed[key], old=old[key],
                snapshot_record=snapshots[key], result_record=results[key],
                archive_record=physical[member], evaluation_path=numeric["evaluation_path"],
                contract=contract,
                gt_variables=gt_variables, gt_targets=gt_targets, probes=probes,
            ))
        except (KeyError, TypeError, ValueError, builder.CleanTaskBuilderError) as exc:
            unresolved.append({"logical_key": terminal["logical_key"], "reason": type(exc).__name__, "detail": str(exc)[:500]})
    if len(plans) + len(unresolved) != expected_count:
        raise AssertionError("plan/unresolved partition is incomplete")
    paths = {
        "plan": output / "pred_simplify_plan.jsonl",
        "unresolved": output / "unresolved.csv",
        "discrepancy": output / "discrepancy.csv",
        "manifest": output / "manifest.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite existing plan output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with paths["plan"].open("w", encoding="utf-8") as handle:
        for record in plans:
            handle.write(canonical_json(record) + "\n")
    with paths["unresolved"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("logical_key", "reason", "detail"))
        writer.writeheader()
        writer.writerows(unresolved)
    discrepancy_fields = (
        "logical_key", "algorithm", "condition", "dataset_id", "seed",
        "evaluation_path", "resolution", "status", "snapshot_equation",
        "snapshot_canonical_expression", "terminal_expression",
        "terminal_source_sha256", "corrected_artifact_sha256", "replay_error",
    )
    with paths["discrepancy"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=discrepancy_fields)
        writer.writeheader()
        writer.writerows(discrepancies)
    manifest = {
        "schema_version": "current_terminal_opus_prediction_plan.v1",
        "api_invoked": False,
        "formal_ready": False,
        "requested_count": expected_count,
        "planned_count": len(plans),
        "unresolved_count": len(unresolved),
        "discrepancy_count": len(discrepancies),
        "discrepancy_by_resolution": dict(sorted(Counter(row["resolution"] for row in discrepancies).items())),
        "unresolved_reasons": dict(sorted(Counter(row["reason"] for row in unresolved).items())),
        "stable_task_id_changed_count": sum(
            row["source_binding"]["stable_evaluation_task_id"] != row["source_binding"]["actual_selected_task_id"]
            for row in plans
        ),
        "prompt_sha256": contract.prompt_sha256,
        "schema_sha256": contract.schema_sha256,
        "dataset_probes_sha256": probes_sha,
        "inputs": {str(path): sha_file(path) for path in (
            collection / "terminal_inputs.jsonl.gz", preflight,
            collection / "terminal_snapshot_payloads.jsonl.gz",
            collection / "selected_result_payloads.jsonl.gz", archive,
            *(TRAJECTORIES / condition / "native_trajectories.jsonl.gz" for condition in CONDITIONS),
        )},
        "outputs": {name: sha_file(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--formal", type=Path, default=FORMAL)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-count", type=int, default=2336)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
