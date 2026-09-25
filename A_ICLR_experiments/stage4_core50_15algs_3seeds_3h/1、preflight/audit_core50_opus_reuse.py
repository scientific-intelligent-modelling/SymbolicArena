import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_paths import relocated_path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    API_TRANSPORT_VERSION,
    CONTRACT_CANONICAL_MODEL,
    CONTRACT_EFFORT,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import (
    _build_task_definition,
    _load_prompt_schema,
    PRED_PRIORITY,
)


sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = Path(__file__).resolve().parent
WORK = ROOT / ".agent/work/GOAL-CORE50"
CONDITIONS = ("clean", "noise001", "noise005")
PRED_ID = re.compile(r"^pred_simplify::([a-z0-9]+)::(g\d{4})::s(520|521|522)::(clean|noise001|noise005)$")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON 行必须为object")
            rows.append(row)
    return rows


def load_current_plans(work: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = work / f"opus_plan_candidates/{condition}_pred_simplify_tasks.jsonl"
        for row in read_jsonl(path):
            logical_id = str(row["logical_id"])
            if logical_id in predictions:
                raise ValueError(f"当前 prediction plan logical_id 重复: {logical_id}")
            predictions[logical_id] = row
    ground_truth_rows = read_jsonl(work / "opus_plan_candidates/gt_simplify_tasks.jsonl")
    ground_truth = {str(row["request"]["dataset_id"]): row for row in ground_truth_rows}
    if len(ground_truth) != 50:
        raise ValueError(f"当前 Ground Truth plan 必须有50项，实际为{len(ground_truth)}")
    return predictions, ground_truth


def result_index(path: Path) -> dict[tuple[str, str, str, int], dict[str, str]]:
    rows = {}
    for row in read_csv(path):
        if row["training_state"] != "copied":
            continue
        key = (row["noise"], row["algorithm"].lower(), row["dataset_id"], int(row["seed"]))
        if key in rows:
            raise ValueError(f"reuse manifest run key 重复: {key}")
        rows[key] = row
    return rows


def task_from_historical_record(record: dict[str, Any], *, contract: Any) -> Any:
    request = dict(record["request"])
    logical_id = str(record["logical_id"])
    task_type = str(record["task_type"])
    condition = str(request.get("noise_tag") or "clean")
    if task_type == "gt_simplify":
        priority = 10
    elif task_type == "pred_simplify":
        priority = PRED_PRIORITY
    else:
        raise ValueError(f"历史 cache task_type 无法使用: {task_type}")
    task = _build_task_definition(
        logical_id=logical_id,
        task_type=task_type,
        priority=priority,
        request=request,
        evidence_hash=str(request["evidence_hash"]),
        contract=contract,
        condition=condition,
    )
    return task


def audit(
    *,
    bindings_path: Path,
    reuse_manifest_path: Path,
    work: Path,
    report_path: Path,
) -> dict[str, Any]:
    current_predictions, current_gt = load_current_plans(work)
    current_runs = result_index(reuse_manifest_path)
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    if not isinstance(bindings, list):
        raise ValueError("Opus bindings 根必须为数组")

    contract = _load_prompt_schema(ROOT)
    counts: Counter[str] = Counter()
    transports: Counter[str] = Counter()
    models: Counter[str] = Counter()
    request_differences: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []

    for binding in bindings:
        task_type = str(binding["task_type"])
        if task_type not in {"pred_simplify", "gt_simplify"}:
            continue
        artifact_path = relocated_path(str(binding["destination"]))
        artifact_bytes = artifact_path.read_bytes()
        artifact_sha = digest_bytes(artifact_bytes)
        if artifact_sha != binding["sha256"]:
            raise ValueError(f"cached Opus artifact SHA256 漂移: {artifact_path}")
        artifact = json.loads(artifact_bytes.decode("utf-8"))
        if artifact.get("evaluation_key") != binding["evaluation_key"]:
            raise ValueError(f"cached Opus evaluation_key 不一致: {artifact_path}")
        if artifact.get("logical_id") != binding["logical_id"]:
            raise ValueError(f"cached Opus logical_id 不一致: {artifact_path}")

        metadata = artifact["metadata"]
        request = artifact["request"]
        historical_task = task_from_historical_record(artifact, contract=contract)
        historical_key_rebuilt = historical_task.evaluation_key == artifact["evaluation_key"]
        request_sha_matches = metadata.get("request_sha256") == digest_bytes(
            canonical_json(request).encode("utf-8")
        )
        if not historical_key_rebuilt or not request_sha_matches:
            invalid.append({
                "logical_id": binding["logical_id"],
                "reason": "historical_task_fingerprint_reconstruction_failed",
                "historical_key_rebuilt": historical_key_rebuilt,
                "request_sha_matches": request_sha_matches,
            })
            counts[f"{task_type}:historical_fingerprint_invalid"] += 1
            continue
        validation_ok = artifact.get("validation", {}).get("ok") is True
        output = artifact.get("structured_output")
        if not validation_ok or not isinstance(output, dict):
            invalid.append({"logical_id": binding["logical_id"], "reason": "historical_validation_failed"})
            counts[f"{task_type}:validation_failed"] += 1
            continue

        current_row = None
        source_result_sha = None
        expression_match = False
        prompt_schema_match = False
        request_identity_match = False
        direct_api_match = (
            metadata.get("requested_model") == CONTRACT_CANONICAL_MODEL
            and metadata.get("requested_effort") == CONTRACT_EFFORT
            and metadata.get("transport_version") == API_TRANSPORT_VERSION
        )

        if task_type == "pred_simplify":
            match = PRED_ID.fullmatch(str(binding["logical_id"]))
            if match is None:
                raise ValueError(f"cached prediction logical_id 非法: {binding['logical_id']}")
            algorithm, dataset_id, seed_text, condition = match.groups()
            run = current_runs.get((condition, algorithm, dataset_id, int(seed_text)))
            if run is None:
                raise ValueError(f"cached prediction 无对应 reuse run: {binding['logical_id']}")
            source_result_sha = str(run["result_sha256"])
            actual_source_result_sha = request.get("ast_source_evidence", {}).get("result_raw_sha256")
            if actual_source_result_sha != source_result_sha:
                invalid.append({
                    "logical_id": binding["logical_id"],
                    "reason": "source_result_sha256_mismatch",
                    "historical": actual_source_result_sha,
                    "current": source_result_sha,
                })
                counts["pred_simplify:source_sha_mismatch"] += 1
                continue
            current_row = current_predictions.get(str(binding["logical_id"]))
            if current_row is None:
                raise ValueError(f"cached prediction 当前 plan 缺少 logical_id: {binding['logical_id']}")
            current_request = current_row["request"]
            expression_match = request.get("expression") == current_request.get("expression")
            request_identity_match = all(
                request.get(field) == current_request.get(field)
                for field in ("algorithm_slug", "dataset_id", "dataset_index", "noise_tag", "seed", "task_id")
            )
            prompt_schema_match = (
                metadata.get("prompt_sha256") == current_row.get("prompt_sha256")
                and metadata.get("schema_sha256") == current_row.get("schema_sha256")
            )
            changed = sorted(
                key for key in set(request) | set(current_request)
                if request.get(key) != current_request.get(key)
            )
            for field in changed:
                request_differences[f"pred_simplify:{field}"] += 1
        else:
            dataset_id = str(request.get("dataset_id"))
            current_row = current_gt.get(dataset_id)
            if current_row is None:
                raise ValueError(f"cached GT task 未命中当前 Ground Truth: {binding['logical_id']}")
            current_request = current_row["request"]
            expression_match = request.get("expression") == current_request.get("expression")
            request_identity_match = request.get("dataset_id") == current_request.get("dataset_id")
            prompt_schema_match = (
                metadata.get("prompt_sha256") == current_row.get("prompt_sha256")
                and metadata.get("schema_sha256") == current_row.get("schema_sha256")
            )
            old_source_sha = request.get("ast_source_evidence", {}).get("ground_truth_source_evidence_sha256")
            current_source_sha = current_request.get("ast_source_evidence", {}).get("ground_truth_source_evidence_sha256")
            source_result_sha = old_source_sha
            if old_source_sha != current_source_sha:
                invalid.append({
                    "logical_id": binding["logical_id"],
                    "reason": "ground_truth_source_evidence_sha256_mismatch",
                    "historical": old_source_sha,
                    "current": current_source_sha,
                })
                counts["gt_simplify:source_sha_mismatch"] += 1
                continue
            changed = sorted(
                key for key in set(request) | set(current_request)
                if request.get(key) != current_request.get(key)
            )
            for field in changed:
                request_differences[f"gt_simplify:{field}"] += 1

        if not expression_match or not prompt_schema_match or not request_identity_match:
            invalid.append({
                "logical_id": binding["logical_id"],
                "reason": "current_formula_or_prompt_binding_mismatch",
                "expression_match": expression_match,
                "prompt_schema_match": prompt_schema_match,
                "request_identity_match": request_identity_match,
            })
            counts[f"{task_type}:current_input_mismatch"] += 1
            continue

        exact_task_key_match = artifact["evaluation_key"] == current_row["evaluation_key"]
        counts[f"{task_type}:historical_validation_ok"] += 1
        counts[f"{task_type}:source_expression_prompt_schema_bound"] += 1
        counts[f"{task_type}:exact_current_evaluation_key"] += int(exact_task_key_match)
        counts[f"{task_type}:legacy_direct_api_contract_match"] += int(direct_api_match)
        transports[str(metadata.get("transport_version"))] += 1
        models[str(metadata.get("requested_model"))] += 1
        cache_rows.append({
            "task_type": task_type,
            "logical_id": binding["logical_id"],
            "historical_evaluation_key": artifact["evaluation_key"],
            "historical_key_rebuilt": historical_key_rebuilt,
            "request_sha_matches": request_sha_matches,
            "current_evaluation_key": current_row["evaluation_key"],
            "source_result_sha256": source_result_sha,
            "cached_artifact_sha256": artifact_sha,
            "expression_match": expression_match,
            "prompt_schema_match": prompt_schema_match,
            "request_identity_match": request_identity_match,
            "exact_current_evaluation_key": exact_task_key_match,
            "historical_transport": metadata.get("transport_version"),
            "historical_model": metadata.get("requested_model"),
            "historical_effort": metadata.get("requested_effort"),
            "reusable_by_source_result_binding": True,
        })

    report = {
        "schema": "core50.opus_cache_audit.v1",
        "status": "passed" if not invalid else "needs_review",
        "binding_count": len(bindings),
        "reusable_cache_count": len(cache_rows),
        "invalid_count": len(invalid),
        "counts": dict(sorted(counts.items())),
        "transport_counts": dict(sorted(transports.items())),
        "model_counts": dict(sorted(models.items())),
        "request_difference_counts": dict(sorted(request_differences.items())),
        "invalid_examples": invalid[:50],
        "cache_rows": cache_rows,
        "inputs": {
            "opus_bindings": {"path": str(bindings_path), "sha256": digest_file(bindings_path)},
            "reuse_manifest": {"path": str(reuse_manifest_path), "sha256": digest_file(reuse_manifest_path)},
            "current_pred_plans": {
                condition: str(work / f"opus_plan_candidates/{condition}_pred_simplify_tasks.jsonl")
                for condition in CONDITIONS
            },
            "current_gt_plan": str(work / "opus_plan_candidates/gt_simplify_tasks.jsonl"),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, default=PREFLIGHT / "reuse_support/opus_bindings.json")
    parser.add_argument("--reuse-manifest", type=Path, default=PREFLIGHT / "reuse_manifest.csv")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--report", type=Path, default=WORK / "opus_cache_audit.json")
    args = parser.parse_args()
    report = audit(
        bindings_path=args.bindings,
        reuse_manifest_path=args.reuse_manifest,
        work=args.work,
        report_path=args.report,
    )
    print(json.dumps({"status": report["status"], "reusable_cache_count": report["reusable_cache_count"], "invalid_count": report["invalid_count"], "counts": report["counts"], "transport_counts": report["transport_counts"], "model_counts": report["model_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
