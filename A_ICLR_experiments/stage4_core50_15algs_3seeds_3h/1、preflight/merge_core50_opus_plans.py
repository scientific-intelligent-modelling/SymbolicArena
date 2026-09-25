import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiment_paths import relocated_path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import (
    _build_task_definition,
    _load_prompt_schema,
    GT_PRIORITY,
    PRED_PRIORITY,
)


sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = Path(__file__).resolve().parent
WORK = ROOT / ".agent/work/GOAL-CORE50"
CONDITIONS = ("clean", "noise001", "noise005")
PRED_ID_TO_RUN = {
    "clean": "clean",
    "noise001": "noise001",
    "noise005": "noise005",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.candidate")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(canonical_json(row) + "\n")
            count += 1
    temporary.replace(path)
    return count, sha256_file(path)


def historical_task(record: dict[str, Any], *, contract: Any) -> dict[str, Any]:
    request = dict(record["request"])
    task_type = str(record["task_type"])
    condition = str(request.get("noise_tag") or "clean")
    priority = GT_PRIORITY if task_type == "gt_simplify" else PRED_PRIORITY
    task = _build_task_definition(
        logical_id=str(record["logical_id"]),
        task_type=task_type,
        priority=priority,
        request=request,
        evidence_hash=str(request["evidence_hash"]),
        contract=contract,
        condition=condition,
    )
    if task.evaluation_key != record["evaluation_key"]:
        raise ValueError(f"{record['logical_id']}: 无法按历史 request 重建 evaluation_key")
    return task.to_json_record()


def _index_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        logical_id = str(row["logical_id"])
        if logical_id in result:
            raise ValueError(f"{label} logical_id 重复: {logical_id}")
        result[logical_id] = row
    return result


def merge_plans(
    *,
    plan_root: Path,
    gt_plan_root: Path,
    cache_audit_path: Path,
    bindings_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"merged plan 输出目录非空，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    if not isinstance(cache_audit, dict):
        raise ValueError("Opus cache audit 顶层必须为object")
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    bindings_by_key = {
        str(item["evaluation_key"]): item
        for item in bindings
        if item.get("task_type") in {"pred_simplify", "gt_simplify"}
    }
    cache_rows = cache_audit["cache_rows"]
    if len(cache_rows) != int(cache_audit["reusable_cache_count"]):
        raise ValueError("cache audit reusable_cache_count 与 rows 数量不一致")
    cache_row_by_logical = {str(row["logical_id"]): row for row in cache_rows}
    if len(cache_row_by_logical) != len(cache_rows):
        raise ValueError("cache audit logical_id 重复")
    contract = _load_prompt_schema(ROOT)

    historical_pred: dict[str, dict[str, Any]] = {}
    historical_gt: dict[str, dict[str, Any]] = {}
    cache_manifest_rows = []
    cache_transport_counts: Counter[str] = Counter()
    for cache in cache_rows:
        binding = bindings_by_key.get(str(cache["historical_evaluation_key"]))
        if binding is None:
            raise ValueError(f"cache binding 缺少 evaluation_key: {cache['logical_id']}")
        artifact_path = relocated_path(str(binding["destination"]))
        if sha256_file(artifact_path) != cache["cached_artifact_sha256"]:
            raise ValueError(f"cached artifact SHA256 在 plan merge 前漂移: {artifact_path}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        task_row = historical_task(artifact, contract=contract)
        if artifact["task_type"] == "pred_simplify":
            logical_id = str(artifact["logical_id"])
            if logical_id in historical_pred:
                raise ValueError(f"历史 prediction cache logical_id 重复: {logical_id}")
            historical_pred[logical_id] = task_row
        else:
            dataset_id = str(artifact["request"]["dataset_id"])
            if dataset_id in historical_gt:
                raise ValueError(f"历史 GT cache dataset_id 重复: {dataset_id}")
            historical_gt[dataset_id] = task_row
        transport = str(artifact["metadata"].get("transport_version"))
        cache_transport_counts[transport] += 1
        cache_manifest_rows.append({
            "task_type": artifact["task_type"],
            "logical_id": artifact["logical_id"],
            "historical_evaluation_key": artifact["evaluation_key"],
            "current_evaluation_key": cache["current_evaluation_key"],
            "source_result_sha256": cache["source_result_sha256"],
            "cached_artifact_path": str(artifact_path.resolve()),
            "cached_artifact_sha256": cache["cached_artifact_sha256"],
            "historical_model": artifact["metadata"].get("requested_model"),
            "historical_transport": transport,
            "reuse_basis": "source_result_sha256_and_expression_match",
            "network_request": False,
        })

    summary: dict[str, Any] = {
        "schema": "core50.opus_plan_merge.v1",
        "status": "candidate",
        "plan_root": str(plan_root.resolve()),
        "gt_plan_root": str(gt_plan_root.resolve()),
        "cache_audit_path": str(cache_audit_path.resolve()),
        "cache_audit_sha256": sha256_file(cache_audit_path),
        "cache_bindings_path": str(bindings_path.resolve()),
        "cache_bindings_sha256": sha256_file(bindings_path),
        "historical_cache_count": len(cache_manifest_rows),
        "historical_cache_transport_counts": dict(sorted(cache_transport_counts.items())),
        "conditions": {},
    }

    mapped_historical_pred: set[str] = set()
    for condition in CONDITIONS:
        current_callable = _index_rows(
            read_jsonl(plan_root / f"{condition}_pred_simplify_tasks.jsonl"),
            label=f"{condition} current pred callable",
        )
        current_full_rows = read_jsonl(plan_root / f"{condition}_pred_full_plan.jsonl")
        current_full = _index_rows(current_full_rows, label=f"{condition} current pred full")
        cache_ids = {}
        for logical_id, task in historical_pred.items():
            if logical_id not in current_callable:
                continue
            current_request = current_callable[logical_id]["request"]
            historical = cache_row_by_logical[logical_id]
            if not (
                historical["expression_match"]
                and historical["prompt_schema_match"]
                and historical["request_identity_match"]
                and current_request.get("ast_source_evidence", {}).get("result_raw_sha256")
                == historical["source_result_sha256"]
            ):
                raise ValueError(f"{logical_id}: cache audit 与当前 plan binding 不一致")
            cache_ids[logical_id] = task
        mapped_historical_pred.update(cache_ids)

        merged_callable = []
        pending_api = []
        for logical_id, current_row in current_callable.items():
            cached = cache_ids.get(logical_id)
            if cached is None:
                merged_callable.append(current_row)
                pending_api.append(current_row)
            else:
                merged_callable.append(cached)

        merged_full = []
        for logical_id, row in current_full.items():
            if logical_id in cache_ids:
                merged_full.append(cache_ids[logical_id])
            else:
                merged_full.append(row)

        order = lambda row: (int(row.get("priority", PRED_PRIORITY)), str(row["logical_id"]))
        merged_callable.sort(key=order)
        merged_full.sort(key=order)
        pending_api.sort(key=order)

        callable_path = output_root / f"{condition}_pred_merged_callable.jsonl"
        full_path = output_root / f"{condition}_pred_merged_full_plan.jsonl"
        api_path = output_root / f"{condition}_pred_pending_api.jsonl"
        callable_count, callable_sha = write_jsonl(callable_path, merged_callable)
        full_count, full_sha = write_jsonl(full_path, merged_full)
        api_count, api_sha = write_jsonl(api_path, pending_api)
        summary["conditions"][condition] = {
            "current_callable_count": len(current_callable),
            "current_full_count": len(current_full),
            "historical_cache_count": len(cache_ids),
            "pending_api_count": len(pending_api),
            "api_attempt_cap": len(pending_api) * 6,
            "retry_cap": len(pending_api) * 5,
            "merged_callable": {"path": str(callable_path), "count": callable_count, "sha256": callable_sha},
            "merged_full_plan": {"path": str(full_path), "count": full_count, "sha256": full_sha},
            "pending_api_plan": {"path": str(api_path), "count": api_count, "sha256": api_sha},
        }
    if mapped_historical_pred != set(historical_pred):
        missing_cache_ids = sorted(set(historical_pred) - mapped_historical_pred)
        raise ValueError(f"当前预测计划缺少历史 cache logical_id: {missing_cache_ids[:10]}")

    current_gt = read_jsonl(gt_plan_root / "gt_simplify_tasks.jsonl")
    gt_cache = {}
    for dataset_id, row in historical_gt.items():
        if dataset_id in gt_cache:
            raise ValueError(f"GT cache dataset_id 重复: {dataset_id}")
        gt_cache[dataset_id] = row
    current_gt_by_name = {str(row["request"]["dataset_id"]): row for row in current_gt}
    if len(current_gt_by_name) != 50:
        raise ValueError(f"当前 GT plan 必须为50项，实际为{len(current_gt_by_name)}")
    merged_gt = []
    pending_gt = []
    mapped_gt_cache: set[str] = set()
    for dataset_id, row in current_gt_by_name.items():
        cached = gt_cache.get(dataset_id)
        if cached is None:
            merged_gt.append(row)
            pending_gt.append(row)
        else:
            merged_gt.append(cached)
            mapped_gt_cache.add(dataset_id)
    if mapped_gt_cache != set(gt_cache):
        raise ValueError(f"当前 Ground Truth plan 缺少历史 cache dataset_id: {sorted(set(gt_cache)-mapped_gt_cache)}")
    merged_gt_full = list(merged_gt)
    gt_order = lambda row: (int(row.get("priority", GT_PRIORITY)), str(row["logical_id"]))
    merged_gt.sort(key=gt_order)
    merged_gt_full.sort(key=gt_order)
    pending_gt.sort(key=gt_order)

    gt_callable_path = output_root / "gt_merged_callable.jsonl"
    gt_full_path = output_root / "gt_merged_full_plan.jsonl"
    gt_api_path = output_root / "gt_pending_api.jsonl"
    gt_callable_count, gt_callable_sha = write_jsonl(gt_callable_path, merged_gt)
    gt_full_count, gt_full_sha = write_jsonl(gt_full_path, merged_gt_full)
    gt_api_count, gt_api_sha = write_jsonl(gt_api_path, pending_gt)
    summary["ground_truth"] = {
        "current_callable_count": len(current_gt),
        "historical_cache_count": len(gt_cache),
        "pending_api_count": len(pending_gt),
        "api_attempt_cap": len(pending_gt) * 6,
        "retry_cap": len(pending_gt) * 5,
        "merged_callable": {"path": str(gt_callable_path), "count": gt_callable_count, "sha256": gt_callable_sha},
        "merged_full_plan": {"path": str(gt_full_path), "count": gt_full_count, "sha256": gt_full_sha},
        "pending_api_plan": {"path": str(gt_api_path), "count": gt_api_count, "sha256": gt_api_sha},
    }

    cache_manifest_path = output_root / "historical_cache_rebind.jsonl"
    cache_manifest_count, cache_manifest_sha = write_jsonl(cache_manifest_path, cache_manifest_rows)
    summary["historical_cache_manifest"] = {
        "path": str(cache_manifest_path),
        "count": cache_manifest_count,
        "sha256": cache_manifest_sha,
    }
    pending_total = sum(item["pending_api_count"] for item in summary["conditions"].values()) + gt_api_count
    summary["api_budget"] = {
        "pending_api_task_count": pending_total,
        "max_retries_per_task": 5,
        "global_retry_cap": pending_total * 5,
        "max_total_attempts": pending_total * 6,
    }
    report_path = output_root / "merge_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", type=Path, default=WORK / "opus_plan_candidates_collected_v1")
    parser.add_argument("--cache-audit", type=Path, default=WORK / "opus_cache_audit.json")
    parser.add_argument("--bindings", type=Path, default=PREFLIGHT / "reuse_support/opus_bindings.json")
    parser.add_argument("--gt-plan-root", type=Path, default=WORK / "opus_plan_candidates")
    parser.add_argument("--output-root", type=Path, default=WORK / "opus_plan_merge_collected_v2")
    args = parser.parse_args()
    summary = merge_plans(
        plan_root=args.plan_root,
        gt_plan_root=args.gt_plan_root,
        cache_audit_path=args.cache_audit,
        bindings_path=args.bindings,
        output_root=args.output_root,
    )
    print(json.dumps({
        "pending_api_task_count": summary["api_budget"]["pending_api_task_count"],
        "global_retry_cap": summary["api_budget"]["global_retry_cap"],
        "max_total_attempts": summary["api_budget"]["max_total_attempts"],
        "historical_cache_count": summary["historical_cache_count"],
        "pending_by_condition": {condition: row["pending_api_count"] for condition, row in summary["conditions"].items()},
        "pending_ground_truth": summary["ground_truth"]["pending_api_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
