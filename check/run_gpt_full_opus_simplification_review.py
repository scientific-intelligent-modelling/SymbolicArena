#!/usr/bin/env python3
"""用 GPT-5.6-Sol 全量复核 Core50 发布包中的 Opus5 化简结果。

发布清单固定包含 6800 条记录。100 条 ``outcome=unable`` 没有 Opus 候选，作为
``non_applicable`` 保留在 inventory 中但不请求 API；其余 6700 条逐条送审。脚本复用
三模型抽查器的 OpenAI Responses 传输、共同定义域近似等价合同、严格 JSON schema、
凭据脱敏、断点记录和请求预算实现。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check import run_three_model_simplification_review as base_review  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO_ROOT / "AAAI_experiments/Core50_final_20260914"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/audits"
    / "gpt56_full_opus_simplification_review_6800_20260915"
)
CONDITIONS = base_review.CONDITIONS
MODEL = base_review.MODELS["gpt-5.6-sol"]
REVIEW_POLICY = "common_domain_approx"
PROMPT_VERSION = "gpt56_full_opus_review.common_domain_approx.native_evidence.v1"
EXPECTED_GROUND_TRUTH = 50
EXPECTED_PER_CONDITION = 2250
EXPECTED_TOTAL = EXPECTED_GROUND_TRUTH + len(CONDITIONS) * EXPECTED_PER_CONDITION
EXPECTED_REVIEWABLE = 6700
CONCURRENCY = 32
MAX_ATTEMPTS = base_review.MAX_ATTEMPTS_PER_TASK
CHECKPOINT_INTERVAL = 100


class FullReviewError(base_review.ReviewError):
    """全量评审输入或断点状态不满足冻结合同。"""


def physical_request_cap(reviewable_count: int) -> int:
    if reviewable_count <= 0:
        raise FullReviewError("reviewable_count 必须是正整数")
    return math.ceil(1.5 * reviewable_count)


def _apply_full_policy(task: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(task)
    base_key = str(row["review_key"])
    row["base_review_key"] = base_key
    row["review_policy"] = REVIEW_POLICY
    row["prompt_version"] = PROMPT_VERSION
    row["review_key"] = base_review._sha256_bytes(
        base_review.canonical_json(
            {
                "base_review_key": base_key,
                "review_policy": REVIEW_POLICY,
                "prompt_version": PROMPT_VERSION,
            }
        ).encode("utf-8")
    )
    return row


def _source_specs(release_root: Path) -> list[tuple[str, str, Path]]:
    return [
        ("ground_truth", "clean", release_root / "ground_truth/opus5_ground_truth.jsonl"),
        *[
            ("prediction", condition, release_root / f"results/{condition}/opus5_prediction.jsonl")
            for condition in CONDITIONS
        ],
    ]


def _load_run_final_indexes(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    indexes: dict[str, dict[str, dict[str, str]]] = {}
    for condition in CONDITIONS:
        path = root / f"results/{condition}/run_final.csv"
        if not path.is_file():
            raise FullReviewError(f"缺少 native 证据表: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        index = {str(row.get("pred_evaluation_key") or ""): row for row in rows}
        if len(index) != len(rows) or "" in index:
            raise FullReviewError(f"{path} pred_evaluation_key 缺失或重复")
        indexes[condition] = index
    return indexes


def _load_gplearn_feature_names(root: Path, condition: str) -> dict[tuple[str, str], list[str]]:
    path = root / f"results/{condition}/raw_results.jsonl.gz"
    if not path.is_file():
        raise FullReviewError(f"缺少 gplearn 变量证据: {path}")
    output: dict[tuple[str, str], list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                envelope = json.loads(line)
                raw_text = envelope.get("result", {}).get("raw_text")
                payload = json.loads(raw_text) if isinstance(raw_text, str) else {}
            except json.JSONDecodeError as error:
                raise FullReviewError(f"{path}:{line_number} raw_text 不是 JSON") from error
            if payload.get("tool") != "gplearn":
                continue
            names = payload.get("feature_names")
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise FullReviewError(f"{path}:{line_number} gplearn feature_names 非法")
            key = (str(payload.get("dataset") or ""), str(payload.get("seed") or ""))
            if not all(key) or key in output:
                raise FullReviewError(f"{path}:{line_number} gplearn 任务身份缺失或重复")
            output[key] = names
    return output


def build_full_manifest(
    release_root: str | Path,
    *,
    expected_ground_truth: int | None = EXPECTED_GROUND_TRUTH,
    expected_per_condition: int | None = EXPECTED_PER_CONDITION,
) -> list[dict[str, Any]]:
    """按发布文件与行号构造 1:1 inventory，不做表达式去重。"""

    root = Path(release_root).expanduser().resolve()
    run_final_indexes = _load_run_final_indexes(root)
    gplearn_features = {
        condition: _load_gplearn_feature_names(root, condition) for condition in CONDITIONS
    }
    tasks: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for source_kind, condition, path in _source_specs(root):
        if not path.is_file():
            raise FullReviewError(f"缺少发布输入: {path}")
        source_sha256 = base_review._sha256_file(path)
        relative_path = str(path.relative_to(root))
        for line_number, row in base_review._read_jsonl(path):
            structured = row.get("structured_output")
            if not isinstance(structured, Mapping):
                raise FullReviewError(f"{path}:{line_number} 缺少 structured_output")
            outcome = str(structured.get("outcome") or "")
            if outcome not in {"simplified", "unchanged", "unable"}:
                raise FullReviewError(f"{path}:{line_number} outcome 非法: {outcome!r}")
            original = row.get("input_expression")
            effective = row.get("effective_expression") or structured.get(
                "simplified_expression"
            )
            if not isinstance(original, str) or not original.strip():
                raise FullReviewError(f"{path}:{line_number} 缺少 input_expression")
            if not isinstance(effective, str) or not effective.strip():
                raise FullReviewError(f"{path}:{line_number} 缺少 effective_expression")
            logical_id = str(row.get("logical_id") or "")
            native_evidence: dict[str, Any] | None = None
            if source_kind == "prediction":
                algorithm, dataset_id, seed = base_review._prediction_identity(
                    logical_id, condition
                )
                linked = run_final_indexes[condition].get(str(row.get("evaluation_key") or ""))
                if linked is None:
                    raise FullReviewError(
                        f"{path}:{line_number} evaluation_key 无法关联 run_final"
                    )
                linked_algorithm = str(
                    linked.get("algorithm_slug") or linked.get("algorithm") or ""
                )
                if linked_algorithm.lower() != algorithm.lower() or linked.get("condition") != condition:
                    raise FullReviewError(f"{path}:{line_number} run_final 身份不一致")
                dataset_id = str(linked.get("dataset_id") or "")
                seed = str(linked.get("seed") or seed)
                if algorithm == "gplearn" and outcome != "unable" and original != effective:
                    feature_names = gplearn_features[condition].get((dataset_id, seed))
                    if feature_names is None:
                        raise FullReviewError(
                            f"{path}:{line_number} 找不到 gplearn feature_names"
                        )
                    native_evidence = {
                        "source": f"results/{condition}/run_final.csv",
                        "pred_evaluation_key": linked["pred_evaluation_key"],
                        "raw_original_equation": linked["raw_original_equation"],
                        "raw_original_equation_sha256": linked["raw_original_equation_sha256"],
                        "canonical_named_expression": linked["canonical_named_expression"],
                        "canonical_named_expression_sha256": linked[
                            "canonical_named_expression_sha256"
                        ],
                        "feature_names": feature_names,
                        "indexed_variable_mapping": {
                            f"X{index}": name for index, name in enumerate(feature_names)
                        },
                        "operator_contract": (
                            "gplearn prefix source is authoritative; div(a,b)=a/b when "
                            "abs(b)>0.001 else 1; log(a)=log(abs(a)) when abs(a)>0.001 "
                            "else 0; sqrt(a)=sqrt(abs(a))"
                        ),
                    }
            else:
                algorithm = "ground_truth"
                dataset_id = base_review._ground_truth_identity(logical_id)
                seed = ""
            source_record_sha256 = base_review._sha256_bytes(
                base_review.canonical_json(row).encode("utf-8")
            )
            base_key = base_review._sha256_bytes(
                base_review.canonical_json(
                    {
                        "source_kind": source_kind,
                        "condition": condition,
                        "source_file": relative_path,
                        "source_line": line_number,
                        "source_record_sha256": source_record_sha256,
                        "logical_id": logical_id,
                        "evaluation_key": str(row.get("evaluation_key") or ""),
                        "original_expression": original,
                        "effective_expression": effective,
                        "native_evidence": native_evidence,
                    }
                ).encode("utf-8")
            )
            occurrence = {
                "source_kind": source_kind,
                "task_type": str(row.get("task_type") or ""),
                "algorithm": algorithm,
                "condition": condition,
                "dataset_id": dataset_id,
                "seed": seed,
                "logical_id": logical_id,
                "evaluation_key": str(row.get("evaluation_key") or ""),
                "source_file": relative_path,
                "source_file_sha256": source_sha256,
                "source_line": line_number,
                "source_record_sha256": source_record_sha256,
            }
            task = {
                "sample_id": f"review_{len(tasks) + 1:06d}",
                "review_key": base_key,
                "pair_sha256": base_review._sha256_bytes(
                    base_review.canonical_json([original, effective]).encode("utf-8")
                ),
                "original_expression": original,
                "effective_expression": effective,
                "outcome": outcome,
                "review_applicable": outcome != "unable",
                "non_applicable_reason": (
                    "unable_no_opus_candidate" if outcome == "unable" else ""
                ),
                "occurrences": [occurrence],
            }
            if native_evidence is not None:
                task["native_evidence"] = native_evidence
            tasks.append(_apply_full_policy(task))
            source_counts["ground_truth" if source_kind == "ground_truth" else condition] += 1

    if expected_ground_truth is not None and source_counts["ground_truth"] != expected_ground_truth:
        raise FullReviewError(
            f"Ground Truth 条数为 {source_counts['ground_truth']}，预期 {expected_ground_truth}"
        )
    if expected_per_condition is not None:
        for condition in CONDITIONS:
            if source_counts[condition] != expected_per_condition:
                raise FullReviewError(
                    f"{condition} 条数为 {source_counts[condition]}，预期 {expected_per_condition}"
                )
    if len({task["sample_id"] for task in tasks}) != len(tasks):
        raise FullReviewError("inventory sample_id 不唯一")
    if len({task["review_key"] for task in tasks}) != len(tasks):
        raise FullReviewError("inventory review_key 不唯一")
    return tasks


def task_path(output_root: str | Path, sample_id: str) -> Path:
    return base_review._task_path(Path(output_root), MODEL.name, sample_id)


def render_full_prompt(task: Mapping[str, Any]) -> str:
    prompt = base_review.render_review_prompt(task, review_policy=REVIEW_POLICY)
    occurrence = task["occurrences"][0]
    algorithm = str(occurrence.get("algorithm") or "")
    supplement: dict[str, Any] = {
        "full_review_prompt_version": PROMPT_VERSION,
        "source_outcome": task.get("outcome"),
    }
    if task.get("native_evidence") is not None:
        supplement["authoritative_native_evidence"] = task["native_evidence"]
        supplement["instruction"] = (
            "Judge the proposed simplification against the authoritative native prefix program, "
            "using canonical_named_expression and indexed_variable_mapping only as its audited "
            "name mapping. Protected operator value semantics must be preserved."
        )
    elif algorithm in {"dso", "udsr"}:
        supplement["operator_contract"] = (
            "This release explicitly declares protected_operators=false for this algorithm; "
            "use ordinary written real-valued operator semantics."
        )
    return prompt + "\n\nFULL RELEASE EVIDENCE:\n" + json.dumps(
        supplement, ensure_ascii=False, sort_keys=True
    )


def _load_full_task_record(path: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "prompt_version": PROMPT_VERSION,
            "review_policy": REVIEW_POLICY,
            "logical_task_id": f"{PROMPT_VERSION}::{MODEL.name}::{task['sample_id']}",
            "sample_id": task["sample_id"],
            "review_key": task["review_key"],
            "pair_sha256": task["pair_sha256"],
            "model": MODEL.name,
            "status": "pending",
            "attempts": [],
        }
    record = _read_record(path)
    assert record is not None
    expected = (
        PROMPT_VERSION,
        MODEL.name,
        task["sample_id"],
        task["review_key"],
        task["pair_sha256"],
    )
    actual = (
        record.get("prompt_version"),
        record.get("model"),
        record.get("sample_id"),
        record.get("review_key"),
        record.get("pair_sha256"),
    )
    if actual != expected or not isinstance(record.get("attempts"), list):
        raise FullReviewError(f"断点记录与当前任务合同不一致: {path}")
    return record


def review_one_task(
    *,
    task: Mapping[str, Any],
    profile: base_review.ApiProfile,
    output_root: str | Path,
    budget: base_review.RequestBudget,
    transport: Callable[[base_review.HttpRequest, float], base_review.HttpResponse] = base_review.default_transport,
    timeout: float = 300.0,
    retry_delay_seconds: float = 2.0,
    allow_terminal_retry: bool = False,
) -> dict[str, Any]:
    path = task_path(output_root, str(task["sample_id"]))
    record = _load_full_task_record(path, task)
    if record.get("status") == "completed":
        return record
    attempts: list[dict[str, Any]] = record["attempts"]
    if allow_terminal_retry:
        if (
            record.get("status") != "failed"
            or len(attempts) != MAX_ATTEMPTS
            or any(attempt.get("status") != "failed" for attempt in attempts)
        ):
            raise FullReviewError("显式恢复仅允许已有两次终态失败的任务")
        attempt_limit = MAX_ATTEMPTS + 1
    else:
        attempt_limit = MAX_ATTEMPTS
    request = base_review.build_http_request(
        MODEL,
        profile,
        render_full_prompt(task),
        review_policy=REVIEW_POLICY,
    )
    safe_metadata = dict(request.safe_metadata)
    safe_metadata["prompt_version"] = PROMPT_VERSION
    request = base_review.HttpRequest(
        url=request.url,
        headers=request.headers,
        body=request.body,
        safe_metadata=safe_metadata,
    )
    while len(attempts) < attempt_limit:
        attempt_number = len(attempts) + 1
        ordinal = budget.reserve(record["logical_task_id"], attempt_number)
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "physical_request_ordinal": ordinal,
            "status": "reserved",
            "request": dict(request.safe_metadata),
        }
        attempts.append(attempt)
        record["status"] = "running"
        base_review._atomic_write_json(path, record)
        response: base_review.HttpResponse | None = None
        try:
            response = transport(request, timeout)
            if not 200 <= response.status < 300:
                raise FullReviewError(f"HTTP {response.status}")
            parsed = base_review.parse_review_json(
                base_review._response_text(MODEL, response.payload),
                review_policy=REVIEW_POLICY,
            )
            attempt.update(
                {
                    "status": "accepted",
                    "http_status": response.status,
                    "response_id": str(response.payload.get("id") or ""),
                    "response_model": str(response.payload.get("model") or ""),
                    "response_store_echo": response.payload.get("store", "not_returned"),
                    "usage": base_review._usage(MODEL, response.payload),
                }
            )
            record["review"] = parsed
            record["status"] = "completed"
            base_review._atomic_write_json(path, record)
            return record
        except BaseException as error:
            attempt.update(
                {
                    "status": "failed",
                    "error": base_review._safe_error(response, error),
                }
            )
            record["status"] = "failed"
            base_review._atomic_write_json(path, record)
            if attempt_number < attempt_limit and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
    return record


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullReviewError(f"断点记录损坏: {path}") from error
    if not isinstance(value, dict):
        raise FullReviewError(f"断点记录不是 JSON object: {path}")
    return value


def _record_state(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "pending"
    if record.get("status") == "completed":
        return "completed"
    attempts = record.get("attempts")
    if isinstance(attempts, list) and len(attempts) >= MAX_ATTEMPTS:
        return "failed"
    return "pending"


def pending_tasks(
    tasks: Sequence[Mapping[str, Any]], output_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(output_root)
    return [
        dict(task)
        for task in tasks
        if task.get("review_applicable") is True
        and _record_state(_read_record(task_path(root, str(task["sample_id"])))) == "pending"
    ]


_COUNT_FIELDS = (
    "expected",
    "applicable",
    "non_applicable",
    "reviewed",
    "completed",
    "failed",
    "pending",
    "pass",
    "fail",
    "undetermined",
    "unavailable",
)


def _empty_counts() -> dict[str, int]:
    return {field: 0 for field in _COUNT_FIELDS}


def _apply_task_counts(counts: dict[str, int], task: Mapping[str, Any], state: str, verdict: str) -> None:
    counts["expected"] += 1
    if task.get("review_applicable") is not True:
        counts["non_applicable"] += 1
        return
    counts["applicable"] += 1
    if state == "pending":
        counts["pending"] += 1
        return
    counts["reviewed"] += 1
    counts[state] += 1
    counts[verdict] += 1


def build_progress_summary(
    tasks: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    *,
    trigger: int | str,
) -> dict[str, Any]:
    """从持久化任务记录重建汇总，避免只相信进程内计数。"""

    root = Path(output_root)
    overall = _empty_counts()
    grouped: dict[str, dict[str, dict[str, int]]] = {
        "condition": {},
        "algorithm": {},
        "outcome": {},
        "condition_algorithm": {},
    }
    verdict_counts = {name: 0 for name in ("fail", "pass", "undetermined", "unavailable")}
    for task in tasks:
        occurrence = task["occurrences"][0]
        record = (
            _read_record(task_path(root, str(task["sample_id"])))
            if task.get("review_applicable") is True
            else None
        )
        state = "non_applicable" if task.get("review_applicable") is not True else _record_state(record)
        if state == "completed":
            review = record.get("review") if isinstance(record, Mapping) else None
            verdict = str(review.get("verdict")) if isinstance(review, Mapping) else "unavailable"
            if verdict not in verdict_counts:
                verdict = "unavailable"
        elif state == "failed":
            verdict = "unavailable"
        else:
            verdict = "unavailable"
        _apply_task_counts(overall, task, state, verdict)
        if state in {"completed", "failed"}:
            verdict_counts[verdict] += 1
        values = {
            "condition": str(occurrence["condition"]),
            "algorithm": str(occurrence["algorithm"]),
            "outcome": str(task["outcome"]),
            "condition_algorithm": f"{occurrence['condition']}::{occurrence['algorithm']}",
        }
        for dimension, value in values.items():
            counts = grouped[dimension].setdefault(value, _empty_counts())
            _apply_task_counts(counts, task, state, verdict)
    for counts in [overall, *(counts for dimension in grouped.values() for counts in dimension.values())]:
        counts["terminal"] = counts["reviewed"]
        counts["attempted_all"] = counts["pending"] == 0
    return {
        "schema_version": 1,
        "review_policy": REVIEW_POLICY,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL.name,
        "trigger": trigger,
        "overall": overall,
        "groups": {**grouped, "verdict": verdict_counts},
    }


def write_progress_checkpoint(
    output_root: str | Path,
    summary: Mapping[str, Any],
    *,
    checkpoint_number: int,
) -> Path:
    root = Path(output_root)
    path = root / "checkpoints" / f"checkpoint_{checkpoint_number:06d}.json"
    base_review._atomic_write_json(path, summary)
    base_review._atomic_write_json(root / "checkpoint_latest.json", summary)
    return path


def completion_status(overall: Mapping[str, Any]) -> str:
    return (
        "completed"
        if (
            int(overall.get("pending", -1)) == 0
            and int(overall.get("failed", -1)) == 0
            and int(overall.get("unavailable", -1)) == 0
        )
        else "partial"
    )


def _write_manifest(path: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(base_review.canonical_json(task) + "\n" for task in tasks)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FullReviewError("input_manifest.jsonl 已存在且与冻结发布输入不一致")
    base_review._atomic_write_text(path, content)


def prepare_manifest(release_root: str | Path, output_root: str | Path) -> list[dict[str, Any]]:
    tasks = build_full_manifest(release_root)
    if len(tasks) != EXPECTED_TOTAL:
        raise FullReviewError(f"inventory 共 {len(tasks)} 条，预期 {EXPECTED_TOTAL}")
    reviewable = sum(task["review_applicable"] is True for task in tasks)
    if reviewable != EXPECTED_REVIEWABLE:
        raise FullReviewError(f"可送审记录共 {reviewable} 条，预期 {EXPECTED_REVIEWABLE}")
    _write_manifest(Path(output_root) / "input_manifest.jsonl", tasks)
    return tasks


def _load_profile(args: argparse.Namespace) -> base_review.ApiProfile:
    if args.reuse_anthropic_token_for_openai:
        anthropic = base_review.load_anthropic_profile(args.anthropic_profile)
        return base_review.derive_openai_profile_from_anthropic(anthropic)
    return base_review.load_openai_profile(
        args.openai_profile, provider=args.openai_provider
    )


def _write_index(output_root: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for task in tasks:
        occurrence = task["occurrences"][0]
        path = task_path(output_root, str(task["sample_id"]))
        record = _read_record(path) if task["review_applicable"] else None
        state = "non_applicable" if not task["review_applicable"] else _record_state(record)
        review = record.get("review") if isinstance(record, Mapping) else None
        rows.append(
            {
                "sample_id": task["sample_id"],
                "condition": occurrence["condition"],
                "algorithm": occurrence["algorithm"],
                "dataset_id": occurrence["dataset_id"],
                "seed": occurrence["seed"],
                "outcome": task["outcome"],
                "review_applicable": task["review_applicable"],
                "status": state,
                "verdict": review.get("verdict", "") if isinstance(review, Mapping) else "",
                "attempt_count": len(record.get("attempts", [])) if isinstance(record, Mapping) else 0,
                "record_path": str(path.relative_to(output_root)) if path.exists() else "",
                "record_sha256": base_review._sha256_file(path) if path.exists() else "",
            }
        )
        if isinstance(record, Mapping):
            for attempt in record.get("attempts", []):
                usage = attempt.get("usage") or {}
                cost_rows.append(
                    {
                        "sample_id": task["sample_id"],
                        "attempt_number": attempt.get("attempt_number"),
                        "physical_request_ordinal": attempt.get("physical_request_ordinal"),
                        "status": attempt.get("status"),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                )
    base_review._write_csv(
        output_root / "review_index.csv",
        [
            "sample_id", "condition", "algorithm", "dataset_id", "seed", "outcome",
            "review_applicable", "status", "verdict", "attempt_count", "record_path",
            "record_sha256",
        ],
        rows,
    )
    base_review._write_csv(
        output_root / "cost_ledger.csv",
        [
            "sample_id", "attempt_number", "physical_request_ordinal", "status",
            "input_tokens", "output_tokens", "total_tokens",
        ],
        cost_rows,
    )
    base_review._write_csv(
        output_root / "failures.csv",
        [
            "sample_id", "condition", "algorithm", "dataset_id", "seed", "outcome",
            "review_applicable", "status", "verdict", "attempt_count", "record_path",
            "record_sha256",
        ],
        (row for row in rows if row["status"] == "failed" or row["verdict"] in {"fail", "undetermined"}),
    )


def run_preflight(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve()
    tasks = prepare_manifest(args.release_root, root)
    profile = _load_profile(args)
    report = {
        "status": "preflight_passed_no_network",
        "inventory_records": len(tasks),
        "reviewable_records": sum(task["review_applicable"] for task in tasks),
        "non_applicable_records": sum(not task["review_applicable"] for task in tasks),
        "non_applicable_reason": "unable_no_opus_candidate",
        "model": MODEL.name,
        "review_policy": REVIEW_POLICY,
        "prompt_version": PROMPT_VERSION,
        "concurrency": args.concurrency,
        "max_attempts_per_task": MAX_ATTEMPTS,
        "physical_request_cap": physical_request_cap(EXPECTED_REVIEWABLE),
        "stream": False,
        "store": False,
        "profile": dict(profile.safe_metadata),
    }
    base_review._atomic_write_json(root / "preflight.json", report)
    base_review._write_checksums(root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def run_batch(args: argparse.Namespace) -> int:
    if not 1 <= args.concurrency <= CONCURRENCY:
        raise FullReviewError(f"concurrency 必须位于 [1,{CONCURRENCY}]")
    if args.limit is not None and args.limit <= 0:
        raise FullReviewError("limit 必须是正整数")
    root = args.output_root.expanduser().resolve()
    tasks = prepare_manifest(args.release_root, root)
    profile = _load_profile(args)
    budget = base_review.RequestBudget(
        root / "request_budget.json",
        cap=physical_request_cap(EXPECTED_REVIEWABLE),
    )
    initial = build_progress_summary(tasks, root, trigger="resume_start")
    candidates = pending_tasks(tasks, root)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    reviewed = int(initial["overall"]["reviewed"])
    next_checkpoint = (reviewed // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
    execution_errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency, thread_name_prefix="gpt56-opus-review"
    ) as executor:
        futures = {
            executor.submit(
                review_one_task,
                task=task,
                profile=profile,
                output_root=root,
                budget=budget,
                timeout=args.timeout,
                retry_delay_seconds=args.retry_delay,
            ): task
            for task in candidates
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except BaseException as error:
                execution_errors.append(
                    f"{task['sample_id']}::{type(error).__name__}"
                )
            state = _record_state(_read_record(task_path(root, str(task["sample_id"]))))
            if state not in {"completed", "failed"}:
                continue
            reviewed += 1
            while reviewed >= next_checkpoint:
                progress = build_progress_summary(tasks, root, trigger=next_checkpoint)
                write_progress_checkpoint(
                    root, progress, checkpoint_number=next_checkpoint
                )
                print(
                    json.dumps(
                        {
                            "checkpoint": next_checkpoint,
                            "overall": progress["overall"],
                            "verdict": progress["groups"]["verdict"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_checkpoint += CHECKPOINT_INTERVAL

    final = build_progress_summary(tasks, root, trigger="invocation_end")
    final.update(
        {
            "status": completion_status(final["overall"]),
            "submitted_this_invocation": len(candidates),
            "execution_errors": execution_errors,
            "concurrency": args.concurrency,
            "physical_request_cap": physical_request_cap(EXPECTED_REVIEWABLE),
            "physical_requests_reserved": budget.used,
            "max_attempts_per_task": MAX_ATTEMPTS,
            "stream": False,
            "store": False,
        }
    )
    base_review._atomic_write_json(root / "summary.json", final)
    base_review._atomic_write_json(root / "checkpoint_latest.json", final)
    _write_index(root, tasks)
    base_review._write_checksums(root)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))
    return 0 if final["status"] == "completed" else 2


def resolve_explicit_task(
    tasks: Sequence[Mapping[str, Any]],
    *,
    sample_id: str | None,
    logical_id: str | None,
) -> dict[str, Any]:
    if (sample_id is None) == (logical_id is None):
        raise FullReviewError("必须且只能显式提供 sample_id 或 logical_id")
    matches = [
        task
        for task in tasks
        if (
            (sample_id is not None and task["sample_id"] == sample_id)
            or (
                logical_id is not None
                and task["occurrences"][0].get("logical_id") == logical_id
            )
        )
    ]
    if len(matches) != 1:
        identity = sample_id if sample_id is not None else logical_id
        raise FullReviewError(f"显式任务身份必须唯一命中，当前为 {len(matches)}: {identity}")
    task = dict(matches[0])
    if task.get("review_applicable") is not True:
        raise FullReviewError("显式任务没有可复核的 Opus 候选")
    return task


def run_retry_one(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve()
    tasks = prepare_manifest(args.release_root, root)
    task = resolve_explicit_task(
        tasks, sample_id=args.sample_id, logical_id=args.logical_id
    )
    profile = _load_profile(args)
    budget = base_review.RequestBudget(
        root / "request_budget.json",
        cap=physical_request_cap(EXPECTED_REVIEWABLE),
    )
    record = review_one_task(
        task=task,
        profile=profile,
        output_root=root,
        budget=budget,
        timeout=args.timeout,
        retry_delay_seconds=0,
        allow_terminal_retry=True,
    )
    final = build_progress_summary(tasks, root, trigger="explicit_terminal_retry")
    final.update(
        {
            "status": completion_status(final["overall"]),
            "retried_sample_id": task["sample_id"],
            "retried_logical_id": task["occurrences"][0]["logical_id"],
            "retry_result": record.get("status"),
            "physical_request_cap": physical_request_cap(EXPECTED_REVIEWABLE),
            "physical_requests_reserved": budget.used,
            "stream": False,
            "store": False,
        }
    )
    base_review._atomic_write_json(root / "summary.json", final)
    base_review._atomic_write_json(root / "checkpoint_latest.json", final)
    _write_index(root, tasks)
    base_review._write_checksums(root)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))
    return 0 if record.get("status") == "completed" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-5.6-Sol 32 并发全量复核 6800 条 Core50 Opus5 记录"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "retry-one"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
        subparser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        subparser.add_argument("--concurrency", type=int, default=CONCURRENCY)
        subparser.add_argument(
            "--anthropic-profile",
            type=Path,
            default=base_review.DEFAULT_ANTHROPIC_PROFILE,
        )
        subparser.add_argument(
            "--openai-profile", type=Path, default=base_review.DEFAULT_OPENAI_PROFILE
        )
        subparser.add_argument("--openai-provider", default="custom")
        subparser.add_argument(
            "--reuse-anthropic-token-for-openai",
            action="store_true",
            help="仅在内存中复用 Routify 凭据并切换到 OpenAI Responses 端口",
        )
        if command in {"run", "retry-one"}:
            subparser.add_argument("--timeout", type=float, default=300.0)
        if command == "run":
            subparser.add_argument("--limit", type=int, default=None)
            subparser.add_argument("--retry-delay", type=float, default=2.0)
        elif command == "retry-one":
            identity = subparser.add_mutually_exclusive_group(required=True)
            identity.add_argument("--sample-id")
            identity.add_argument("--logical-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return run_preflight(args)
    if args.command == "retry-one":
        return run_retry_one(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
