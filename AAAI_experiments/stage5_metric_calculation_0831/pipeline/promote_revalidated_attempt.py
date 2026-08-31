"""把因本地验证器缺陷失败的既有 Claude attempt 审计后补冻。

该流程不会发起新的模型调用，也不会增加物理 attempt 计数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    validate_claude_envelope,
    validate_structured_output,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (
    SEMANTIC_VALIDATOR_VERSION,
    _infer_task_kind,
    _validate_simplify_semantics,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    _load_predecessor_attempt_manifest,
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskStateStore,
)


class PromotionError(RuntimeError):
    """既有 attempt 不满足可审计补冻条件。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"JSON 文件不可读: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"JSON 文件不是 object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def promote_revalidated_attempt(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempt_json: str | Path,
    frozen_dir: str | Path,
    predecessor_attempt_manifest: str | Path | None,
    audit_reason: str,
    report_json: str | Path | None = None,
) -> dict[str, object]:
    if not audit_reason.strip():
        raise PromotionError("audit_reason 不得为空")
    attempt_path = Path(attempt_json).resolve()
    attempt_sha256 = _sha256_file(attempt_path)
    attempt = _read_object(attempt_path)
    attempt_id = attempt.get("attempt_id")
    evaluation_key = attempt.get("evaluation_key")
    metadata = attempt.get("metadata")
    validation = attempt.get("validation")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise PromotionError("attempt_id 缺失")
    if not isinstance(evaluation_key, str) or not evaluation_key:
        raise PromotionError("evaluation_key 缺失")
    if not isinstance(metadata, Mapping) or metadata.get("error_class") != "validation_failed":
        raise PromotionError("仅允许补冻原 error_class=validation_failed 的 attempt")
    if not isinstance(validation, Mapping) or validation.get("ok") is not False:
        raise PromotionError("原 attempt 不是验证失败状态")
    structured_raw = validation.get("structured_output")

    loaded_plan = load_plan_jsonl(plan_jsonl)
    by_key = {entry.evaluation_key: entry for entry in loaded_plan.entries}
    entry = by_key.get(evaluation_key)
    if entry is None:
        raise PromotionError("attempt evaluation_key 不在冻结 plan 中")
    definition = entry.definition
    task_kind = _infer_task_kind(
        definition.task_spec.task_type,
        definition.task_kind,
    )
    if task_kind != "simplify":
        raise PromotionError("当前补冻器仅允许重验证 simplify 任务")
    if attempt.get("request") != definition.request:
        raise PromotionError("attempt request 与冻结 plan 不一致")
    if isinstance(structured_raw, Mapping):
        structured_output = validate_structured_output(task_kind, structured_raw)
    else:
        envelope = attempt.get("envelope")
        if not isinstance(envelope, Mapping):
            raise PromotionError("原 attempt 缺少可重解析的 Claude envelope")
        structured_output = validate_claude_envelope(
            envelope,
            task_kind=task_kind,
            schema=definition.schema,
        )
    semantic_evidence = _validate_simplify_semantics(definition, structured_output)
    if semantic_evidence.get("decision") not in {"equivalent", "not_applicable"}:
        raise PromotionError("重验证未形成可接受的等价闭环")

    predecessor = None
    if predecessor_attempt_manifest is not None:
        loaded_predecessor = _load_predecessor_attempt_manifest(
            predecessor_attempt_manifest
        )
        predecessor = PredecessorAttemptManifest(
            path=str(loaded_predecessor.path),
            sha256=loaded_predecessor.sha256,
            attempt_count=loaded_predecessor.attempt_count,
        )
    store = TaskStateStore(
        state_db,
        predecessor_attempt_manifest=predecessor,
    )
    state = store.task_state(evaluation_key)
    if state not in {"retry_wait", "exhausted"}:
        raise PromotionError(f"任务状态 {state!r} 不允许补冻")

    validator_path = Path(__file__).with_name("symbolic_evidence.py")
    promoted_at = time.time()
    promoted_metadata = dict(metadata)
    promoted_metadata.update(
        {
            "original_error_class": metadata.get("error_class"),
            "original_attempt_audit_path": str(attempt_path),
            "original_attempt_audit_sha256": attempt_sha256,
            "promotion_audit_reason": audit_reason,
            "promoted_at": promoted_at,
            "semantic_validator_version": SEMANTIC_VALIDATOR_VERSION,
            "semantic_validator_sha256": _sha256_file(validator_path),
            "error_class": None,
            "retryable": False,
        }
    )
    promoted_validation = {
        "ok": True,
        "error_class": None,
        "error_message": None,
        "structured_output": structured_output,
        "semantic_evidence": semantic_evidence,
        "promotion": {
            "source_attempt_audit_sha256": attempt_sha256,
            "reason": audit_reason,
            "new_model_call": False,
        },
    }
    frozen_payload = {
        "attempt_id": attempt_id,
        "evaluation_key": evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "request": attempt.get("request"),
        "prompt": attempt.get("prompt"),
        "command": attempt.get("command"),
        "stdout": attempt.get("stdout"),
        "stderr": attempt.get("stderr"),
        "envelope": attempt.get("envelope"),
        "structured_output": structured_output,
        "validation": promoted_validation,
        "metadata": promoted_metadata,
    }
    frozen_path = Path(frozen_dir).resolve() / f"{evaluation_key}.json"
    if frozen_path.exists():
        raise PromotionError(f"目标 frozen 文件已存在: {frozen_path}")
    _atomic_write_json(frozen_path, frozen_payload)
    frozen_sha256 = _sha256_file(frozen_path)
    try:
        store.promote_failed_attempt(
            attempt_id,
            result_path=str(frozen_path),
            result_sha256=frozen_sha256,
            allowed_error_classes=("validation_failed",),
            audit_reason=audit_reason,
            now=promoted_at,
        )
    except StateContractError as exc:
        raise PromotionError(f"状态库拒绝补冻: {exc}") from exc

    report = {
        "status": "promoted",
        "logical_id": entry.logical_id,
        "evaluation_key": evaluation_key,
        "attempt_id": attempt_id,
        "attempt_count_delta": 0,
        "new_model_call": False,
        "source_attempt_audit_path": str(attempt_path),
        "source_attempt_audit_sha256": attempt_sha256,
        "frozen_path": str(frozen_path),
        "frozen_sha256": frozen_sha256,
        "semantic_decision": semantic_evidence.get("decision"),
        "symbolic_decision": semantic_evidence.get("symbolic_decision"),
        "proof_basis": semantic_evidence.get("proof_basis"),
        "probe_count": semantic_evidence.get("probe_count"),
        "audit_reason": audit_reason,
    }
    if report_json is not None:
        _atomic_write_json(Path(report_json), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补冻通过修正版验证器的既有 Claude attempt")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempt-json", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--predecessor-attempt-manifest", type=Path, default=None)
    parser.add_argument("--audit-reason", required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    promote_revalidated_attempt(
        plan_jsonl=args.plan_jsonl,
        state_db=args.state_db,
        attempt_json=args.attempt_json,
        frozen_dir=args.frozen_dir,
        predecessor_attempt_manifest=args.predecessor_attempt_manifest,
        audit_reason=args.audit_reason,
        report_json=args.report_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
