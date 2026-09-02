"""合并两轮模型裁决与本地符号证据，生成最终公式审计报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .claude_contract import (
    canonical_json,
    validate_formula_audit_scope,
    validate_structured_output,
)
from .run_claude_plan import load_plan_jsonl


JsonDict = dict[str, Any]
DEFAULT_MAXIMUM_PHYSICAL_ATTEMPTS = 1575
DEFAULT_INPUT_PRICE_CNY_PER_MILLION = Decimal("3")
DEFAULT_OUTPUT_PRICE_CNY_PER_MILLION = Decimal("15")
DEFAULT_CACHE_READ_PRICE_CNY_PER_MILLION = Decimal("0.3")

_STRICT_SIMPLIFICATION_EQUIVALENT_PROOFS = {"artifact_identity"}
_STRICT_REFERENCE_EQUIVALENT_PROOFS = {
    "artifact_identity",
    "symbolic_difference_zero",
}
_STRICT_NOT_EQUIVALENT_PROOFS = {
    "symbolic_nonzero_constant_difference",
    "symbolic_nonzero_exact_difference",
}
_DIMENSIONS = ("algorithm", "condition", "seed", "audit_scope")


class FormulaAuditFinalizationError(RuntimeError):
    """最终汇总输入不完整、身份漂移或突破硬预算。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormulaAuditFinalizationError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaAuditFinalizationError(f"{context} 必须是非空字符串")
    return value.strip()


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[JsonDict]:
    if not path.is_file():
        raise FormulaAuditFinalizationError(f"JSONL 不存在: {path}")
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FormulaAuditFinalizationError(f"{path}:{line_number} 不得为空行")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormulaAuditFinalizationError(
                    f"{path}:{line_number} JSON 非法"
                ) from exc
            rows.append(dict(_require_mapping(payload, context=f"{path}:{line_number}")))
    if not rows and not allow_empty:
        raise FormulaAuditFinalizationError(f"JSONL 不能为空: {path}")
    return rows


def _unique_by(
    rows: Sequence[Mapping[str, Any]], key: str, *, context: str
) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for row in rows:
        value = _require_string(row.get(key), context=f"{context}.{key}")
        if value in result:
            raise FormulaAuditFinalizationError(f"{context} 出现重复 {key}: {value}")
        result[value] = dict(row)
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_plan(path: Path, *, allow_empty: bool = False) -> list[JsonDict]:
    rows = _read_jsonl(path, allow_empty=allow_empty)
    if rows:
        loaded = load_plan_jsonl(path)
        if {entry.evaluation_key for entry in loaded.entries} != {
            str(row.get("evaluation_key")) for row in rows
        }:
            raise FormulaAuditFinalizationError(f"plan 解析身份漂移: {path}")
    return rows


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FormulaAuditFinalizationError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _resolve_result_path(raw_path: str, *, state_db: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    relative = state_db.parent / path
    return relative if relative.is_file() else path.resolve()


def _validate_request_evidence_hash(request: Mapping[str, Any], *, context: str) -> None:
    declared = _require_string(request.get("evidence_hash"), context=f"{context}.evidence_hash")
    without_hash = {key: value for key, value in request.items() if key != "evidence_hash"}
    if declared != _sha256_json(without_hash):
        raise FormulaAuditFinalizationError(f"{context}.evidence_hash 漂移")


def _load_state_round(
    *,
    label: str,
    plan_rows: Sequence[Mapping[str, Any]],
    state_db: Path,
) -> tuple[dict[str, str], dict[str, JsonDict], list[JsonDict]]:
    plan_by_key = _unique_by(plan_rows, "evaluation_key", context=f"{label}_plan")
    keys = tuple(plan_by_key)
    task_rows: dict[str, JsonDict] = {}
    attempts: list[JsonDict] = []
    try:
        with _connect_read_only(state_db) as connection:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                task_query = f"""SELECT t.evaluation_key, t.logical_id, t.spec_json,
                                           t.state, f.attempt_id AS frozen_attempt_id,
                                           f.result_path, f.result_sha256
                                    FROM tasks t
                                    LEFT JOIN frozen_results f
                                      ON f.evaluation_key=t.evaluation_key
                                    WHERE t.evaluation_key IN ({placeholders})"""
                for row in connection.execute(task_query, chunk):
                    task_rows[str(row["evaluation_key"])] = dict(row)
                attempt_query = f"""SELECT attempt_id, evaluation_key, attempt_number,
                                              status, error_class
                                       FROM attempts
                                       WHERE evaluation_key IN ({placeholders})
                                       ORDER BY evaluation_key, attempt_number"""
                attempts.extend(dict(row) for row in connection.execute(attempt_query, chunk))
    except sqlite3.Error as exc:
        raise FormulaAuditFinalizationError(f"{label} 状态库读取失败: {exc}") from exc
    if set(task_rows) != set(plan_by_key):
        missing = sorted(set(plan_by_key) - set(task_rows))
        raise FormulaAuditFinalizationError(
            f"{label} 状态库缺少任务 {missing[:3]}，共 {len(missing)} 条"
        )

    states: dict[str, str] = {}
    frozen: dict[str, JsonDict] = {}
    for key, plan_row in plan_by_key.items():
        row = task_rows[key]
        try:
            stored_spec = json.loads(str(row["spec_json"]))
        except json.JSONDecodeError as exc:
            raise FormulaAuditFinalizationError(f"{label}.{key}.spec_json 非法") from exc
        if stored_spec != plan_row.get("task_spec"):
            raise FormulaAuditFinalizationError(f"{label}.{key} task_spec 与 plan 不一致")
        state = _require_string(row.get("state"), context=f"{label}.{key}.state")
        if state == "running":
            raise FormulaAuditFinalizationError(f"{label} 任务仍在运行: {key}")
        if state not in {"frozen", "exhausted", "pending", "retry_wait"}:
            raise FormulaAuditFinalizationError(f"{label} 任务状态不可汇总: {key}={state}")
        states[key] = state
        if state != "frozen":
            if row.get("result_path") is not None:
                raise FormulaAuditFinalizationError(f"{label} 非 frozen 任务存在结果: {key}")
            continue
        result_path = _resolve_result_path(str(row["result_path"]), state_db=state_db)
        if not result_path.is_file() or _sha256_file(result_path) != row["result_sha256"]:
            raise FormulaAuditFinalizationError(f"{label} frozen 文件缺失或 SHA 漂移: {key}")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormulaAuditFinalizationError(f"{label} frozen 文件不可读: {key}") from exc
        frozen_payload = dict(_require_mapping(payload, context=f"{label}.{key}.frozen"))
        expected = {
            "evaluation_key": key,
            "logical_id": plan_row.get("logical_id"),
            "task_type": "formula_audit",
            "task_kind": "formula_audit",
            "attempt_id": row.get("frozen_attempt_id"),
            "request": plan_row.get("request"),
        }
        for field, expected_value in expected.items():
            if frozen_payload.get(field) != expected_value:
                raise FormulaAuditFinalizationError(f"{label}.{key}.frozen.{field} 漂移")
        output = _require_mapping(
            frozen_payload.get("structured_output"),
            context=f"{label}.{key}.structured_output",
        )
        validation = _require_mapping(
            frozen_payload.get("validation"), context=f"{label}.{key}.validation"
        )
        if validation.get("ok") is not True or validation.get("structured_output") != output:
            raise FormulaAuditFinalizationError(
                f"{label}.{key} frozen validation 与 structured_output 不一致"
            )
        try:
            validated = validate_structured_output("formula_audit", output)
            validate_formula_audit_scope(
                _require_mapping(plan_row.get("request"), context=f"{label}.{key}.request"),
                validated,
            )
        except ValueError as exc:
            raise FormulaAuditFinalizationError(
                f"{label}.{key} structured_output 非法: {exc}"
            ) from exc
        frozen_payload["structured_output"] = validated
        frozen_payload["_result_path"] = str(result_path)
        frozen[key] = frozen_payload
    return states, frozen, attempts


def _attempt_artifact_candidates(
    *,
    state_db: Path,
    explicit_dir: Path | None,
    frozen_payloads: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_dir is not None:
        if not explicit_dir.is_dir():
            raise FormulaAuditFinalizationError(f"attempts 目录不存在: {explicit_dir}")
        candidates.append(explicit_dir)
    candidates.append(state_db.parent)
    for payload in frozen_payloads.values():
        result_path = Path(str(payload["_result_path"]))
        candidates.extend((result_path.parent, result_path.parent.parent))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_dir():
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _find_attempt_artifacts(
    *,
    attempts: Sequence[Mapping[str, Any]],
    state_db: Path,
    explicit_dir: Path | None,
    frozen_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    wanted = {str(row["attempt_id"]) for row in attempts}
    found: dict[str, Path] = {}
    for root in _attempt_artifact_candidates(
        state_db=state_db,
        explicit_dir=explicit_dir,
        frozen_payloads=frozen_payloads,
    ):
        for attempt_id in sorted(wanted - set(found)):
            direct = root / f"{attempt_id}.json"
            if direct.is_file():
                found[attempt_id] = direct
        if wanted == set(found):
            break
        for path in root.glob("*attempt*/**/*.json"):
            if path.stem in wanted and path.stem not in found:
                found[path.stem] = path
        if wanted == set(found):
            break
    return found


def _nonnegative_int(value: object, *, context: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormulaAuditFinalizationError(f"{context} 必须是非负整数")
    parsed = int(value)
    if parsed != value or parsed < 0:
        raise FormulaAuditFinalizationError(f"{context} 必须是非负整数")
    return parsed


def _usage_from_payload(payload: Mapping[str, Any], *, context: str) -> JsonDict:
    metadata = _require_mapping(payload.get("metadata"), context=f"{context}.metadata")
    raw_usage = metadata.get("usage")
    if raw_usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "usage_available": False,
        }
    usage = _require_mapping(raw_usage, context=f"{context}.metadata.usage")
    cache_creation = usage.get("cache_creation_input_tokens")
    if cache_creation is None and isinstance(usage.get("cache_creation"), Mapping):
        cache_creation = sum(
            _nonnegative_int(value, context=f"{context}.usage.cache_creation")
            for value in usage["cache_creation"].values()
        )
    return {
        "input_tokens": _nonnegative_int(
            usage.get("input_tokens"), context=f"{context}.usage.input_tokens"
        ),
        "output_tokens": _nonnegative_int(
            usage.get("output_tokens"), context=f"{context}.usage.output_tokens"
        ),
        "cache_read_input_tokens": _nonnegative_int(
            usage.get("cache_read_input_tokens"),
            context=f"{context}.usage.cache_read_input_tokens",
        ),
        "cache_creation_input_tokens": _nonnegative_int(
            cache_creation, context=f"{context}.usage.cache_creation_input_tokens"
        ),
        "usage_available": True,
    }


def _sum_usage(rows: Iterable[Mapping[str, Any]]) -> JsonDict:
    materialized = list(rows)
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    result = {field: sum(int(row[field]) for row in materialized) for field in fields}
    result["usage_available_count"] = sum(bool(row["usage_available"]) for row in materialized)
    result["usage_missing_count"] = sum(not bool(row["usage_available"]) for row in materialized)
    result["usage_complete"] = result["usage_missing_count"] == 0
    return result


def _round_usage(
    *,
    label: str,
    attempts: Sequence[Mapping[str, Any]],
    state_db: Path,
    explicit_attempts_dir: Path | None,
    frozen_payloads: Mapping[str, Mapping[str, Any]],
    plan_by_key: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    paths = _find_attempt_artifacts(
        attempts=attempts,
        state_db=state_db,
        explicit_dir=explicit_attempts_dir,
        frozen_payloads=frozen_payloads,
    )
    frozen_by_attempt = {
        str(payload["attempt_id"]): payload for payload in frozen_payloads.values()
    }
    usages: list[JsonDict] = []
    for row in attempts:
        attempt_id = str(row["attempt_id"])
        path = paths.get(attempt_id)
        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise FormulaAuditFinalizationError(
                    f"{label} attempt 文件不可读: {attempt_id}"
                ) from exc
            attempt_payload = _require_mapping(payload, context=f"{label}.{attempt_id}")
        elif attempt_id in frozen_by_attempt:
            attempt_payload = frozen_by_attempt[attempt_id]
        else:
            raise FormulaAuditFinalizationError(
                f"{label} 缺少 attempt 审计文件，无法精确统计 usage: {attempt_id}"
            )
        if attempt_payload.get("attempt_id") != attempt_id:
            raise FormulaAuditFinalizationError(f"{label} attempt_id 漂移: {attempt_id}")
        if attempt_payload.get("evaluation_key") != row["evaluation_key"]:
            raise FormulaAuditFinalizationError(f"{label} attempt evaluation_key 漂移: {attempt_id}")
        expected_request = plan_by_key[str(row["evaluation_key"])].get("request")
        if attempt_payload.get("request") != expected_request:
            raise FormulaAuditFinalizationError(f"{label} attempt request 漂移: {attempt_id}")
        metadata = _require_mapping(
            attempt_payload.get("metadata"), context=f"{label}.{attempt_id}.metadata"
        )
        request_sha256 = metadata.get("request_sha256")
        if request_sha256 is not None and request_sha256 != _sha256_json(expected_request):
            raise FormulaAuditFinalizationError(
                f"{label} attempt request_sha256 漂移: {attempt_id}"
            )
        usage = _usage_from_payload(attempt_payload, context=f"{label}.{attempt_id}")
        usages.append({"attempt_id": attempt_id, **usage})
    return {
        "physical_attempt_count": len(attempts),
        **_sum_usage(usages),
    }


def _external_usage(
    *,
    attempts_root: Path | None,
    explicit_count: int | None,
) -> JsonDict:
    if attempts_root is not None and explicit_count is not None:
        raise FormulaAuditFinalizationError(
            "external_attempts_root 与 external_attempt_count 只能提供一个"
        )
    if attempts_root is None and explicit_count is None:
        raise FormulaAuditFinalizationError(
            "必须提供 external_attempts_root 或 external_attempt_count（无外部调用时填 0）"
        )
    if explicit_count is not None:
        if explicit_count < 0:
            raise FormulaAuditFinalizationError("external_attempt_count 不能为负数")
        return {
            "source": "explicit_count",
            "physical_attempt_count": explicit_count,
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "usage_available_count": 0,
            "usage_missing_count": explicit_count,
            "usage_complete": explicit_count == 0,
        }
    if attempts_root is None:
        raise FormulaAuditFinalizationError("external attempts 输入状态非法")
    if not attempts_root.is_dir():
        raise FormulaAuditFinalizationError(f"external attempts 目录不存在: {attempts_root}")
    payloads: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(attempts_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormulaAuditFinalizationError(f"external attempt 不可读: {path}") from exc
        attempt = _require_mapping(payload, context=str(path))
        attempt_id = _require_string(attempt.get("attempt_id"), context=f"{path}.attempt_id")
        if attempt_id in seen_ids:
            raise FormulaAuditFinalizationError(f"external attempt_id 重复: {attempt_id}")
        seen_ids.add(attempt_id)
        payloads.append(attempt)
    usages = [
        _usage_from_payload(payload, context=f"external.{payload['attempt_id']}")
        for payload in payloads
    ]
    return {
        "source": "attempts_root",
        "attempts_root": str(attempts_root),
        "physical_attempt_count": len(payloads),
        **_sum_usage(usages),
    }


def _priced_usage(
    usage: Mapping[str, Any],
    *,
    input_price: Decimal,
    output_price: Decimal,
    cache_read_price: Decimal,
) -> JsonDict:
    if usage.get("input_tokens") is None:
        return {**usage, "priced_cost_cny": None}
    cost = (
        Decimal(int(usage["input_tokens"])) * input_price
        + Decimal(int(usage["output_tokens"])) * output_price
        + Decimal(int(usage["cache_read_input_tokens"])) * cache_read_price
    ) / Decimal(1_000_000)
    return {**usage, "priced_cost_cny": float(cost)}


def _round2_base(logical_id: str) -> str:
    if not logical_id.endswith("::v2"):
        raise FormulaAuditFinalizationError(f"round2 logical_id 必须以 ::v2 结尾: {logical_id}")
    return logical_id[: -len("::v2")]


def _verify_round2_mapping(
    *,
    round1_by_logical: Mapping[str, Mapping[str, Any]],
    round2_rows: Sequence[Mapping[str, Any]],
    required_round2_keys: set[str],
) -> dict[str, JsonDict]:
    mapped: dict[str, JsonDict] = {}
    for row in round2_rows:
        logical_id = _require_string(row.get("logical_id"), context="round2.logical_id")
        base = _round2_base(logical_id)
        round1 = round1_by_logical.get(base)
        if round1 is None:
            raise FormulaAuditFinalizationError(f"round2 缺少 round1 predecessor: {logical_id}")
        round1_key = str(round1["evaluation_key"])
        if round1_key not in required_round2_keys:
            raise FormulaAuditFinalizationError(f"round2 任务未被 requires_round2 触发: {logical_id}")
        first_request = _require_mapping(round1.get("request"), context=f"{base}.request")
        second_request = _require_mapping(row.get("request"), context=f"{logical_id}.request")
        _validate_request_evidence_hash(first_request, context=f"{base}.request")
        _validate_request_evidence_hash(second_request, context=f"{logical_id}.request")
        first_core = {
            key: value
            for key, value in first_request.items()
            if key not in {"review_round", "evidence_hash"}
        }
        second_core = {
            key: value
            for key, value in second_request.items()
            if key not in {"review_round", "evidence_hash"}
        }
        if first_request.get("review_round") != 1 or second_request.get("review_round") != 2:
            raise FormulaAuditFinalizationError(f"round1/round2 review_round 漂移: {base}")
        if first_core != second_core:
            raise FormulaAuditFinalizationError(f"round2 request 与 round1 语义输入不一致: {base}")
        if round1_key in mapped:
            raise FormulaAuditFinalizationError(f"round1 存在多个 round2 映射: {base}")
        mapped[round1_key] = dict(row)
    return mapped


def _verify_retry_mapping(
    *,
    round2_rows: Sequence[Mapping[str, Any]],
    round2_states: Mapping[str, str],
    retry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    round2_by_logical = _unique_by(round2_rows, "logical_id", context="round2_plan")
    expected_predecessors = {
        str(row["evaluation_key"])
        for row in round2_rows
        if round2_states.get(str(row["evaluation_key"])) == "exhausted"
    }
    mapped: dict[str, JsonDict] = {}
    for row in retry_rows:
        logical_id = _require_string(row.get("logical_id"), context="round2_retry.logical_id")
        if not logical_id.endswith("::retry1"):
            raise FormulaAuditFinalizationError(
                f"round2 retry logical_id 必须以 ::retry1 结尾: {logical_id}"
            )
        predecessor_logical_id = logical_id[: -len("::retry1")]
        predecessor = round2_by_logical.get(predecessor_logical_id)
        if predecessor is None:
            raise FormulaAuditFinalizationError(
                f"round2 retry 缺少原 round2 predecessor: {logical_id}"
            )
        predecessor_key = str(predecessor["evaluation_key"])
        if row.get("retry_predecessor_evaluation_key") != predecessor_key:
            raise FormulaAuditFinalizationError(
                f"round2 retry predecessor evaluation_key 漂移: {logical_id}"
            )
        if round2_states.get(predecessor_key) != "exhausted":
            raise FormulaAuditFinalizationError(
                f"round2 retry 只能映射 exhausted 原任务: {logical_id}"
            )
        predecessor_request = _require_mapping(
            predecessor.get("request"), context=f"{predecessor_logical_id}.request"
        )
        retry_request = _require_mapping(row.get("request"), context=f"{logical_id}.request")
        _validate_request_evidence_hash(
            predecessor_request, context=f"{predecessor_logical_id}.request"
        )
        _validate_request_evidence_hash(retry_request, context=f"{logical_id}.request")
        if predecessor_request.get("review_round") != 2 or retry_request.get("review_round") != 2:
            raise FormulaAuditFinalizationError(
                f"round2 retry review_round 必须保持为 2: {logical_id}"
            )
        if retry_request != predecessor_request:
            raise FormulaAuditFinalizationError(
                f"round2 retry request 与原 round2 不完全一致: {logical_id}"
            )
        if row.get("prompt_version") != "formula_audit.v3":
            raise FormulaAuditFinalizationError(
                f"round2 retry 必须使用 formula_audit.v3 prompt: {logical_id}"
            )
        if predecessor_key in mapped:
            raise FormulaAuditFinalizationError(
                f"原 round2 存在多个 retry1 映射: {predecessor_logical_id}"
            )
        mapped[predecessor_key] = dict(row)
    if set(mapped) != expected_predecessors:
        missing = sorted(expected_predecessors - set(mapped))
        extra = sorted(set(mapped) - expected_predecessors)
        raise FormulaAuditFinalizationError(
            f"round2 retry plan 与 exhausted 集合不一致: missing={missing[:3]}, extra={extra[:3]}"
        )
    return mapped


def _model_decision(output: Mapping[str, Any] | None, *, field: str) -> str | None:
    if output is None:
        return None
    value = output.get(field)
    return str(value) if isinstance(value, str) else None


def _strict_local_resolution(
    local: Mapping[str, Any] | None,
    *,
    simplification: bool,
) -> tuple[str, str] | None:
    if local is None:
        return None
    decision = local.get("decision")
    proof_basis = local.get("proof_basis")
    counterexample = local.get("counterexample")
    if decision == "not_equivalent" and counterexample is not None:
        return "not_equivalent", "local_counterexample"
    if decision == "not_equivalent" and proof_basis in _STRICT_NOT_EQUIVALENT_PROOFS:
        return "not_equivalent", "local_strict_proof"
    equivalent_proofs = (
        _STRICT_SIMPLIFICATION_EQUIVALENT_PROOFS
        if simplification
        else _STRICT_REFERENCE_EQUIVALENT_PROOFS
    )
    if decision == "equivalent" and proof_basis in equivalent_proofs:
        return "equivalent", "local_strict_proof"
    return None


def _resolve_decision(
    *,
    local: Mapping[str, Any] | None,
    round2: str | None,
    round1: str | None,
    simplification: bool,
    round2_source: str = "round2",
) -> tuple[str, str]:
    local_resolution = _strict_local_resolution(
        local,
        simplification=simplification,
    )
    if local_resolution is not None:
        decision, source = local_resolution
        if simplification:
            decision = "preserved" if decision == "equivalent" else "not_preserved"
        return decision, source
    decisive = (
        {"preserved", "not_preserved"}
        if simplification
        else {"equivalent", "not_equivalent"}
    )
    if round2 in decisive:
        return str(round2), round2_source
    if round1 in decisive:
        return str(round1), "round1"
    return "undetermined", "no_determinate_evidence"


def _stored_simplification(outcome: object) -> str:
    value = _require_string(outcome, context="stored_simplification_outcome")
    if value in {"simplified", "unchanged"}:
        return "preserved"
    if value == "unable":
        return "undetermined"
    raise FormulaAuditFinalizationError(f"stored simplification outcome 非法: {value}")


def _offline_severity(
    *,
    scope: str,
    final_simplification: str,
    final_reference: str,
    final_quality: str,
    stored_simplification: str,
    stored_reference: str,
) -> tuple[str, list[str]]:
    if final_simplification == "not_preserved":
        return "critical", ["simplification_not_preserved"]
    if final_simplification == "undetermined" or final_reference == "undetermined":
        return "undetermined", ["no_determinate_evidence"]
    if (
        stored_simplification in {"preserved", "not_preserved"}
        and final_simplification != stored_simplification
    ):
        return (
            "critical" if scope == "ground_truth_simplification" else "major",
            ["stored_simplification_conflict"],
        )
    if (
        scope == "prediction_formula"
        and stored_reference in {"equivalent", "not_equivalent"}
        and final_reference != stored_reference
    ):
        return "major", ["stored_equivalence_conflict"]
    if final_quality == "more_complex":
        return (
            "major" if scope == "ground_truth_simplification" else "minor",
            ["candidate_more_complex"],
        )
    abstention_reasons: list[str] = []
    if (
        stored_simplification == "undetermined"
        and final_simplification in {"preserved", "not_preserved"}
    ):
        abstention_reasons.append("stored_simplification_abstention_resolved")
    if (
        scope == "prediction_formula"
        and stored_reference == "undetermined"
        and final_reference in {"equivalent", "not_equivalent"}
    ):
        abstention_reasons.append("stored_equivalence_abstention_resolved")
    return "pass", abstention_reasons or ["no_offline_conflict"]


def _quality(round1: Mapping[str, Any] | None, round2: Mapping[str, Any] | None) -> str:
    for output in (round2, round1):
        if output is not None and output.get("simplification_quality") in {
            "strictly_simpler",
            "equally_simple",
            "more_complex",
        }:
            return str(output["simplification_quality"])
    return "undetermined"


def _dimension_value(row: Mapping[str, Any], dimension: str) -> str:
    value = row.get(dimension)
    return "not_applicable" if value is None or value == "" else str(value)


def _build_statistics(final_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    result: JsonDict = {"overall": {}, "dimensions": {}}

    def summarize(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
        return {
            "total": len(rows),
            "severity_counts": dict(
                sorted(Counter(str(row["offline_severity"]) for row in rows).items())
            ),
            "simplification_decision_counts": dict(
                sorted(
                    Counter(str(row["final_simplification_decision"]) for row in rows).items()
                )
            ),
            "reference_decision_counts": dict(
                sorted(Counter(str(row["final_reference_decision"]) for row in rows).items())
            ),
            "issue_count": sum(bool(row["issue_types"]) for row in rows),
        }

    result["overall"] = summarize(final_rows)
    for dimension in _DIMENSIONS:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in final_rows:
            groups[_dimension_value(row, dimension)].append(row)
        result["dimensions"][dimension] = {
            value: summarize(rows) for value, rows in sorted(groups.items())
        }
    return result


def _statistics_csv_rows(statistics: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    dimensions = _require_mapping(statistics.get("dimensions"), context="statistics.dimensions")
    for dimension in _DIMENSIONS:
        groups = _require_mapping(dimensions.get(dimension), context=f"statistics.{dimension}")
        for value, raw_summary in groups.items():
            summary = _require_mapping(raw_summary, context=f"statistics.{dimension}.{value}")
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "total": summary["total"],
                    "issue_count": summary["issue_count"],
                    "severity_counts_json": canonical_json(summary["severity_counts"]),
                    "simplification_decision_counts_json": canonical_json(
                        summary["simplification_decision_counts"]
                    ),
                    "reference_decision_counts_json": canonical_json(
                        summary["reference_decision_counts"]
                    ),
                }
            )
    return rows


def _write_statistics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dimension",
        "value",
        "total",
        "issue_count",
        "severity_counts_json",
        "simplification_decision_counts_json",
        "reference_decision_counts_json",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown_report(
    *,
    final_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> str:
    severity = _require_mapping(
        _require_mapping(statistics["overall"], context="statistics.overall").get(
            "severity_counts"
        ),
        context="statistics.overall.severity_counts",
    )
    lines = [
        "# Formula Audit Final Report",
        "",
        "## Summary",
        "",
        f"- Final records: {len(final_rows)}",
        f"- Records with issues: {len(issues)}",
        f"- Severity: `{canonical_json(severity)}`",
        f"- Physical attempts: {budget['total_physical_attempt_count']} / {budget['maximum_total_physical_attempts']}",
        f"- Priced cost (CNY): {budget['priced_cost_cny']}",
        f"- Cache creation tokens (unpriced): {budget['cache_creation_input_tokens_unpriced']}",
        "",
        "## Attempt Accounting",
        "",
        "| Source | Attempts | Input | Output | Cache read | Cache creation | Cost CNY |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("external", "round1", "round2", "round2_retry"):
        item = budget[label]
        lines.append(
            f"| {label} | {item['physical_attempt_count']} | {item['input_tokens']} | "
            f"{item['output_tokens']} | {item['cache_read_input_tokens']} | "
            f"{item['cache_creation_input_tokens']} | {item['priced_cost_cny']} |"
        )
    lines.extend(
        [
            "",
            "## Resolution Policy",
            "",
            "Local deterministic counterexamples or strict symbolic proofs take precedence. "
            "Otherwise a successful round 2 retry takes precedence over the original round 2, "
            "which takes precedence over round 1. Without determinate evidence, "
            "the final decision is undetermined. Model-reported severity is never used.",
            "",
            "## Dimension Overview",
            "",
            "| Dimension | Value | Records | Issues |",
            "|---|---|---:|---:|",
        ]
    )
    dimensions = _require_mapping(statistics["dimensions"], context="statistics.dimensions")
    for dimension in _DIMENSIONS:
        groups = _require_mapping(dimensions[dimension], context=f"statistics.{dimension}")
        for value, summary_value in groups.items():
            summary = _require_mapping(summary_value, context=f"statistics.{dimension}.{value}")
            lines.append(
                f"| {dimension} | {value} | {summary['total']} | {summary['issue_count']} |"
            )
    lines.append("")
    return "\n".join(lines)


def finalize_formula_audit(
    *,
    round1_plan_jsonl: str | Path,
    round1_comparisons_jsonl: str | Path,
    round1_state_db: str | Path,
    local_verification_jsonl: str | Path,
    sample_manifest_jsonl: str | Path,
    output_root: str | Path,
    round2_plan_jsonl: str | Path | None = None,
    round2_state_db: str | Path | None = None,
    retry_plan_jsonl: str | Path | None = None,
    retry_state_db: str | Path | None = None,
    round1_attempts_dir: str | Path | None = None,
    round2_attempts_dir: str | Path | None = None,
    retry_attempts_dir: str | Path | None = None,
    external_attempts_root: str | Path | None = None,
    external_attempt_count: int | None = None,
    maximum_total_physical_attempts: int = DEFAULT_MAXIMUM_PHYSICAL_ATTEMPTS,
    input_price_cny_per_million: Decimal | float | str = DEFAULT_INPUT_PRICE_CNY_PER_MILLION,
    output_price_cny_per_million: Decimal | float | str = DEFAULT_OUTPUT_PRICE_CNY_PER_MILLION,
    cache_read_price_cny_per_million: Decimal | float | str = DEFAULT_CACHE_READ_PRICE_CNY_PER_MILLION,
) -> JsonDict:
    """生成最终逐条裁决、问题、维度统计、成本统计和 Markdown 报告。"""

    if maximum_total_physical_attempts <= 0:
        raise FormulaAuditFinalizationError("maximum_total_physical_attempts 必须为正整数")
    if (round2_plan_jsonl is None) != (round2_state_db is None):
        raise FormulaAuditFinalizationError("round2 plan 与 state DB 必须同时提供或同时省略")
    if (retry_plan_jsonl is None) != (retry_state_db is None):
        raise FormulaAuditFinalizationError("round2 retry plan 与 state DB 必须同时提供或同时省略")
    if retry_plan_jsonl is not None and round2_plan_jsonl is None:
        raise FormulaAuditFinalizationError("round2 retry 必须与原 round2 plan/state 一起提供")
    if retry_attempts_dir is not None and retry_plan_jsonl is None:
        raise FormulaAuditFinalizationError("retry_attempts_dir 需要同时提供 retry plan/state")
    prices = {
        "input": Decimal(str(input_price_cny_per_million)),
        "output": Decimal(str(output_price_cny_per_million)),
        "cache_read": Decimal(str(cache_read_price_cny_per_million)),
    }
    if any(value < 0 for value in prices.values()):
        raise FormulaAuditFinalizationError("token 单价不能为负数")

    round1_plan_path = Path(round1_plan_jsonl).resolve()
    round1_rows = _load_plan(round1_plan_path)
    round1_by_key = _unique_by(round1_rows, "evaluation_key", context="round1_plan")
    round1_by_logical = _unique_by(round1_rows, "logical_id", context="round1_plan")
    for row in round1_rows:
        _validate_request_evidence_hash(
            _require_mapping(row.get("request"), context="round1.request"),
            context=f"{row['logical_id']}.request",
        )
    comparisons_by_key = _unique_by(
        _read_jsonl(Path(round1_comparisons_jsonl).resolve()),
        "round1_evaluation_key",
        context="round1_comparisons",
    )
    manifest_by_key = _unique_by(
        _read_jsonl(Path(sample_manifest_jsonl).resolve()),
        "audit_evaluation_key",
        context="sample_manifest",
    )
    if set(comparisons_by_key) != set(round1_by_key) or set(manifest_by_key) != set(
        round1_by_key
    ):
        raise FormulaAuditFinalizationError(
            "round1 plan、comparisons 与 sample manifest 任务集合不一致"
        )
    required_round2_keys = {
        key for key, row in comparisons_by_key.items() if row.get("requires_round2") is True
    }
    local_by_key = _unique_by(
        _read_jsonl(Path(local_verification_jsonl).resolve(), allow_empty=True),
        "round1_evaluation_key",
        context="local_verification",
    )
    if set(local_by_key) != required_round2_keys:
        raise FormulaAuditFinalizationError(
            "local verification 与 comparisons.requires_round2 集合不一致"
        )
    for key, row in local_by_key.items():
        logical_id = round1_by_key[key].get("logical_id")
        if row.get("audit_logical_id") != logical_id:
            raise FormulaAuditFinalizationError(f"local verification logical_id 漂移: {key}")
        declared_hash = _require_string(
            row.get("local_evidence_hash"), context=f"{key}.local_evidence_hash"
        )
        hash_payload = {
            field: value for field, value in row.items() if field != "local_evidence_hash"
        }
        if declared_hash != _sha256_json(hash_payload):
            raise FormulaAuditFinalizationError(
                f"local verification evidence hash 漂移: {key}"
            )

    round1_state_path = Path(round1_state_db).resolve()
    round1_states, round1_frozen, round1_attempts = _load_state_round(
        label="round1",
        plan_rows=round1_rows,
        state_db=round1_state_path,
    )
    if round2_plan_jsonl is not None:
        round2_plan_path = Path(round2_plan_jsonl).resolve()
        round2_rows = _load_plan(round2_plan_path, allow_empty=True)
        round2_state_path = Path(str(round2_state_db)).resolve()
        round2_states, round2_frozen, round2_attempts = _load_state_round(
            label="round2",
            plan_rows=round2_rows,
            state_db=round2_state_path,
        )
    else:
        round2_plan_path = None
        round2_state_path = None
        round2_rows = []
        round2_states = {}
        round2_frozen = {}
        round2_attempts = []
    round2_by_round1 = _verify_round2_mapping(
        round1_by_logical=round1_by_logical,
        round2_rows=round2_rows,
        required_round2_keys=required_round2_keys,
    )
    if retry_plan_jsonl is not None:
        retry_plan_path = Path(retry_plan_jsonl).resolve()
        retry_rows = _load_plan(retry_plan_path, allow_empty=True)
        retry_state_path = Path(str(retry_state_db)).resolve()
        retry_states, retry_frozen, retry_attempts = _load_state_round(
            label="round2_retry",
            plan_rows=retry_rows,
            state_db=retry_state_path,
        )
        retry_by_round2 = _verify_retry_mapping(
            round2_rows=round2_rows,
            round2_states=round2_states,
            retry_rows=retry_rows,
        )
    else:
        retry_plan_path = None
        retry_state_path = None
        retry_rows = []
        retry_states = {}
        retry_frozen = {}
        retry_attempts = []
        retry_by_round2 = {}

    round1_usage = _round_usage(
        label="round1",
        attempts=round1_attempts,
        state_db=round1_state_path,
        explicit_attempts_dir=(
            Path(round1_attempts_dir).resolve() if round1_attempts_dir is not None else None
        ),
        frozen_payloads=round1_frozen,
        plan_by_key=round1_by_key,
    )
    round2_usage = _round_usage(
        label="round2",
        attempts=round2_attempts,
        state_db=round2_state_path or round1_state_path,
        explicit_attempts_dir=(
            Path(round2_attempts_dir).resolve() if round2_attempts_dir is not None else None
        ),
        frozen_payloads=round2_frozen,
        plan_by_key=_unique_by(round2_rows, "evaluation_key", context="round2_plan"),
    )
    retry_usage = _round_usage(
        label="round2_retry",
        attempts=retry_attempts,
        state_db=retry_state_path or round1_state_path,
        explicit_attempts_dir=(
            Path(retry_attempts_dir).resolve() if retry_attempts_dir is not None else None
        ),
        frozen_payloads=retry_frozen,
        plan_by_key=_unique_by(retry_rows, "evaluation_key", context="round2_retry_plan"),
    )
    external_usage = _external_usage(
        attempts_root=(
            Path(external_attempts_root).resolve()
            if external_attempts_root is not None
            else None
        ),
        explicit_count=external_attempt_count,
    )
    total_attempts = sum(
        int(item["physical_attempt_count"])
        for item in (external_usage, round1_usage, round2_usage, retry_usage)
    )
    if total_attempts > maximum_total_physical_attempts:
        raise FormulaAuditFinalizationError(
            f"总物理调用突破硬上限: {total_attempts}/{maximum_total_physical_attempts}"
        )
    priced_round1 = _priced_usage(round1_usage, **{
        "input_price": prices["input"],
        "output_price": prices["output"],
        "cache_read_price": prices["cache_read"],
    })
    priced_round2 = _priced_usage(round2_usage, **{
        "input_price": prices["input"],
        "output_price": prices["output"],
        "cache_read_price": prices["cache_read"],
    })
    priced_retry = _priced_usage(retry_usage, **{
        "input_price": prices["input"],
        "output_price": prices["output"],
        "cache_read_price": prices["cache_read"],
    })
    priced_external = _priced_usage(external_usage, **{
        "input_price": prices["input"],
        "output_price": prices["output"],
        "cache_read_price": prices["cache_read"],
    })

    final_rows: list[JsonDict] = []
    issues: list[JsonDict] = []
    for round1_key, round1_row in sorted(
        round1_by_key.items(), key=lambda item: str(item[1]["logical_id"])
    ):
        logical_id = str(round1_row["logical_id"])
        manifest = manifest_by_key[round1_key]
        comparison = comparisons_by_key[round1_key]
        if manifest.get("audit_logical_id") != logical_id or comparison.get(
            "audit_logical_id"
        ) != logical_id:
            raise FormulaAuditFinalizationError(f"round1 identity 漂移: {logical_id}")
        declared_comparison_state = comparison.get("round1_state")
        if (
            declared_comparison_state is not None
            and declared_comparison_state != round1_states[round1_key]
        ):
            raise FormulaAuditFinalizationError(
                f"comparison/state round1_state 漂移: {logical_id}"
            )
        request = _require_mapping(round1_row.get("request"), context=f"{logical_id}.request")
        scope = _require_string(request.get("audit_scope"), context=f"{logical_id}.scope")
        round1_output = (
            _require_mapping(round1_frozen[round1_key]["structured_output"], context="round1")
            if round1_key in round1_frozen
            else None
        )
        round2_row = round2_by_round1.get(round1_key)
        round2_key = str(round2_row["evaluation_key"]) if round2_row is not None else None
        round2_output = (
            _require_mapping(round2_frozen[round2_key]["structured_output"], context="round2")
            if round2_key is not None and round2_key in round2_frozen
            else None
        )
        retry_row = retry_by_round2.get(round2_key) if round2_key is not None else None
        retry_key = str(retry_row["evaluation_key"]) if retry_row is not None else None
        retry_output = (
            _require_mapping(
                retry_frozen[retry_key]["structured_output"], context="round2_retry"
            )
            if retry_key is not None and retry_key in retry_frozen
            else None
        )
        effective_round2_output = retry_output if retry_output is not None else round2_output
        effective_round2_source = (
            "round2_retry"
            if retry_output is not None
            else ("round2" if round2_output is not None else None)
        )
        local_record = local_by_key.get(round1_key)
        local_simplification = (
            _require_mapping(local_record.get("local_simplification"), context="local_simplification")
            if local_record is not None
            else None
        )
        local_reference = (
            _require_mapping(local_record.get("local_reference"), context="local_reference")
            if local_record is not None
            else None
        )
        r1_simplification = _model_decision(
            round1_output, field="simplification_equivalence"
        )
        original_r2_simplification = _model_decision(
            round2_output, field="simplification_equivalence"
        )
        retry_simplification = _model_decision(
            retry_output, field="simplification_equivalence"
        )
        r2_simplification = _model_decision(
            effective_round2_output, field="simplification_equivalence"
        )
        final_simplification, simplification_source = _resolve_decision(
            local=local_simplification,
            round2=r2_simplification,
            round1=r1_simplification,
            simplification=True,
            round2_source=effective_round2_source or "round2",
        )
        if scope == "ground_truth_simplification":
            r1_reference = "not_applicable"
            original_r2_reference = (
                "not_applicable" if round2_output is not None else None
            )
            retry_reference = "not_applicable" if retry_output is not None else None
            r2_reference = (
                "not_applicable" if effective_round2_output is not None else None
            )
            final_reference, reference_source = "not_applicable", "scope_not_applicable"
        else:
            r1_reference = _model_decision(round1_output, field="reference_equivalence")
            original_r2_reference = _model_decision(
                round2_output, field="reference_equivalence"
            )
            retry_reference = _model_decision(
                retry_output, field="reference_equivalence"
            )
            r2_reference = _model_decision(
                effective_round2_output, field="reference_equivalence"
            )
            final_reference, reference_source = _resolve_decision(
                local=local_reference,
                round2=r2_reference,
                round1=r1_reference,
                simplification=False,
                round2_source=effective_round2_source or "round2",
            )
        final_quality = _quality(round1_output, effective_round2_output)
        stored_simplification = _stored_simplification(
            manifest.get("stored_simplification_outcome")
        )
        stored_reference = _require_string(
            manifest.get("stored_equivalence_decision"),
            context=f"{logical_id}.stored_equivalence_decision",
        )
        severity, severity_reasons = _offline_severity(
            scope=scope,
            final_simplification=final_simplification,
            final_reference=final_reference,
            final_quality=final_quality,
            stored_simplification=stored_simplification,
            stored_reference=stored_reference,
        )
        issue_types: list[str] = []
        simplification_disagreement = (
            r1_simplification is not None
            and r2_simplification is not None
            and r1_simplification != r2_simplification
        )
        reference_disagreement = (
            r1_reference is not None
            and r2_reference is not None
            and r1_reference != r2_reference
        )
        if simplification_disagreement:
            issue_types.append("round_disagreement:simplification")
        if reference_disagreement:
            issue_types.append("round_disagreement:reference")
        if round2_row is not None and round2_output is None:
            issue_types.append(f"round2_result_unavailable:{round2_states[round2_key]}")
        if retry_row is not None and retry_output is None:
            issue_types.append(
                f"round2_retry_result_unavailable:{retry_states[retry_key]}"
            )
        if retry_output is not None:
            issue_types.append("round2_retry_recovered")
        if comparison.get("requires_round2") is True and round2_row is None:
            issue_types.append("round2_not_planned")
        for label, local in (
            ("simplification", local_simplification),
            ("reference", local_reference),
        ):
            if local is not None and local.get("decision") == "error":
                issue_types.append(f"local_verification_error:{label}")
        if simplification_source.startswith("local_") and any(
            value in {"preserved", "not_preserved"} and value != final_simplification
            for value in (r1_simplification, r2_simplification)
        ):
            issue_types.append("local_override:simplification")
        if reference_source.startswith("local_") and any(
            value in {"equivalent", "not_equivalent"} and value != final_reference
            for value in (r1_reference, r2_reference)
        ):
            issue_types.append("local_override:reference")
        if final_simplification == "undetermined" or final_reference == "undetermined":
            issue_types.append("final_undetermined")
        if (
            manifest.get("stored_simplification_outcome") == "unable"
            and final_simplification in {"preserved", "not_preserved"}
        ):
            issue_types.append("coverage_recovery:stored_simplification_unable")
        if (
            stored_reference == "undetermined"
            and final_reference in {"equivalent", "not_equivalent"}
        ):
            issue_types.append("coverage_recovery:stored_equivalence_undetermined")
        if severity in {"critical", "major", "minor"}:
            issue_types.append(f"offline_severity:{severity}")
        issue_types = list(dict.fromkeys(issue_types))
        final_row: JsonDict = {
            "audit_logical_id": logical_id,
            "round1_evaluation_key": round1_key,
            "round2_evaluation_key": round2_key,
            "round2_retry_evaluation_key": retry_key,
            "source_identity": {
                key: manifest[key]
                for key in (
                    "logical_key",
                    "source_formula_logical_id",
                    "source_formula_evaluation_key",
                    "source_formula_result_sha256",
                )
                if key in manifest
            },
            "audit_scope": scope,
            "algorithm": manifest.get("algorithm"),
            "condition": manifest.get("condition"),
            "seed": manifest.get("seed"),
            "dataset_id": manifest.get("dataset_id"),
            "expressions": {
                "original": request.get("original_expression"),
                "candidate_simplified": request.get("candidate_simplified_expression"),
                "reference_simplified": request.get("reference_simplified_expression"),
            },
            "stored_decisions": {
                "simplification_outcome": manifest.get("stored_simplification_outcome"),
                "simplification_decision": stored_simplification,
                "reference_equivalence": stored_reference,
            },
            "round1_state": round1_states[round1_key],
            "round2_state": round2_states.get(round2_key) if round2_key is not None else None,
            "round2_retry_state": retry_states.get(retry_key) if retry_key is not None else None,
            "round1_model_evidence": dict(round1_output) if round1_output else None,
            "round2_model_evidence": dict(round2_output) if round2_output else None,
            "round2_retry_model_evidence": dict(retry_output) if retry_output else None,
            "effective_round2_source": effective_round2_source,
            "effective_round2_model_evidence": (
                dict(effective_round2_output) if effective_round2_output else None
            ),
            "round1_simplification_decision": r1_simplification,
            "round2_simplification_decision": r2_simplification,
            "original_round2_simplification_decision": original_r2_simplification,
            "round2_retry_simplification_decision": retry_simplification,
            "final_simplification_decision": final_simplification,
            "simplification_resolution_source": simplification_source,
            "round1_reference_decision": r1_reference,
            "round2_reference_decision": r2_reference,
            "original_round2_reference_decision": original_r2_reference,
            "round2_retry_reference_decision": retry_reference,
            "final_reference_decision": final_reference,
            "reference_resolution_source": reference_source,
            "round_disagreement": {
                "simplification": simplification_disagreement,
                "reference": reference_disagreement,
            },
            "local_criteria": {
                "simplification": local_simplification,
                "reference": local_reference,
                "local_evidence_hash": (
                    local_record.get("local_evidence_hash") if local_record is not None else None
                ),
            },
            "final_simplification_quality": final_quality,
            "model_reported_severity": {
                "round1": round1_output.get("severity") if round1_output else None,
                "round2": (
                    effective_round2_output.get("severity")
                    if effective_round2_output
                    else None
                ),
                "round2_original": round2_output.get("severity") if round2_output else None,
                "round2_retry": retry_output.get("severity") if retry_output else None,
                "used_for_final": False,
            },
            "offline_severity": severity,
            "offline_severity_reasons": severity_reasons,
            "issue_types": issue_types,
        }
        final_row["final_record_sha256"] = _sha256_json(final_row)
        final_rows.append(final_row)
        if issue_types:
            issues.append(
                {
                    "audit_logical_id": logical_id,
                    "round1_evaluation_key": round1_key,
                    "round2_evaluation_key": round2_key,
                    "round2_retry_evaluation_key": retry_key,
                    "audit_scope": scope,
                    "algorithm": manifest.get("algorithm"),
                    "condition": manifest.get("condition"),
                    "seed": manifest.get("seed"),
                    "dataset_id": manifest.get("dataset_id"),
                    "expressions": dict(final_row["expressions"]),
                    "stored_decisions": dict(final_row["stored_decisions"]),
                    "round1_model_evidence": (
                        dict(round1_output) if round1_output else None
                    ),
                    "round2_model_evidence": (
                        dict(round2_output) if round2_output else None
                    ),
                    "round2_retry_model_evidence": (
                        dict(retry_output) if retry_output else None
                    ),
                    "effective_round2_source": effective_round2_source,
                    "effective_round2_model_evidence": (
                        dict(effective_round2_output) if effective_round2_output else None
                    ),
                    "local_criteria": dict(final_row["local_criteria"]),
                    "offline_severity": severity,
                    "issue_types": issue_types,
                    "resolution": {
                        "simplification": final_simplification,
                        "simplification_source": simplification_source,
                        "reference": final_reference,
                        "reference_source": reference_source,
                    },
                }
            )

    statistics = _build_statistics(final_rows)
    priced_known = [
        priced_round1["priced_cost_cny"],
        priced_round2["priced_cost_cny"],
        priced_retry["priced_cost_cny"],
    ]
    if priced_external["priced_cost_cny"] is not None:
        priced_known.append(priced_external["priced_cost_cny"])
    cache_creation_unpriced = sum(
        int(item.get("cache_creation_input_tokens") or 0)
        for item in (priced_external, priced_round1, priced_round2, priced_retry)
    )
    budget: JsonDict = {
        "maximum_total_physical_attempts": maximum_total_physical_attempts,
        "total_physical_attempt_count": total_attempts,
        "within_hard_limit": True,
        "external": priced_external,
        "round1": priced_round1,
        "round2": priced_round2,
        "round2_retry": priced_retry,
        "prices_cny_per_million_tokens": {
            "input": float(prices["input"]),
            "output": float(prices["output"]),
            "cache_read": float(prices["cache_read"]),
            "cache_creation": None,
        },
        "priced_cost_cny": float(sum(Decimal(str(value)) for value in priced_known)),
        "priced_cost_cny_known_usage_only": float(
            sum(Decimal(str(value)) for value in priced_known)
        ),
        "two_round_usage_complete": bool(
            priced_round1["usage_complete"]
            and priced_round2["usage_complete"]
            and priced_retry["usage_complete"]
        ),
        "all_sources_usage_complete": bool(
            priced_external["usage_complete"]
            and priced_round1["usage_complete"]
            and priced_round2["usage_complete"]
            and priced_retry["usage_complete"]
        ),
        "cache_creation_input_tokens_unpriced": cache_creation_unpriced,
        "external_usage_included_in_cost": priced_external["priced_cost_cny"] is not None,
    }

    output_path = Path(output_root).resolve()
    final_path = output_path / "results/formula_audit_final.jsonl"
    issues_path = output_path / "results/formula_audit_issues.jsonl"
    statistics_path = output_path / "reports/formula_audit_statistics.json"
    statistics_csv_path = output_path / "reports/formula_audit_statistics.csv"
    markdown_path = output_path / "reports/formula_audit_final_report.md"
    _atomic_write_jsonl(final_path, final_rows)
    _atomic_write_jsonl(issues_path, issues)
    statistics_document = {
        "statistics": statistics,
        "budget": budget,
        "inputs": {
            "round1_plan_sha256": _sha256_file(round1_plan_path),
            "round2_plan_sha256": (
                _sha256_file(round2_plan_path) if round2_plan_path is not None else None
            ),
            "round2_retry_plan_sha256": (
                _sha256_file(retry_plan_path) if retry_plan_path is not None else None
            ),
            "round1_comparisons_sha256": _sha256_file(
                Path(round1_comparisons_jsonl).resolve()
            ),
            "local_verification_sha256": _sha256_file(
                Path(local_verification_jsonl).resolve()
            ),
            "sample_manifest_sha256": _sha256_file(
                Path(sample_manifest_jsonl).resolve()
            ),
        },
    }
    _atomic_write_json(statistics_path, statistics_document)
    _write_statistics_csv(statistics_csv_path, _statistics_csv_rows(statistics))
    _atomic_write_text(
        markdown_path,
        _markdown_report(
            final_rows=final_rows,
            issues=issues,
            statistics=statistics,
            budget=budget,
        ),
    )
    return {
        "status": "ok",
        "final_count": len(final_rows),
        "issue_count": len(issues),
        "budget": budget,
        "outputs": {
            "final_jsonl": str(final_path),
            "issues_jsonl": str(issues_path),
            "statistics_json": str(statistics_path),
            "statistics_csv": str(statistics_csv_path),
            "markdown_report": str(markdown_path),
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def _nonnegative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能为负整数")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成公式审计最终裁决与成本报告")
    parser.add_argument("--round1-plan-jsonl", type=Path, required=True)
    parser.add_argument("--round1-comparisons-jsonl", type=Path, required=True)
    parser.add_argument("--round1-state-db", type=Path, required=True)
    parser.add_argument("--local-verification-jsonl", type=Path, required=True)
    parser.add_argument("--sample-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round2-plan-jsonl", type=Path)
    parser.add_argument("--round2-state-db", type=Path)
    parser.add_argument("--retry-plan-jsonl", type=Path)
    parser.add_argument("--retry-state-db", type=Path)
    parser.add_argument("--round1-attempts-dir", type=Path)
    parser.add_argument("--round2-attempts-dir", type=Path)
    parser.add_argument("--retry-attempts-dir", type=Path)
    parser.add_argument("--external-attempts-root", type=Path)
    parser.add_argument("--external-attempt-count", type=_nonnegative_int_arg)
    parser.add_argument(
        "--maximum-total-physical-attempts",
        type=_positive_int,
        default=DEFAULT_MAXIMUM_PHYSICAL_ATTEMPTS,
    )
    parser.add_argument("--input-price-cny-per-million", default="3")
    parser.add_argument("--output-price-cny-per-million", default="15")
    parser.add_argument("--cache-read-price-cny-per-million", default="0.3")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = finalize_formula_audit(
            round1_plan_jsonl=args.round1_plan_jsonl,
            round1_comparisons_jsonl=args.round1_comparisons_jsonl,
            round1_state_db=args.round1_state_db,
            local_verification_jsonl=args.local_verification_jsonl,
            sample_manifest_jsonl=args.sample_manifest_jsonl,
            output_root=args.output_root,
            round2_plan_jsonl=args.round2_plan_jsonl,
            round2_state_db=args.round2_state_db,
            retry_plan_jsonl=args.retry_plan_jsonl,
            retry_state_db=args.retry_state_db,
            round1_attempts_dir=args.round1_attempts_dir,
            round2_attempts_dir=args.round2_attempts_dir,
            retry_attempts_dir=args.retry_attempts_dir,
            external_attempts_root=args.external_attempts_root,
            external_attempt_count=args.external_attempt_count,
            maximum_total_physical_attempts=args.maximum_total_physical_attempts,
            input_price_cny_per_million=args.input_price_cny_per_million,
            output_price_cny_per_million=args.output_price_cny_per_million,
            cache_read_price_cny_per_million=args.cache_read_price_cny_per_million,
        )
    except (FormulaAuditFinalizationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
