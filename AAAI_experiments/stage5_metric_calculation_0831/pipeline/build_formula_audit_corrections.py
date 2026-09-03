"""基于 clean 公式审计结论构建版本化修正 overlay。

该模块故意不修改既有 frozen plan/index 或 aggregate 脚本。``draft`` 阶段
只验证来源并生成 corrected views、确定性 evidence 和结构重判计划；``final``
阶段在六（或本次审计实际产生的全部）结构任务结果齐全后，原子发布一个新的
corrections manifest。所有输入文件的字节级 SHA256 都写入 manifest，便于在
聚合前拒绝陈旧或被替换的输入。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from .claude_contract import canonical_json, evaluation_key, render_prompt
from .symbolic_evidence import SymbolicEvidenceError, build_pair_evidence


JsonDict = dict[str, Any]

AUDIT_SCHEMA_VERSION = "audit_corrections.v1"
AUDIT_ROW_SCHEMA_VERSION = "audit_corrections.overlay_row.v1"
CONDITION = "clean"
KNOWN_SEVERITIES = {"pass", "minor", "major", "critical", "undetermined"}
DETERMINATE_REFERENCE = {"equivalent", "not_equivalent"}
SIMPLIFICATION_DECISIONS = {"preserved", "not_preserved", "undetermined"}
STRUCTURE_DECISIONS = {
    "mathematically_equivalent",
    "same_canonical_structure",
    "different_structure",
    "undetermined",
}
SEEDS = (520, 521, 522)
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))

PRED_LOGICAL_ID_RE = re.compile(
    r"^pred_simplify::([a-z0-9_]+)::(g\d{4})::s(520|521|522)::clean(?:::v[1-9]\d*)?$"
)
GT_LOGICAL_ID_RE = re.compile(r"^gt_simplify::([^:]+)(?:::(v2))?$")
STRUCTURE_LOGICAL_ID_RE = re.compile(
    r"^stab_structure::([a-z0-9_]+)::(g\d{4})::s(520|521)-s(520|521|522)$"
)

# clean 审计中最终 effective expression 发生身份变化的规则会覆盖
# critical 和 undetermined 两种严重度；不能只筛 critical。
_FALLBACK_FINAL_DECISIONS = {"not_preserved", "undetermined"}


class CorrectionOverlayError(RuntimeError):
    """修正来源、身份或下游闭环不满足安全契约。"""


def _canonical(value: object) -> str:
    return canonical_json(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:  # pragma: no cover - exercised through public wrapper
        raise CorrectionOverlayError(f"无法读取文件 {path}: {exc}") from exc


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join((_canonical(dict(row)) + "\n").encode("utf-8") for row in rows)
    _atomic_write(path, payload)


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionOverlayError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorrectionOverlayError(f"{path} 顶层必须是 JSON object")
    return dict(value)


def _read_jsonl(path: Path, *, allow_empty: bool = True) -> list[JsonDict]:
    if not path.is_file():
        raise CorrectionOverlayError(f"JSONL 不存在: {path}")
    rows: list[JsonDict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorrectionOverlayError(f"无法读取 JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorrectionOverlayError(f"{path}:{line_number} JSON 解析失败") from exc
        if not isinstance(value, dict):
            raise CorrectionOverlayError(f"{path}:{line_number} 顶层必须是 JSON object")
        rows.append(dict(value))
    if not rows and not allow_empty:
        raise CorrectionOverlayError(f"JSONL 不能为空: {path}")
    return rows


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorrectionOverlayError(f"{context} 必须是非空字符串")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve()


def _request(row: Mapping[str, Any]) -> JsonDict:
    direct = row.get("request")
    if isinstance(direct, Mapping):
        return dict(direct)
    normalized = row.get("normalized_input")
    if isinstance(normalized, Mapping) and isinstance(normalized.get("request"), Mapping):
        return dict(normalized["request"])
    # Frozen index rows do not contain a request and are intentionally accepted
    # by callers only where a plan row is already available.
    return {}


def _unique_rows(rows: Sequence[Mapping[str, Any]], *, key: str, context: str) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for row in rows:
        value = _require_string(row.get(key), context=f"{context}.{key}")
        if value in result:
            raise CorrectionOverlayError(f"{context} 出现重复 {key}: {value}")
        result[value] = dict(row)
    return result


def _file_record(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": len(rows),
    }


def _parse_pred(logical_id: str) -> tuple[str, str, int]:
    match = PRED_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise CorrectionOverlayError(f"pred logical_id 非 canonical: {logical_id}")
    return match.group(1), match.group(2), int(match.group(3))


def _parse_gt(logical_id: str) -> str:
    match = GT_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise CorrectionOverlayError(f"GT logical_id 非 canonical: {logical_id}")
    return match.group(1)


def _parse_structure(logical_id: str) -> tuple[str, str, tuple[int, int]]:
    match = STRUCTURE_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise CorrectionOverlayError(f"structure logical_id 非 canonical: {logical_id}")
    left, right = sorted((int(match.group(3)), int(match.group(4))))
    pair = (left, right)
    if pair not in SEED_PAIRS:
        raise CorrectionOverlayError(f"structure seed pair 非法: {logical_id}")
    return match.group(1), match.group(2), pair


def _base_input_paths(base_inputs: Mapping[str, str | Path | None] | None) -> dict[str, Path]:
    if not base_inputs:
        raise CorrectionOverlayError("必须显式提供 base_inputs")
    result: dict[str, Path] = {}
    for key, value in base_inputs.items():
        if value is None:
            continue
        result[str(key)] = _resolve_path(value)
    for required in ("gt_plan", "gt_index", "pred_plan", "pred_index"):
        if required not in result:
            raise CorrectionOverlayError(f"base_inputs 缺少 {required}")
    return result


def _load_bases(base_inputs: Mapping[str, Path]) -> tuple[dict[str, JsonDict], dict[str, dict[str, JsonDict]], dict[str, dict[str, JsonDict]]]:
    files: dict[str, JsonDict] = {}
    rows_by_kind: dict[str, dict[str, JsonDict]] = {}
    raw_rows: dict[str, list[JsonDict]] = {}
    for kind, path in base_inputs.items():
        if path.suffix.lower() in {".json", ".yaml", ".yml"} and not path.name.endswith(".jsonl"):
            try:
                payload = _read_json(path)
            except CorrectionOverlayError:
                # summary YAML/JSON is tracked by bytes; semantic validation is
                # intentionally left to the owning pipeline.
                files[kind] = {"path": str(path), "sha256": _sha256_file(path), "row_count": None}
                continue
            files[kind] = {"path": str(path), "sha256": _sha256_file(path), "row_count": None}
            files[kind]["object_sha256"] = _sha256_json(payload)
            continue
        rows = _read_jsonl(path)
        raw_rows[kind] = rows
        files[kind] = _file_record(path, rows)
        if kind.endswith("_plan") or kind.endswith("_index") or kind in {"evidence", "audit_final"}:
            key = "logical_id"
            if kind == "evidence":
                key = "pred_logical_id"
            rows_by_kind[kind] = _unique_rows(rows, key=key, context=kind)
    # Return raw rows as a private side channel for callers that need evidence
    # duplicates checked by another key (e.g. logical_key).
    return files, rows_by_kind, raw_rows


def _plan_and_index_consistency(
    *,
    kind: str,
    plans: Mapping[str, JsonDict],
    indexes: Mapping[str, JsonDict],
) -> None:
    if not plans or not indexes:
        return
    for logical_id, index in indexes.items():
        plan = plans.get(logical_id)
        if plan is None:
            raise CorrectionOverlayError(f"{kind} index 存在 plan 缺失: {logical_id}")
        if index.get("evaluation_key") != plan.get("evaluation_key"):
            raise CorrectionOverlayError(f"{kind} index/plan evaluation_key 漂移: {logical_id}")


def _validate_audit_row(row: Mapping[str, Any], *, line_number: int) -> None:
    logical_id = _require_string(row.get("audit_logical_id"), context=f"audit:{line_number}.audit_logical_id")
    if row.get("condition") != CONDITION:
        return
    declared = _require_string(row.get("final_record_sha256"), context=f"{logical_id}.final_record_sha256")
    actual = _sha256_json({key: value for key, value in row.items() if key != "final_record_sha256"})
    if declared != actual:
        raise CorrectionOverlayError(f"{logical_id} final_record_sha256 校验失败")
    severity = row.get("offline_severity")
    if severity not in KNOWN_SEVERITIES:
        raise CorrectionOverlayError(f"{logical_id} offline_severity 未知: {severity!r}")


def _source_for_audit(row: Mapping[str, Any]) -> tuple[str, str, JsonDict]:
    identity = row.get("source_identity")
    if not isinstance(identity, Mapping):
        raise CorrectionOverlayError(f"{row.get('audit_logical_id')} 缺少 source_identity")
    logical_id = _require_string(identity.get("source_formula_logical_id"), context="source_identity.logical_id")
    evaluation = _require_string(identity.get("source_formula_evaluation_key"), context=f"{logical_id}.source_evaluation_key")
    result_sha = _require_string(identity.get("source_formula_result_sha256"), context=f"{logical_id}.source_result_sha256")
    return logical_id, evaluation, {"result_sha256": result_sha}


def _original_expression(request: Mapping[str, Any]) -> str | None:
    for key in ("original_expression", "expression"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _expression_from_index(index: Mapping[str, Any]) -> str | None:
    value = index.get("effective_expression")
    return value if isinstance(value, str) and value.strip() else None


def _audit_corrections(
    audit_rows: Sequence[JsonDict],
    *,
    gt_plans: Mapping[str, JsonDict],
    gt_indexes: Mapping[str, JsonDict],
    pred_plans: Mapping[str, JsonDict],
    pred_indexes: Mapping[str, JsonDict],
    severity_allowlist: set[str] | None,
) -> tuple[dict[str, JsonDict], dict[str, JsonDict], list[JsonDict], list[JsonDict], dict[str, JsonDict]]:
    gt_patch: dict[str, JsonDict] = {}
    pred_patch: dict[str, JsonDict] = {}
    corrections: list[JsonDict] = []
    eq_overrides: list[JsonDict] = []
    audit_by_source: dict[str, JsonDict] = {}

    for line_number, audit in enumerate(audit_rows, start=1):
        _validate_audit_row(audit, line_number=line_number)
        if audit.get("condition") != CONDITION:
            continue
        source_logical, source_eval, source_extra = _source_for_audit(audit)
        if source_logical in audit_by_source:
            raise CorrectionOverlayError(f"audit source logical_id 重复: {source_logical}")
        audit_by_source[source_logical] = audit
        if source_logical.startswith("gt_simplify::"):
            plan = gt_plans.get(source_logical)
            index = gt_indexes.get(source_logical)
        elif source_logical.startswith("pred_simplify::"):
            plan = pred_plans.get(source_logical)
            index = pred_indexes.get(source_logical)
        else:
            raise CorrectionOverlayError(f"无法识别 audit source logical_id: {source_logical}")
        if plan is None or index is None:
            raise CorrectionOverlayError(f"audit source 在 base plan/index 中不存在: {source_logical}")
        if plan.get("evaluation_key") != source_eval or index.get("evaluation_key") != source_eval:
            raise CorrectionOverlayError(f"audit source evaluation_key 漂移: {source_logical}")
        if index.get("result_sha256") != source_extra["result_sha256"]:
            raise CorrectionOverlayError(f"audit source result_sha256 漂移: {source_logical}")
        expressions = audit.get("expressions")
        if not isinstance(expressions, Mapping):
            raise CorrectionOverlayError(f"{source_logical} 缺少 expressions")
        original = expressions.get("original")
        if not isinstance(original, str) or not original.strip():
            raise CorrectionOverlayError(f"{source_logical}.expressions.original 缺失")
        plan_original = _original_expression(_request(plan))
        if plan_original is not None and plan_original != original:
            raise CorrectionOverlayError(f"{source_logical} audit original 与 plan request 不一致")
        stored = audit.get("stored_decisions")
        stored_map = dict(stored) if isinstance(stored, Mapping) else {}
        final_simplification = audit.get("final_simplification_decision")
        severity = str(audit.get("offline_severity"))
        enabled = severity_allowlist is None or severity in severity_allowlist

        # 只要旧值是 preserved 而最终审计不再确认 preserved，就回到原式。
        # 这同时覆盖 critical not_preserved 和四条 undetermined 结论。
        if (
            enabled
            and stored_map.get("simplification_decision") == "preserved"
            and final_simplification in _FALLBACK_FINAL_DECISIONS
        ):
            patch = {
                "effective_expression": original,
                "expression_resolution": "original_identity_fallback_after_audit",
                "source_audit_logical_id": audit["audit_logical_id"],
                "source_final_record_sha256": audit["final_record_sha256"],
                "severity": severity,
                "suggested_simplified_expression": expressions.get("candidate_simplified"),
            }
            target = gt_patch if source_logical.startswith("gt_simplify::") else pred_patch
            target[source_logical] = patch
            corrections.append(
                _correction_row(
                    audit=audit,
                    target_kind="gt" if source_logical.startswith("gt_simplify::") else "pred",
                    target_logical_id=source_logical,
                    action="replace_expression",
                    before={"effective_expression": _expression_from_index(index)},
                    after={"effective_expression": original},
                    base_row=index,
                    basis={
                        "final_simplification_decision": final_simplification,
                        "stored_simplification_decision": stored_map.get("simplification_decision"),
                        "resolution_source": audit.get("simplification_resolution_source"),
                    },
                )
            )

        final_reference = audit.get("final_reference_decision")
        old_reference = stored_map.get("reference_equivalence")
        # 正式口径消费全部 clean 最终 reference 结论：确定性翻转、确定性
        # -> undetermined 以及 undetermined -> not_equivalent 均记录 successor。
        if (
            enabled
            and old_reference in {"equivalent", "not_equivalent", "undetermined"}
            and final_reference in {"equivalent", "not_equivalent", "undetermined"}
            and old_reference != final_reference
            and source_logical.startswith("pred_simplify::")
        ):
            algorithm, dataset_index, seed = _parse_pred(source_logical)
            eq_id = f"equivalence::{algorithm}::{dataset_index}::s{seed}::clean"
            eq_overrides.append(
                {
                    "target_logical_id": eq_id,
                    "source_audit_logical_id": audit["audit_logical_id"],
                    "source_final_record_sha256": audit["final_record_sha256"],
                    "before_decision": old_reference,
                    "after_decision": final_reference,
                    "severity": severity,
                }
            )
            corrections.append(
                _correction_row(
                    audit=audit,
                    target_kind="equivalence",
                    target_logical_id=eq_id,
                    action="set_equivalence",
                    before={"decision": old_reference},
                    after={"decision": final_reference},
                    base_row=None,
                    basis={
                        "final_reference_decision": final_reference,
                        "stored_reference_decision": old_reference,
                        "resolution_source": audit.get("reference_resolution_source"),
                    },
                )
            )
    return gt_patch, pred_patch, corrections, eq_overrides, audit_by_source


def _correction_row(
    *,
    audit: Mapping[str, Any],
    target_kind: str,
    target_logical_id: str,
    action: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    base_row: Mapping[str, Any] | None,
    basis: Mapping[str, Any],
) -> JsonDict:
    source_hash = _require_string(audit.get("final_record_sha256"), context="audit.final_record_sha256")
    base_hash = _sha256_json(base_row) if base_row is not None else None
    correction_id = _sha256_json(
        {
            "source_final_record_sha256": source_hash,
            "target_logical_id": target_logical_id,
            "action": action,
            "base_row_sha256": base_hash,
        }
    )
    return {
        "schema_version": AUDIT_ROW_SCHEMA_VERSION,
        "correction_id": correction_id,
        "target_kind": target_kind,
        "target_logical_id": target_logical_id,
        "condition": CONDITION,
        "source_audit_logical_id": audit["audit_logical_id"],
        "source_final_record_sha256": source_hash,
        "severity": audit.get("offline_severity"),
        "action": action,
        "base_row_sha256": base_hash,
        "before": dict(before),
        "after": dict(after),
        "deterministic_basis": dict(basis),
        "requires_rebuild": action == "replace_expression",
    }


def _probe(request: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]] | None, str | None, str | None, list[str]]:
    raw = request.get("dataset_probe_evidence")
    if not isinstance(raw, Mapping):
        deterministic = request.get("deterministic_evidence")
        if isinstance(deterministic, Mapping):
            raw = deterministic.get("dataset_probe")
    if not isinstance(raw, Mapping):
        return None, None, None, list(request.get("variables", [])) if isinstance(request.get("variables"), list) else []
    points = raw.get("points")
    points_list = list(points) if isinstance(points, list) else None
    source = request.get("probe_source", raw.get("schema_version"))
    sample = request.get("probe_sample_sha256", raw.get("sample_sha256"))
    variables = raw.get("variables", request.get("variables", []))
    return (
        points_list,
        str(source) if isinstance(source, str) else None,
        str(sample) if isinstance(sample, str) else None,
        [str(item) for item in variables] if isinstance(variables, list) else [],
    )


def _pair_payload(
    *,
    phase: str,
    lhs_logical_id: str,
    lhs_eval: str,
    lhs_expression: str,
    lhs_request: Mapping[str, Any],
    rhs_logical_id: str,
    rhs_eval: str,
    rhs_expression: str,
    rhs_request: Mapping[str, Any],
    seed: int,
    overlay_revision: str,
) -> JsonDict:
    lhs_points, lhs_source, lhs_sample, lhs_variables = _probe(lhs_request)
    rhs_points, rhs_source, rhs_sample, rhs_variables = _probe(rhs_request)
    points = lhs_points or rhs_points
    if lhs_points is not None and rhs_points is not None and lhs_points != rhs_points:
        raise CorrectionOverlayError(f"{lhs_logical_id} / {rhs_logical_id} probe points 不一致")
    variables = lhs_variables or rhs_variables
    if lhs_variables and rhs_variables and lhs_variables != rhs_variables:
        raise CorrectionOverlayError(f"{lhs_logical_id} / {rhs_logical_id} probe variables 不一致")
    functions = sorted(
        {
            str(value)
            for request in (lhs_request, rhs_request)
            for value in (request.get("allowed_functions") or [])
            if isinstance(value, str)
        }
    )
    try:
        pair = build_pair_evidence(
            lhs_expression,
            rhs_expression,
            allowed_variables=variables or None,
            allowed_functions=functions or None,
            seed=seed,
            probe_points=points,
            probe_source=lhs_source or rhs_source,
            probe_sample_sha256=lhs_sample or rhs_sample,
            include_tree_distance=True,
        )
    except SymbolicEvidenceError as exc:
        raise CorrectionOverlayError(
            f"{lhs_logical_id} / {rhs_logical_id} pair evidence 构建失败: {exc}"
        ) from exc
    payload: JsonDict = {
        "schema_version": "symbolic_pair_evidence.v2",
        "phase": phase,
        "pair_seed": seed,
        "overlay_revision_sha256": overlay_revision,
        "pair_evidence": pair,
        "lhs_binding": {
            "role": "lhs",
            "frozen_logical_id": lhs_logical_id,
            "frozen_evaluation_key": lhs_eval,
            "frozen_effective_expression": lhs_expression,
        },
        "rhs_binding": {
            "role": "rhs",
            "frozen_logical_id": rhs_logical_id,
            "frozen_evaluation_key": rhs_eval,
            "frozen_effective_expression": rhs_expression,
        },
        "allowed_variables": variables,
        "allowed_functions": functions,
    }
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def _effective_views(
    rows: Mapping[str, JsonDict],
    *,
    patch: Mapping[str, JsonDict],
    base_file_sha256: str,
    overlay_revision: str,
) -> list[JsonDict]:
    result: list[JsonDict] = []
    for logical_id, source in rows.items():
        row = dict(source)
        row["base_file_sha256"] = base_file_sha256
        row["base_row_sha256"] = _sha256_json(source)
        row["overlay_revision_sha256"] = overlay_revision
        correction = patch.get(logical_id)
        if correction is None:
            row["overlay_action"] = "unchanged"
        else:
            row.update(
                {
                    "effective_expression": correction["effective_expression"],
                    "expression_resolution": correction["expression_resolution"],
                    "overlay_action": "replace_expression",
                    "source_audit_logical_id": correction["source_audit_logical_id"],
                    "source_final_record_sha256": correction["source_final_record_sha256"],
                    # 仅记录候选用于审计，绝不将其作为 effective expression。
                    "suggested_simplified_expression": correction.get("suggested_simplified_expression"),
                }
            )
        row["output_row_sha256"] = _sha256_json(row)
        result.append(row)
    return result


def _affected_structure_ids(pred_patch: Mapping[str, JsonDict]) -> list[str]:
    result: set[str] = set()
    for logical_id in pred_patch:
        algorithm, dataset_index, seed = _parse_pred(logical_id)
        for left, right in SEED_PAIRS:
            if seed in (left, right):
                result.add(f"stab_structure::{algorithm}::{dataset_index}::s{left}-s{right}")
    return sorted(result)


def _find_pred_source(
    pred_plans: Mapping[str, JsonDict],
    *,
    algorithm: str,
    dataset_index: str,
    seed: int,
) -> str:
    """按三元身份解析 pred successor 后缀，避免把 ``::v2`` 丢掉。"""

    candidates: list[str] = []
    for logical_id in pred_plans:
        try:
            current_algorithm, current_dataset, current_seed = _parse_pred(logical_id)
        except CorrectionOverlayError:
            continue
        if (current_algorithm, current_dataset, current_seed) == (algorithm, dataset_index, seed):
            candidates.append(logical_id)
    if not candidates:
        raise CorrectionOverlayError(
            f"pred source 缺失: {algorithm}/{dataset_index}/s{seed}"
        )
    # 同一身份若同时存在 predecessor/successor，优先 canonical 无后缀，
    # 否则按版本号升序取最后一个，保证 plan/index 使用同一版本。
    candidates.sort(key=lambda value: ("::v" in value, value))
    return candidates[0]


def _structure_contract() -> tuple[Path, str, str, str, dict[str, Any]]:
    stage = Path(__file__).resolve().parents[1]
    prompt_path = stage / "config/prompts/structure.v1.txt"
    schema_path = stage / "config/schemas/structure.v1.json"
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionOverlayError(f"structure prompt/schema 不可读: {exc}") from exc
    if not isinstance(schema, dict):
        raise CorrectionOverlayError("structure schema 顶层必须是 object")
    return (
        prompt_path.resolve(),
        prompt,
        _sha256_file(prompt_path),
        _sha256_file(schema_path),
        schema,
    )


def _structure_plan_rows(
    *,
    structure_ids: Sequence[str],
    pred_plans: Mapping[str, JsonDict],
    pred_indexes: Mapping[str, JsonDict],
    old_structure_plans: Mapping[str, JsonDict],
    pred_patch: Mapping[str, JsonDict],
    overlay_revision: str,
) -> list[JsonDict]:
    prompt_path, prompt, prompt_sha, schema_sha, schema = _structure_contract()
    schema_path = prompt_path.parent.parent / "schemas" / "structure.v1.json"
    rows: list[JsonDict] = []
    for logical_id in structure_ids:
        algorithm, dataset_index, (seed_a, seed_b) = _parse_structure(logical_id)
        left_id = _find_pred_source(
            pred_plans, algorithm=algorithm, dataset_index=dataset_index, seed=seed_a
        )
        right_id = _find_pred_source(
            pred_plans, algorithm=algorithm, dataset_index=dataset_index, seed=seed_b
        )
        left_plan = pred_plans.get(left_id)
        right_plan = pred_plans.get(right_id)
        left_index = pred_indexes.get(left_id)
        right_index = pred_indexes.get(right_id)
        if not left_plan or not right_plan or not left_index or not right_index:
            raise CorrectionOverlayError(f"structure source 缺失: {logical_id}")
        left_request = _request(left_plan)
        right_request = _request(right_plan)
        left_expression = str((pred_patch.get(left_id) or {}).get("effective_expression") or _expression_from_index(left_index) or "")
        right_expression = str((pred_patch.get(right_id) or {}).get("effective_expression") or _expression_from_index(right_index) or "")
        if not left_expression or not right_expression:
            raise CorrectionOverlayError(f"{logical_id} source expression 缺失")
        pair = _pair_payload(
            phase="structure",
            lhs_logical_id=left_id,
            lhs_eval=str(left_index.get("evaluation_key")),
            lhs_expression=left_expression,
            lhs_request=left_request,
            rhs_logical_id=right_id,
            rhs_eval=str(right_index.get("evaluation_key")),
            rhs_expression=right_expression,
            rhs_request=right_request,
            seed=seed_a * 1000 + seed_b,
            overlay_revision=overlay_revision,
        )
        old = old_structure_plans.get(logical_id, {})
        request = _request(old)
        if not request:
            request = {
                "algorithm": algorithm,
                "algorithm_slug": algorithm,
                "dataset_id": right_request.get("dataset_id", left_request.get("dataset_id", dataset_index)),
                "dataset_index": dataset_index,
                "noise_tag": CONDITION,
                "seed_a": seed_a,
                "seed_b": seed_b,
            }
        request.update(
            {
                "algorithm": request.get("algorithm", algorithm),
                "algorithm_slug": algorithm,
                "dataset_index": dataset_index,
                "noise_tag": CONDITION,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "effective_prediction_a_expression": left_expression,
                "effective_prediction_b_expression": right_expression,
                "simplified_prediction_a_expression": left_expression,
                "simplified_prediction_b_expression": right_expression,
                "prediction_a_logical_id": left_id,
                "prediction_b_logical_id": right_id,
                "prediction_a_plan_evaluation_key": left_plan.get("evaluation_key"),
                "prediction_b_plan_evaluation_key": right_plan.get("evaluation_key"),
                "prediction_a_frozen_evaluation_key": left_index.get("evaluation_key"),
                "prediction_b_frozen_evaluation_key": right_index.get("evaluation_key"),
                "prediction_a_result_sha256": left_index.get("result_sha256"),
                "prediction_b_result_sha256": right_index.get("result_sha256"),
                "deterministic_evidence": pair,
                "deterministic_pair_evidence": pair,
                "overlay_revision_sha256": overlay_revision,
                "model": "claude-opus-5",
                "effort": "xhigh",
                "stream": False,
            }
        )
        request["evidence_hash"] = pair["evidence_sha256"]
        normalized_input = {
            "request": request,
            "prompt_sha256": prompt_sha,
            "schema_sha256": schema_sha,
        }
        input_hash = _sha256_json(normalized_input)
        task_key = evaluation_key(
            task_type="stab_structure",
            logical_id=logical_id,
            prompt_version="structure.v1",
            schema_version="structure.v1",
            prompt_sha256=prompt_sha,
            schema_sha256=schema_sha,
            normalized_input=normalized_input,
            evidence_hash=str(pair["evidence_sha256"]),
        )
        # 这是独立 successor 计划，不能把 predecessor evaluation keys 当作
        # state DB dependencies，否则新库会永久停在 pending。两侧身份、结果和
        # 表达式 hash 已写进 request/pair evidence 作为可审计绑定。
        dependencies: list[str] = []
        row: JsonDict = {
            "logical_id": logical_id,
            "evaluation_key": task_key,
            "task_type": "stab_structure",
            "task_kind": "structure",
            "condition": CONDITION,
            "priority": 40,
            "input_hash": input_hash,
            "prompt_version": "structure.v1",
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_sha,
            "prompt_template": prompt,
            "schema_version": "structure.v1",
            "schema_path": str(schema_path.resolve()),
            "schema_sha256": schema_sha,
            "schema_content": schema,
            "dependencies": dependencies,
            "normalized_input": normalized_input,
            "request": request,
            "execution": {"model": "claude-opus-5", "effort": "xhigh", "stream": False, "max_attempts": 1},
            "overlay_revision_sha256": overlay_revision,
        }
        try:
            row["rendered_prompt"] = render_prompt(prompt, request, schema)
        except Exception as exc:  # pragma: no cover - catches contract drift explicitly
            raise CorrectionOverlayError(f"{logical_id} structure prompt 渲染失败: {exc}") from exc
        row["task_spec"] = {
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "task_type": "stab_structure",
            "condition": CONDITION,
            "priority": 40,
            "input_hash": input_hash,
            "prompt_version": "structure.v1",
            "schema_version": "structure.v1",
            "dependencies": dependencies,
        }
        rows.append(row)
    rows.sort(key=lambda item: str(item["logical_id"]))
    return rows


def _structure_stale_rows(structure_ids: Sequence[str], structure_indexes: Mapping[str, JsonDict], structure_file_sha: str | None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for logical_id in structure_ids:
        base = structure_indexes.get(logical_id)
        rows.append(
            {
                "schema_version": AUDIT_ROW_SCHEMA_VERSION,
                "target_kind": "structure",
                "target_logical_id": logical_id,
                "action": "mark_stale",
                "base_file_sha256": structure_file_sha,
                "base_row_sha256": _sha256_json(base) if base is not None else None,
                "requires_rebuild": True,
            }
        )
    return rows


def _rebuild_evidence(
    *,
    pred_ids: Iterable[str],
    gt_by_dataset: Mapping[str, JsonDict],
    pred_plans: Mapping[str, JsonDict],
    pred_indexes: Mapping[str, JsonDict],
    gt_patch: Mapping[str, JsonDict],
    pred_patch: Mapping[str, JsonDict],
    base_evidence: Mapping[str, JsonDict],
    overlay_revision: str,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for pred_id in sorted(set(pred_ids)):
        algorithm, dataset_index, seed = _parse_pred(pred_id)
        pred_plan = pred_plans.get(pred_id)
        pred_index = pred_indexes.get(pred_id)
        if pred_plan is None or pred_index is None:
            raise CorrectionOverlayError(f"evidence pred source 缺失: {pred_id}")
        pred_request = _request(pred_plan)
        dataset_id = pred_request.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise CorrectionOverlayError(f"{pred_id} request.dataset_id 缺失")
        gt_id = next((logical for logical, row in gt_by_dataset.items() if _request(row).get("dataset_id") == dataset_id), None)
        # gt_by_dataset 是按 GT logical_id 建立的；兼容调用者传入按 dataset_id 的映射。
        if gt_id is None and dataset_id in gt_by_dataset:
            gt_id = dataset_id
        if gt_id is None:
            raise CorrectionOverlayError(f"{pred_id} 无法绑定 GT dataset_id={dataset_id}")
        gt_plan = gt_by_dataset.get(gt_id)
        if gt_plan is None:
            raise CorrectionOverlayError(f"GT source 缺失: {gt_id}")
        gt_index = gt_plan.get("__index") if isinstance(gt_plan.get("__index"), Mapping) else None
        if gt_index is None:
            raise CorrectionOverlayError(f"GT source index 缺失: {gt_id}")
        gt_expression = str((gt_patch.get(gt_id) or {}).get("effective_expression") or _expression_from_index(gt_index) or "")
        pred_expression = str((pred_patch.get(pred_id) or {}).get("effective_expression") or _expression_from_index(pred_index) or "")
        pair = _pair_payload(
            phase="equivalence",
            lhs_logical_id=gt_id,
            lhs_eval=str(gt_index.get("evaluation_key")),
            lhs_expression=gt_expression,
            lhs_request=_request(gt_plan),
            rhs_logical_id=pred_id,
            rhs_eval=str(pred_index.get("evaluation_key")),
            rhs_expression=pred_expression,
            rhs_request=pred_request,
            seed=seed,
            overlay_revision=overlay_revision,
        )
        pair_evidence = pair["pair_evidence"]
        old = dict(base_evidence.get(pred_id, {}))
        algorithm_name = str(pred_request.get("algorithm", algorithm))
        out: JsonDict = {
            "logical_key": f"{algorithm_name}::{dataset_id}::s{seed}::{CONDITION}",
            "gt_logical_id": gt_id,
            "pred_logical_id": pred_id,
            "evidence_hash": pair["evidence_sha256"],
            "ground_truth": {
                "simplified_expression": gt_expression,
                "artifact_sha256": pair_evidence["lhs_artifact"]["artifact_sha256"],
            },
            "prediction": {
                "simplified_expression": pred_expression,
                "artifact_sha256": pair_evidence["rhs_artifact"]["artifact_sha256"],
            },
            "tree": {"tree_similarity": pair_evidence["tree"]["tree_similarity"]},
            "variable": {"f1": pair_evidence["variable"]["f1"]},
            "operator": {"f1": pair_evidence["operator"]["f1"]},
            "pair_evidence": pair,
            "base_evidence_row_sha256": _sha256_json(old) if old else None,
            "overlay_revision_sha256": overlay_revision,
        }
        out["output_row_sha256"] = _sha256_json(out)
        rows.append(out)
    rows.sort(key=lambda item: str(item["logical_key"]))
    return rows


def _manifest_output_record(path: Path, *, row_count: int | None = None) -> JsonDict:
    return {"path": str(path), "sha256": _sha256_file(path), "row_count": row_count}


def build_draft(
    *,
    audit_final_jsonl: str | Path,
    base_inputs: Mapping[str, str | Path | None],
    output_root: str | Path,
    severity_allowlist: Sequence[str] | None = None,
    undetermined_policy: str = "fail",
    abstention_recovery: str = "retain",
    dependent_policy: str = "invalidate",
) -> JsonDict:
    """构建 draft overlay；不会触碰任何 base 输入。"""

    if undetermined_policy not in {"fail", "retain", "quarantine"}:
        raise CorrectionOverlayError("undetermined_policy 必须为 fail/retain/quarantine")
    if abstention_recovery not in {"retain", "apply"}:
        raise CorrectionOverlayError("abstention_recovery 必须为 retain/apply")
    if dependent_policy not in {"invalidate", "rebuild"}:
        raise CorrectionOverlayError("dependent_policy 必须为 invalidate/rebuild")
    output = _resolve_path(output_root)
    if output.exists() and any(output.iterdir()):
        raise CorrectionOverlayError(f"output_root 非空，拒绝覆盖: {output}")
    bases = _base_input_paths(base_inputs)
    audit_path = _resolve_path(audit_final_jsonl)
    audit_rows = _read_jsonl(audit_path, allow_empty=False)
    files, rows_by_kind, raw_rows = _load_bases(bases)
    _plan_and_index_consistency(kind="gt", plans=rows_by_kind.get("gt_plan", {}), indexes=rows_by_kind.get("gt_index", {}))
    _plan_and_index_consistency(kind="pred", plans=rows_by_kind.get("pred_plan", {}), indexes=rows_by_kind.get("pred_index", {}))
    _plan_and_index_consistency(kind="equivalence", plans=rows_by_kind.get("equivalence_plan", {}), indexes=rows_by_kind.get("equivalence_index", {}))
    _plan_and_index_consistency(kind="structure", plans=rows_by_kind.get("structure_plan", {}), indexes=rows_by_kind.get("structure_index", {}))
    audit_sha = _sha256_file(audit_path)
    files["audit_final"] = {"path": str(audit_path), "sha256": audit_sha, "row_count": len(audit_rows)}
    severity_set = None if severity_allowlist is None else set(severity_allowlist)
    if severity_set is not None and not severity_set.issubset(KNOWN_SEVERITIES):
        raise CorrectionOverlayError("severity_allowlist 包含未知值")
    revision_payload = {
        "audit_final_sha256": audit_sha,
        "base_inputs": {kind: value["sha256"] for kind, value in sorted(files.items())},
        "condition": CONDITION,
        "policy": {
            "severity_allowlist": sorted(severity_set) if severity_set is not None else "all",
            "undetermined_policy": undetermined_policy,
            "abstention_recovery": abstention_recovery,
            "dependent_policy": dependent_policy,
        },
    }
    overlay_revision = _sha256_json(revision_payload)
    gt_plans = rows_by_kind.get("gt_plan", {})
    gt_indexes = rows_by_kind.get("gt_index", {})
    pred_plans = rows_by_kind.get("pred_plan", {})
    pred_indexes = rows_by_kind.get("pred_index", {})
    gt_patch, pred_patch, corrections, eq_overrides, _ = _audit_corrections(
        audit_rows,
        gt_plans=gt_plans,
        gt_indexes=gt_indexes,
        pred_plans=pred_plans,
        pred_indexes=pred_indexes,
        severity_allowlist=severity_set,
    )
    gt_views = _effective_views(
        gt_indexes,
        patch=gt_patch,
        base_file_sha256=files["gt_index"]["sha256"],
        overlay_revision=overlay_revision,
    )
    pred_views = _effective_views(
        pred_indexes,
        patch=pred_patch,
        base_file_sha256=files["pred_index"]["sha256"],
        overlay_revision=overlay_revision,
    )

    # 按 dataset_id 绑定 GT plan，并挂上对应 index 供 evidence 重建使用。
    gt_by_dataset: dict[str, JsonDict] = {}
    for logical_id, plan in gt_plans.items():
        request = _request(plan)
        dataset_id = request.get("dataset_id")
        if isinstance(dataset_id, str):
            bound = dict(plan)
            bound["__index"] = gt_indexes.get(logical_id)
            gt_by_dataset[logical_id] = bound
    changed_gt_datasets = {
        _request(gt_plans[logical_id]).get("dataset_id")
        for logical_id in gt_patch
        if logical_id in gt_plans
    }
    affected_pred_ids = set(pred_patch)
    for pred_id, plan in pred_plans.items():
        if _request(plan).get("dataset_id") in changed_gt_datasets:
            affected_pred_ids.add(pred_id)
    base_evidence = rows_by_kind.get("evidence", {})
    evidence_rows = _rebuild_evidence(
        pred_ids=affected_pred_ids,
        gt_by_dataset=gt_by_dataset,
        pred_plans=pred_plans,
        pred_indexes=pred_indexes,
        gt_patch=gt_patch,
        pred_patch=pred_patch,
        base_evidence=base_evidence,
        overlay_revision=overlay_revision,
    ) if affected_pred_ids else []

    structure_ids = _affected_structure_ids(pred_patch)
    structure_plans = _structure_plan_rows(
        structure_ids=structure_ids,
        pred_plans=pred_plans,
        pred_indexes=pred_indexes,
        old_structure_plans=rows_by_kind.get("structure_plan", {}),
        pred_patch=pred_patch,
        overlay_revision=overlay_revision,
    ) if structure_ids else []
    stale_rows = _structure_stale_rows(
        structure_ids,
        rows_by_kind.get("structure_index", {}),
        files.get("structure_index", {}).get("sha256"),
    )

    corrections.sort(key=lambda row: (str(row.get("target_kind")), str(row.get("target_logical_id"))))
    for row in corrections:
        row["overlay_revision_sha256"] = overlay_revision
    manifest_dir = output / "manifest"
    views_dir = output / "views"
    evidence_dir = output / "evidence"
    plans_dir = output / "plans"
    _write_jsonl(manifest_dir / "corrections.jsonl", corrections)
    _write_jsonl(manifest_dir / "equivalence_overrides.jsonl", eq_overrides)
    _write_jsonl(manifest_dir / "structure_stale.jsonl", stale_rows)
    _write_jsonl(views_dir / "gt_effective.jsonl", gt_views)
    _write_jsonl(views_dir / "pred_effective.jsonl", pred_views)
    _write_jsonl(evidence_dir / "evidence_rebuilt.jsonl", evidence_rows)
    _write_jsonl(plans_dir / "structure_overlay_plan.jsonl", structure_plans)

    counts = {
        "audit_clean_rows": sum(1 for row in audit_rows if row.get("condition") == CONDITION),
        "expression_fallbacks": len(gt_patch) + len(pred_patch),
        "gt_expression_fallbacks": len(gt_patch),
        "pred_expression_fallbacks": len(pred_patch),
        "reference_overrides": len(eq_overrides),
        "evidence_rebuilds": len(evidence_rows),
        "stale_structure": len(structure_ids),
        "structure_replacements": 0,
    }
    outputs = {
        "gt_effective": _manifest_output_record(views_dir / "gt_effective.jsonl", row_count=len(gt_views)),
        "pred_effective": _manifest_output_record(views_dir / "pred_effective.jsonl", row_count=len(pred_views)),
        "corrections": _manifest_output_record(manifest_dir / "corrections.jsonl", row_count=len(corrections)),
        "equivalence_overrides": _manifest_output_record(manifest_dir / "equivalence_overrides.jsonl", row_count=len(eq_overrides)),
        "structure_stale": _manifest_output_record(manifest_dir / "structure_stale.jsonl", row_count=len(stale_rows)),
        "evidence_rebuilt": _manifest_output_record(evidence_dir / "evidence_rebuilt.jsonl", row_count=len(evidence_rows)),
        "structure_overlay_plan": _manifest_output_record(plans_dir / "structure_overlay_plan.jsonl", row_count=len(structure_plans)),
    }
    manifest: JsonDict = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "phase": "draft",
        "status": "ready_for_structure" if structure_ids else "ready",
        "condition": CONDITION,
        "overlay_revision_sha256": overlay_revision,
        "audit_final": files["audit_final"],
        "base_inputs": files,
        "policy": {
            "severity_allowlist": sorted(severity_set) if severity_set is not None else "all",
            "undetermined_policy": undetermined_policy,
            "abstention_recovery": abstention_recovery,
            "dependent_policy": dependent_policy,
        },
        "counts": counts,
        "outputs": outputs,
        "structure_ids": structure_ids,
        "overlay_sha256": None,
    }
    manifest["overlay_sha256"] = _sha256_json({key: value for key, value in manifest.items() if key != "overlay_sha256"})
    _write_json(manifest_dir / "draft_manifest.json", manifest)
    manifest["outputs"]["draft_manifest"] = _manifest_output_record(manifest_dir / "draft_manifest.json")
    return manifest


def _load_structure_results(paths: Sequence[Path]) -> tuple[dict[str, JsonDict], dict[str, str]]:
    results: dict[str, JsonDict] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate.name in {"draft_manifest.json", "corrections_manifest.json"}:
                continue
            if candidate.suffix != ".jsonl" and not candidate.is_file():
                continue
            if candidate.suffix == ".jsonl":
                rows = _read_jsonl(candidate, allow_empty=True)
                for row in rows:
                    logical_id = row.get("logical_id")
                    if isinstance(logical_id, str):
                        if logical_id in results:
                            raise CorrectionOverlayError(f"structure result logical_id 重复: {logical_id}")
                        results[logical_id] = row
                        hashes[logical_id] = _sha256_file(candidate)
                continue
            row = _read_json(candidate)
            logical_id = row.get("logical_id")
            if isinstance(logical_id, str):
                if logical_id in results:
                    raise CorrectionOverlayError(f"structure result logical_id 重复: {logical_id}")
                results[logical_id] = row
                hashes[logical_id] = _sha256_file(candidate)
    return results, hashes


def finalize_draft(
    *,
    draft_manifest_json: str | Path,
    structure_results: Sequence[str | Path] | None = None,
    output_root: str | Path | None = None,
) -> JsonDict:
    """消费 structure 结果，发布 corrections manifest；缺结果时显式 blocked。"""

    draft_path = _resolve_path(draft_manifest_json)
    draft = _read_json(draft_path)
    if draft.get("schema_version") != AUDIT_SCHEMA_VERSION or draft.get("phase") != "draft":
        raise CorrectionOverlayError("draft manifest schema/phase 不匹配")
    output = _resolve_path(output_root or draft_path.parent.parent)
    expected = [str(item) for item in draft.get("structure_ids", [])]
    result_paths = [_resolve_path(path) for path in (structure_results or [])]
    results, result_hashes = _load_structure_results(result_paths)
    missing = sorted(set(expected) - set(results))
    extra = sorted(set(results) - set(expected)) if expected else sorted(results)
    final: JsonDict = dict(draft)
    final["phase"] = "final"
    final["status"] = "blocked" if missing or extra else "ok"
    final["structure_results"] = {
        logical_id: {
            "result_container_sha256": result_hashes[logical_id],
            "evaluation_key": results[logical_id].get("evaluation_key"),
            "structured_output_sha256": _sha256_json(results[logical_id].get("structured_output")),
        }
        for logical_id in sorted(results)
        if logical_id in expected
    }
    final["missing_structure_ids"] = missing
    final["unexpected_structure_ids"] = extra
    final["counts"] = dict(draft.get("counts") or {})
    final["counts"]["structure_replacements"] = 0 if missing or extra else len(expected)
    if not missing and not extra:
        plan_path = _resolve_path(str((draft.get("outputs") or {}).get("structure_overlay_plan", {}).get("path")))
        plan_rows = _read_jsonl(plan_path, allow_empty=True)
        by_id = _unique_rows(plan_rows, key="logical_id", context="structure_overlay_plan")
        replacement_rows: list[JsonDict] = []
        for logical_id in expected:
            plan = by_id.get(logical_id)
            result = results[logical_id]
            if plan is None:
                raise CorrectionOverlayError(f"structure plan 缺少 {logical_id}")
            if result.get("evaluation_key") != plan.get("evaluation_key"):
                raise CorrectionOverlayError(f"structure result evaluation_key 漂移: {logical_id}")
            structured = result.get("structured_output")
            if not isinstance(structured, Mapping) or structured.get("decision") not in STRUCTURE_DECISIONS:
                raise CorrectionOverlayError(f"structure result structured_output 非法: {logical_id}")
            replacement_rows.append(
                {
                    "logical_id": logical_id,
                    "evaluation_key": plan["evaluation_key"],
                    "task_type": "stab_structure",
                    "condition": CONDITION,
                    "state": "frozen",
                    "effective_expression": None,
                    "expression_resolution": None,
                    "structured_output": dict(structured),
                    # 兼容 frozen_result_index 的旧字段，同时明确该值是
                    # 整个 result JSON 容器的 hash，而非单条模型输出 hash。
                    "result_sha256": result_hashes[logical_id],
                    "result_container_sha256": result_hashes[logical_id],
                    "overlay_revision_sha256": draft.get("overlay_revision_sha256"),
                }
            )
        replacement_path = output / "results" / "structure_overlay_frozen_index.jsonl"
        _write_jsonl(replacement_path, replacement_rows)
        final.setdefault("outputs", {})["structure_overlay_index"] = _manifest_output_record(
            replacement_path, row_count=len(replacement_rows)
        )
    manifest_path = output / "manifest" / "corrections_manifest.json"
    final["overlay_sha256"] = _sha256_json({key: value for key, value in final.items() if key != "overlay_sha256"})
    _write_json(manifest_path, final)
    final.setdefault("outputs", {})["corrections_manifest"] = _manifest_output_record(manifest_path)
    return final


def _parse_base_inputs(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise CorrectionOverlayError(f"--base-input 应为 name=path: {value}")
        key, path = value.split("=", 1)
        if not key or not path:
            raise CorrectionOverlayError(f"--base-input 无效: {value}")
        result[key] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建公式审计 corrections overlay")
    sub = parser.add_subparsers(dest="phase", required=True)
    draft = sub.add_parser("draft")
    draft.add_argument("--audit-final-jsonl", type=Path, required=True)
    draft.add_argument("--base-input", action="append", default=[], metavar="NAME=PATH")
    draft.add_argument("--output-root", type=Path, required=True)
    draft.add_argument("--severity-allowlist", default=None)
    draft.add_argument("--undetermined-policy", choices=["fail", "retain", "quarantine"], default="fail")
    draft.add_argument("--abstention-recovery", choices=["retain", "apply"], default="retain")
    draft.add_argument("--dependent-policy", choices=["invalidate", "rebuild"], default="invalidate")
    final = sub.add_parser("final")
    final.add_argument("--draft-manifest-json", type=Path, required=True)
    final.add_argument("--structure-result", action="append", default=[], type=Path)
    final.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.phase == "draft":
            allowlist = None
            if args.severity_allowlist:
                allowlist = [item.strip() for item in args.severity_allowlist.split(",") if item.strip()]
            report = build_draft(
                audit_final_jsonl=args.audit_final_jsonl,
                base_inputs=_parse_base_inputs(args.base_input),
                output_root=args.output_root,
                severity_allowlist=allowlist,
                undetermined_policy=args.undetermined_policy,
                abstention_recovery=args.abstention_recovery,
                dependent_policy=args.dependent_policy,
            )
        else:
            report = finalize_draft(
                draft_manifest_json=args.draft_manifest_json,
                structure_results=args.structure_result,
                output_root=args.output_root,
            )
    except CorrectionOverlayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
