"""对公式审计复判触发项执行本地确定性符号复核。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_contract import canonical_json
from .run_claude_plan import load_plan_jsonl
from .symbolic_evidence import (
    SimplificationContractError,
    SymbolicEvidenceError,
    build_pair_evidence,
    validate_simplification,
)


JsonDict = dict[str, Any]
DEFAULT_WORKERS = 2

_SOURCE_IDENTITY_KEYS = (
    "logical_key",
    "condition",
    "algorithm",
    "dataset_id",
    "dataset_index",
    "seed",
    "source_formula_logical_id",
    "source_formula_evaluation_key",
    "source_formula_result_sha256",
    "source_equivalence_logical_id",
    "source_gt_logical_id",
    "source_gt_artifact_sha256",
    "source_pred_artifact_sha256",
)


class FormulaAuditLocalVerificationError(RuntimeError):
    """本地复核的输入产物不一致或不满足契约。"""


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormulaAuditLocalVerificationError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaAuditLocalVerificationError(f"{context} 必须是非空字符串")
    return value.strip()


def _require_string_list(value: object, *, context: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise FormulaAuditLocalVerificationError(f"{context} 必须是非空字符串组成的数组")
    return list(value)


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        raise FormulaAuditLocalVerificationError(f"JSONL 不存在: {path}")
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FormulaAuditLocalVerificationError(
                    f"{path}:{line_number} 不得为空行"
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormulaAuditLocalVerificationError(
                    f"{path}:{line_number} JSON 非法"
                ) from exc
            rows.append(dict(_require_mapping(payload, context=f"{path}:{line_number}")))
    return rows


def _unique_by(
    rows: Sequence[Mapping[str, Any]], key: str, *, context: str
) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for row in rows:
        value = _require_string(row.get(key), context=f"{context}.{key}")
        if value in result:
            raise FormulaAuditLocalVerificationError(
                f"{context} 出现重复 {key}: {value}"
            )
        result[value] = dict(row)
    return result


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_for(logical_id: str) -> int:
    digest = hashlib.sha256(logical_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _compact_counterexample(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return str(value)
    allowed = {
        "values",
        "original",
        "simplified",
        "abs_error",
        "rel_error",
        "tolerance",
        "split",
        "row_index",
    }
    return {key: value[key] for key in sorted(allowed) if key in value}


def _compact_evidence(
    evidence: Mapping[str, Any],
    *,
    decision: str | None = None,
    error: Mapping[str, str] | None = None,
    evidence_sha256: str | None = None,
) -> JsonDict:
    symbolic_difference = evidence.get("symbolic_difference")
    difference = symbolic_difference if isinstance(symbolic_difference, Mapping) else {}
    counterexample = evidence.get("counterexample", difference.get("counterexample"))
    return {
        "decision": decision or evidence.get("decision", "undetermined"),
        "proof_basis": evidence.get("proof_basis", difference.get("proof_basis")),
        "counterexample": _compact_counterexample(counterexample),
        "probe_count": int(evidence.get("probe_count", 0)),
        "max_abs_error": evidence.get("max_abs_error", difference.get("max_abs_error")),
        "max_rel_error": evidence.get("max_rel_error", difference.get("max_rel_error")),
        "evidence_sha256": evidence_sha256 or str(
            evidence.get("evidence_sha256") or _sha256_json(evidence)
        ),
        "error": dict(error) if error is not None else None,
    }


def _error_evidence(exc: Exception) -> JsonDict:
    error = {"class": type(exc).__name__, "message": str(exc)[:1000]}
    return {
        "decision": "error",
        "proof_basis": None,
        "counterexample": None,
        "probe_count": 0,
        "max_abs_error": None,
        "max_rel_error": None,
        "evidence_sha256": _sha256_json(error),
        "error": error,
    }


def _verify_simplification(
    *,
    original: str,
    candidate: str,
    variables: Sequence[str],
    allowed_functions: Sequence[str],
    seed: int,
) -> JsonDict:
    try:
        evidence = validate_simplification(
            original,
            candidate,
            variables,
            allowed_functions,
            seed,
        )
    except SimplificationContractError as exc:
        return _compact_evidence(
            exc.evidence,
            decision="not_equivalent",
            error={"class": type(exc).__name__, "message": str(exc)[:1000]},
        )
    except (SymbolicEvidenceError, SyntaxError) as exc:
        return _error_evidence(exc)
    return _compact_evidence(evidence)


def _verify_reference(
    *,
    scope: str,
    candidate: str,
    reference: object,
    variables: Sequence[str],
    allowed_functions: Sequence[str],
    seed: int,
) -> JsonDict:
    if scope == "ground_truth_simplification":
        evidence = {"decision": "not_applicable", "scope": scope}
        return {
            "decision": "not_applicable",
            "proof_basis": "not_applicable",
            "counterexample": None,
            "probe_count": 0,
            "max_abs_error": None,
            "max_rel_error": None,
            "evidence_sha256": _sha256_json(evidence),
            "error": None,
        }
    reference_expression = _require_string(
        reference, context="request.reference_simplified_expression"
    )
    try:
        evidence = build_pair_evidence(
            candidate,
            reference_expression,
            allowed_variables=variables,
            allowed_functions=allowed_functions,
            seed=seed,
            include_tree_distance=False,
        )
    except (SymbolicEvidenceError, SyntaxError) as exc:
        return _error_evidence(exc)
    return _compact_evidence(evidence)


def _source_identity(manifest: Mapping[str, Any]) -> JsonDict:
    return {
        key: manifest[key]
        for key in _SOURCE_IDENTITY_KEYS
        if key in manifest and manifest[key] is not None
    }


def _verify_one(
    *,
    plan_row: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> JsonDict:
    logical_id = _require_string(plan_row.get("logical_id"), context="plan.logical_id")
    evaluation_key = _require_string(
        plan_row.get("evaluation_key"), context=f"{logical_id}.evaluation_key"
    )
    request = _require_mapping(plan_row.get("request"), context=f"{logical_id}.request")
    scope = _require_string(request.get("audit_scope"), context=f"{logical_id}.audit_scope")
    if scope not in {"ground_truth_simplification", "prediction_formula"}:
        raise FormulaAuditLocalVerificationError(f"{logical_id} audit_scope 非法: {scope}")
    variables = _require_string_list(
        request.get("variables"), context=f"{logical_id}.variables"
    )
    allowed_functions = _require_string_list(
        request.get("allowed_functions"), context=f"{logical_id}.allowed_functions"
    )
    original = _require_string(
        request.get("original_expression"), context=f"{logical_id}.original_expression"
    )
    candidate = _require_string(
        request.get("candidate_simplified_expression"),
        context=f"{logical_id}.candidate_simplified_expression",
    )
    seed = _seed_for(logical_id)
    simplification = _verify_simplification(
        original=original,
        candidate=candidate,
        variables=variables,
        allowed_functions=allowed_functions,
        seed=seed,
    )
    reference = _verify_reference(
        scope=scope,
        candidate=candidate,
        reference=request.get("reference_simplified_expression"),
        variables=variables,
        allowed_functions=allowed_functions,
        seed=seed,
    )
    record_without_hash: JsonDict = {
        "audit_logical_id": logical_id,
        "round1_evaluation_key": evaluation_key,
        "source_identity": _source_identity(manifest),
        "deterministic_seed": seed,
        "local_simplification": simplification,
        "local_reference": reference,
    }
    return {
        **record_without_hash,
        "local_evidence_hash": _sha256_json(record_without_hash),
    }


def _verify_work_item(
    item: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> JsonDict:
    plan_row, manifest = item
    return _verify_one(plan_row=plan_row, manifest=manifest)


def verify_formula_audit_disagreements(
    *,
    round1_plan_jsonl: str | Path,
    sample_manifest_jsonl: str | Path,
    comparisons_jsonl: str | Path,
    triggers_jsonl: str | Path,
    output_jsonl: str | Path,
    workers: int = DEFAULT_WORKERS,
) -> JsonDict:
    """复核所有 requires_round2 条目，并写出不含大型 AST 的紧凑证据。"""

    if workers <= 0:
        raise FormulaAuditLocalVerificationError("workers 必须为正整数")
    plan_path = Path(round1_plan_jsonl).resolve()
    loaded_plan = load_plan_jsonl(plan_path)
    plan_by_key = _unique_by(_read_jsonl(plan_path), "evaluation_key", context="round1_plan")
    if set(plan_by_key) != {entry.evaluation_key for entry in loaded_plan.entries}:
        raise FormulaAuditLocalVerificationError("round1 plan 解析身份漂移")
    manifest_by_key = _unique_by(
        _read_jsonl(Path(sample_manifest_jsonl).resolve()),
        "audit_evaluation_key",
        context="sample_manifest",
    )
    comparisons = _read_jsonl(Path(comparisons_jsonl).resolve())
    comparisons_by_key = _unique_by(
        comparisons, "round1_evaluation_key", context="comparisons"
    )
    triggers_by_key = _unique_by(
        _read_jsonl(Path(triggers_jsonl).resolve()),
        "round1_evaluation_key",
        context="triggers",
    )
    required_keys = {
        key for key, row in comparisons_by_key.items() if row.get("requires_round2") is True
    }
    if set(triggers_by_key) != required_keys:
        raise FormulaAuditLocalVerificationError(
            "triggers 与 comparisons.requires_round2 集合不一致"
        )
    if not required_keys.issubset(plan_by_key) or not required_keys.issubset(manifest_by_key):
        raise FormulaAuditLocalVerificationError("复核项缺少 plan 或 manifest 绑定")

    ordered_keys = sorted(required_keys, key=lambda key: str(plan_by_key[key]["logical_id"]))
    for key in ordered_keys:
        logical_id = plan_by_key[key].get("logical_id")
        if manifest_by_key[key].get("audit_logical_id") != logical_id:
            raise FormulaAuditLocalVerificationError(f"manifest logical_id 漂移: {key}")
        if comparisons_by_key[key].get("audit_logical_id") != logical_id:
            raise FormulaAuditLocalVerificationError(f"comparison logical_id 漂移: {key}")
        if triggers_by_key[key].get("audit_logical_id") != logical_id:
            raise FormulaAuditLocalVerificationError(f"trigger logical_id 漂移: {key}")

    work_items = [(plan_by_key[key], manifest_by_key[key]) for key in ordered_keys]
    # 每个进程的主线程可使用 symbolic_evidence 的严格 signal 超时保护；线程池会
    # 降级为无符号证明的保守路径，导致同一表达式在并发与串行下结论不一致。
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        output_rows = list(executor.map(_verify_work_item, work_items))
    output_path = Path(output_jsonl).resolve()
    _atomic_write_jsonl(output_path, output_rows)
    decision_counts: dict[str, int] = {}
    for row in output_rows:
        pair = f"{row['local_simplification']['decision']}|{row['local_reference']['decision']}"
        decision_counts[pair] = decision_counts.get(pair, 0) + 1
    return {
        "status": "ok",
        "model_invoked": False,
        "workers": workers,
        "verified_count": len(output_rows),
        "decision_pair_counts": dict(sorted(decision_counts.items())),
        "output_jsonl": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地确定性复核公式审计分歧")
    parser.add_argument("--round1-plan-jsonl", type=Path, required=True)
    parser.add_argument("--sample-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--comparisons-jsonl", type=Path, required=True)
    parser.add_argument("--triggers-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_formula_audit_disagreements(
            round1_plan_jsonl=args.round1_plan_jsonl,
            sample_manifest_jsonl=args.sample_manifest_jsonl,
            comparisons_jsonl=args.comparisons_jsonl,
            triggers_jsonl=args.triggers_jsonl,
            output_jsonl=args.output_jsonl,
            workers=args.workers,
        )
    except (FormulaAuditLocalVerificationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
