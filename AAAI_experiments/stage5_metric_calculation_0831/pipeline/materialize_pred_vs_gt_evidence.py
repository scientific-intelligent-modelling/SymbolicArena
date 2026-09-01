"""从 equivalence plan 物化 clean_pred_vs_gt_evidence.jsonl。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    PlanContractError,
    _row_to_definition,
)


STAGE_ROOT_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831")
DEFAULT_OUTPUT_JSONL = STAGE_ROOT_RELATIVE / "results/clean_pred_vs_gt_evidence.jsonl"
DEFAULT_REPORT_JSON = STAGE_ROOT_RELATIVE / "reports/clean_pred_vs_gt_evidence.json"
EQUIVALENCE_LOGICAL_ID_RE = re.compile(r"^equivalence::([a-z0-9_]+)::(g\d{4})::s(520|521|522)::clean$")
GT_LOGICAL_ID_RE = re.compile(r"^gt_simplify::([^:]+)$")
PRED_LOGICAL_ID_RE = re.compile(r"^pred_simplify::([a-z0-9_]+)::(g\d{4})::s(520|521|522)::clean$")


class MaterializePredVsGtEvidenceError(RuntimeError):
    """evidence 物化或契约校验失败。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_line(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(path, b"".join(_json_line(dict(row)) for row in rows))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _read_jsonl_rows(path: Path) -> tuple[bytes, list[tuple[int, dict[str, Any]]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MaterializePredVsGtEvidenceError(f"无法读取 plan JSONL {path}: {exc}") from exc
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializePredVsGtEvidenceError(f"{path}:{line_number} JSON 解析失败: {exc}") from exc
            if not isinstance(payload, dict):
                raise MaterializePredVsGtEvidenceError(f"{path}:{line_number} 顶层必须是 JSON object")
            rows.append((line_number, payload))
    if not rows:
        raise MaterializePredVsGtEvidenceError("equivalence plan JSONL 不能为空")
    return raw, rows


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterializePredVsGtEvidenceError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise MaterializePredVsGtEvidenceError(f"{context} 必须是非空字符串")
    return value


def _require_float01(value: object, *, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise MaterializePredVsGtEvidenceError(f"{context} 缺失或不是数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MaterializePredVsGtEvidenceError(f"{context} 不是合法数值") from exc
    if number < 0.0 or number > 1.0:
        raise MaterializePredVsGtEvidenceError(f"{context} 必须位于 [0,1]")
    return number


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaterializePredVsGtEvidenceError(f"{context} 必须是整数")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    text = _require_string(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise MaterializePredVsGtEvidenceError(f"{context} 必须是小写十六进制 SHA256")
    return text


def _parse_equivalence_logical_id(logical_id: str) -> tuple[str, str, int]:
    match = EQUIVALENCE_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise MaterializePredVsGtEvidenceError(f"equivalence logical_id 非 canonical: {logical_id!r}")
    return match.group(1), match.group(2), int(match.group(3))


def _parse_gt_logical_id(logical_id: str) -> str:
    match = GT_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise MaterializePredVsGtEvidenceError(f"gt logical_id 非 canonical: {logical_id!r}")
    return match.group(1)


def _parse_pred_logical_id(logical_id: str) -> tuple[str, str, int]:
    match = PRED_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise MaterializePredVsGtEvidenceError(f"pred logical_id 非 canonical: {logical_id!r}")
    return match.group(1), match.group(2), int(match.group(3))


def _logical_key(algorithm: str, dataset_id: str, seed: int) -> str:
    return f"{algorithm}::{dataset_id}::s{seed}::clean"


def materialize_pred_vs_gt_evidence(
    *,
    equivalence_plan_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    expected_row_count: int | None = None,
) -> dict[str, Any]:
    raw_plan, rows = _read_jsonl_rows(equivalence_plan_jsonl)
    plan_sha256 = _sha256_bytes(raw_plan)
    output_rows: list[dict[str, Any]] = []
    seen_logical_keys: set[str] = set()
    callable_count = 0
    skipped_non_applicable_count = 0

    for line_number, row in rows:
        context = f"{equivalence_plan_jsonl}:{line_number}"
        try:
            planned = _row_to_definition(row, line_number=line_number)
        except PlanContractError as exc:
            raise MaterializePredVsGtEvidenceError(str(exc)) from exc
        if planned.definition.task_spec.task_type != "equivalence":
            raise MaterializePredVsGtEvidenceError(f"{context} task_type 必须为 equivalence")
        logical_id = planned.logical_id
        _parse_equivalence_logical_id(logical_id)
        request = planned.definition.request
        deterministic_evidence = request.get("deterministic_evidence")
        if deterministic_evidence is None:
            status = row.get("status")
            if status == "planned_non_applicable":
                skipped_non_applicable_count += 1
                continue
            raise MaterializePredVsGtEvidenceError(f"{context} 缺少 deterministic_evidence")
        callable_count += 1

        evidence = dict(_require_mapping(deterministic_evidence, context=f"{context}.deterministic_evidence"))
        if _require_string(evidence.get("schema_version"), context=f"{context}.schema_version") != "symbolic_pair_evidence.v2":
            raise MaterializePredVsGtEvidenceError(f"{context}.schema_version 必须为 symbolic_pair_evidence.v2")
        if _require_string(evidence.get("phase"), context=f"{context}.phase") != "equivalence":
            raise MaterializePredVsGtEvidenceError(f"{context}.phase 必须为 equivalence")
        request_evidence_hash = _require_sha256(
            request.get("evidence_hash"),
            context=f"{context}.request.evidence_hash",
        )
        evidence_sha256 = _require_sha256(
            evidence.get("evidence_sha256"),
            context=f"{context}.deterministic_evidence.evidence_sha256",
        )
        if request_evidence_hash != evidence_sha256:
            raise MaterializePredVsGtEvidenceError(f"{context} request.evidence_hash 与 deterministic_evidence.evidence_sha256 不一致")

        pair_evidence = _require_mapping(
            evidence.get("pair_evidence"),
            context=f"{context}.pair_evidence",
        )
        lhs_binding = _require_mapping(evidence.get("lhs_binding"), context=f"{context}.lhs_binding")
        rhs_binding = _require_mapping(evidence.get("rhs_binding"), context=f"{context}.rhs_binding")
        if _require_string(lhs_binding.get("role"), context=f"{context}.lhs_binding.role") != "lhs":
            raise MaterializePredVsGtEvidenceError(f"{context}.lhs_binding.role 必须为 lhs")
        if _require_string(rhs_binding.get("role"), context=f"{context}.rhs_binding.role") != "rhs":
            raise MaterializePredVsGtEvidenceError(f"{context}.rhs_binding.role 必须为 rhs")

        gt_logical_id = _require_string(lhs_binding.get("frozen_logical_id"), context=f"{context}.lhs_binding.frozen_logical_id")
        pred_logical_id = _require_string(rhs_binding.get("frozen_logical_id"), context=f"{context}.rhs_binding.frozen_logical_id")
        dataset_id = _parse_gt_logical_id(gt_logical_id)
        algorithm_slug, dataset_index, seed = _parse_pred_logical_id(pred_logical_id)
        request_algorithm_slug = _require_string(request.get("algorithm_slug"), context=f"{context}.request.algorithm_slug")
        request_dataset_index = _require_string(request.get("dataset_index"), context=f"{context}.request.dataset_index")
        request_seed = _require_int(request.get("seed"), context=f"{context}.request.seed")
        if (algorithm_slug, dataset_index, seed) != (request_algorithm_slug, request_dataset_index, request_seed):
            raise MaterializePredVsGtEvidenceError(f"{context} pred_logical_id 与 request 身份不一致")
        if _require_string(request.get("ground_truth_logical_id"), context=f"{context}.request.ground_truth_logical_id") != gt_logical_id:
            raise MaterializePredVsGtEvidenceError(f"{context} gt_logical_id 与 request 漂移")
        if _require_string(request.get("prediction_logical_id"), context=f"{context}.request.prediction_logical_id") != pred_logical_id:
            raise MaterializePredVsGtEvidenceError(f"{context} pred_logical_id 与 request 漂移")
        if _require_string(request.get("dataset_id"), context=f"{context}.request.dataset_id") != dataset_id:
            raise MaterializePredVsGtEvidenceError(f"{context} dataset_id 与 GT logical_id 漂移")
        algorithm = _require_string(request.get("algorithm"), context=f"{context}.request.algorithm")
        logical_key = _logical_key(algorithm, dataset_id, seed)
        if logical_key in seen_logical_keys:
            raise MaterializePredVsGtEvidenceError(f"出现重复 logical_key: {logical_key}")
        seen_logical_keys.add(logical_key)

        output_rows.append(
            {
                "logical_key": logical_key,
                "gt_logical_id": gt_logical_id,
                "pred_logical_id": pred_logical_id,
                "evidence_hash": evidence_sha256,
                "ground_truth": {
                    "simplified_expression": _require_string(
                        lhs_binding.get("frozen_simplified_expression"),
                        context=f"{context}.lhs_binding.frozen_simplified_expression",
                    ),
                    "artifact_sha256": _require_sha256(
                        lhs_binding.get("plan_symbolic_artifact_sha256"),
                        context=f"{context}.lhs_binding.plan_symbolic_artifact_sha256",
                    ),
                },
                "prediction": {
                    "simplified_expression": _require_string(
                        rhs_binding.get("frozen_simplified_expression"),
                        context=f"{context}.rhs_binding.frozen_simplified_expression",
                    ),
                    "artifact_sha256": _require_sha256(
                        rhs_binding.get("plan_symbolic_artifact_sha256"),
                        context=f"{context}.rhs_binding.plan_symbolic_artifact_sha256",
                    ),
                },
                "tree": {
                    "tree_similarity": _require_float01(
                        _require_mapping(pair_evidence.get("tree"), context=f"{context}.pair_evidence.tree").get(
                            "tree_similarity"
                        ),
                        context=f"{context}.pair_evidence.tree.tree_similarity",
                    )
                },
                "variable": {
                    "f1": _require_float01(
                        _require_mapping(
                            pair_evidence.get("variable"),
                            context=f"{context}.pair_evidence.variable",
                        ).get("f1"),
                        context=f"{context}.pair_evidence.variable.f1",
                    )
                },
                "operator": {
                    "f1": _require_float01(
                        _require_mapping(
                            pair_evidence.get("operator"),
                            context=f"{context}.pair_evidence.operator",
                        ).get("f1"),
                        context=f"{context}.pair_evidence.operator.f1",
                    )
                },
            }
        )

    output_rows.sort(key=lambda item: item["logical_key"])
    if expected_row_count is not None and len(output_rows) != expected_row_count:
        raise MaterializePredVsGtEvidenceError(
            f"evidence 行数不符: 期望 {expected_row_count}，实际 {len(output_rows)}"
        )

    _write_jsonl(output_jsonl, output_rows)
    report = {
        "status": "ok",
        "contract_ok": True,
        "inputs": {
            "equivalence_plan_jsonl": str(equivalence_plan_jsonl.resolve()),
            "equivalence_plan_sha256": plan_sha256,
            "input_row_count": len(rows),
        },
        "outputs": {
            "evidence_jsonl": str(output_jsonl.resolve()),
            "evidence_jsonl_sha256": _sha256_file(output_jsonl),
            "evidence_jsonl_row_count": len(output_rows),
        },
        "counts": {
            "callable_row_count": callable_count,
            "skipped_non_applicable_row_count": skipped_non_applicable_count,
        },
    }
    _write_json(report_json, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    stage_root = repo_root / STAGE_ROOT_RELATIVE
    parser = argparse.ArgumentParser(description="从 equivalence plan 物化 clean_pred_vs_gt_evidence.jsonl")
    parser.add_argument("--equivalence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=stage_root / DEFAULT_OUTPUT_JSONL.relative_to(STAGE_ROOT_RELATIVE))
    parser.add_argument("--report-json", type=Path, default=stage_root / DEFAULT_REPORT_JSON.relative_to(STAGE_ROOT_RELATIVE))
    parser.add_argument("--expected-row-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = materialize_pred_vs_gt_evidence(
            equivalence_plan_jsonl=args.equivalence_plan_jsonl.resolve(),
            output_jsonl=args.output_jsonl.resolve(),
            report_json=args.report_json.resolve(),
            expected_row_count=args.expected_row_count,
        )
    except MaterializePredVsGtEvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["outputs"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
