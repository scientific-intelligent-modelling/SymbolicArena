#!/usr/bin/env python3
"""构建噪声条件下由预测公式替换触发的 equivalence 增量计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .build_clean_downstream_hybrid_plan import (
    CleanDownstreamHybridPlanError,
    _register_state,
    _successor_row,
    _task_spec,
    _validate_apply_state_db_path,
)
from .claude_contract import canonical_json
from .run_claude_plan import PlanContractError, load_plan_jsonl


JsonDict = dict[str, Any]
SCHEMA_VERSION = "noise_equivalence_hybrid_plan.v1"
FORBIDDEN_SOURCE = "all_15alg_fullcpu_v1"
_VERSION_SUFFIX = re.compile(r"::v([1-9]\d*)$")
_PRED_BASE = re.compile(
    r"^pred_simplify::(?P<algorithm>[a-z0-9_]+)::(?P<dataset>g\d{4})::"
    r"s(?P<seed>520|521|522)::(?P<condition>noise001|noise005)$"
)
_EQ_BASE = re.compile(
    r"^equivalence::(?P<algorithm>[a-z0-9_]+)::(?P<dataset>g\d{4})::"
    r"s(?P<seed>520|521|522)::(?P<condition>noise001|noise005)$"
)


class NoiseEquivalenceHybridPlanError(RuntimeError):
    """输入计划或预测替换绑定不足以生成严格 equivalence successor。"""


def _base_logical_id(value: object) -> str:
    return _VERSION_SUFFIX.sub("", str(value))


def _next_logical_id(value: object) -> str:
    logical_id = str(value)
    match = _VERSION_SUFFIX.search(logical_id)
    version = 2 if match is None else int(match.group(1)) + 1
    return f"{_base_logical_id(logical_id)}::v{version}"


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise NoiseEquivalenceHybridPlanError(f"{label} 必须是小写十六进制 SHA256")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json_object(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NoiseEquivalenceHybridPlanError(f"{label} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise NoiseEquivalenceHybridPlanError(f"{label} 顶层必须是 JSON object")
    return payload


def _reject_forbidden_source(path: Path, *, label: str) -> None:
    if FORBIDDEN_SOURCE in str(path):
        raise NoiseEquivalenceHybridPlanError(
            f"{label} 命中禁止来源 {FORBIDDEN_SOURCE}"
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if FORBIDDEN_SOURCE in line:
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label}:{line_number} 命中禁止来源 {FORBIDDEN_SOURCE}"
                    )
    except (OSError, UnicodeError) as exc:
        raise NoiseEquivalenceHybridPlanError(f"{label} 不可读: {exc}") from exc


def _load_callable_plan(
    path: Path,
    *,
    label: str,
    condition: str,
    expected_total_count: int | None,
    allow_non_applicable: bool,
) -> tuple[list[tuple[str, JsonDict]], str]:
    _reject_forbidden_source(path, label=label)
    try:
        loaded = load_plan_jsonl(path)
    except PlanContractError as exc:
        raise NoiseEquivalenceHybridPlanError(f"{label} 契约失败: {exc}") from exc
    if expected_total_count is not None and len(loaded.entries) != expected_total_count:
        raise NoiseEquivalenceHybridPlanError(
            f"{label} 数量不符: {len(loaded.entries)} != {expected_total_count}"
        )
    validated_keys = {entry.evaluation_key for entry in loaded.entries}
    if len(validated_keys) != len(loaded.entries):
        raise NoiseEquivalenceHybridPlanError(f"{label} evaluation_key 不唯一")

    rows: list[tuple[str, JsonDict]] = []
    seen_keys: set[str] = set()
    seen_bases: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label}:{line_number} 顶层必须是 JSON object"
                    )
                logical_id = str(row.get("logical_id", ""))
                base = _base_logical_id(logical_id)
                match = _EQ_BASE.fullmatch(base)
                if (
                    match is None
                    or match.group("condition") != condition
                    or row.get("condition") != condition
                    or row.get("task_type") != "equivalence"
                ):
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label}:{line_number} condition/logical identity 漂移"
                    )
                status = row.get("status")
                if status not in {None, "planned_non_applicable"}:
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label}:{line_number}.status 非法: {status}"
                    )
                if status == "planned_non_applicable" and not allow_non_applicable:
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label}:{line_number} 包含 fresh non-applicable"
                    )
                key = str(row.get("evaluation_key", ""))
                if key in seen_keys or base in seen_bases:
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label} evaluation/base logical identity 不唯一"
                    )
                dependencies = row.get("dependencies")
                task_spec = row.get("task_spec")
                if (
                    not isinstance(dependencies, list)
                    or len(dependencies) != 2
                    or not isinstance(task_spec, Mapping)
                    or task_spec.get("dependencies") != dependencies
                ):
                    raise NoiseEquivalenceHybridPlanError(
                        f"{label}:{line_number} dependencies/task_spec 不一致"
                    )
                seen_keys.add(key)
                seen_bases.add(base)
                rows.append((raw_line, row))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NoiseEquivalenceHybridPlanError(f"{label} 回读失败: {exc}") from exc
    if seen_keys != validated_keys or len(rows) != len(loaded.entries):
        raise NoiseEquivalenceHybridPlanError(f"{label} 回读 identity/count 漂移")
    return rows, loaded.plan_sha256


def _load_pred_bindings(
    path: Path,
    *,
    condition: str,
    expected_refresh_count: int | None,
) -> tuple[list[JsonDict], dict[str, str]]:
    _reject_forbidden_source(path, label="pred hybrid manifest")
    payload = _read_json_object(path, label="pred hybrid manifest")
    if payload.get("status") != "ok":
        raise NoiseEquivalenceHybridPlanError("pred hybrid manifest status 必须为 ok")
    conditions = payload.get("conditions")
    selected = conditions.get(condition) if isinstance(conditions, Mapping) else None
    raw_bindings = selected.get("bindings") if isinstance(selected, Mapping) else None
    if not isinstance(raw_bindings, list):
        raise NoiseEquivalenceHybridPlanError(
            f"pred hybrid manifest 缺少 conditions.{condition}.bindings"
        )
    if expected_refresh_count is not None and len(raw_bindings) != expected_refresh_count:
        raise NoiseEquivalenceHybridPlanError(
            f"pred binding 数量不符: {len(raw_bindings)} != {expected_refresh_count}"
        )

    bindings: list[JsonDict] = []
    dependency_mapping: dict[str, str] = {}
    seen_new_keys: set[str] = set()
    seen_bases: set[str] = set()
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise NoiseEquivalenceHybridPlanError(
                f"pred binding[{index}] 必须是 JSON object"
            )
        binding = dict(raw)
        base = str(binding.get("base_logical_id", ""))
        match = _PRED_BASE.fullmatch(base)
        predecessor_logical_id = str(binding.get("predecessor_logical_id", ""))
        successor_logical_id = str(binding.get("successor_logical_id", ""))
        old_key = _require_sha256(
            binding.get("predecessor_evaluation_key"),
            label=f"pred binding[{index}].predecessor_evaluation_key",
        )
        new_key = _require_sha256(
            binding.get("successor_evaluation_key"),
            label=f"pred binding[{index}].successor_evaluation_key",
        )
        if (
            match is None
            or match.group("condition") != condition
            or _base_logical_id(predecessor_logical_id) != base
            or successor_logical_id != _next_logical_id(predecessor_logical_id)
        ):
            raise NoiseEquivalenceHybridPlanError(
                f"pred binding[{index}] condition/logical identity 漂移"
            )
        if (
            old_key == new_key
            or old_key in dependency_mapping
            or new_key in seen_new_keys
            or base in seen_bases
        ):
            raise NoiseEquivalenceHybridPlanError(
                f"pred binding[{index}] evaluation key/SHA identity 不唯一"
            )
        dependency_mapping[old_key] = new_key
        seen_new_keys.add(new_key)
        seen_bases.add(base)
        bindings.append(binding)
    return bindings, dependency_mapping


def _reject_non_applicable(path: Path | None) -> int:
    if path is None:
        return 0
    _reject_forbidden_source(path, label="fresh non-applicable plan")
    try:
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeError) as exc:
        raise NoiseEquivalenceHybridPlanError(
            f"fresh non-applicable plan 不可读: {exc}"
        ) from exc
    if count:
        raise NoiseEquivalenceHybridPlanError(
            f"fresh equivalence 存在 {count} 条 non-applicable，当前严格模式拒绝物化"
        )
    return count


def build_noise_equivalence_hybrid_plan(
    *,
    condition: str,
    predecessor_equivalence_plan_jsonl: str | Path,
    fresh_equivalence_plan_jsonl: str | Path,
    pred_hybrid_manifest_json: str | Path,
    output_full_plan_jsonl: str | Path,
    output_active_plan_jsonl: str | Path,
    output_refresh_plan_jsonl: str | Path,
    supersession_manifest_json: str | Path,
    report_json: str | Path,
    fresh_non_applicable_jsonl: str | Path | None = None,
    expected_total_count: int | None = 2250,
    expected_refresh_count: int | None = None,
    apply_state_db: str | Path | None = None,
) -> JsonDict:
    """用 fresh equivalence 证据替换被新 pred 绑定影响的行。"""

    if condition not in {"noise001", "noise005"}:
        raise NoiseEquivalenceHybridPlanError("condition 必须是 noise001 或 noise005")
    paths = {
        "predecessor": Path(predecessor_equivalence_plan_jsonl).resolve(),
        "fresh": Path(fresh_equivalence_plan_jsonl).resolve(),
        "pred_manifest": Path(pred_hybrid_manifest_json).resolve(),
        "output_full": Path(output_full_plan_jsonl).resolve(),
        "output_active": Path(output_active_plan_jsonl).resolve(),
        "output_refresh": Path(output_refresh_plan_jsonl).resolve(),
        "supersession_manifest": Path(supersession_manifest_json).resolve(),
        "report": Path(report_json).resolve(),
    }
    if fresh_non_applicable_jsonl is not None:
        paths["fresh_nonapp"] = Path(fresh_non_applicable_jsonl).resolve()
    input_names = {"predecessor", "fresh", "pred_manifest", "fresh_nonapp"}
    output_names = {
        "output_full",
        "output_active",
        "output_refresh",
        "supersession_manifest",
        "report",
    }
    inputs = {path for name, path in paths.items() if name in input_names}
    outputs = {path for name, path in paths.items() if name in output_names}
    if len(outputs) != len(output_names) or inputs.intersection(outputs):
        raise NoiseEquivalenceHybridPlanError("输入输出路径必须互异且禁止覆盖输入")

    nonapp_count = _reject_non_applicable(paths.get("fresh_nonapp"))
    pred_bindings, pred_mapping = _load_pred_bindings(
        paths["pred_manifest"],
        condition=condition,
        expected_refresh_count=expected_refresh_count,
    )
    predecessor_rows, predecessor_sha = _load_callable_plan(
        paths["predecessor"],
        label="predecessor equivalence plan",
        condition=condition,
        expected_total_count=expected_total_count,
        allow_non_applicable=True,
    )
    fresh_rows, fresh_sha = _load_callable_plan(
        paths["fresh"],
        label="fresh equivalence plan",
        condition=condition,
        expected_total_count=expected_total_count,
        allow_non_applicable=False,
    )
    fresh_by_base = {
        _base_logical_id(row["logical_id"]): row for _, row in fresh_rows
    }
    predecessor_bases = {
        _base_logical_id(row["logical_id"]) for _, row in predecessor_rows
    }
    if set(fresh_by_base) != predecessor_bases:
        raise NoiseEquivalenceHybridPlanError(
            "predecessor/fresh equivalence base logical identity 集合不一致"
        )

    pred_binding_by_old = {
        str(binding["predecessor_evaluation_key"]): binding
        for binding in pred_bindings
    }
    output_lines: list[str] = []
    output_rows: list[JsonDict] = []
    refresh_lines: list[str] = []
    supersessions: list[JsonDict] = []
    state_bindings: list[JsonDict] = []
    matched_old_pred_keys: set[str] = set()
    for raw_line, predecessor in predecessor_rows:
        base = _base_logical_id(predecessor["logical_id"])
        old_dependencies = tuple(str(item) for item in predecessor["dependencies"])
        pred_binding = pred_binding_by_old.get(old_dependencies[1])
        if pred_binding is None:
            output_lines.append(raw_line)
            output_rows.append(predecessor)
            continue

        pred_base = str(pred_binding["base_logical_id"])
        pred_match = _PRED_BASE.fullmatch(pred_base)
        eq_match = _EQ_BASE.fullmatch(base)
        if pred_match is None or eq_match is None or pred_match.groupdict() != eq_match.groupdict():
            raise NoiseEquivalenceHybridPlanError(
                f"{base} 与 pred binding logical identity 不一致"
            )
        fresh = fresh_by_base[base]
        new_dependencies = tuple(str(item) for item in fresh["dependencies"])
        if old_dependencies[0] != new_dependencies[0]:
            raise NoiseEquivalenceHybridPlanError(f"{base} GT dependency 发生变化")
        if old_dependencies[1] != str(pred_binding["predecessor_evaluation_key"]):
            raise NoiseEquivalenceHybridPlanError(
                f"{base} 旧 dependency[1] 未绑定 pred predecessor"
            )
        if new_dependencies[1] != str(pred_binding["successor_evaluation_key"]):
            raise NoiseEquivalenceHybridPlanError(
                f"{base} 新 dependency[1] 未绑定 pred successor"
            )
        try:
            successor = _successor_row(
                predecessor,
                fresh,
                expected_task_type="equivalence",
            )
        except CleanDownstreamHybridPlanError as exc:
            raise NoiseEquivalenceHybridPlanError(f"{base} successor 构造失败: {exc}") from exc
        successor_line = canonical_json(successor) + "\n"
        output_lines.append(successor_line)
        output_rows.append(successor)
        refresh_lines.append(successor_line)
        matched_old_pred_keys.add(old_dependencies[1])
        supersession = {
            "base_logical_id": base,
            "condition": condition,
            "predecessor_evaluation_key": predecessor["evaluation_key"],
            "predecessor_logical_id": predecessor["logical_id"],
            "successor_evaluation_key": successor["evaluation_key"],
            "successor_logical_id": successor["logical_id"],
            "gt_dependency": old_dependencies[0],
            "pred_predecessor_evaluation_key": old_dependencies[1],
            "pred_successor_evaluation_key": new_dependencies[1],
            "identity": f"{condition}_equivalence_pred_replacement::{base}",
            "reason": f"{condition} pred successor changed equivalence evidence",
        }
        supersessions.append(supersession)
        state_bindings.append(
            {
                **supersession,
                "task_type": "equivalence",
                "successor_spec": _task_spec(successor),
            }
        )
    if matched_old_pred_keys != set(pred_mapping):
        missing = sorted(set(pred_mapping) - matched_old_pred_keys)
        raise NoiseEquivalenceHybridPlanError(
            f"pred bindings 未与 equivalence 一一闭合: missing={missing[:3]}"
        )
    if expected_refresh_count is not None and len(supersessions) != expected_refresh_count:
        raise NoiseEquivalenceHybridPlanError(
            f"refresh 数量不符: {len(supersessions)} != {expected_refresh_count}"
        )

    logical_ids = [str(row["logical_id"]) for row in output_rows]
    evaluation_keys = [str(row["evaluation_key"]) for row in output_rows]
    if (
        len(output_rows) != len(predecessor_rows)
        or len(set(logical_ids)) != len(logical_ids)
        or len(set(evaluation_keys)) != len(evaluation_keys)
    ):
        raise NoiseEquivalenceHybridPlanError(
            "hybrid condition/logical identity/count/SHA 不唯一"
        )
    remaining_nonapp = [
        str(row["logical_id"])
        for row in output_rows
        if row.get("status") == "planned_non_applicable"
    ]
    if remaining_nonapp:
        raise NoiseEquivalenceHybridPlanError(
            "hybrid active plan 仍含 predecessor non-applicable: "
            f"{remaining_nonapp[:3]}"
        )

    full_content = "".join(output_lines)
    refresh_content = "".join(refresh_lines)
    _atomic_write(paths["output_full"], full_content)
    _atomic_write(paths["output_active"], full_content)
    _atomic_write(paths["output_refresh"], refresh_content)
    try:
        verified_full = load_plan_jsonl(paths["output_full"])
        verified_active = load_plan_jsonl(paths["output_active"])
        verified_refresh = load_plan_jsonl(paths["output_refresh"])
    except PlanContractError as exc:
        raise NoiseEquivalenceHybridPlanError(f"输出计划回读失败: {exc}") from exc
    if (
        len(verified_full.entries) != len(output_rows)
        or len(verified_active.entries) != len(output_rows)
        or len(verified_refresh.entries) != len(supersessions)
        or verified_full.plan_sha256 != verified_active.plan_sha256
    ):
        raise NoiseEquivalenceHybridPlanError("输出计划 count/SHA 回读漂移")

    state_registration: JsonDict = {
        "requested": False,
        "mutated": False,
        "state_db": None,
    }
    if apply_state_db is not None:
        try:
            state_path = _validate_apply_state_db_path(Path(apply_state_db))
            state_registration = _register_state(
                state_path,
                bindings=state_bindings,
                equivalence_predecessor_sha=predecessor_sha,
                equivalence_successor_sha=verified_full.plan_sha256,
                structure_predecessor_sha=predecessor_sha,
                structure_successor_sha=verified_full.plan_sha256,
            )
        except CleanDownstreamHybridPlanError as exc:
            raise NoiseEquivalenceHybridPlanError(f"状态库 supersession 注册失败: {exc}") from exc

    supersession_manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "condition": condition,
        "counts": {"supersessions": len(supersessions)},
        "predecessor_plan_sha256": predecessor_sha,
        "fresh_plan_sha256": fresh_sha,
        "hybrid_plan_sha256": verified_full.plan_sha256,
        "refresh_plan_sha256": verified_refresh.plan_sha256,
        "pred_hybrid_manifest_sha256": _sha256_file(paths["pred_manifest"]),
        "supersessions": supersessions,
    }
    _atomic_write_json(paths["supersession_manifest"], supersession_manifest)

    report: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "condition": condition,
        "model_invoked": False,
        "counts": {
            "total": len(output_rows),
            "changed": len(supersessions),
            "preserved": len(output_rows) - len(supersessions),
            "refresh_callable": len(supersessions),
            "fresh_non_applicable": nonapp_count,
        },
        "inputs": {
            "predecessor_equivalence_plan": {
                "path": str(paths["predecessor"]),
                "sha256": predecessor_sha,
            },
            "fresh_equivalence_plan": {
                "path": str(paths["fresh"]),
                "sha256": fresh_sha,
            },
            "pred_hybrid_manifest": {
                "path": str(paths["pred_manifest"]),
                "sha256": _sha256_file(paths["pred_manifest"]),
            },
            "fresh_non_applicable": (
                {
                    "path": str(paths["fresh_nonapp"]),
                    "sha256": _sha256_file(paths["fresh_nonapp"]),
                }
                if "fresh_nonapp" in paths
                else None
            ),
        },
        "outputs": {
            "full_hybrid_plan": {
                "path": str(paths["output_full"]),
                "sha256": verified_full.plan_sha256,
            },
            "active_hybrid_plan": {
                "path": str(paths["output_active"]),
                "sha256": verified_active.plan_sha256,
            },
            "refresh_callable_plan": {
                "path": str(paths["output_refresh"]),
                "sha256": verified_refresh.plan_sha256,
            },
            "supersession_manifest": {
                "path": str(paths["supersession_manifest"]),
                "sha256": _sha256_file(paths["supersession_manifest"]),
            },
        },
        "state_registration": state_registration,
    }
    _atomic_write_json(paths["report"], report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("noise001", "noise005"), required=True)
    parser.add_argument("--predecessor-equivalence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-equivalence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-non-applicable-jsonl", type=Path)
    parser.add_argument("--pred-hybrid-manifest-json", type=Path, required=True)
    parser.add_argument("--output-full-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-active-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-refresh-plan-jsonl", type=Path, required=True)
    parser.add_argument("--supersession-manifest-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-total-count", type=int, default=2250)
    parser.add_argument("--expected-refresh-count", type=int)
    parser.add_argument("--apply-state-db", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_noise_equivalence_hybrid_plan(
            condition=args.condition,
            predecessor_equivalence_plan_jsonl=args.predecessor_equivalence_plan_jsonl,
            fresh_equivalence_plan_jsonl=args.fresh_equivalence_plan_jsonl,
            fresh_non_applicable_jsonl=args.fresh_non_applicable_jsonl,
            pred_hybrid_manifest_json=args.pred_hybrid_manifest_json,
            output_full_plan_jsonl=args.output_full_plan_jsonl,
            output_active_plan_jsonl=args.output_active_plan_jsonl,
            output_refresh_plan_jsonl=args.output_refresh_plan_jsonl,
            supersession_manifest_json=args.supersession_manifest_json,
            report_json=args.report_json,
            expected_total_count=args.expected_total_count,
            expected_refresh_count=args.expected_refresh_count,
            apply_state_db=args.apply_state_db,
        )
    except NoiseEquivalenceHybridPlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
