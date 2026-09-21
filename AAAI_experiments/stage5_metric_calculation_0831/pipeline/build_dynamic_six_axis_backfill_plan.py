"""构建逐分钟六轴历史回填计划，不执行任何网络或模型请求。

输入由逐 run 数值宽表、冻结分钟快照以及 clean EFF 证据组成。输出是可按稳定键
断点续跑的 run-minute 骨架、去重后的符号任务和 task-minute seed-pair 计划。
clean 只有在来源分钟和 canonical EFF 数值都能与冻结快照严格对齐时才释放表达式；
否则 fail-closed 为 unresolved，绝不猜测公式。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

from .metrics import phi_nmse
from .performance_replay import EVALUATION_PATH
from .prepare_eff import TRAJECTORY_EVIDENCE_SCHEMA_VERSION


CONDITIONS = ("clean", "noise001", "noise005")
FORBIDDEN_SOURCE_TOKENS = ("all_15alg_fullcpu_v1",)
SCHEMA_VERSION = "stage5.dynamic_six_axis_backfill_plan.v1"
FLOAT_TOLERANCE = 1.0e-12
SOURCE_MINUTE_RE = re.compile(
    r"(?:snapshot|final|carry_forward|internal_best_carry_forward|"
    r"budget_end_internal_best|audited_repair):(\d+)"
)


class DynamicBackfillPlanError(ValueError):
    """输入违反逐分钟回填计划契约。"""


@dataclass(frozen=True)
class SourceBundle:
    condition: str
    tier: str
    precedence: int
    path: Path


@dataclass(frozen=True)
class CleanEvidence:
    tier: str
    path: Path


@dataclass(frozen=True)
class CompactSnapshot:
    minute: int
    expression: str | None
    id_quality: float | None
    ood_quality: float | None
    quality: float | None
    status: str
    selected_path: str | None
    selected_sha256: str | None
    backfilled_from_minute: int | None


@dataclass(frozen=True)
class CompactRun:
    logical_key: str
    condition: str
    algorithm: str
    dataset_id: str
    seed: int
    task_id: str
    tier: str
    bundle_path: str
    bundle_sha256: str
    source_record_path: str | None
    snapshots: Mapping[int, CompactSnapshot]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_forbidden(value: object, *, context: str) -> None:
    text = _canonical_json(value) if isinstance(value, (dict, list, tuple)) else str(value)
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise DynamicBackfillPlanError(f"{context} 命中禁止来源 {token!r}")


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DynamicBackfillPlanError(f"{path}:{line_number} JSON 无效") from exc
            if not isinstance(row, dict):
                raise DynamicBackfillPlanError(f"{path}:{line_number} 不是 JSON object")
            yield row


def _logical_key(source: Mapping[str, Any]) -> str:
    try:
        return (
            f"{source['algorithm']}::{source['dataset_id']}::"
            f"s{int(source['seed'])}::{source['noise_tag']}"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicBackfillPlanError("source 缺少合法身份字段") from exc


def _finite_quality(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _snapshot_quality(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return phi_nmse(number) if math.isfinite(number) and number >= 0.0 else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _compact_snapshot(snapshot: Mapping[str, Any]) -> CompactSnapshot:
    try:
        minute = int(snapshot["minute"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicBackfillPlanError("snapshot.minute 无效") from exc
    expression_value = snapshot.get("expression")
    expression = expression_value.strip() if isinstance(expression_value, str) and expression_value.strip() else None
    id_quality = _snapshot_quality(snapshot.get("id_nmse"))
    ood_quality = _snapshot_quality(snapshot.get("ood_nmse"))
    quality = (
        (id_quality + ood_quality) / 2.0
        if id_quality is not None and ood_quality is not None
        else None
    )
    return CompactSnapshot(
        minute=minute,
        expression=expression,
        id_quality=id_quality,
        ood_quality=ood_quality,
        quality=quality,
        status=str(snapshot.get("status") or "unknown"),
        selected_path=str(snapshot["selected_path"]) if snapshot.get("selected_path") else None,
        selected_sha256=str(snapshot["selected_sha256"]) if snapshot.get("selected_sha256") else None,
        backfilled_from_minute=_optional_int(snapshot.get("backfilled_from_minute")),
    )


def _load_source_runs(specs: Iterable[SourceBundle]) -> tuple[dict[str, CompactRun], list[dict[str, Any]]]:
    selected: dict[str, tuple[int, CompactRun]] = {}
    artifacts: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: (item.precedence, str(item.path))):
        if spec.condition not in CONDITIONS:
            raise DynamicBackfillPlanError(f"非法 condition: {spec.condition}")
        path = spec.path.resolve()
        _check_forbidden(path, context="source bundle 路径")
        if not path.is_file():
            raise DynamicBackfillPlanError(f"source bundle 不存在: {path}")
        file_sha = _sha256_file(path)
        row_count = 0
        seen_in_file: set[str] = set()
        for row in _iter_jsonl(path):
            row_count += 1
            source = row.get("source")
            if not isinstance(source, Mapping):
                raise DynamicBackfillPlanError(f"{path} 的记录缺少 source")
            _check_forbidden(source, context=f"{path} source")
            key = _logical_key(source)
            if key in seen_in_file:
                raise DynamicBackfillPlanError(f"{path} 内 logical_key 重复: {key}")
            seen_in_file.add(key)
            if str(source.get("noise_tag")) != spec.condition:
                continue
            raw_snapshots = row.get("snapshots")
            if not isinstance(raw_snapshots, list):
                raise DynamicBackfillPlanError(f"{key} 缺少 snapshots")
            snapshots: dict[int, CompactSnapshot] = {}
            for raw_snapshot in raw_snapshots:
                if not isinstance(raw_snapshot, Mapping):
                    raise DynamicBackfillPlanError(f"{key} snapshot 不是 object")
                snapshot = _compact_snapshot(raw_snapshot)
                if snapshot.minute in snapshots:
                    raise DynamicBackfillPlanError(f"{key} minute={snapshot.minute} 重复")
                snapshots[snapshot.minute] = snapshot
            compact = CompactRun(
                logical_key=key,
                condition=spec.condition,
                algorithm=str(source["algorithm"]),
                dataset_id=str(source["dataset_id"]),
                seed=int(source["seed"]),
                task_id=str(source.get("task_id") or ""),
                tier=spec.tier,
                bundle_path=str(path),
                bundle_sha256=file_sha,
                source_record_path=str(source["path"]) if source.get("path") else None,
                snapshots=snapshots,
            )
            previous = selected.get(key)
            if previous is not None and previous[0] == spec.precedence:
                raise DynamicBackfillPlanError(f"同优先级 source bundle 重复 logical_key: {key}")
            if previous is None or spec.precedence > previous[0]:
                selected[key] = (spec.precedence, compact)
        artifacts.append({"path": str(path), "sha256": file_sha, "record_count": row_count, "tier": spec.tier, "precedence": spec.precedence})
    return {key: value[1] for key, value in selected.items()}, artifacts


def _load_clean_evidence(specs: Iterable[CleanEvidence], horizon: int) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        path = spec.path.resolve()
        _check_forbidden(path, context="clean evidence 路径")
        file_sha = _sha256_file(path)
        count = 0
        for row in _iter_jsonl(path):
            count += 1
            key = str(row.get("logical_key") or "")
            qualities = row.get("quality_trajectory")
            sources = row.get("trajectory_sources")
            expressions = row.get("selected_expression_trajectory")
            id_qualities = row.get("id_quality_trajectory")
            ood_qualities = row.get("ood_quality_trajectory")
            valid_outputs = row.get("valid_output_trajectory")
            if not key:
                raise DynamicBackfillPlanError(f"{path} clean evidence 缺少 logical_key")
            arrays = {
                "quality_trajectory": qualities,
                "trajectory_sources": sources,
                "selected_expression_trajectory": expressions,
                "id_quality_trajectory": id_qualities,
                "ood_quality_trajectory": ood_qualities,
                "valid_output_trajectory": valid_outputs,
            }
            contract_errors = [
                f"{name} 缺失或不是数组"
                for name, value in arrays.items()
                if not isinstance(value, list)
            ]
            contract_errors.extend(
                f"{name} 不是 {horizon} 点"
                for name, value in arrays.items()
                if isinstance(value, list) and len(value) != horizon
            )
            if row.get("evaluation_path") != EVALUATION_PATH:
                contract_errors.append("evaluation_path 不是 canonical replay")
            if (
                row.get("trajectory_evidence_schema_version")
                != TRAJECTORY_EVIDENCE_SCHEMA_VERSION
            ):
                contract_errors.append("trajectory evidence schema 版本不匹配")
            evidence_key = (spec.tier, key)
            if evidence_key in evidence:
                raise DynamicBackfillPlanError(f"clean evidence 重复: {evidence_key}")
            evidence[evidence_key] = {
                **row,
                "_contract_errors": contract_errors,
                "_evidence_path": str(path),
                "_evidence_sha256": file_sha,
            }
        artifacts.append({"path": str(path), "sha256": file_sha, "record_count": count, "tier": spec.tier})
    return evidence, artifacts


def _read_numeric_rows(path: Path, *, condition: str, horizon: int) -> list[dict[str, str]]:
    resolved = path.resolve()
    _check_forbidden(resolved, context="numeric 路径")
    with _open_text(resolved) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DynamicBackfillPlanError(f"numeric 表为空: {resolved}")
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("logical_key") or "")
        _check_forbidden(row, context=f"numeric row {key}")
        if key in seen:
            raise DynamicBackfillPlanError(f"numeric logical_key 重复: {key}")
        seen.add(key)
        if row.get("condition", row.get("noise_tag")) != condition:
            raise DynamicBackfillPlanError(f"numeric {key} condition 不匹配")
        for minute in range(1, horizon + 1):
            if _finite_quality(row.get(f"q_{minute:04d}")) is None:
                raise DynamicBackfillPlanError(f"numeric {key} 缺少合法 q_{minute:04d}")
    return rows


def _close(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and math.isclose(left, right, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def _source_evidence(run: CompactRun, snapshot: CompactSnapshot | None, *, source_minute: int | None, source_label: str) -> dict[str, Any]:
    return {
        "bundle_path": run.bundle_path,
        "bundle_sha256": run.bundle_sha256,
        "source_tier": run.tier,
        "source_record_path": run.source_record_path,
        "source_label": source_label,
        "source_minute": source_minute,
        "snapshot_path": snapshot.selected_path if snapshot else None,
        "snapshot_sha256": snapshot.selected_sha256 if snapshot else None,
    }


def _unresolved_point(run: CompactRun, minute: int, q: float, reason: str, *, source_label: str = "") -> dict[str, Any]:
    return {
        "expression": None,
        "expression_status": "unresolved",
        "valid_output": None,
        "invalid_reason": None,
        "unresolved_reason": reason,
        "id_quality": None,
        "ood_quality": None,
        "q": q,
        "source_evidence": _source_evidence(run, None, source_minute=None, source_label=source_label),
    }


def _invalid_point(run: CompactRun, q: float, *, reason: str, source_label: str) -> dict[str, Any]:
    if not _close(q, 0.0):
        return _unresolved_point(run, 0, q, f"无有效表达式但 q={q} 非零", source_label=source_label)
    return {
        "expression": None,
        "expression_status": "invalid",
        "valid_output": False,
        "invalid_reason": reason,
        "unresolved_reason": None,
        "id_quality": 0.0,
        "ood_quality": 0.0,
        "q": q,
        "source_evidence": _source_evidence(run, None, source_minute=None, source_label=source_label),
    }


def _resolve_noise_points(run: CompactRun, numeric: Mapping[str, str], horizon: int) -> list[dict[str, Any]]:
    best_snapshot: CompactSnapshot | None = None
    best_quality = 0.0
    points: list[dict[str, Any]] = []
    for minute in range(1, horizon + 1):
        snapshot = run.snapshots.get(minute)
        if snapshot is not None and snapshot.quality is not None and snapshot.quality > best_quality:
            best_snapshot = snapshot
            best_quality = snapshot.quality
        q = float(numeric[f"q_{minute:04d}"])
        id_q = _finite_quality(numeric.get(f"id_q_{minute:04d}"))
        ood_q = _finite_quality(numeric.get(f"ood_q_{minute:04d}"))
        if id_q is None or ood_q is None:
            points.append(_unresolved_point(run, minute, q, "noise numeric 缺少 ID/OOD quality"))
            continue
        if not _close(q, (id_q + ood_q) / 2.0):
            points.append(_unresolved_point(run, minute, q, "noise q 与 ID/OOD 均值不一致"))
            continue
        if best_snapshot is None:
            points.append(_invalid_point(run, q, reason="截至当前分钟无可评估快照", source_label="observed_best_so_far:none"))
            continue
        if not (_close(id_q, best_snapshot.id_quality) and _close(ood_q, best_snapshot.ood_quality) and _close(q, best_snapshot.quality)):
            points.append(_unresolved_point(run, minute, q, "noise 数值轨迹与冻结 best-so-far 快照不一致"))
            continue
        if best_snapshot.expression is None:
            point = _unresolved_point(run, minute, q, "数值最优快照缺少表达式", source_label=f"observed_best_so_far:{best_snapshot.minute}")
            point["id_quality"] = id_q
            point["ood_quality"] = ood_q
            points.append(point)
            continue
        points.append({
            "expression": best_snapshot.expression,
            "expression_status": "resolved",
            "valid_output": True,
            "invalid_reason": None,
            "unresolved_reason": None,
            "id_quality": id_q,
            "ood_quality": ood_q,
            "q": q,
            "source_evidence": _source_evidence(run, best_snapshot, source_minute=best_snapshot.minute, source_label=f"observed_best_so_far:{best_snapshot.minute}"),
        })
    return points


def _source_minute(source_label: str) -> int | None:
    matches = SOURCE_MINUTE_RE.findall(source_label)
    return int(matches[-1]) if matches else None


def _resolve_clean_points(run: CompactRun, numeric: Mapping[str, str], evidence: Mapping[str, Any] | None, horizon: int) -> list[dict[str, Any]]:
    if evidence is None:
        return [_unresolved_point(run, minute, float(numeric[f"q_{minute:04d}"]), "缺少对应 source_tier 的 clean canonical EFF 证据") for minute in range(1, horizon + 1)]
    contract_errors = evidence.get("_contract_errors")
    if isinstance(contract_errors, list) and contract_errors:
        reason = "clean canonical EFF 证据契约失败: " + "; ".join(
            str(item) for item in contract_errors
        )
        return [
            _unresolved_point(
                run,
                minute,
                float(numeric[f"q_{minute:04d}"]),
                reason,
            )
            for minute in range(1, horizon + 1)
        ]
    evidence_q = evidence["quality_trajectory"]
    evidence_sources = evidence["trajectory_sources"]
    expressions = evidence["selected_expression_trajectory"]
    id_qualities = evidence["id_quality_trajectory"]
    ood_qualities = evidence["ood_quality_trajectory"]
    valid_outputs = evidence["valid_output_trajectory"]
    points: list[dict[str, Any]] = []
    for minute in range(1, horizon + 1):
        q = float(numeric[f"q_{minute:04d}"])
        source_label = str(evidence_sources[minute - 1])
        if not _close(q, _finite_quality(evidence_q[minute - 1])):
            points.append(_unresolved_point(run, minute, q, "clean numeric 与 canonical EFF 证据不一致", source_label=source_label))
            continue
        id_quality = _finite_quality(id_qualities[minute - 1])
        ood_quality = _finite_quality(ood_qualities[minute - 1])
        if id_quality is None or ood_quality is None or not _close(
            q, (id_quality + ood_quality) / 2.0
        ):
            points.append(_unresolved_point(run, minute, q, "clean canonical ID/OOD 分量与 combined q 不一致", source_label=source_label))
            continue
        valid_output = valid_outputs[minute - 1]
        if not isinstance(valid_output, bool):
            points.append(_unresolved_point(run, minute, q, "clean canonical valid_output 不是布尔值", source_label=source_label))
            continue
        source_minute = _source_minute(source_label)
        snapshot = run.snapshots.get(source_minute) if source_minute is not None else None
        expression_value = expressions[minute - 1]
        expression = (
            expression_value.strip()
            if isinstance(expression_value, str) and expression_value.strip()
            else None
        )
        canonical_evidence = {
            "path": evidence.get("_evidence_path"),
            "sha256": evidence.get("_evidence_sha256"),
            "schema_version": evidence.get("trajectory_evidence_schema_version"),
            "evaluation_path": evidence.get("evaluation_path"),
            "point_index": minute - 1,
        }
        if not valid_output:
            if not (_close(q, 0.0) and _close(id_quality, 0.0) and _close(ood_quality, 0.0)):
                points.append(_unresolved_point(run, minute, q, "clean 无效 canonical 候选没有显式零分", source_label=source_label))
                continue
            point = _invalid_point(
                run,
                q,
                reason=source_label or "canonical_invalid_output",
                source_label=source_label,
            )
            point["source_evidence"]["canonical_evidence"] = canonical_evidence
            points.append(point)
            continue
        if expression is None:
            points.append(_unresolved_point(run, minute, q, "clean 有效 canonical 候选缺少所选表达式", source_label=source_label))
            continue
        if source_minute is None:
            points.append(_unresolved_point(run, minute, q, "clean trajectory_source 缺少实际候选来源分钟", source_label=source_label))
            continue
        source_evidence = _source_evidence(
            run,
            snapshot,
            source_minute=source_minute,
            source_label=source_label,
        )
        source_evidence["canonical_evidence"] = canonical_evidence
        points.append({
            "expression": expression,
            "expression_status": "resolved",
            "valid_output": True,
            "invalid_reason": None,
            "unresolved_reason": None,
            "id_quality": id_quality,
            "ood_quality": ood_quality,
            "q": q,
            "source_evidence": source_evidence,
        })
    return points


class _AtomicJsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.handle: TextIO | None = None
        self.count = 0

    def __enter__(self) -> "_AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temp.open("w", encoding="utf-8", newline="")
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        assert self.handle is not None
        self.handle.write(_canonical_json(row) + "\n")
        self.count += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.handle is not None
        self.handle.close()
        if exc_type is None:
            self.temp.replace(self.path)
        elif self.temp.exists():
            self.temp.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _expression_task(expression: str) -> tuple[str, dict[str, Any]]:
    key = "pred_simplify::" + _sha256_text(_canonical_json({"expression": expression, "contract": "symbolic_simplify.v1"}))
    return key, {"task_key": key, "task_type": "pred_simplify", "state": "pending", "expression": expression, "contract": "symbolic_simplify.v1"}


def _equivalence_task(dataset_id: str, simplify_key: str) -> tuple[str, dict[str, Any]]:
    key = "equivalence::" + _sha256_text(_canonical_json({"dataset_id": dataset_id, "pred_simplify_key": simplify_key, "contract": "symbolic_equivalence.v1"}))
    return key, {"task_key": key, "task_type": "equivalence", "state": "pending", "dataset_id": dataset_id, "dependencies": [f"gt_simplify::{dataset_id}", simplify_key], "contract": "symbolic_equivalence.v1"}


def build_dynamic_six_axis_backfill_plan(
    *,
    numeric_paths: Mapping[str, Path],
    source_bundles: Sequence[SourceBundle],
    clean_evidence: Sequence[CleanEvidence],
    output_dir: Path,
    horizon: int = 180,
    expected_runs_by_condition: Mapping[str, int] | None = None,
    expected_seeds: Sequence[int] = (520, 521, 522),
) -> dict[str, Any]:
    """生成本地计划。所有输出以 manifest 最后提交，便于可靠续跑。"""

    if horizon <= 0:
        raise DynamicBackfillPlanError("horizon 必须为正整数")
    conditions = tuple(condition for condition in CONDITIONS if condition in numeric_paths)
    if not conditions:
        raise DynamicBackfillPlanError("未提供任何 numeric 输入")
    missing_sources = set(conditions) - {spec.condition for spec in source_bundles}
    if missing_sources:
        raise DynamicBackfillPlanError(f"缺少 source bundle: {sorted(missing_sources)}")

    runs, source_artifacts = _load_source_runs(spec for spec in source_bundles if spec.condition in conditions)
    evidence, evidence_artifacts = _load_clean_evidence(clean_evidence, horizon)
    output_dir = output_dir.resolve()
    _check_forbidden(output_dir, context="输出路径")

    run_minute_path = output_dir / "run_minute_plan.jsonl"
    symbolic_path = output_dir / "symbolic_task_plan.jsonl"
    stab_path = output_dir / "task_minute_stab_pair_plan.jsonl"
    unresolved_path = output_dir / "unresolved.jsonl"
    manifest_path = output_dir / "manifest.json"

    expression_tasks: dict[str, dict[str, Any]] = {}
    equivalence_tasks: dict[str, dict[str, Any]] = {}
    condition_counts: dict[str, dict[str, int]] = {}
    unresolved_reasons: Counter[str] = Counter()
    inputs: list[dict[str, Any]] = [*source_artifacts, *evidence_artifacts]
    points_by_group: dict[tuple[str, str, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)

    with _AtomicJsonlWriter(run_minute_path) as run_writer, _AtomicJsonlWriter(unresolved_path) as unresolved_writer:
        for condition in conditions:
            numeric_path = numeric_paths[condition].resolve()
            numeric_sha = _sha256_file(numeric_path)
            numeric_rows = _read_numeric_rows(numeric_path, condition=condition, horizon=horizon)
            inputs.append({"path": str(numeric_path), "sha256": numeric_sha, "record_count": len(numeric_rows), "kind": "numeric"})
            expected = expected_runs_by_condition.get(condition) if expected_runs_by_condition else None
            if expected is not None and len(numeric_rows) != expected:
                raise DynamicBackfillPlanError(f"{condition} run 数={len(numeric_rows)}，期望 {expected}")
            state_counts: Counter[str] = Counter()
            for numeric in sorted(numeric_rows, key=lambda row: (str(row["algorithm"]), str(row["dataset_id"]), int(row["seed"]))):
                key = str(numeric["logical_key"])
                run = runs.get(key)
                if run is None:
                    raise DynamicBackfillPlanError(f"numeric run 缺少冻结 source: {key}")
                tier = str(numeric.get("source_tier") or run.tier)
                if condition == "clean":
                    points = _resolve_clean_points(run, numeric, evidence.get((tier, key)), horizon)
                else:
                    points = _resolve_noise_points(run, numeric, horizon)
                q_star = max(float(numeric[f"q_{minute:04d}"]) for minute in range(1, horizon + 1))
                cumulative = 0.0
                for minute, point in enumerate(points, start=1):
                    relative = float(point["q"]) / q_star if q_star > 0.0 else 0.0
                    cumulative += relative
                    expression_key = None
                    equivalence_key = None
                    if point["expression_status"] == "resolved":
                        expression_key, task = _expression_task(str(point["expression"]))
                        expression_tasks.setdefault(expression_key, task)
                        equivalence_key, task = _equivalence_task(run.dataset_id, expression_key)
                        equivalence_tasks.setdefault(equivalence_key, task)
                    run_minute_key = f"{key}::m{minute:04d}"
                    row = {
                        "run_minute_key": run_minute_key,
                        "condition": condition,
                        "algorithm": run.algorithm,
                        "dataset_id": run.dataset_id,
                        "seed": run.seed,
                        "minute": minute,
                        **point,
                        "sym_score": None,
                        "min_score": None,
                        "stab_score": None,
                        "relative_progress": relative,
                        "cumulative_eff": cumulative / minute,
                        "pred_simplify_key": expression_key,
                        "equivalence_key": equivalence_key,
                        "numeric_source": {"path": str(numeric_path), "sha256": numeric_sha, "logical_key": key},
                    }
                    run_writer.write(row)
                    points_by_group[(condition, run.algorithm, run.dataset_id, minute)][run.seed] = row
                    state_counts[str(point["expression_status"])] += 1
                    if point["expression_status"] == "unresolved":
                        reason = str(point["unresolved_reason"])
                        unresolved_reasons[reason] += 1
                        unresolved_writer.write({"run_minute_key": run_minute_key, "condition": condition, "reason": reason, "source_evidence": point["source_evidence"]})
            condition_counts[condition] = {"run_count": len(numeric_rows), "run_minute_count": len(numeric_rows) * horizon, **dict(sorted(state_counts.items()))}

    with _AtomicJsonlWriter(symbolic_path) as symbolic_writer:
        for task in sorted(expression_tasks.values(), key=lambda row: str(row["task_key"])):
            symbolic_writer.write(task)
        for task in sorted(equivalence_tasks.values(), key=lambda row: str(row["task_key"])):
            symbolic_writer.write(task)

    seeds = tuple(int(seed) for seed in expected_seeds)
    pair_list = tuple((left, right) for index, left in enumerate(seeds) for right in seeds[index + 1 :])
    with _AtomicJsonlWriter(stab_path) as stab_writer:
        for group, by_seed in sorted(points_by_group.items()):
            condition, algorithm, dataset_id, minute = group
            missing = sorted(set(seeds) - set(by_seed))
            if missing:
                raise DynamicBackfillPlanError(f"STAB group 缺少 seeds: {group}, missing={missing}")
            for left, right in pair_list:
                left_row, right_row = by_seed[left], by_seed[right]
                statuses = {left_row["expression_status"], right_row["expression_status"]}
                eligible = statuses == {"resolved"}
                state = "pending" if eligible else ("not_applicable" if "invalid" in statuses and "unresolved" not in statuses else "blocked")
                comparison_key = None
                if eligible:
                    pair = sorted((str(left_row["pred_simplify_key"]), str(right_row["pred_simplify_key"])))
                    comparison_key = "stab_structure::" + _sha256_text(_canonical_json({"dataset_id": dataset_id, "pair": pair, "contract": "symbolic_structure.v1"}))
                left_id = left_row["id_quality"]
                right_id = right_row["id_quality"]
                left_ood = left_row["ood_quality"]
                right_ood = right_row["ood_quality"]
                numerical_disagreement = (
                    (abs(float(left_id) - float(right_id)) + abs(float(left_ood) - float(right_ood))) / 2.0
                    if None not in (left_id, right_id, left_ood, right_ood)
                    else None
                )
                stab_writer.write({
                    "task_minute_pair_key": f"{condition}::{algorithm}::{dataset_id}::m{minute:04d}::s{left}-s{right}",
                    "condition": condition,
                    "algorithm": algorithm,
                    "dataset_id": dataset_id,
                    "minute": minute,
                    "seed_left": left,
                    "seed_right": right,
                    "run_minute_left": left_row["run_minute_key"],
                    "run_minute_right": right_row["run_minute_key"],
                    "eligible": eligible,
                    "state": state,
                    "id_quality_left": left_id,
                    "id_quality_right": right_id,
                    "ood_quality_left": left_ood,
                    "ood_quality_right": right_ood,
                    "numerical_disagreement": numerical_disagreement,
                    "valid_output_left": left_row["valid_output"],
                    "valid_output_right": right_row["valid_output"],
                    "structure_comparison_key": comparison_key,
                    "dependencies": [key for key in (left_row["pred_simplify_key"], right_row["pred_simplify_key"]) if key],
                })

    expected_run_minutes = sum(values["run_count"] * horizon for values in condition_counts.values())
    expected_pairs = len(points_by_group) * len(pair_list)
    target_run_count = sum(
        (
            expected_runs_by_condition[condition]
            if expected_runs_by_condition is not None
            and condition in expected_runs_by_condition
            else condition_counts[condition]["run_count"]
        )
        for condition in conditions
    )
    target_run_minutes = target_run_count * horizon
    if target_run_count % len(seeds) != 0:
        raise DynamicBackfillPlanError(
            f"目标 run 数 {target_run_count} 不能按 {len(seeds)} 个 seed 分组"
        )
    target_groups = target_run_count // len(seeds) * horizon
    target_pairs = target_groups * len(pair_list)
    output_info = {}
    for name, path, count in (
        ("run_minute_plan", run_minute_path, expected_run_minutes),
        ("symbolic_task_plan", symbolic_path, len(expression_tasks) + len(equivalence_tasks)),
        ("task_minute_stab_pair_plan", stab_path, expected_pairs),
        ("unresolved", unresolved_path, sum(unresolved_reasons.values())),
    ):
        output_info[name] = {"path": str(path), "sha256": _sha256_file(path), "record_count": count}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if not unresolved_reasons else "complete_with_unresolved",
        "horizon_minutes": horizon,
        "conditions": condition_counts,
        "targets": {
            "expected": {
                "run_count": target_run_count,
                "run_minute_count": target_run_minutes,
                "task_minute_group_count": target_groups,
                "seed_pair_count_per_group": len(pair_list),
                "task_minute_pair_count": target_pairs,
            },
            "actual": {
                "run_count": sum(values["run_count"] for values in condition_counts.values()),
                "run_minute_count": expected_run_minutes,
                "task_minute_group_count": len(points_by_group),
                "seed_pair_count_per_group": len(pair_list),
                "task_minute_pair_count": expected_pairs,
            },
            "contract_satisfied": (
                expected_run_minutes == target_run_minutes
                and len(points_by_group) == target_groups
                and expected_pairs == target_pairs
            ),
        },
        "axis_fields": {
            "ID": "id_quality",
            "OOD": "ood_quality",
            "SYM": "sym_score",
            "MIN": "min_score",
            "EFF": "cumulative_eff",
            "STAB": "stab_score",
        },
        "resume_contract": {
            "run_minute_key": "stable primary key for run-minute scores",
            "symbolic_task_key": "stable deduplicated key; completed keys may be skipped",
            "task_minute_pair_key": "stable primary key for STAB seed pairs",
            "manifest_written_last": True,
        },
        "deduplication": {"pred_simplify_task_count": len(expression_tasks), "equivalence_task_count": len(equivalence_tasks)},
        "unresolved": {"record_count": sum(unresolved_reasons.values()), "reason_counts": dict(sorted(unresolved_reasons.items()))},
        "forbidden_source_check": {"passed": True, "tokens": list(FORBIDDEN_SOURCE_TOKENS)},
        "inputs": sorted(inputs, key=lambda row: (str(row.get("path")), str(row.get("tier", "")))),
        "outputs": output_info,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _parse_mapping(values: Sequence[str], *, parts: int, label: str) -> list[list[str]]:
    parsed: list[list[str]] = []
    for value in values:
        fields = value.split(":", maxsplit=parts - 1)
        if len(fields) != parts:
            raise DynamicBackfillPlanError(f"{label} 参数格式无效: {value}")
        parsed.append(fields)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="append", required=True, metavar="CONDITION:PATH")
    parser.add_argument("--source", action="append", required=True, metavar="CONDITION:TIER:PRECEDENCE:PATH")
    parser.add_argument("--clean-evidence", action="append", default=[], metavar="TIER:PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=180)
    args = parser.parse_args(argv)
    numeric = {condition: Path(path) for condition, path in _parse_mapping(args.numeric, parts=2, label="numeric")}
    sources = [SourceBundle(condition, tier, int(precedence), Path(path)) for condition, tier, precedence, path in _parse_mapping(args.source, parts=4, label="source")]
    clean_evidence = [CleanEvidence(tier, Path(path)) for tier, path in _parse_mapping(args.clean_evidence, parts=2, label="clean-evidence")]
    manifest = build_dynamic_six_axis_backfill_plan(numeric_paths=numeric, source_bundles=sources, clean_evidence=clean_evidence, output_dir=args.output_dir, horizon=args.horizon)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
