"""为正式六轴评测准备 clean 条件下的运行级 EFF 输入。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .freeze_binding import FreezeBindingContractError, validate_freeze_binding_summary
from .metrics import MetricContractError, efficiency_from_qualities
from .trajectories import TrajectoryContractError, reconstruct_trajectory
from .trajectory_repairs import (
    TrajectoryRepairContractError,
    apply_repair_manifest,
    load_repair_manifest,
)


HORIZON = 180
NOISE_TAG = "clean"
EXPECTED_HOSTS = 8
EXPECTED_TASKS = 2250
EXPECTED_POINTS = 405000
EXPECTED_EXISTING_POINTS = 404985
EXPECTED_MISSING_POINTS = 15
EXPECTED_AUDITED_REPAIR_POINTS = 15
EXPECTED_FUTURE_BACKFILL_IGNORED_POINTS = 34
EXPECTED_CHECKPOINT_NORMALIZATION_POINTS = 19


class EffPreparationContractError(ValueError):
    """冻结 bundle、修复清单或 EFF 准备过程不满足正式契约。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage5_root() -> Path:
    return _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _raise(message: str) -> None:
    raise EffPreparationContractError(message)


def _resolve_path(base: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (base / path).resolve()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EffPreparationContractError(f"{label} 不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        _raise(f"{label} 顶层必须是 JSON object: {path}")
    return payload


def _logical_key(source: Mapping[str, Any]) -> str:
    try:
        seed = int(source["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EffPreparationContractError(f"source.seed 无效: {source!r}") from exc
    return (
        f"{source.get('algorithm')}::{source.get('dataset_id')}::"
        f"s{seed}::{source.get('noise_tag')}"
    )


def _stable_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["algorithm"]),
        str(row["dataset_id"]),
        int(row["seed"]),
        str(row["task_id"]),
        str(row["host"]),
    )


def load_freeze_binding_report(
    path: Path,
    *,
    repo_root: Path,
    expected_hosts: int | None,
    expected_tasks: int | None,
    expected_points: int | None,
    expected_existing_points: int | None,
    expected_missing_points: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _resolve_path(repo_root, path)
    payload = _load_json_object(resolved, label="freeze_binding report")
    try:
        validate_freeze_binding_summary(
            payload,
            expected_hosts=expected_hosts,
            expected_tasks=expected_tasks,
            expected_points=expected_points,
            expected_existing_points=expected_existing_points,
            expected_missing_points=expected_missing_points,
        )
    except FreezeBindingContractError as exc:
        raise EffPreparationContractError(f"freeze_binding report 契约失败: {exc}") from exc
    if payload.get("noise_tag") != NOISE_TAG:
        _raise(f"freeze_binding report noise_tag 必须为 {NOISE_TAG}")
    if int(payload.get("horizon", 0)) != HORIZON:
        _raise(f"freeze_binding report horizon 必须为 {HORIZON}")
    report_info = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }
    return payload, report_info


def _verified_input_files(
    summary: Mapping[str, Any],
    *,
    repo_root: Path,
    section_name: str,
) -> dict[str, dict[str, Any]]:
    input_files = summary.get("input_files")
    if not isinstance(input_files, Mapping):
        _raise("freeze_binding report 缺少 input_files")
    entries = input_files.get(section_name)
    if not isinstance(entries, list) or not entries:
        _raise(f"freeze_binding report 缺少 {section_name}")
    verified: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            _raise(f"{section_name} 条目必须是 object")
        host = str(item.get("host") or "")
        raw_path = item.get("path")
        expected_sha = str(item.get("sha256") or "")
        if not host or not raw_path or not expected_sha:
            _raise(f"{section_name} 条目缺少 host/path/sha256")
        resolved = _resolve_path(repo_root, raw_path)
        if not resolved.is_file():
            _raise(f"{section_name} 文件不存在: {resolved}")
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            _raise(f"{section_name} SHA 不匹配: {resolved}")
        if host in verified:
            _raise(f"{section_name} 出现重复 host: {host}")
        verified[host] = {
            "host": host,
            "path": str(resolved),
            "sha256": actual_sha,
            "size_bytes": resolved.stat().st_size,
        }
    return dict(sorted(verified.items()))


def _iter_bundle_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EffPreparationContractError(
                    f"bundle {path.name}:{line_number} 不是合法 JSON"
                ) from exc
            if not isinstance(payload, dict):
                _raise(f"bundle {path.name}:{line_number} 顶层必须是 JSON object")
            yield payload


def _count_prefixed_sources(sources: Iterable[str], prefix: str) -> int:
    return sum(1 for source in sources if str(source).startswith(prefix))


def _future_backfill_minutes(raw_snapshots: list[Mapping[str, Any]], target_minute: int) -> list[int]:
    matched: list[int] = []
    for minute, snapshot in enumerate(raw_snapshots, start=1):
        if snapshot.get("status") != "ok":
            continue
        if snapshot.get("record_type") != "periodic_backfill":
            continue
        try:
            backfilled_from_minute = int(snapshot.get("backfilled_from_minute"))
        except (TypeError, ValueError):
            continue
        if backfilled_from_minute == target_minute and minute < target_minute:
            matched.append(minute)
    return matched


def _normalize_checkpoint_drift(
    parsed_snapshots: dict[int, dict[str, Any]],
    *,
    raw_snapshots: list[Mapping[str, Any]],
    logical_key: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    normalized = {minute: dict(payload) for minute, payload in parsed_snapshots.items()}
    audit: list[dict[str, Any]] = []
    for minute, payload in normalized.items():
        record_type = payload.get("record_type")
        index = payload.get("checkpoint_index")
        if record_type == "final_best" and minute == HORIZON and index == "final":
            continue
        try:
            parsed_index = int(index)
        except (TypeError, ValueError) as exc:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} checkpoint_index 无效: {index!r}"
            ) from exc
        if parsed_index == minute:
            continue
        if record_type != "periodic_best" or parsed_index > minute:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} 的 checkpoint_index={parsed_index} 不匹配"
            )
        leaked_minutes = _future_backfill_minutes(raw_snapshots, minute)
        if not leaked_minutes:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} 的 checkpoint_index={parsed_index} 缺少 future backfill 佐证"
            )
        payload["checkpoint_index_original"] = parsed_index
        payload["checkpoint_index"] = minute
        payload["checkpoint_index_normalization"] = {
            "reason": "future_backfill_source_minute",
            "future_backfill_minutes": leaked_minutes,
        }
        audit.append(
            {
                "minute": minute,
                "original_checkpoint_index": parsed_index,
                "normalized_checkpoint_index": minute,
                "future_backfill_minutes": leaked_minutes,
            }
        )
    return normalized, audit


def _csv_row_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "logical_key": record["logical_key"],
        "algorithm": record["algorithm"],
        "dataset_id": record["dataset_id"],
        "seed": record["seed"],
        "task_id": record["task_id"],
        "host": record["host"],
        "noise_tag": record["noise_tag"],
        "m_eff": f"{float(record['m_eff']):.17g}",
        "best_quality": f"{float(record['best_quality']):.17g}",
        "audited_repair_points": record["audited_repair_points"],
        "future_backfill_ignored_points": record["future_backfill_ignored_points"],
        "checkpoint_normalization_points": len(record["checkpoint_normalizations"]),
        "bundle_sha256": record["bundle_sha256"],
        "bundle_report_sha256": record["bundle_report_sha256"],
        "freeze_binding_report_sha256": record["freeze_binding_report_sha256"],
        "repair_manifest_sha256": record["repair_manifest_sha256"],
    }
    for minute, value in enumerate(record["quality_trajectory"], start=1):
        row[f"q_{minute:04d}"] = f"{float(value):.17g}"
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _raise("不允许写空 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_csv_row_from_record(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(flat_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_eff_preparation(
    *,
    freeze_binding_report: Path,
    repair_manifest: Path,
    repo_root: Path | None = None,
    expected_hosts: int | None = EXPECTED_HOSTS,
    expected_tasks: int | None = EXPECTED_TASKS,
    expected_points: int | None = EXPECTED_POINTS,
    expected_existing_points: int | None = EXPECTED_EXISTING_POINTS,
    expected_missing_points: int | None = EXPECTED_MISSING_POINTS,
    expected_audited_repair_points: int | None = EXPECTED_AUDITED_REPAIR_POINTS,
    expected_future_backfill_ignored_points: int | None = EXPECTED_FUTURE_BACKFILL_IGNORED_POINTS,
    expected_checkpoint_normalization_points: int | None = EXPECTED_CHECKPOINT_NORMALIZATION_POINTS,
    limit_runs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_root = repo_root.resolve() if repo_root is not None else _repo_root()
    binding_summary, binding_report_info = load_freeze_binding_report(
        freeze_binding_report,
        repo_root=repo_root,
        expected_hosts=expected_hosts,
        expected_tasks=expected_tasks,
        expected_points=expected_points,
        expected_existing_points=expected_existing_points,
        expected_missing_points=expected_missing_points,
    )
    freeze_bundles = _verified_input_files(
        binding_summary,
        repo_root=repo_root,
        section_name="freeze_records",
    )
    freeze_reports = _verified_input_files(
        binding_summary,
        repo_root=repo_root,
        section_name="freeze_reports",
    )
    if set(freeze_bundles) != set(freeze_reports):
        _raise("freeze bundle 与 bundle report host 集合不一致")

    manifest = load_repair_manifest(
        _resolve_path(repo_root, repair_manifest),
        repo_root=repo_root,
    )
    repair_manifest_info = {
        "path": str(_resolve_path(repo_root, repair_manifest)),
        "sha256": str(manifest["manifest_sha256"]),
    }

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    audited_repair_points = 0
    future_backfill_ignored_points = 0
    checkpoint_normalization_points = 0
    checkpoint_normalization_details: list[dict[str, Any]] = []
    missing_points_after_repairs = 0
    original_missing_points = 0
    processed_runs = 0

    for host, bundle_info in freeze_bundles.items():
        bundle_path = Path(bundle_info["path"])
        report_info = freeze_reports[host]
        for record in _iter_bundle_records(bundle_path):
            if limit_runs is not None and processed_runs >= limit_runs:
                break
            processed_runs += 1
            source = record.get("source")
            if not isinstance(source, Mapping):
                unresolved.append({"host": host, "reason": "record 缺少 source"})
                continue
            logical_key = _logical_key(source)
            if logical_key in seen_keys:
                unresolved.append({"logical_key": logical_key, "reason": "logical_key 重复"})
                continue
            seen_keys.add(logical_key)
            if str(source.get("host")) != host:
                unresolved.append({"logical_key": logical_key, "reason": "source.host 与 bundle host 不一致"})
                continue
            if str(source.get("noise_tag")) != NOISE_TAG:
                unresolved.append({"logical_key": logical_key, "reason": "仅允许 clean 记录"})
                continue
            raw_snapshots = record.get("snapshots")
            if not isinstance(raw_snapshots, list):
                unresolved.append({"logical_key": logical_key, "reason": "record 缺少 snapshots 数组"})
                continue
            original_missing_points += sum(
                1 for snapshot in raw_snapshots if isinstance(snapshot, Mapping) and snapshot.get("status") == "missing"
            )
            try:
                repaired_snapshots, repair_audit = apply_repair_manifest(
                    record,
                    manifest=manifest,
                    repo_root=repo_root,
                )
                repaired_snapshots, checkpoint_normalizations = _normalize_checkpoint_drift(
                    repaired_snapshots,
                    raw_snapshots=raw_snapshots,
                    logical_key=logical_key,
                )
                trajectory = reconstruct_trajectory(repaired_snapshots, horizon=HORIZON)
                quality_trajectory = [float(point.quality) for point in trajectory]
                trajectory_sources = [str(point.source) for point in trajectory]
                m_eff = float(efficiency_from_qualities(quality_trajectory, horizon=HORIZON))
            except (
                EffPreparationContractError,
                MetricContractError,
                TrajectoryContractError,
                TrajectoryRepairContractError,
            ) as exc:
                unresolved.append({"logical_key": logical_key, "reason": str(exc)})
                continue

            if not (0.0 <= m_eff <= 1.0):
                unresolved.append({"logical_key": logical_key, "reason": f"m_eff 越界: {m_eff}"})
                continue

            missing_after = HORIZON - len(trajectory)
            if missing_after != 0:
                missing_points_after_repairs += missing_after

            audited_count = _count_prefixed_sources(trajectory_sources, "audited_repair:")
            future_ignored_count = _count_prefixed_sources(trajectory_sources, "future_backfill_ignored:")
            audited_repair_points += audited_count
            future_backfill_ignored_points += future_ignored_count
            checkpoint_normalization_points += len(checkpoint_normalizations)
            checkpoint_normalization_details.extend(
                {"logical_key": logical_key, **item}
                for item in checkpoint_normalizations
            )

            row = {
                "logical_key": logical_key,
                "algorithm": str(source.get("algorithm")),
                "dataset_id": str(source.get("dataset_id")),
                "seed": int(source["seed"]),
                "task_id": str(source.get("task_id")),
                "host": host,
                "noise_tag": str(source.get("noise_tag")),
                "bundle_path": str(bundle_path),
                "bundle_sha256": bundle_info["sha256"],
                "bundle_report_path": str(report_info["path"]),
                "bundle_report_sha256": report_info["sha256"],
                "freeze_binding_report_path": binding_report_info["path"],
                "freeze_binding_report_sha256": binding_report_info["sha256"],
                "repair_manifest_path": repair_manifest_info["path"],
                "repair_manifest_sha256": repair_manifest_info["sha256"],
                "quality_trajectory": quality_trajectory,
                "trajectory_sources": trajectory_sources,
                "best_quality": max(quality_trajectory, default=0.0),
                "m_eff": m_eff,
                "audited_repair_points": audited_count,
                "future_backfill_ignored_points": future_ignored_count,
                "checkpoint_normalizations": checkpoint_normalizations,
                "repair_applied": bool(repair_audit.get("repair_applied")),
                "repair_minutes": list(repair_audit.get("applied_minutes", [])),
            }
            rows.append(row)
        if limit_runs is not None and processed_runs >= limit_runs:
            break

    rows.sort(key=_stable_row_sort_key)

    full_contract_checked = limit_runs is None
    if limit_runs is None and expected_tasks is not None and len(rows) != expected_tasks:
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": f"成功 run 数应为 {expected_tasks}，实际为 {len(rows)}",
            }
        )
    if limit_runs is None and expected_audited_repair_points is not None and audited_repair_points != expected_audited_repair_points:
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": (
                    f"audited_repair_points 应为 {expected_audited_repair_points}，"
                    f"实际为 {audited_repair_points}"
                ),
            }
        )
    if (
        limit_runs is None
        and expected_future_backfill_ignored_points is not None
        and future_backfill_ignored_points != expected_future_backfill_ignored_points
    ):
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": (
                    f"future_backfill_ignored_points 应为 {expected_future_backfill_ignored_points}，"
                    f"实际为 {future_backfill_ignored_points}"
                ),
            }
        )
    if limit_runs is None and missing_points_after_repairs != 0:
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": f"修复后仍有缺失点: {missing_points_after_repairs}",
            }
        )
    if (
        limit_runs is None
        and expected_checkpoint_normalization_points is not None
        and checkpoint_normalization_points != expected_checkpoint_normalization_points
    ):
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": (
                    "checkpoint_normalization_points 应为 "
                    f"{expected_checkpoint_normalization_points}，实际为 "
                    f"{checkpoint_normalization_points}"
                ),
            }
        )

    formal_eff_ready = (
        full_contract_checked
        and expected_tasks is not None
        and len(rows) == expected_tasks
        and missing_points_after_repairs == 0
        and not unresolved
    )
    report = {
        "condition": NOISE_TAG,
        "horizon": HORIZON,
        "inputs": {
            "freeze_binding_report": binding_report_info,
            "freeze_records": list(freeze_bundles.values()),
            "freeze_reports": list(freeze_reports.values()),
            "repair_manifest": repair_manifest_info,
        },
        "summary": {
            "processed_run_count": processed_runs,
            "success_count": len(rows),
            "unresolved_run_count": len(unresolved),
            "full_contract_checked": full_contract_checked,
            "limit_runs": limit_runs,
            "original_missing_points": original_missing_points,
            "missing_points_after_repairs": missing_points_after_repairs,
            "audited_repair_points": audited_repair_points,
            "future_backfill_ignored_points": future_backfill_ignored_points,
            "checkpoint_normalization_points": checkpoint_normalization_points,
            "formal_eff_ready": formal_eff_ready,
            "m_eff_min": min((row["m_eff"] for row in rows), default=0.0),
            "m_eff_max": max((row["m_eff"] for row in rows), default=0.0),
        },
        "checkpoint_normalization_details": checkpoint_normalization_details,
        "unresolved": unresolved,
    }
    return rows, report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    stage5_root = _stage5_root()
    parser = argparse.ArgumentParser(description="从冻结 clean 轨迹准备正式 EFF 运行级输入")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="仓库根目录，用于解析相对路径",
    )
    parser.add_argument(
        "--freeze-binding-report",
        type=Path,
        default=stage5_root / "reports/freeze_binding.json",
    )
    parser.add_argument(
        "--repair-manifest",
        type=Path,
        default=stage5_root / "manifests/trajectory_repairs.v1.json",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=stage5_root / "reports/eff_preparation_runs.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=stage5_root / "reports/eff_preparation_runs.csv",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=stage5_root / "reports/eff_preparation.json",
    )
    parser.add_argument("--expected-hosts", type=int, default=EXPECTED_HOSTS)
    parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)
    parser.add_argument("--expected-points", type=int, default=EXPECTED_POINTS)
    parser.add_argument("--expected-existing-points", type=int, default=EXPECTED_EXISTING_POINTS)
    parser.add_argument("--expected-missing-points", type=int, default=EXPECTED_MISSING_POINTS)
    parser.add_argument(
        "--expected-audited-repair-points",
        type=int,
        default=EXPECTED_AUDITED_REPAIR_POINTS,
    )
    parser.add_argument(
        "--expected-future-backfill-ignored-points",
        type=int,
        default=EXPECTED_FUTURE_BACKFILL_IGNORED_POINTS,
    )
    parser.add_argument(
        "--expected-checkpoint-normalization-points",
        type=int,
        default=EXPECTED_CHECKPOINT_NORMALIZATION_POINTS,
    )
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, report = build_eff_preparation(
            freeze_binding_report=args.freeze_binding_report,
            repair_manifest=args.repair_manifest,
            repo_root=args.repo_root,
            expected_hosts=args.expected_hosts,
            expected_tasks=args.expected_tasks,
            expected_points=args.expected_points,
            expected_existing_points=args.expected_existing_points,
            expected_missing_points=args.expected_missing_points,
            expected_audited_repair_points=args.expected_audited_repair_points,
            expected_future_backfill_ignored_points=args.expected_future_backfill_ignored_points,
            expected_checkpoint_normalization_points=args.expected_checkpoint_normalization_points,
            limit_runs=args.limit_runs,
        )
        output_jsonl = args.output_jsonl.resolve()
        output_csv = args.output_csv.resolve()
        output_report = args.output_report.resolve()
        _write_jsonl(output_jsonl, rows)
        _write_csv(output_csv, rows)
        report = {
            **report,
            "status": "ok",
            "contract_ok": len(report["unresolved"]) == 0,
            "outputs": {
                "eff_jsonl": str(output_jsonl),
                "eff_jsonl_sha256": sha256_file(output_jsonl),
                "eff_jsonl_row_count": len(rows),
                "eff_csv": str(output_csv),
                "eff_csv_sha256": sha256_file(output_csv),
                "eff_csv_row_count": len(rows),
            },
        }
        _write_json(output_report, report)
    except (
        EffPreparationContractError,
        FreezeBindingContractError,
        MetricContractError,
        TrajectoryContractError,
        TrajectoryRepairContractError,
    ) as exc:
        report = {
            "condition": NOISE_TAG,
            "horizon": HORIZON,
            "fatal_error": str(exc),
            "status": "error",
            "contract_ok": False,
            "summary": {
                "processed_run_count": 0,
                "success_count": 0,
                "unresolved_run_count": 1,
                "full_contract_checked": args.limit_runs is None,
                "limit_runs": args.limit_runs,
                "formal_eff_ready": False,
            },
            "unresolved": [{"logical_key": "__fatal__", "reason": str(exc)}],
        }
        _write_json(args.output_report.resolve(), report)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.print_summary:
            print(rendered)
        else:
            print(rendered)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.print_summary:
        print(rendered)
    else:
        print(rendered)
    if report["summary"]["unresolved_run_count"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
