from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class TrajectoryCoverageContractError(ValueError):
    """轨迹覆盖率输入不满足 Stage5 契约。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TrajectoryCoverageContractError(f"{field_name} 不是合法整数: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryCoverageContractError(f"{field_name} 不是合法整数: {value!r}") from exc


def _finite_nonnegative(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _approx_equal(left: float | None, right: float | None, *, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= (atol + rtol * abs(right))


def _logical_key(algorithm: str, dataset_id: str, seed: int, noise_tag: str) -> str:
    return f"{algorithm}::{dataset_id}::s{seed}::{noise_tag}"


def _stable_task_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(item["algorithm"]),
        str(item["dataset_id"]),
        int(item["seed"]),
        str(item["task_id"]),
        str(item["host"]),
    )


def _stable_point_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(item["logical_key"]),
        int(item["minute"]),
        str(item["host"]),
        str(item["task_id"]),
    )


def _load_source_rows(source_runs_csv: Path, *, noise_tag: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("noise_tag") != noise_tag:
                continue
            seed = _parse_int(row.get("seed"), field_name="seed")
            logical_key = row.get("logical_key") or _logical_key(
                str(row["algorithm"]),
                str(row["dataset_id"]),
                seed,
                str(row["noise_tag"]),
            )
            normalized = {
                "logical_key": logical_key,
                "batch": str(row["batch"]),
                "algorithm": str(row["algorithm"]),
                "dataset_id": str(row["dataset_id"]),
                "seed": seed,
                "noise_tag": str(row["noise_tag"]),
                "task_id": str(row["task_id"]),
                "host": str(row["host"]),
                "status": str(row["status"]),
                "path": str(row["path"]),
                "id_nmse": float(row["id_nmse"]),
                "ood_nmse": float(row["ood_nmse"]),
            }
            if logical_key in rows:
                raise TrajectoryCoverageContractError(f"source manifest 出现重复 logical_key: {logical_key}")
            rows[logical_key] = normalized
    return rows


def _extract_host_from_path(path: Path, *, noise_tag: str, suffix: str) -> str:
    prefix = f"{noise_tag}_inventory_"
    name = path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise TrajectoryCoverageContractError(f"无法从文件名解析 host: {path}")
    return name[len(prefix) : len(name) - len(suffix)]


def _iter_inventory_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrajectoryCoverageContractError(
                    f"{path} 第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise TrajectoryCoverageContractError(f"{path} 第 {line_number} 行顶层不是 object")
            yield payload


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrajectoryCoverageContractError(f"report 顶层不是 object: {path}")
    return payload


def _point_detail(
    *,
    logical_key: str,
    minute: int,
    source: Mapping[str, Any],
    classification: str,
    inventory_path: Path,
    snapshot: Mapping[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    detail = {
        "logical_key": logical_key,
        "algorithm": str(source["algorithm"]),
        "dataset_id": str(source["dataset_id"]),
        "seed": int(source["seed"]),
        "noise_tag": str(source["noise_tag"]),
        "task_id": str(source["task_id"]),
        "host": str(source["host"]),
        "minute": minute,
        "classification": classification,
        "inventory_file": inventory_path.name,
        "record_type": snapshot.get("record_type"),
        "payload_status": snapshot.get("payload_status"),
        "backfilled_from_minute": snapshot.get("backfilled_from_minute"),
    }
    if note is not None:
        detail["note"] = note
    return detail


def _classify_snapshot(
    snapshot: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    logical_key: str,
    inventory_path: Path,
    future_backfill_points: list[dict[str, Any]],
) -> str:
    minute = _parse_int(snapshot.get("minute"), field_name="minute")
    status = snapshot.get("status")
    if status == "missing":
        return "missing_file"
    if status != "ok":
        raise TrajectoryCoverageContractError(
            f"{logical_key} minute_{minute:04d} 出现未支持 snapshot status: {status!r}"
        )

    record_type = snapshot.get("record_type")
    if record_type == "periodic_backfill":
        source_minute = _parse_int(
            snapshot.get("backfilled_from_minute"),
            field_name=f"{logical_key} minute_{minute:04d} 的 backfilled_from_minute",
        )
        if source_minute > minute:
            future_backfill_points.append(
                _point_detail(
                    logical_key=logical_key,
                    minute=minute,
                    source=source,
                    classification="future_backfill",
                    inventory_path=inventory_path,
                    snapshot=snapshot,
                    note=f"backfilled_from_minute={source_minute} 大于当前 minute={minute}",
                )
            )
            return "future_backfill_ignored"

    has_expression = bool(snapshot.get("has_expression"))
    id_nmse = _finite_nonnegative(snapshot.get("id_nmse"))
    ood_nmse = _finite_nonnegative(snapshot.get("ood_nmse"))
    payload_status = snapshot.get("payload_status")

    if id_nmse is not None or ood_nmse is not None:
        if not has_expression:
            raise TrajectoryCoverageContractError(
                f"{logical_key} minute_{minute:04d} 有数值指标但没有表达式"
            )
        if id_nmse is None or ood_nmse is None:
            return "evaluator_error"
        return "direct_valid"

    if payload_status == "error" or has_expression:
        return "evaluator_error"
    return "explicit_no_output"


def _new_host_summary(host: str) -> dict[str, Any]:
    return {
        "host": host,
        "tasks": 0,
        "expected_points": 0,
        "available_points": 0,
        "point_classification_counts": {
            "direct_valid": 0,
            "evaluator_error": 0,
            "explicit_no_output": 0,
            "future_backfill_ignored": 0,
            "missing_file": 0,
        },
    }


def build_trajectory_coverage_summary(
    *,
    source_runs_csv: Path,
    inventory_paths: Iterable[Path],
    report_paths: Iterable[Path],
    noise_tag: str = "clean",
    horizon: int = 180,
) -> dict[str, Any]:
    if horizon <= 0:
        raise TrajectoryCoverageContractError("horizon 必须为正整数")

    inventory_list = sorted(Path(path) for path in inventory_paths)
    report_list = sorted(Path(path) for path in report_paths)
    if not inventory_list:
        raise TrajectoryCoverageContractError("inventory_paths 不能为空")
    if not report_list:
        raise TrajectoryCoverageContractError("report_paths 不能为空")

    source_rows = _load_source_rows(source_runs_csv, noise_tag=noise_tag)
    source_keys = set(source_rows)

    per_host: dict[str, dict[str, Any]] = {}
    report_by_host = {
        _extract_host_from_path(path, noise_tag=noise_tag, suffix=".report.json"): _load_report(path)
        for path in report_list
    }
    seen_inventory_hosts: set[str] = set()
    seen_tasks: dict[str, dict[str, Any]] = {}
    counts = Counter()
    missing_points: list[dict[str, Any]] = []
    future_backfill_points: list[dict[str, Any]] = []
    report_mismatches: list[dict[str, Any]] = []
    inventory_identity_mismatches: list[dict[str, Any]] = []
    final_dataset_mismatches: list[dict[str, Any]] = []
    final_seed_mismatches: list[dict[str, Any]] = []
    final_id_nmse_mismatches: list[dict[str, Any]] = []
    final_ood_nmse_mismatches: list[dict[str, Any]] = []

    for inventory_path in inventory_list:
        expected_host = _extract_host_from_path(inventory_path, noise_tag=noise_tag, suffix=".jsonl.gz")
        file_tasks = 0
        file_host: str | None = None
        for record in _iter_inventory_records(inventory_path):
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise TrajectoryCoverageContractError(f"{inventory_path} 存在缺失 source 的记录")
            algorithm = str(source.get("algorithm"))
            dataset_id = str(source.get("dataset_id"))
            seed = _parse_int(source.get("seed"), field_name=f"{inventory_path} source.seed")
            source_noise_tag = str(source.get("noise_tag"))
            task_id = str(source.get("task_id"))
            host = str(source.get("host"))
            logical_key = _logical_key(algorithm, dataset_id, seed, source_noise_tag)

            if source_noise_tag != noise_tag:
                raise TrajectoryCoverageContractError(
                    f"{logical_key} 的 noise_tag={source_noise_tag!r} 与请求 {noise_tag!r} 不一致"
                )
            if file_host is None:
                file_host = host
            elif file_host != host:
                raise TrajectoryCoverageContractError(
                    f"{inventory_path} 混入多个 host: {file_host!r} 与 {host!r}"
                )
            if host != expected_host:
                raise TrajectoryCoverageContractError(
                    f"{inventory_path} 文件名 host={expected_host!r} 与记录 host={host!r} 不一致"
                )
            if logical_key in seen_tasks:
                raise TrajectoryCoverageContractError(f"inventory 出现重复 logical_key: {logical_key}")

            manifest_row = source_rows.get(logical_key)
            if manifest_row is None:
                inventory_identity_mismatches.append(
                    {
                        "logical_key": logical_key,
                        "reason": "unexpected_task_key",
                        "inventory_file": inventory_path.name,
                    }
                )
            else:
                if manifest_row["dataset_id"] != dataset_id:
                    final_dataset_mismatches.append(
                        {
                            "logical_key": logical_key,
                            "expected_dataset_id": manifest_row["dataset_id"],
                            "actual_dataset_id": dataset_id,
                        }
                    )
                if manifest_row["seed"] != seed:
                    final_seed_mismatches.append(
                        {
                            "logical_key": logical_key,
                            "expected_seed": manifest_row["seed"],
                            "actual_seed": seed,
                        }
                    )
                for field in ("algorithm", "noise_tag", "task_id", "host", "batch", "path"):
                    expected_value = manifest_row[field]
                    actual_value = str(source.get(field))
                    if expected_value != actual_value:
                        inventory_identity_mismatches.append(
                            {
                                "logical_key": logical_key,
                                "field": field,
                                "expected": expected_value,
                                "actual": actual_value,
                            }
                        )
                result_summary = record.get("result", {}).get("payload_summary", {})
                if not isinstance(result_summary, Mapping):
                    raise TrajectoryCoverageContractError(f"{logical_key} 缺失 result.payload_summary")
                actual_id = _finite_nonnegative(result_summary.get("id_nmse"))
                actual_ood = _finite_nonnegative(result_summary.get("ood_nmse"))
                if not _approx_equal(manifest_row["id_nmse"], actual_id):
                    final_id_nmse_mismatches.append(
                        {
                            "logical_key": logical_key,
                            "expected_id_nmse": manifest_row["id_nmse"],
                            "actual_id_nmse": actual_id,
                        }
                    )
                if not _approx_equal(manifest_row["ood_nmse"], actual_ood):
                    final_ood_nmse_mismatches.append(
                        {
                            "logical_key": logical_key,
                            "expected_ood_nmse": manifest_row["ood_nmse"],
                            "actual_ood_nmse": actual_ood,
                        }
                    )

            snapshots = record.get("snapshots")
            if not isinstance(snapshots, list):
                raise TrajectoryCoverageContractError(f"{logical_key} 的 snapshots 不是列表")
            if len(snapshots) != horizon:
                raise TrajectoryCoverageContractError(
                    f"{logical_key} 的 snapshots 数量应为 {horizon}，实际为 {len(snapshots)}"
                )

            host_summary = per_host.setdefault(host, _new_host_summary(host))
            host_summary["tasks"] += 1
            host_summary["expected_points"] += horizon
            file_tasks += 1
            seen_tasks[logical_key] = {
                "logical_key": logical_key,
                "algorithm": algorithm,
                "dataset_id": dataset_id,
                "seed": seed,
                "noise_tag": source_noise_tag,
                "task_id": task_id,
                "host": host,
            }
            for snapshot in snapshots:
                classification = _classify_snapshot(
                    snapshot,
                    source=seen_tasks[logical_key],
                    logical_key=logical_key,
                    inventory_path=inventory_path,
                    future_backfill_points=future_backfill_points,
                )
                counts[classification] += 1
                host_summary["point_classification_counts"][classification] += 1
                if classification != "missing_file":
                    host_summary["available_points"] += 1
                else:
                    minute = _parse_int(snapshot.get("minute"), field_name=f"{logical_key} missing minute")
                    missing_points.append(
                        _point_detail(
                            logical_key=logical_key,
                            minute=minute,
                            source=seen_tasks[logical_key],
                            classification="missing_file",
                            inventory_path=inventory_path,
                            snapshot=snapshot,
                        )
                    )
        if file_host is None:
            raise TrajectoryCoverageContractError(f"{inventory_path} 不包含任何记录")
        seen_inventory_hosts.add(file_host)
        report = report_by_host.get(file_host)
        if report is None:
            report_mismatches.append(
                {
                    "host": file_host,
                    "reason": "missing_report",
                    "inventory_file": inventory_path.name,
                }
            )
            continue
        file_summary = per_host[file_host]
        expected_report = {
            "tasks": file_summary["tasks"],
            "expected_snapshots": file_summary["expected_points"],
            "available_snapshots": file_summary["available_points"],
            "missing_snapshots": file_summary["point_classification_counts"]["missing_file"],
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
        }
        for field, expected_value in expected_report.items():
            actual_value = report.get(field)
            if actual_value != expected_value:
                report_mismatches.append(
                    {
                        "host": file_host,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                        "report_file": f"{noise_tag}_inventory_{file_host}.report.json",
                    }
                )
        file_summary["report"] = report

    unexpected_report_hosts = sorted(set(report_by_host) - seen_inventory_hosts)
    for host in unexpected_report_hosts:
        report_mismatches.append(
            {
                "host": host,
                "reason": "report_without_inventory",
                "report_file": f"{noise_tag}_inventory_{host}.report.json",
            }
        )

    missing_task_keys = sorted(source_keys - set(seen_tasks))
    unexpected_task_keys = sorted(set(seen_tasks) - source_keys)
    for logical_key in unexpected_task_keys:
        inventory_identity_mismatches.append(
            {
                "logical_key": logical_key,
                "reason": "unexpected_task_key",
            }
        )

    available_points = (
        counts["direct_valid"]
        + counts["evaluator_error"]
        + counts["explicit_no_output"]
        + counts["future_backfill_ignored"]
    )
    summary = {
        "noise_tag": noise_tag,
        "horizon": horizon,
        "source_manifest": {
            "source_runs_csv": str(source_runs_csv),
            "task_count": len(source_rows),
            "unique_task_keys": len(source_rows),
            "expected_points": len(source_rows) * horizon,
        },
        "inventory": {
            "inventory_files": [path.name for path in inventory_list],
            "report_files": [path.name for path in report_list],
            "task_count": len(seen_tasks),
            "unique_task_keys": len(seen_tasks),
            "expected_points": len(source_rows) * horizon,
            "available_points": available_points,
            "missing_points": counts["missing_file"],
            "point_classification_counts": {
                "direct_valid": counts["direct_valid"],
                "evaluator_error": counts["evaluator_error"],
                "explicit_no_output": counts["explicit_no_output"],
                "future_backfill_ignored": counts["future_backfill_ignored"],
                "missing_file": counts["missing_file"],
            },
            "missing_point_details": sorted(missing_points, key=_stable_point_sort_key),
            "future_backfill_points": sorted(future_backfill_points, key=_stable_point_sort_key),
        },
        "per_host": {
            host: {
                "tasks": payload["tasks"],
                "expected_points": payload["expected_points"],
                "available_points": payload["available_points"],
                "point_classification_counts": dict(payload["point_classification_counts"]),
                "report": payload.get("report"),
            }
            for host, payload in sorted(per_host.items())
        },
        "final_result_audit": {
            "missing_task_keys": missing_task_keys,
            "unexpected_task_keys": unexpected_task_keys,
            "inventory_identity_mismatches": inventory_identity_mismatches,
            "dataset_mismatches": final_dataset_mismatches,
            "seed_mismatches": final_seed_mismatches,
            "id_nmse_mismatches": final_id_nmse_mismatches,
            "ood_nmse_mismatches": final_ood_nmse_mismatches,
        },
        "report_mismatches": report_mismatches,
    }
    identity_issues = (
        summary["final_result_audit"]["missing_task_keys"],
        summary["final_result_audit"]["unexpected_task_keys"],
        summary["final_result_audit"]["inventory_identity_mismatches"],
        summary["final_result_audit"]["dataset_mismatches"],
        summary["final_result_audit"]["seed_mismatches"],
        summary["final_result_audit"]["id_nmse_mismatches"],
        summary["final_result_audit"]["ood_nmse_mismatches"],
        summary["report_mismatches"],
    )
    identity_contract_ok = not any(identity_issues)
    summary["readiness"] = {
        "identity_contract_ok": identity_contract_ok,
        "formal_eff_ready": identity_contract_ok and counts["missing_file"] == 0,
        "blocking_missing_points": counts["missing_file"],
        "future_backfill_ignored_points": counts["future_backfill_ignored"],
    }
    summary["contract_ok"] = identity_contract_ok
    return summary


def validate_trajectory_coverage_summary(
    summary: Mapping[str, Any],
    *,
    reject_future_backfill: bool = False,
) -> None:
    inventory = summary.get("inventory")
    final_audit = summary.get("final_result_audit")
    if not isinstance(inventory, Mapping) or not isinstance(final_audit, Mapping):
        raise TrajectoryCoverageContractError("summary 缺少 inventory 或 final_result_audit")

    issues: list[str] = []
    if reject_future_backfill and inventory.get("future_backfill_points"):
        issues.append(
            f"发现 {len(inventory['future_backfill_points'])} 个 future backfill 点"
        )
    for field in (
        "missing_task_keys",
        "unexpected_task_keys",
        "inventory_identity_mismatches",
        "dataset_mismatches",
        "seed_mismatches",
        "id_nmse_mismatches",
        "ood_nmse_mismatches",
    ):
        values = final_audit.get(field)
        if isinstance(values, list) and values:
            issues.append(f"{field}={len(values)}")
    report_mismatches = summary.get("report_mismatches")
    if isinstance(report_mismatches, list) and report_mismatches:
        issues.append(f"report_mismatches={len(report_mismatches)}")
    if issues:
        raise TrajectoryCoverageContractError("；".join(issues))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    stage5_root = repo_root / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description="汇总 Stage5 分钟轨迹覆盖率")
    parser.add_argument(
        "--source-runs-csv",
        type=Path,
        default=stage5_root / "manifests/source_runs.csv",
    )
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=stage5_root / "source_snapshot/trajectory_inventory",
    )
    parser.add_argument("--noise-tag", choices=("clean", "noise001", "noise005"), default="clean")
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=stage5_root / "reports/trajectory_coverage.json",
    )
    parser.add_argument("--reject-future-backfill", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory_paths = sorted(args.inventory_dir.glob(f"{args.noise_tag}_inventory_*.jsonl.gz"))
    report_paths = sorted(args.inventory_dir.glob(f"{args.noise_tag}_inventory_*.report.json"))
    summary = build_trajectory_coverage_summary(
        source_runs_csv=args.source_runs_csv.resolve(),
        inventory_paths=inventory_paths,
        report_paths=report_paths,
        noise_tag=args.noise_tag,
        horizon=args.horizon,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.print_summary:
        print(rendered)
    try:
        validate_trajectory_coverage_summary(
            summary,
            reject_future_backfill=args.reject_future_backfill,
        )
    except TrajectoryCoverageContractError as exc:
        print(f"trajectory coverage contract violation: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
