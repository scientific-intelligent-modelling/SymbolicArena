from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class FreezeBindingContractError(ValueError):
    """inventory 与 freeze 冻结产物的绑定契约错误。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _parse_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise FreezeBindingContractError(f"{field_name} 不是合法整数: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FreezeBindingContractError(f"{field_name} 不是合法整数: {value!r}") from exc


def _logical_key(source: Mapping[str, Any]) -> str:
    return "::".join(
        (
            str(source.get("algorithm")),
            str(source.get("dataset_id")),
            f"s{_parse_int(source.get('seed'), field_name='source.seed')}",
            str(source.get("noise_tag")),
        )
    )


def _extract_host_from_name(path: Path, *, noise_tag: str, kind: str, suffix: str) -> str:
    prefix = f"{noise_tag}_{kind}_"
    name = path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise FreezeBindingContractError(f"无法从文件名解析 host: {path}")
    return name[len(prefix) : len(name) - len(suffix)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FreezeBindingContractError(f"{path} 顶层不是 JSON object")
    return payload


def _iter_jsonl_gz(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FreezeBindingContractError(
                    f"{path} 第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise FreezeBindingContractError(f"{path} 第 {line_number} 行顶层不是 object")
            yield line_number, payload


def _file_info(path: Path, *, noise_tag: str, kind: str, role: str) -> dict[str, Any]:
    raw = path.read_bytes()
    host = _extract_host_from_name(
        path,
        noise_tag=noise_tag,
        kind=kind,
        suffix=".jsonl.gz" if role == "records" else ".report.json",
    )
    return {
        "host": host,
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _source_without_sha(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(source)
    payload.pop("source_row_sha256", None)
    return payload


def _source_row_sha256(source: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(_source_without_sha(source)))


def _mapping_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for field in sorted(set(left) | set(right)):
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value != right_value:
            diffs.append(
                {
                    "field": field,
                    "inventory": left_value,
                    "freeze": right_value,
                }
            )
    return diffs


def _new_counts(host: str) -> dict[str, Any]:
    return {
        "host": host,
        "tasks": 0,
        "expected_points": 0,
        "existing_points": 0,
        "missing_points": 0,
        "result_ok": 0,
        "result_missing_or_invalid": 0,
    }


def _append_drift(
    drifts: list[dict[str, Any]],
    *,
    truncated: list[int],
    max_details: int,
    payload: dict[str, Any],
) -> None:
    if len(drifts) < max_details:
        drifts.append(payload)
    else:
        truncated[0] += 1


def _normalize_report(
    path: Path,
    *,
    kind: str,
    noise_tag: str,
    legacy_inventory_hosts: list[str],
) -> tuple[str, dict[str, Any]]:
    payload = _read_json(path)
    host = _extract_host_from_name(path, noise_tag=noise_tag, kind=kind, suffix=".report.json")
    if kind == "inventory":
        raw_verify_inner = payload.get("verify_inner")
        if "verify_inner" not in payload:
            payload["verify_inner"] = True
            payload["verify_inner_legacy_defaulted"] = True
            legacy_inventory_hosts.append(host)
        else:
            payload["verify_inner_legacy_defaulted"] = False
        payload["verify_inner_raw"] = raw_verify_inner
    return host, payload


def _compact_inventory_record(
    record: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    host: str,
    horizon: int,
    counts: dict[str, Any],
    missing_points: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise FreezeBindingContractError(f"{path} 第 {line_number} 行缺少 source")
    logical_key = _logical_key(source)
    if str(source.get("host")) != host:
        raise FreezeBindingContractError(
            f"{path} 第 {line_number} 行 host={source.get('host')!r} 与文件名 host={host!r} 不一致"
        )
    result = record.get("result")
    if not isinstance(result, Mapping):
        raise FreezeBindingContractError(f"{logical_key} 缺少 result")
    snapshots = record.get("snapshots")
    if not isinstance(snapshots, list):
        raise FreezeBindingContractError(f"{logical_key} 的 snapshots 不是列表")
    if len(snapshots) != horizon:
        raise FreezeBindingContractError(
            f"{logical_key} 的 snapshots 数量应为 {horizon}，实际为 {len(snapshots)}"
        )

    compact_snapshots: list[dict[str, Any]] = []
    counts["tasks"] += 1
    counts["expected_points"] += horizon
    if result.get("status") == "ok":
        counts["result_ok"] += 1
    else:
        counts["result_missing_or_invalid"] += 1

    for expected_minute, snapshot in enumerate(snapshots, start=1):
        if not isinstance(snapshot, Mapping):
            raise FreezeBindingContractError(
                f"{logical_key} minute_{expected_minute:04d} 不是 JSON object"
            )
        minute = _parse_int(
            snapshot.get("minute"),
            field_name=f"{logical_key} snapshot.minute",
        )
        if minute != expected_minute:
            raise FreezeBindingContractError(
                f"{logical_key} minute 序号不连续：期望 {expected_minute}，实际 {minute}"
            )
        status = str(snapshot.get("status"))
        selected_sha = snapshot.get("selected_sha256")
        compact_snapshots.append(
            {
                "minute": minute,
                "status": status,
                "selected_sha256": selected_sha,
            }
        )
        if status == "ok":
            counts["existing_points"] += 1
        elif status == "missing":
            counts["missing_points"] += 1
            missing_points.append(
                {
                    "logical_key": logical_key,
                    "host": host,
                    "task_id": str(source.get("task_id")),
                    "minute": minute,
                }
            )
        else:
            raise FreezeBindingContractError(
                f"{logical_key} minute_{minute:04d} 出现未支持 snapshot status: {status!r}"
            )

    source_sha_field = source.get("source_row_sha256")
    if not isinstance(source_sha_field, str) or not source_sha_field:
        raise FreezeBindingContractError(f"{logical_key} 缺少 source_row_sha256")
    return logical_key, {
        "source": dict(source),
        "source_row_sha256": source_sha_field,
        "source_row_sha256_recomputed": _source_row_sha256(source),
        "result_status": str(result.get("status")),
        "result_sha256": result.get("sha256"),
        "snapshots": compact_snapshots,
    }


def _validate_freeze_raw_text(
    *,
    logical_key: str,
    record_kind: str,
    minute: int | None,
    status: str,
    raw_text: object,
    expected_sha: object,
    drifts: list[dict[str, Any]],
    truncated: list[int],
    max_details: int,
) -> None:
    if status != "ok":
        return
    if not isinstance(raw_text, str):
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_details,
            payload={
                "type": f"{record_kind}_missing_raw_text",
                "logical_key": logical_key,
                "minute": minute,
            },
        )
        return
    raw_sha = _sha256_text(raw_text)
    if raw_sha != expected_sha:
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_details,
            payload={
                "type": f"{record_kind}_raw_sha_mismatch",
                "logical_key": logical_key,
                "minute": minute,
                "expected_sha256": expected_sha,
                "actual_raw_sha256": raw_sha,
            },
        )


def _compare_host_records(
    *,
    inventory_path: Path,
    freeze_path: Path,
    host: str,
    horizon: int,
    max_drift_details: int,
    drifts: list[dict[str, Any]],
    truncated: list[int],
    missing_points: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory_counts = _new_counts(host)
    freeze_counts = _new_counts(host)
    inventory_rows: dict[str, dict[str, Any]] = {}

    for line_number, record in _iter_jsonl_gz(inventory_path):
        logical_key, compact = _compact_inventory_record(
            record,
            path=inventory_path,
            line_number=line_number,
            host=host,
            horizon=horizon,
            counts=inventory_counts,
            missing_points=missing_points,
        )
        if logical_key in inventory_rows:
            raise FreezeBindingContractError(f"{inventory_path} 出现重复 logical_key: {logical_key}")
        inventory_rows[logical_key] = compact

    for line_number, record in _iter_jsonl_gz(freeze_path):
        logical_key, compact = _compact_inventory_record(
            record,
            path=freeze_path,
            line_number=line_number,
            host=host,
            horizon=horizon,
            counts=freeze_counts,
            missing_points=[],
        )
        inventory_compact = inventory_rows.pop(logical_key, None)
        if inventory_compact is None:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "unexpected_freeze_logical_key",
                    "host": host,
                    "logical_key": logical_key,
                },
            )
            continue

        source_diffs = _mapping_diff(inventory_compact["source"], compact["source"])
        if source_diffs:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "source_mismatch",
                    "host": host,
                    "logical_key": logical_key,
                    "diffs": source_diffs,
                },
            )
        if inventory_compact["source_row_sha256"] != compact["source_row_sha256"]:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "source_row_sha_mismatch",
                    "host": host,
                    "logical_key": logical_key,
                    "inventory": inventory_compact["source_row_sha256"],
                    "freeze": compact["source_row_sha256"],
                },
            )
        if inventory_compact["source_row_sha256_recomputed"] != inventory_compact["source_row_sha256"]:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "inventory_source_row_sha_invalid",
                    "host": host,
                    "logical_key": logical_key,
                    "expected": inventory_compact["source_row_sha256_recomputed"],
                    "actual": inventory_compact["source_row_sha256"],
                },
            )
        if compact["source_row_sha256_recomputed"] != compact["source_row_sha256"]:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "freeze_source_row_sha_invalid",
                    "host": host,
                    "logical_key": logical_key,
                    "expected": compact["source_row_sha256_recomputed"],
                    "actual": compact["source_row_sha256"],
                },
            )

        inventory_result = record.get("result")
        if not isinstance(inventory_result, Mapping):
            raise FreezeBindingContractError(f"{logical_key} freeze 结果缺少 result")
        if inventory_compact["result_status"] != compact["result_status"]:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "result_status_mismatch",
                    "host": host,
                    "logical_key": logical_key,
                    "inventory": inventory_compact["result_status"],
                    "freeze": compact["result_status"],
                },
            )
        if inventory_compact["result_sha256"] != compact["result_sha256"]:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "result_sha_mismatch",
                    "host": host,
                    "logical_key": logical_key,
                    "inventory": inventory_compact["result_sha256"],
                    "freeze": compact["result_sha256"],
                },
            )
        _validate_freeze_raw_text(
            logical_key=logical_key,
            record_kind="result",
            minute=None,
            status=compact["result_status"],
            raw_text=inventory_result.get("raw_text"),
            expected_sha=compact["result_sha256"],
            drifts=drifts,
            truncated=truncated,
            max_details=max_drift_details,
        )

        freeze_snapshots = record.get("snapshots")
        if not isinstance(freeze_snapshots, list):
            raise FreezeBindingContractError(f"{logical_key} freeze snapshots 不是列表")
        for inventory_snapshot, freeze_snapshot in zip(inventory_compact["snapshots"], compact["snapshots"]):
            if inventory_snapshot["minute"] != freeze_snapshot["minute"]:
                _append_drift(
                    drifts,
                    truncated=truncated,
                    max_details=max_drift_details,
                    payload={
                        "type": "snapshot_minute_mismatch",
                        "host": host,
                        "logical_key": logical_key,
                        "inventory": inventory_snapshot["minute"],
                        "freeze": freeze_snapshot["minute"],
                    },
                )
                continue
            minute = inventory_snapshot["minute"]
            if inventory_snapshot["status"] != freeze_snapshot["status"]:
                _append_drift(
                    drifts,
                    truncated=truncated,
                    max_details=max_drift_details,
                    payload={
                        "type": "snapshot_status_mismatch",
                        "host": host,
                        "logical_key": logical_key,
                        "minute": minute,
                        "inventory": inventory_snapshot["status"],
                        "freeze": freeze_snapshot["status"],
                    },
                )
                continue
            if inventory_snapshot["selected_sha256"] != freeze_snapshot["selected_sha256"]:
                _append_drift(
                    drifts,
                    truncated=truncated,
                    max_details=max_drift_details,
                    payload={
                        "type": "snapshot_selected_sha_mismatch",
                        "host": host,
                        "logical_key": logical_key,
                        "minute": minute,
                        "inventory": inventory_snapshot["selected_sha256"],
                        "freeze": freeze_snapshot["selected_sha256"],
                    },
                )
            freeze_raw_snapshot = freeze_snapshots[minute - 1]
            if not isinstance(freeze_raw_snapshot, Mapping):
                raise FreezeBindingContractError(
                    f"{logical_key} freeze minute_{minute:04d} 不是 JSON object"
                )
            _validate_freeze_raw_text(
                logical_key=logical_key,
                record_kind="snapshot",
                minute=minute,
                status=freeze_snapshot["status"],
                raw_text=freeze_raw_snapshot.get("raw_text"),
                expected_sha=freeze_snapshot["selected_sha256"],
                drifts=drifts,
                truncated=truncated,
                max_details=max_drift_details,
            )

    for logical_key in sorted(inventory_rows):
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={
                "type": "missing_freeze_logical_key",
                "host": host,
                "logical_key": logical_key,
            },
        )

    return inventory_counts, freeze_counts


def _collect_paths(directory: Path, *, noise_tag: str, kind: str, suffix: str) -> dict[str, Path]:
    matched = {}
    for path in sorted(directory.glob(f"{noise_tag}_{kind}_*{suffix}")):
        host = _extract_host_from_name(path, noise_tag=noise_tag, kind=kind, suffix=suffix)
        if host in matched:
            raise FreezeBindingContractError(f"{directory} 出现重复 host 文件: {path.name}")
        matched[host] = path
    return matched


def _check_report_contract(
    *,
    host: str,
    kind: str,
    report: Mapping[str, Any],
    counts: Mapping[str, Any],
    drifts: list[dict[str, Any]],
    truncated: list[int],
    max_drift_details: int,
) -> None:
    expected = {
        "tasks": counts["tasks"],
        "expected_snapshots": counts["expected_points"],
        "available_snapshots": counts["existing_points"],
        "missing_snapshots": counts["missing_points"],
        "conflicting_snapshots": 0,
        "parse_errors": 0,
        "result_missing_or_invalid": counts["result_missing_or_invalid"],
        "freeze_raw": kind == "freeze",
        "verify_inner": kind == "inventory",
    }
    for field, expected_value in expected.items():
        actual_value = report.get(field)
        if actual_value != expected_value:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "report_field_mismatch",
                    "host": host,
                    "kind": kind,
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                },
            )


def build_freeze_binding_summary(
    *,
    inventory_dir: Path,
    freeze_dir: Path,
    noise_tag: str = "clean",
    horizon: int = 180,
    max_drift_details: int = 200,
) -> dict[str, Any]:
    if horizon <= 0:
        raise FreezeBindingContractError("horizon 必须为正整数")
    inventory_records = _collect_paths(
        inventory_dir,
        noise_tag=noise_tag,
        kind="inventory",
        suffix=".jsonl.gz",
    )
    inventory_reports = _collect_paths(
        inventory_dir,
        noise_tag=noise_tag,
        kind="inventory",
        suffix=".report.json",
    )
    freeze_records = _collect_paths(
        freeze_dir,
        noise_tag=noise_tag,
        kind="freeze",
        suffix=".jsonl.gz",
    )
    freeze_reports = _collect_paths(
        freeze_dir,
        noise_tag=noise_tag,
        kind="freeze",
        suffix=".report.json",
    )
    if not inventory_records:
        raise FreezeBindingContractError("inventory 记录文件不能为空")
    if not freeze_records:
        raise FreezeBindingContractError("freeze 记录文件不能为空")

    drifts: list[dict[str, Any]] = []
    truncated = [0]
    legacy_inventory_hosts: list[str] = []
    per_host: dict[str, Any] = {}
    inventory_missing_points: list[dict[str, Any]] = []
    inventory_totals = _new_counts("inventory_total")
    freeze_totals = _new_counts("freeze_total")

    for host in sorted(set(inventory_reports) - set(inventory_records)):
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={"type": "inventory_report_without_records", "host": host},
        )
    for host in sorted(set(freeze_reports) - set(freeze_records)):
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={"type": "freeze_report_without_records", "host": host},
        )

    inventory_report_payloads = dict(
        _normalize_report(
            path,
            kind="inventory",
            noise_tag=noise_tag,
            legacy_inventory_hosts=legacy_inventory_hosts,
        )
        for path in inventory_reports.values()
    )
    freeze_report_payloads = dict(
        _normalize_report(
            path,
            kind="freeze",
            noise_tag=noise_tag,
            legacy_inventory_hosts=[],
        )
        for path in freeze_reports.values()
    )

    for host in sorted(set(inventory_records) | set(freeze_records)):
        inventory_path = inventory_records.get(host)
        freeze_path = freeze_records.get(host)
        if inventory_path is None or freeze_path is None:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={
                    "type": "host_record_set_mismatch",
                    "host": host,
                    "inventory_present": inventory_path is not None,
                    "freeze_present": freeze_path is not None,
                },
            )
            continue
        inventory_counts, freeze_counts = _compare_host_records(
            inventory_path=inventory_path,
            freeze_path=freeze_path,
            host=host,
            horizon=horizon,
            max_drift_details=max_drift_details,
            drifts=drifts,
            truncated=truncated,
            missing_points=inventory_missing_points,
        )
        for key in ("tasks", "expected_points", "existing_points", "missing_points", "result_ok", "result_missing_or_invalid"):
            inventory_totals[key] += inventory_counts[key]
            freeze_totals[key] += freeze_counts[key]

        inventory_report = inventory_report_payloads.get(host)
        freeze_report = freeze_report_payloads.get(host)
        if inventory_report is None:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={"type": "missing_inventory_report", "host": host},
            )
        else:
            _check_report_contract(
                host=host,
                kind="inventory",
                report=inventory_report,
                counts=inventory_counts,
                drifts=drifts,
                truncated=truncated,
                max_drift_details=max_drift_details,
            )
        if freeze_report is None:
            _append_drift(
                drifts,
                truncated=truncated,
                max_details=max_drift_details,
                payload={"type": "missing_freeze_report", "host": host},
            )
        else:
            _check_report_contract(
                host=host,
                kind="freeze",
                report=freeze_report,
                counts=freeze_counts,
                drifts=drifts,
                truncated=truncated,
                max_drift_details=max_drift_details,
            )

        per_host[host] = {
            "inventory": {
                "counts": inventory_counts,
                "records_file": _file_info(
                    inventory_path,
                    noise_tag=noise_tag,
                    kind="inventory",
                    role="records",
                ),
                "report_file": _file_info(
                    inventory_reports[host],
                    noise_tag=noise_tag,
                    kind="inventory",
                    role="report",
                )
                if host in inventory_reports
                else None,
                "report": inventory_report,
            },
            "freeze": {
                "counts": freeze_counts,
                "records_file": _file_info(
                    freeze_path,
                    noise_tag=noise_tag,
                    kind="freeze",
                    role="records",
                ),
                "report_file": _file_info(
                    freeze_reports[host],
                    noise_tag=noise_tag,
                    kind="freeze",
                    role="report",
                )
                if host in freeze_reports
                else None,
                "report": freeze_report,
            },
        }

    if inventory_totals["tasks"] != freeze_totals["tasks"]:
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={
                "type": "total_task_count_mismatch",
                "inventory": inventory_totals["tasks"],
                "freeze": freeze_totals["tasks"],
            },
        )
    if inventory_totals["expected_points"] != freeze_totals["expected_points"]:
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={
                "type": "total_expected_points_mismatch",
                "inventory": inventory_totals["expected_points"],
                "freeze": freeze_totals["expected_points"],
            },
        )
    if inventory_totals["existing_points"] != freeze_totals["existing_points"]:
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={
                "type": "total_existing_points_mismatch",
                "inventory": inventory_totals["existing_points"],
                "freeze": freeze_totals["existing_points"],
            },
        )
    if inventory_totals["missing_points"] != freeze_totals["missing_points"]:
        _append_drift(
            drifts,
            truncated=truncated,
            max_details=max_drift_details,
            payload={
                "type": "total_missing_points_mismatch",
                "inventory": inventory_totals["missing_points"],
                "freeze": freeze_totals["missing_points"],
            },
        )

    return {
        "noise_tag": noise_tag,
        "horizon": horizon,
        "inventory_counts": {
            "hosts": len(inventory_records),
            "tasks": inventory_totals["tasks"],
            "expected_points": inventory_totals["expected_points"],
            "existing_points": inventory_totals["existing_points"],
            "missing_points": inventory_totals["missing_points"],
        },
        "freeze_counts": {
            "hosts": len(freeze_records),
            "tasks": freeze_totals["tasks"],
            "expected_points": freeze_totals["expected_points"],
            "existing_points": freeze_totals["existing_points"],
            "missing_points": freeze_totals["missing_points"],
        },
        "missing_point_details": sorted(
            inventory_missing_points,
            key=lambda item: (
                str(item["logical_key"]),
                int(item["minute"]),
                str(item["host"]),
                str(item["task_id"]),
            ),
        ),
        "legacy_inventory_verify_inner_defaulted_hosts": sorted(set(legacy_inventory_hosts)),
        "input_files": {
            "inventory_records": [
                _file_info(path, noise_tag=noise_tag, kind="inventory", role="records")
                for path in sorted(inventory_records.values())
            ],
            "inventory_reports": [
                _file_info(path, noise_tag=noise_tag, kind="inventory", role="report")
                for path in sorted(inventory_reports.values())
            ],
            "freeze_records": [
                _file_info(path, noise_tag=noise_tag, kind="freeze", role="records")
                for path in sorted(freeze_records.values())
            ],
            "freeze_reports": [
                _file_info(path, noise_tag=noise_tag, kind="freeze", role="report")
                for path in sorted(freeze_reports.values())
            ],
        },
        "per_host": per_host,
        "binding": {
            "drift_count": len(drifts) + truncated[0],
            "drift_details": drifts,
            "truncated_drift_details": truncated[0],
        },
        "contract_ok": not drifts and truncated[0] == 0,
    }


def validate_freeze_binding_summary(
    summary: Mapping[str, Any],
    *,
    expected_hosts: int | None = 8,
    expected_tasks: int | None = 2250,
    expected_points: int | None = 405000,
    expected_existing_points: int | None = 404985,
    expected_missing_points: int | None = 15,
) -> None:
    issues: list[str] = []
    binding = summary.get("binding")
    inventory_counts = summary.get("inventory_counts")
    freeze_counts = summary.get("freeze_counts")
    if not isinstance(binding, Mapping) or not isinstance(inventory_counts, Mapping) or not isinstance(freeze_counts, Mapping):
        raise FreezeBindingContractError("summary 缺少 binding 或 counts")
    if int(binding.get("drift_count", 0)) != 0:
        issues.append(f"drift_count={binding.get('drift_count')}")
    for label, expected in (
        ("hosts", expected_hosts),
        ("tasks", expected_tasks),
        ("expected_points", expected_points),
        ("existing_points", expected_existing_points),
        ("missing_points", expected_missing_points),
    ):
        if expected is None:
            continue
        inventory_value = inventory_counts.get(label)
        freeze_value = freeze_counts.get(label)
        if inventory_value != expected:
            issues.append(f"inventory_{label}={inventory_value}")
        if freeze_value != expected:
            issues.append(f"freeze_{label}={freeze_value}")
    if summary.get("contract_ok") is not True:
        issues.append("contract_ok=false")
    if issues:
        raise FreezeBindingContractError("；".join(issues))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    stage5_root = _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description="校验 Stage5 inventory 与 freeze 的严格字节绑定")
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=stage5_root / "source_snapshot/trajectory_inventory",
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=stage5_root / "source_snapshot/trajectory_freeze",
    )
    parser.add_argument("--noise-tag", choices=("clean", "noise001", "noise005"), default="clean")
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=stage5_root / "reports/freeze_binding.json",
    )
    parser.add_argument("--expected-hosts", type=int, default=8)
    parser.add_argument("--expected-tasks", type=int, default=2250)
    parser.add_argument("--expected-points", type=int, default=405000)
    parser.add_argument("--expected-existing-points", type=int, default=404985)
    parser.add_argument("--expected-missing-points", type=int, default=15)
    parser.add_argument("--max-drift-details", type=int, default=200)
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_freeze_binding_summary(
            inventory_dir=args.inventory_dir,
            freeze_dir=args.freeze_dir,
            noise_tag=args.noise_tag,
            horizon=args.horizon,
            max_drift_details=args.max_drift_details,
        )
        validate_freeze_binding_summary(
            summary,
            expected_hosts=args.expected_hosts,
            expected_tasks=args.expected_tasks,
            expected_points=args.expected_points,
            expected_existing_points=args.expected_existing_points,
            expected_missing_points=args.expected_missing_points,
        )
    except FreezeBindingContractError as exc:
        summary = {
            "contract_ok": False,
            "binding": {
                "drift_count": 1,
                "drift_details": [{"type": "fatal_error", "message": str(exc)}],
                "truncated_drift_details": 0,
            },
            "fatal_error": str(exc),
        }
        exit_code = 1
    else:
        exit_code = 0

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.print_summary:
        print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
