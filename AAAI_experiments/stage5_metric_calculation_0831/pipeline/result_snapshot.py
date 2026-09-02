"""Stage5 result.json 快照清单准备与本地镜像验收。"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence


SUPPORTED_CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = frozenset({520, 521, 522})
TASK_ID_RE = re.compile(
    r"^(?P<slug>[a-z0-9]+)_s(?P<seed>520|521|522)_"
    r"(?P<condition>clean|noise001|noise005)_g(?P<dataset_index>\d{4})$"
)
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "batch",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "status",
        "id_nmse",
        "ood_nmse",
        "path",
        "logical_key",
    }
)
INDEX_FIELDS = (
    "logical_key",
    "batch",
    "noise_tag",
    "algorithm",
    "dataset_id",
    "seed",
    "task_id",
    "host",
    "remote_result_path",
    "local_result_path",
    "local_sha256",
    "status",
    "equation_present",
    "canonical_artifact_present",
    "id_nmse_source",
    "id_nmse_result",
    "ood_nmse_source",
    "ood_nmse_result",
    "metrics_match",
)


class ResultSnapshotError(ValueError):
    """源清单、镜像文件或既有输出不符合冻结契约。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_nmse(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ResultSnapshotError(f"{context} 必须为有限非负 NMSE")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultSnapshotError(f"{context} 必须为有限非负 NMSE") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ResultSnapshotError(f"{context} 必须为有限非负 NMSE")
    return parsed


def _same_nmse(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _safe_component(value: str, *, context: str) -> str:
    if not value or SAFE_COMPONENT_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ResultSnapshotError(f"{context} 不是安全路径组件: {value!r}")
    return value


def _stripped_remote_path(value: str, *, context: str) -> str:
    if not value.startswith("/") or value.startswith("//") or "\x00" in value or "\n" in value:
        raise ResultSnapshotError(f"{context} 必须是规范远端绝对路径")
    path = PurePosixPath(value)
    if any(part in {".", ".."} for part in path.parts):
        raise ResultSnapshotError(f"{context} 含不安全路径片段")
    stripped = value[1:]
    if not stripped or str(PurePosixPath("/") / stripped) != value:
        raise ResultSnapshotError(f"{context} 不是规范远端绝对路径")
    return stripped


def _read_source_rows(
    source_runs_csv: Path, *, condition: str, expected_count: int | None
) -> list[dict[str, str]]:
    if condition not in SUPPORTED_CONDITIONS:
        raise ResultSnapshotError(f"未知 condition: {condition!r}")
    with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_SOURCE_FIELDS.difference(reader.fieldnames or ()))
        if missing:
            raise ResultSnapshotError(f"source_runs.csv 缺少字段: {missing}")
        rows = [dict(row) for row in reader if row.get("noise_tag") == condition]
    if expected_count is not None and len(rows) != expected_count:
        raise ResultSnapshotError(
            f"{condition} source_runs 行数不符: 期望 {expected_count}，实际 {len(rows)}"
        )
    seen_logical: set[str] = set()
    seen_tasks: set[str] = set()
    seen_remote: set[tuple[str, str]] = set()
    for row in rows:
        task_id = row["task_id"]
        match = TASK_ID_RE.fullmatch(task_id)
        if match is None or match.group("condition") != condition:
            raise ResultSnapshotError(f"task_id 与 condition 不一致: {task_id!r}")
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise ResultSnapshotError(f"{task_id}: seed 非法") from exc
        if seed not in SEEDS or seed != int(match.group("seed")):
            raise ResultSnapshotError(f"{task_id}: seed 与 task_id 不一致")
        _safe_component(row["host"], context=f"{task_id}.host")
        _safe_component(row["algorithm"], context=f"{task_id}.algorithm")
        stripped = _stripped_remote_path(row["path"], context=f"{task_id}.path")
        expected_key = f"{row['algorithm']}::{row['dataset_id']}::s{seed}::{condition}"
        if row["logical_key"] != expected_key:
            raise ResultSnapshotError(f"{task_id}: logical_key 非 canonical")
        if row["logical_key"] in seen_logical:
            raise ResultSnapshotError(f"source_runs 出现重复 logical_key: {row['logical_key']}")
        if task_id in seen_tasks:
            raise ResultSnapshotError(f"source_runs 出现重复 task_id: {task_id}")
        remote_key = (row["host"], stripped)
        if remote_key in seen_remote:
            raise ResultSnapshotError(f"source_runs 出现重复远端路径: {remote_key}")
        seen_logical.add(row["logical_key"])
        seen_tasks.add(task_id)
        seen_remote.add(remote_key)
        _parse_nmse(row["id_nmse"], context=f"{task_id}.id_nmse")
        _parse_nmse(row["ood_nmse"], context=f"{task_id}.ood_nmse")
        if not row["status"]:
            raise ResultSnapshotError(f"{task_id}.status 为空")
    return sorted(rows, key=lambda item: item["logical_key"])


def _write_bytes_without_drift(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ResultSnapshotError(f"拒绝覆盖漂移文件: {path}")
        return
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        tmp_path = Path(handle.name)
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != data:
            raise ResultSnapshotError(f"并发生成了漂移文件: {path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def prepare_rsync_lists(
    *,
    source_runs_csv: Path,
    condition: str,
    output_root: Path,
    expected_count: int | None = 2250,
) -> dict[str, Any]:
    """生成供 `rsync --relative --files-from` 使用的按主机相对路径清单。"""
    source_runs_csv = source_runs_csv.resolve()
    rows = _read_source_rows(
        source_runs_csv, condition=condition, expected_count=expected_count
    )
    by_host: dict[str, list[str]] = {}
    for row in rows:
        by_host.setdefault(row["host"], []).append(
            _stripped_remote_path(row["path"], context=f"{row['task_id']}.path")
        )
    condition_dir = (output_root.resolve() / condition)
    for host, paths in sorted(by_host.items()):
        payload = "".join(f"{path}\n" for path in sorted(paths)).encode("utf-8")
        _write_bytes_without_drift(condition_dir / f"{host}.txt", payload)
    existing_hosts = {
        path.stem for path in condition_dir.glob("*.txt") if path.is_file()
    } if condition_dir.exists() else set()
    expected_hosts = set(by_host)
    if existing_hosts != expected_hosts:
        raise ResultSnapshotError(
            f"rsync list 目录含非本次主机文件，拒绝隐式删除: {sorted(existing_hosts - expected_hosts)}"
        )
    return {
        "phase": "prepare",
        "condition": condition,
        "source_runs_csv": str(source_runs_csv),
        "source_runs_sha256": _sha256_file(source_runs_csv),
        "output_dir": str(condition_dir),
        "row_count": len(rows),
        "host_count": len(by_host),
        "host_counts": {host: len(paths) for host, paths in sorted(by_host.items())},
        "status": "ok",
    }


def _load_result(path: Path, *, row: Mapping[str, str]) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    try:
        payload_raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultSnapshotError(f"{row['task_id']}: result.json 解析失败") from exc
    if not isinstance(payload_raw, Mapping):
        raise ResultSnapshotError(f"{row['task_id']}: result.json 根必须是对象")
    payload = dict(payload_raw)
    if payload.get("status") != row["status"]:
        raise ResultSnapshotError(f"{row['task_id']}: result status 与 source 不一致")
    equation = payload.get("equation")
    if equation is not None and not isinstance(equation, str):
        raise ResultSnapshotError(f"{row['task_id']}: equation 必须为字符串或 null")
    canonical = payload.get("canonical_artifact")
    if canonical is not None and not isinstance(canonical, Mapping):
        raise ResultSnapshotError(f"{row['task_id']}: canonical_artifact 必须为对象或 null")
    if isinstance(canonical, Mapping):
        for field in (
            "instantiated_expression", "normalized_expression", "return_expression_source"
        ):
            value = canonical.get(field)
            if value is not None and not isinstance(value, str):
                raise ResultSnapshotError(
                    f"{row['task_id']}: canonical_artifact.{field} 类型非法"
                )
    source_id = _parse_nmse(row["id_nmse"], context=f"{row['task_id']}.source.id_nmse")
    source_ood = _parse_nmse(row["ood_nmse"], context=f"{row['task_id']}.source.ood_nmse")
    try:
        id_block = payload["id_test"]
        ood_block = payload["ood_test"]
    except KeyError as exc:
        raise ResultSnapshotError(f"{row['task_id']}: result 缺少数值测试块") from exc
    if not isinstance(id_block, Mapping) or not isinstance(ood_block, Mapping):
        raise ResultSnapshotError(f"{row['task_id']}: 数值测试块类型非法")
    result_id = _parse_nmse(id_block.get("nmse"), context=f"{row['task_id']}.id_test.nmse")
    result_ood = _parse_nmse(ood_block.get("nmse"), context=f"{row['task_id']}.ood_test.nmse")
    if not _same_nmse(source_id, result_id) or not _same_nmse(source_ood, result_ood):
        raise ResultSnapshotError(f"{row['task_id']}: ID/OOD NMSE 与 source 不一致")
    return payload, _sha256_bytes(raw_bytes)


def _materialize_without_drift(source: Path, destination: Path, *, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != expected_sha256:
            raise ResultSnapshotError(f"拒绝覆盖漂移文件: {destination}")
        return "existing_identical"
    with NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        tmp_path = Path(handle.name)
    tmp_path.unlink()
    method = "hardlink"
    try:
        try:
            os.link(source, tmp_path)
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
                raise
            shutil.copy2(source, tmp_path)
            method = "copy"
        if _sha256_file(tmp_path) != expected_sha256:
            raise ResultSnapshotError(f"临时物化文件 SHA 漂移: {destination}")
        try:
            os.link(tmp_path, destination)
        except FileExistsError:
            if not destination.is_file() or _sha256_file(destination) != expected_sha256:
                raise ResultSnapshotError(f"并发生成了漂移文件: {destination}")
            method = "existing_identical"
    finally:
        tmp_path.unlink(missing_ok=True)
    return method


def _index_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def finalize_snapshot(
    *,
    source_runs_csv: Path,
    condition: str,
    mirror_root: Path,
    snapshot_root: Path,
    index_csv: Path,
    expected_count: int | None = 2250,
) -> dict[str, Any]:
    """验收本地 rsync mirror，并安全物化 result 快照与兼容索引。"""
    source_runs_csv = source_runs_csv.resolve()
    mirror_root = mirror_root.resolve()
    snapshot_root = snapshot_root.resolve()
    rows = _read_source_rows(
        source_runs_csv, condition=condition, expected_count=expected_count
    )
    index_rows: list[dict[str, Any]] = []
    methods: dict[str, int] = {}
    for row in rows:
        stripped = _stripped_remote_path(row["path"], context=f"{row['task_id']}.path")
        mirror_path = mirror_root / condition / row["host"] / stripped
        if not mirror_path.is_file():
            raise ResultSnapshotError(f"镜像 result.json 不存在: {mirror_path}")
        payload, source_sha = _load_result(mirror_path, row=row)
        destination = (
            snapshot_root / "results" / condition / row["algorithm"] / row["task_id"] / "result.json"
        )
        method = _materialize_without_drift(
            mirror_path, destination, expected_sha256=source_sha
        )
        methods[method] = methods.get(method, 0) + 1
        local_sha = _sha256_file(destination)
        if local_sha != source_sha:
            raise ResultSnapshotError(f"{row['task_id']}: 物化后 SHA 漂移")
        id_nmse = _parse_nmse(row["id_nmse"], context=f"{row['task_id']}.id_nmse")
        ood_nmse = _parse_nmse(row["ood_nmse"], context=f"{row['task_id']}.ood_nmse")
        result_id = float(payload["id_test"]["nmse"])
        result_ood = float(payload["ood_test"]["nmse"])
        equation = payload.get("equation")
        canonical = payload.get("canonical_artifact")
        index_rows.append(
            {
                "logical_key": row["logical_key"],
                "batch": row["batch"],
                "noise_tag": condition,
                "algorithm": row["algorithm"],
                "dataset_id": row["dataset_id"],
                "seed": int(row["seed"]),
                "task_id": row["task_id"],
                "host": row["host"],
                "remote_result_path": row["path"],
                "local_result_path": str(destination),
                "local_sha256": local_sha,
                "status": payload["status"],
                "equation_present": bool(isinstance(equation, str) and equation.strip()),
                "canonical_artifact_present": isinstance(canonical, Mapping),
                "id_nmse_source": format(id_nmse, ".17g"),
                "id_nmse_result": format(result_id, ".17g"),
                "ood_nmse_source": format(ood_nmse, ".17g"),
                "ood_nmse_result": format(result_ood, ".17g"),
                "metrics_match": True,
            }
        )
    index_payload = _index_bytes(index_rows)
    _write_bytes_without_drift(index_csv.resolve(), index_payload)
    return {
        "phase": "finalize",
        "condition": condition,
        "source_runs_csv": str(source_runs_csv),
        "source_runs_sha256": _sha256_file(source_runs_csv),
        "mirror_root": str(mirror_root),
        "snapshot_root": str(snapshot_root),
        "index_csv": str(index_csv.resolve()),
        "index_sha256": _sha256_bytes(index_payload),
        "row_count": len(index_rows),
        "materialization_counts": dict(sorted(methods.items())),
        "status": "ok",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage5 result.json 本地快照工具")
    subparsers = parser.add_subparsers(dest="phase", required=True)
    prepare = subparsers.add_parser("prepare", help="生成按主机拆分的 rsync 清单")
    finalize = subparsers.add_parser("finalize", help="验收 mirror 并生成快照与索引")
    for subparser in (prepare, finalize):
        subparser.add_argument("--source-runs-csv", type=Path, required=True)
        subparser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
        subparser.add_argument("--expected-count", type=int, default=2250)
    prepare.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--mirror-root", type=Path, required=True)
    finalize.add_argument("--snapshot-root", type=Path, required=True)
    finalize.add_argument("--index-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "prepare":
        report = prepare_rsync_lists(
            source_runs_csv=args.source_runs_csv,
            condition=args.condition,
            output_root=args.output_root,
            expected_count=args.expected_count,
        )
    else:
        report = finalize_snapshot(
            source_runs_csv=args.source_runs_csv,
            condition=args.condition,
            mirror_root=args.mirror_root,
            snapshot_root=args.snapshot_root,
            index_csv=args.index_csv,
            expected_count=args.expected_count,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
