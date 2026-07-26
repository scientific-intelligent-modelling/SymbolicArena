#!/usr/bin/env python3
"""Build and verify the final NeurIPS rebuttal experiment archive index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_TASKS = 5976
EXPECTED_RUNS = 13944
EXPECTED_DATASETS = 664
EXPECTED_NEW_ALGORITHMS = {"fepysr", "jaxsr", "symbolfit"}
EXPECTED_SEEDS = {520, 521, 522}
EXPECTED_ALGORITHMS = {
    "dso",
    "fepysr",
    "imcts",
    "jaxsr",
    "pyoperon",
    "symbolfit",
    "udsr",
}
ARCHIVE_ROOT_FILES = (
    "BATCH_NAME.txt",
    "EXPERIMENT_PATHS.md",
    "README.md",
    "SOURCE_MAP.tsv",
    "heartbeat.json",
)
ARCHIVE_ROOT_DIRS = (
    "analysis",
    "audit",
    "collect",
    "deploy",
    "harvest",
    "manifest",
    "monitoring",
    "params",
    "params_smoke",
    "preflight",
    "provenance",
    "queues",
    "runs",
    "smoke",
    "symf",
)
ARCHIVE_FILENAMES = {"MANIFEST.tsv", "CHECKSUMS.sha256"}
EXCLUDED_NAMES = {"remote-experiments", "runtime_queue"}
MANIFEST_HEADER = "relative_path\tsize_bytes"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON 顶层必须是对象：{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(batch_dir: Path, path: Path) -> str:
    relative = path.relative_to(batch_dir).as_posix()
    if not relative or relative.startswith("../"):
        raise RuntimeError(f"归档路径越出实验目录：{path}")
    if any(character in relative for character in "\t\r\n"):
        raise RuntimeError(f"归档路径包含 TSV 不支持的字符：{relative!r}")
    return relative


def _is_volatile(relative: str) -> bool:
    name = Path(relative).name
    return (
        name in ARCHIVE_FILENAMES
        or name in EXCLUDED_NAMES
        or name.endswith(".lock")
        or name.endswith(".pid")
        or name.endswith(".tmp")
        or ".tmp." in name
        or name == "__pycache__"
        or name.endswith(".pyc")
    )


def _inspect_scope_path(batch_dir: Path, path: Path) -> bool:
    relative = _relative_path(batch_dir, path)
    if _is_volatile(relative):
        return False
    if path.is_symlink():
        raise RuntimeError(f"归档范围内不允许软链接：{relative}")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise RuntimeError(f"无法检查归档路径：{relative}: {exc}") from exc
    if stat.S_ISDIR(mode):
        return False
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"归档范围内只允许普通文件：{relative}")
    return True


def collect_archive_files(batch_dir: Path) -> list[Path]:
    """Return the deterministic, declared archive file scope."""

    batch_dir = batch_dir.resolve()
    if not batch_dir.is_dir():
        raise RuntimeError(f"实验目录不存在：{batch_dir}")

    files: list[Path] = []
    for filename in ARCHIVE_ROOT_FILES:
        path = batch_dir / filename
        if path.exists() or path.is_symlink():
            if _inspect_scope_path(batch_dir, path):
                files.append(path)

    for dirname in ARCHIVE_ROOT_DIRS:
        root = batch_dir / dirname
        if not root.exists() and not root.is_symlink():
            continue
        if root.is_symlink():
            raise RuntimeError(f"归档范围内不允许软链接：{dirname}")
        if not root.is_dir():
            raise RuntimeError(f"归档根必须是目录：{dirname}")
        for current_root, directories, filenames in os.walk(
            root,
            followlinks=False,
        ):
            current = Path(current_root)
            kept_directories: list[str] = []
            for directory in directories:
                path = current / directory
                relative = _relative_path(batch_dir, path)
                if _is_volatile(relative):
                    continue
                if path.is_symlink():
                    raise RuntimeError(f"归档范围内不允许软链接：{relative}")
                kept_directories.append(directory)
            directories[:] = kept_directories
            for filename in filenames:
                path = current / filename
                if _inspect_scope_path(batch_dir, path):
                    files.append(path)

    return sorted(
        files,
        key=lambda path: _relative_path(batch_dir, path).encode("utf-8"),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_analysis(batch_dir: Path) -> None:
    summary = _load_json(batch_dir / "analysis" / "analysis_summary.json")
    new3 = summary.get("new3", {})
    audit_gate = summary.get("audit_gate", {})
    stage3 = summary.get("stage3", {})
    _require(summary.get("final_ready") is True, "analysis 尚未 final_ready")
    _require(
        new3.get("expected_tasks") == EXPECTED_TASKS
        and new3.get("present_results") == EXPECTED_TASKS
        and new3.get("missing_results") == 0
        and new3.get("unreadable_results") == 0
        and new3.get("identity_mismatches") == 0
        and new3.get("valid_outputs") == EXPECTED_TASKS
        and new3.get("metric_complete") == EXPECTED_TASKS,
        "新三算法分析没有达到 5976 条完整门禁",
    )
    _require(
        audit_gate.get("valid") is True,
        "analysis 引用的 audit gate 未通过",
    )
    _require(
        stage3.get("rows") == 7968
        and stage3.get("expected_rows") == 7968
        and stage3.get("keyspace_valid") is True
        and stage3.get("raw_digest_rows") == 7968,
        "Stage 3 四算法输入网格不完整",
    )


def _validate_audits(batch_dir: Path) -> None:
    gate = _load_json(batch_dir / "audit" / "audit_gate_summary.json")
    _require(
        gate.get("audit_passed") is True
        and gate.get("expected_total_tasks") == EXPECTED_TASKS
        and gate.get("task_audit_rows") == EXPECTED_TASKS
        and gate.get("failure_rows") == 0
        and gate.get("issues") == [],
        "正式 audit gate 没有达到 5976/5976",
    )

    audit_root = batch_dir / "monitoring" / "completed_audit"
    summaries = sorted(audit_root.glob("*/summary.json"))
    _require(bool(summaries), "缺少 completed audit 证据")
    final_summaries = []
    for path in summaries:
        payload = _load_json(path)
        if (
            payload.get("expected_done_tasks") == EXPECTED_TASKS
            and payload.get("audited_done_tasks") == EXPECTED_TASKS
        ):
            final_summaries.append((path, payload))
    _require(bool(final_summaries), "缺少 5976/5976 completed audit")
    path, summary = final_summaries[-1]
    _require(
        summary.get("validated_results") == EXPECTED_TASKS
        and summary.get("all_hosts_passed") is True
        and summary.get("issue_count") == 0
        and summary.get("passed") is True,
        f"最终 completed audit 未通过：{path}",
    )
    snapshot = path.parent / "state.snapshot.json"
    _require(snapshot.is_file(), f"最终审计缺少 state 快照：{snapshot}")
    _require(
        _sha256(snapshot) == summary.get("state_sha256"),
        "最终审计 state 快照 SHA-256 不匹配",
    )


def _validate_task_and_result_counts(batch_dir: Path) -> None:
    with (batch_dir / "manifest" / "tasks.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        tasks = list(csv.DictReader(handle))
    _require(len(tasks) == EXPECTED_TASKS, "正式任务 manifest 不是 5976 条")
    task_ids = [row.get("task_id", "") for row in tasks]
    _require(
        all(task_ids) and len(set(task_ids)) == EXPECTED_TASKS,
        "正式任务 manifest 的 task_id 不完整或重复",
    )

    expected_keys: set[tuple[str, int, str]] = set()
    expected_result_paths: set[str] = set()
    for row in tasks:
        algorithm = row.get("algorithm", "")
        dataset_id = row.get("dataset_id", "")
        noise_tag = row.get("noise_tag", "")
        try:
            seed = int(row.get("seed", ""))
        except ValueError as exc:
            raise RuntimeError("正式任务 manifest 包含非法 seed") from exc
        _require(
            algorithm in EXPECTED_NEW_ALGORITHMS,
            f"正式任务 manifest 包含非目标算法：{algorithm}",
        )
        _require(seed in EXPECTED_SEEDS, f"正式任务 manifest 包含非法 seed：{seed}")
        _require(noise_tag == "clean", "正式任务 manifest 包含非 clean 任务")
        _require(bool(dataset_id), "正式任务 manifest 缺少 dataset_id")
        expected_task_id = f"{algorithm}__seed{seed}__clean__{dataset_id}"
        _require(
            row.get("task_id") == expected_task_id,
            f"正式任务 task_id 与身份字段不一致：{row.get('task_id')}",
        )
        key = (algorithm, seed, dataset_id)
        _require(key not in expected_keys, f"正式任务身份重复：{key}")
        expected_keys.add(key)
        expected_result_paths.add(
            (
                Path("runs")
                / algorithm
                / f"seed{seed}"
                / "clean"
                / dataset_id
                / "result.json"
            ).as_posix()
        )

    expected_grid_size = (
        len(EXPECTED_NEW_ALGORITHMS) * len(EXPECTED_SEEDS) * EXPECTED_DATASETS
    )
    _require(
        len(expected_keys) == expected_grid_size == EXPECTED_TASKS,
        "正式任务 manifest 不是完整 algorithm × seed × dataset 网格",
    )
    result_paths = {
        path.relative_to(batch_dir).as_posix()
        for path in (batch_dir / "runs").rglob("result.json")
    }
    _require(
        len(result_paths) == EXPECTED_TASKS,
        f"本地规范 result.json 数量不是 5976：{len(result_paths)}",
    )
    missing = expected_result_paths - result_paths
    extra = result_paths - expected_result_paths
    _require(
        not missing and not extra,
        (
            "本地规范 result.json 未与任务 manifest 一一对应："
            f"missing={len(missing)}, extra={len(extra)}"
        ),
    )


def _validate_symf(batch_dir: Path) -> None:
    source_root = batch_dir / "symf" / "full664_7alg" / "by_source"
    final_summaries = sorted(
        source_root.glob("*/final/symbolic_metrics_formal_summary.json")
    )
    _require(
        len(final_summaries) == 1,
        f"正式 SYM-F source 数量必须为 1，实际为 {len(final_summaries)}",
    )
    summary_path = final_summaries[0]
    summary = _load_json(summary_path)
    _require(
        summary.get("runs") == EXPECTED_RUNS
        and summary.get("datasets") == EXPECTED_DATASETS
        and summary.get("algorithms") == len(EXPECTED_ALGORITHMS)
        and summary.get("params_datasets") == EXPECTED_DATASETS,
        "正式 SYM-F summary 未达到 13944/664/7 门禁",
    )

    source_dir = summary_path.parents[1]
    leaderboard_dir = source_dir / "leaderboard"
    source_csv = leaderboard_dir / "full664_7alg_leaderboard_with_symf.csv"
    source_summary_path = source_csv.with_suffix(".summary.json")
    source_summary = _load_json(source_summary_path)
    _require(
        source_summary.get("algorithms") == len(EXPECTED_ALGORITHMS)
        and set(source_summary.get("algorithm_keys", [])) == EXPECTED_ALGORITHMS,
        "正式 7 算法榜单算法集合不完整",
    )
    _require(
        source_summary.get("output_sha256") == _sha256(source_csv),
        "正式 7 算法榜单 SHA-256 与 summary 不一致",
    )

    latest_csv = batch_dir / "analysis" / "full664_7alg_leaderboard_with_symf.csv"
    latest_summary = latest_csv.with_suffix(".summary.json")
    _require(
        latest_csv.is_file() and latest_summary.is_file(),
        "analysis 缺少最终带 SYM-F 榜单镜像",
    )
    _require(
        _sha256(latest_csv) == _sha256(source_csv)
        and _sha256(latest_summary) == _sha256(source_summary_path),
        "analysis 榜单镜像与内容寻址权威版本不一致",
    )
    with latest_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == len(EXPECTED_ALGORITHMS), "最终榜单不是 7 行")
    _require(
        {row.get("Algorithm key", "") for row in rows} == EXPECTED_ALGORITHMS,
        "最终榜单算法键不完整",
    )


def validate_final_artifacts(batch_dir: Path) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    status = _load_json(batch_dir / "deploy" / "finalization_status.json")
    _require(
        status.get("state") == "finished"
        and status.get("exit_code") == 0
        and bool(status.get("ended_at")),
        "远端 finalization 尚未成功结束",
    )
    _validate_analysis(batch_dir)
    _validate_audits(batch_dir)
    _validate_task_and_result_counts(batch_dir)
    _validate_symf(batch_dir)
    return {
        "finalization": "finished",
        "tasks": EXPECTED_TASKS,
        "symf_runs": EXPECTED_RUNS,
        "datasets": EXPECTED_DATASETS,
        "algorithms": len(EXPECTED_ALGORITHMS),
    }


def _build_entries(batch_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for path in collect_archive_files(batch_dir):
        before = path.stat(follow_symlinks=False)
        digest = _sha256(path)
        after = path.stat(follow_symlinks=False)
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            relative = _relative_path(batch_dir, path)
            raise RuntimeError(f"归档生成期间文件发生变化：{relative}")
        entries.append(
            {
                "relative_path": _relative_path(batch_dir, path),
                "size_bytes": before.st_size,
                "sha256": digest,
            }
        )
    return entries


def write_archive(
    batch_dir: Path,
    *,
    validate_final: bool = True,
) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    gate = validate_final_artifacts(batch_dir) if validate_final else None
    entries = _build_entries(batch_dir)

    manifest_lines = [MANIFEST_HEADER]
    manifest_lines.extend(
        f"{entry['relative_path']}\t{entry['size_bytes']}" for entry in entries
    )
    manifest_path = batch_dir / "MANIFEST.tsv"
    _atomic_write_text(manifest_path, "\n".join(manifest_lines) + "\n")

    checksum_lines = [f"{_sha256(manifest_path)}  ./MANIFEST.tsv"]
    checksum_lines.extend(
        f"{entry['sha256']}  ./{entry['relative_path']}" for entry in entries
    )
    checksum_path = batch_dir / "CHECKSUMS.sha256"
    _atomic_write_text(checksum_path, "\n".join(checksum_lines) + "\n")
    return {
        "valid": True,
        "files": len(entries),
        "bytes": sum(entry["size_bytes"] for entry in entries),
        "manifest": str(manifest_path),
        "checksums": str(checksum_path),
        "gate": gate,
    }


def _read_manifest(batch_dir: Path) -> list[dict[str, Any]]:
    path = batch_dir / "MANIFEST.tsv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _require(
                reader.fieldnames == ["relative_path", "size_bytes"],
                "MANIFEST.tsv 表头不正确",
            )
            rows = list(reader)
    except OSError as exc:
        raise RuntimeError(f"无法读取 MANIFEST.tsv：{exc}") from exc

    entries: list[dict[str, Any]] = []
    for row in rows:
        relative = row.get("relative_path", "")
        _require(
            bool(relative)
            and not relative.startswith("/")
            and ".." not in Path(relative).parts
            and not any(character in relative for character in "\t\r\n"),
            f"MANIFEST.tsv 包含非法相对路径：{relative!r}",
        )
        try:
            size = int(row.get("size_bytes", ""))
        except ValueError as exc:
            raise RuntimeError(f"MANIFEST.tsv 字节数无效：{relative}") from exc
        _require(size >= 0, f"MANIFEST.tsv 字节数为负：{relative}")
        entries.append({"relative_path": relative, "size_bytes": size})

    paths = [entry["relative_path"] for entry in entries]
    _require(len(paths) == len(set(paths)), "MANIFEST.tsv 路径重复")
    _require(
        paths == sorted(paths, key=str.encode),
        "MANIFEST.tsv 没有按路径字节序排序",
    )
    return entries


def _read_checksums(batch_dir: Path) -> list[tuple[str, str]]:
    path = batch_dir / "CHECKSUMS.sha256"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"无法读取 CHECKSUMS.sha256：{exc}") from exc
    entries = []
    for line in lines:
        try:
            digest, relative = line.split("  ./", maxsplit=1)
        except ValueError as exc:
            raise RuntimeError("CHECKSUMS.sha256 格式无效") from exc
        _require(
            len(digest) == 64
            and digest == digest.lower()
            and all(character in "0123456789abcdef" for character in digest),
            f"CHECKSUMS.sha256 包含非法摘要：{digest!r}",
        )
        _require(relative and not relative.startswith("/"), "校验路径无效")
        entries.append((relative, digest))
    return entries


def verify_archive(
    batch_dir: Path,
    *,
    require_exact_scope: bool,
    validate_final: bool = True,
) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    gate = validate_final_artifacts(batch_dir) if validate_final else None
    manifest_entries = _read_manifest(batch_dir)
    manifest_paths = [entry["relative_path"] for entry in manifest_entries]

    if require_exact_scope:
        actual_paths = [
            _relative_path(batch_dir, path) for path in collect_archive_files(batch_dir)
        ]
        _require(
            manifest_paths == actual_paths,
            "MANIFEST.tsv 与当前归档范围不一致",
        )

    for entry in manifest_entries:
        path = batch_dir / entry["relative_path"]
        _require(
            path.is_file() and not path.is_symlink(),
            f"归档文件缺失或不是普通文件：{entry['relative_path']}",
        )
        _require(
            path.stat().st_size == entry["size_bytes"],
            f"归档文件大小不匹配：{entry['relative_path']}",
        )

    checksums = _read_checksums(batch_dir)
    expected_checksum_paths = ["MANIFEST.tsv", *manifest_paths]
    _require(
        [relative for relative, _ in checksums] == expected_checksum_paths,
        "CHECKSUMS.sha256 覆盖范围或顺序不正确",
    )
    for relative, expected_digest in checksums:
        path = batch_dir / relative
        _require(path.is_file(), f"校验文件不存在：{relative}")
        _require(
            _sha256(path) == expected_digest,
            f"SHA-256 校验失败：{relative}",
        )

    return {
        "valid": True,
        "files": len(manifest_entries),
        "bytes": sum(entry["size_bytes"] for entry in manifest_entries),
        "exact_scope": require_exact_scope,
        "gate": gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成或验证 NeurIPS rebuttal 最终归档清单",
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--require-exact-scope",
        action="store_true",
        help="验证 MANIFEST 与声明归档范围完全一致",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_exact_scope and not args.verify:
        raise SystemExit("--require-exact-scope 只能与 --verify 一起使用")
    if args.verify and not args.require_exact_scope:
        raise SystemExit("--verify 必须同时指定 --require-exact-scope")
    if args.write:
        summary = write_archive(args.batch_dir)
    else:
        summary = verify_archive(
            args.batch_dir,
            require_exact_scope=args.require_exact_scope,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
