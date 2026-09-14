#!/usr/bin/env python3
"""Build the auditable Core-50 raw snapshot archive without remote access.

The archive deliberately separates physical files from the 180-point logical
trajectory.  A carry-forward minute binds to an earlier physical object; it is
never materialized as a fabricated ``minute_N.json``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import zstandard


class ArchiveBuildError(RuntimeError):
    """Raised when frozen evidence cannot satisfy the authoritative index."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_lines(path: Path) -> Iterator[tuple[bytes, dict]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip(b"\r\n")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArchiveBuildError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ArchiveBuildError(f"{path}:{line_number}: record is not an object")
            yield raw, payload


def _raw_text(value: object, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise ArchiveBuildError(f"{label}: raw_text is missing")
    return value.encode("utf-8")


def _verify_raw(raw: bytes, expected: object, *, label: str) -> str:
    actual = sha256_bytes(raw)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ArchiveBuildError(f"{label}: invalid declared SHA256")
    if actual != expected:
        raise ArchiveBuildError(f"{label}: SHA256 mismatch: {actual} != {expected}")
    return actual


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    path: Path
    historical: bool = False


@dataclass(frozen=True)
class RawObject:
    task_id: str
    record_type: str
    raw: bytes
    sha256: str
    source_path: str
    minute: int | None
    evidence_kind: str
    container_path: str
    logical_key: str | None = None


@dataclass(frozen=True)
class UnavailableObject:
    task_id: str
    minute: int | None
    source_path: str
    reason: str
    container_path: str
    scope: str


@dataclass(frozen=True)
class RunHeader:
    logical_key: str
    task_id: str
    condition: str
    algorithm: str
    dataset_id: str
    seed: int
    host: str
    bundle_path: str
    bundle_sha256: str


def iter_task_freeze(spec: SourceSpec) -> Iterator[RawObject | UnavailableObject]:
    """Read the task-oriented freeze schema used by clean/noise/CPU overlays."""

    for _, record in _json_lines(spec.path):
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("task_id"), str):
            raise ArchiveBuildError(f"{spec.path}: task freeze record lacks source.task_id")
        task_id = source["task_id"]
        result = record.get("result")
        if isinstance(result, dict) and result.get("raw_text") is not None:
            raw = _raw_text(result["raw_text"], label=f"{task_id}:result")
            digest = _verify_raw(raw, result.get("sha256"), label=f"{task_id}:result")
            yield RawObject(
                task_id, "result", raw, digest, str(result.get("path") or ""), None,
                "task_freeze_result", str(spec.path),
            )
        snapshots = record.get("snapshots")
        if not isinstance(snapshots, list):
            raise ArchiveBuildError(f"{task_id}: snapshots is not a list")
        for fallback_minute, snapshot in enumerate(snapshots, start=1):
            if not isinstance(snapshot, dict):
                raise ArchiveBuildError(f"{task_id}: invalid snapshot record")
            minute = int(snapshot.get("minute") or fallback_minute)
            source_path = str(
                snapshot.get("selected_path")
                or snapshot.get("outer_path")
                or snapshot.get("inner_path")
                or ""
            )
            if snapshot.get("status") != "ok" or snapshot.get("raw_text") is None:
                if spec.historical:
                    yield UnavailableObject(
                        task_id, minute, source_path,
                        str(snapshot.get("status") or "missing_raw_text"), str(spec.path),
                        "historical_base_freeze",
                    )
                    continue
                raise ArchiveBuildError(f"{task_id}: minute {minute} is unavailable")
            raw = _raw_text(snapshot["raw_text"], label=f"{task_id}:minute:{minute}")
            digest = _verify_raw(
                raw, snapshot.get("selected_sha256"), label=f"{task_id}:minute:{minute}"
            )
            yield RawObject(
                task_id, "snapshot", raw, digest, source_path, minute,
                "task_freeze_snapshot", str(spec.path),
            )


def iter_flat_raw_bundle(spec: SourceSpec) -> Iterator[RawObject]:
    """Read ``eff_native_raw_record.v1`` result/snapshot records."""

    for _, record in _json_lines(spec.path):
        task_id = record.get("task_id")
        record_type = record.get("record_type")
        if not isinstance(task_id, str) or record_type not in {"result", "snapshot"}:
            raise ArchiveBuildError(f"{spec.path}: invalid flat raw record identity")
        minute = record.get("minute")
        minute = int(minute) if minute is not None else None
        raw = _raw_text(record.get("raw_text"), label=f"{task_id}:{record_type}:{minute}")
        digest = _verify_raw(raw, record.get("sha256"), label=f"{task_id}:{record_type}:{minute}")
        yield RawObject(
            task_id, record_type, raw, digest, str(record.get("source_path") or ""), minute,
            "flat_raw_bundle", str(spec.path),
        )


def iter_symbolfit_trajectory(spec: SourceSpec, *, task_id: str | None = None) -> Iterator[RawObject]:
    """Read one SymbolFit per-run JSONL; each line is the frozen raw payload."""

    inferred_task = task_id or spec.path.name.removesuffix(".jsonl.gz")
    seen: set[int] = set()
    for raw, record in _json_lines(spec.path):
        minute_value = record.get("checkpoint_index", record.get("elapsed_minutes"))
        if minute_value is None:
            raise ArchiveBuildError(f"{spec.path}: SymbolFit line lacks checkpoint minute")
        minute = int(minute_value)
        if minute in seen:
            raise ArchiveBuildError(f"{spec.path}: duplicate SymbolFit minute {minute}")
        seen.add(minute)
        yield RawObject(
            inferred_task, "snapshot", raw, sha256_bytes(raw), str(spec.path), minute,
            "symbolfit_frozen_payload", str(spec.path),
        )


def iter_recovery_evidence(spec: SourceSpec) -> Iterator[RawObject]:
    """Read progress.json and best-history raw text from recovery evidence."""

    for _, record in _json_lines(spec.path):
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("task_id"), str):
            raise ArchiveBuildError(f"{spec.path}: recovery record lacks source.task_id")
        if record.get("status") != "ok":
            continue  # A later evidence overlay may resolve this attempt.
        task_id = source["task_id"]
        if record.get("progress_raw_text") is not None:
            raw = _raw_text(record["progress_raw_text"], label=f"{task_id}:progress")
            digest = _verify_raw(raw, record.get("progress_sha256"), label=f"{task_id}:progress")
            yield RawObject(
                task_id, "recovery", raw, digest, str(record.get("progress_path") or ""), None,
                "recovery_progress", str(spec.path), source.get("logical_key"),
            )
        candidates = record.get("candidates") or []
        if not isinstance(candidates, list):
            raise ArchiveBuildError(f"{task_id}: recovery candidates is not a list")
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                raise ArchiveBuildError(f"{task_id}: invalid recovery candidate")
            raw = _raw_text(candidate.get("raw_text"), label=f"{task_id}:candidate:{index}")
            digest = _verify_raw(
                raw, candidate.get("sha256"), label=f"{task_id}:candidate:{index}"
            )
            yield RawObject(
                task_id, "recovery", raw, digest, str(candidate.get("path") or ""), None,
                "recovery_candidate", str(spec.path), source.get("logical_key"),
            )


READERS = {
    "task_freeze": iter_task_freeze,
    "flat_raw": iter_flat_raw_bundle,
    "recovery": iter_recovery_evidence,
}


def _normal_path(path: str | Path, repo_root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return str(candidate.resolve(strict=False))


def _safe(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "unknown"


def _run_root(run: RunHeader) -> str:
    return "/".join(
        [
            "runs", _safe(run.condition), _safe(run.algorithm), _safe(run.dataset_id),
            f"seed_{run.seed}", _safe(run.task_id),
        ]
    )


def _member_for(raw: RawObject, run: RunHeader) -> str:
    root = _run_root(run)
    if raw.record_type == "snapshot":
        if raw.minute is None:
            raise ArchiveBuildError(f"{raw.task_id}: snapshot lacks minute")
        return f"{root}/progress/minute_{raw.minute:04d}.json"
    if raw.record_type == "result":
        return f"{root}/result.json"
    basename = _safe(Path(raw.source_path).name or raw.sha256[:16])
    return f"{root}/recovery/{basename}"


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    archive.addfile(_tar_info(name, len(data)), io.BytesIO(data))


def _add_path(archive: tarfile.TarFile, name: str, path: Path) -> None:
    with path.open("rb") as handle:
        archive.addfile(_tar_info(name, path.stat().st_size), handle)


def _open_gzip_csv(path: Path, fieldnames: Sequence[str]):
    binary = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    return binary, gz, text, writer


def _close_gzip_csv(handles: tuple) -> None:
    binary, gz, text, _ = handles
    text.flush()
    text.detach()
    gz.close()
    binary.close()


def _read_authoritative(
    paths: Sequence[Path], repo_root: Path, horizon: int, expected_runs: int | None
) -> tuple[
    dict[str, RunHeader], dict[str, tuple[str, str]], dict[tuple[str, str], str],
    dict[tuple[str, str], list[str]], set[tuple[str, str, str]], dict[str, set[str]],
    dict[str, list[str]], dict[str, int]
]:
    runs: dict[str, RunHeader] = {}
    selected: dict[str, tuple[str, str]] = {}
    ordinary_locator: dict[tuple[str, str], str] = {}
    ordinary_sha_locator: dict[tuple[str, str], list[str]] = {}
    required_pairs: set[tuple[str, str, str]] = set()
    required_sha_by_logical: dict[str, set[str]] = {}
    logical_keys_by_task: dict[str, list[str]] = {}
    container_counts: dict[str, int] = {}
    for path in paths:
        for _, record in _json_lines(path):
            task_id = record.get("task_id")
            logical_key = record.get("logical_key")
            if not isinstance(task_id, str) or not isinstance(logical_key, str):
                raise ArchiveBuildError(f"{path}: authoritative run lacks identity")
            if logical_key in runs:
                raise ArchiveBuildError(f"duplicate authoritative run: {task_id}/{logical_key}")
            source_paths = record.get("source_path")
            source_shas = record.get("source_sha256")
            trajectory = record.get("trajectory_source")
            incumbent = record.get("incumbent_source_minute")
            valid = record.get("valid_output")
            arrays = [source_paths, source_shas, trajectory, incumbent, valid]
            if any(not isinstance(values, list) or len(values) != horizon for values in arrays):
                raise ArchiveBuildError(f"{logical_key}: authoritative arrays are not {horizon} points")
            bundle_path = str(record.get("bundle_path") or "")
            bundle_sha = str(record.get("bundle_sha256") or "")
            header = RunHeader(
                logical_key=logical_key,
                task_id=task_id,
                condition=str(record.get("condition") or ""),
                algorithm=str(record.get("algorithm") or ""),
                dataset_id=str(record.get("dataset_id") or ""),
                seed=int(record.get("seed")),
                host=str(record.get("host") or ""),
                bundle_path=bundle_path,
                bundle_sha256=bundle_sha,
            )
            runs[logical_key] = header
            normalized_bundle = _normal_path(bundle_path, repo_root)
            selected[logical_key] = (normalized_bundle, bundle_sha)
            locator_key = (normalized_bundle, task_id)
            previous_logical = ordinary_locator.setdefault(locator_key, logical_key)
            if previous_logical != logical_key:
                raise ArchiveBuildError(
                    f"ordinary source identity is ambiguous: {normalized_bundle}/{task_id}"
                )
            logical_keys_by_task.setdefault(task_id, []).append(logical_key)
            ordinary_sha_locator.setdefault((bundle_sha, task_id), []).append(logical_key)
            container_counts[normalized_bundle] = container_counts.get(normalized_bundle, 0) + 1
            logical_shas = required_sha_by_logical.setdefault(logical_key, set())
            for source_path, source_sha in zip(source_paths, source_shas):
                if source_sha in (None, ""):
                    continue
                if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
                    raise ArchiveBuildError(f"{logical_key}: invalid source SHA")
                logical_shas.add(source_sha)
                if source_path not in (None, ""):
                    required_pairs.add(
                        (logical_key, _normal_path(str(source_path), repo_root), source_sha)
                    )
    if expected_runs is not None and len(runs) != expected_runs:
        raise ArchiveBuildError(f"authoritative runs={len(runs)} != {expected_runs}")
    return (
        runs,
        selected,
        ordinary_locator,
        ordinary_sha_locator,
        required_pairs,
        required_sha_by_logical,
        logical_keys_by_task,
        container_counts,
    )


def discover_default_sources(repo_root: Path, release_root: Path) -> list[SourceSpec]:
    stage = repo_root / "AAAI_experiments/stage5_metric_calculation_0831"
    specs: list[SourceSpec] = []
    for path in sorted((stage / "source_snapshot/trajectory_freeze").glob("clean_freeze_*.jsonl.gz")):
        specs.append(SourceSpec("task_freeze", path, historical=True))
    for path in sorted((stage / "work/noise_trajectory_freeze_v1/collected").glob("noise_freeze_*.jsonl.gz")):
        specs.append(SourceSpec("task_freeze", path, historical=True))
    overlay = stage / "work/clean_rerun_overlay_v1/clean_eff_overlay.jsonl.gz"
    if overlay.is_file():
        specs.append(SourceSpec("task_freeze", overlay))
    for path in sorted((stage / "work/remote_collect/all_conditions_cpu_v2/collected").glob("noise_freeze_*.jsonl.gz")):
        specs.append(SourceSpec("task_freeze", path))
    for path in sorted((release_root / "provenance/eff_native_raw_bundles_20260914/host_bundles").glob("*.jsonl.gz")):
        specs.append(SourceSpec("flat_raw", path))
    symbolfit_root = stage / "work/final_release_20260913/symbolfit_noise_latest/hosts"
    for path in sorted(symbolfit_root.glob("*/trajectories/*.jsonl.gz")):
        specs.append(SourceSpec("symbolfit", path))
    recovery_roots = [
        release_root / "provenance/eff_recovery_existing/collected",
        stage / "work/final_release_20260913/release_v2/eff_noise_revision/recovery_llmsr/collected",
        release_root / "provenance/eff_revision_v3/recovery_llmsr_one",
    ]
    for root in recovery_roots:
        for path in sorted(root.glob("evidence*.jsonl.gz")):
            specs.append(SourceSpec("recovery", path))
    return specs


RUN_FIELDS = [
    "logical_key", "task_id", "condition", "algorithm", "dataset_id", "seed", "host",
    "bundle_path", "bundle_sha256",
]
PHYSICAL_FIELDS = [
    "archive_member", "sha256", "size_bytes", "source_path", "task_id", "logical_key",
    "condition", "algorithm", "dataset_id", "seed", "minute", "record_type",
    "evidence_kind", "container_path",
]
BINDING_FIELDS = [
    "logical_key", "task_id", "condition", "algorithm", "dataset_id", "seed", "minute",
    "trajectory_source", "incumbent_source_minute", "valid_output", "source_path",
    "source_sha256", "archive_member", "binding_status",
]
UNAVAILABLE_FIELDS = [
    "scope", "task_id", "minute", "source_path", "reason", "container_path",
]
CONTAINER_FIELDS = [
    "kind", "path", "sha256", "size_bytes", "historical", "selected_run_count", "status",
]


def build_archive(
    *,
    repo_root: Path,
    release_root: Path,
    output: Path,
    authoritative_paths: Sequence[Path] | None = None,
    source_specs: Sequence[SourceSpec] | None = None,
    horizon: int = 180,
    expected_runs: int | None = 6750,
    expected_historical_missing: int | None = 17,
    compression_level: int = 10,
    compression_threads: int = 4,
) -> dict:
    repo_root = repo_root.resolve()
    release_root = release_root.resolve()
    output = output.resolve()
    if output.exists():
        raise ArchiveBuildError(f"archive output already exists: {output}")
    if compression_threads <= 0:
        raise ArchiveBuildError("compression_threads must be positive")
    if authoritative_paths is None:
        authoritative_paths = [
            release_root / f"provenance/eff_revision_v3/{condition}/native_trajectories.jsonl.gz"
            for condition in ("clean", "noise001", "noise005")
        ]
    authoritative_paths = [Path(path).resolve() for path in authoritative_paths]
    for path in authoritative_paths:
        if not path.is_file():
            raise ArchiveBuildError(f"authoritative trajectory missing: {path}")
    specs = list(source_specs or discover_default_sources(repo_root, release_root))
    if not specs:
        raise ArchiveBuildError("no raw source containers discovered")

    (
        runs,
        selected,
        ordinary_locator,
        ordinary_sha_locator,
        required_pairs,
        required_sha_by_logical,
        logical_keys_by_task,
        container_counts,
    ) = _read_authoritative(authoritative_paths, repo_root, horizon, expected_runs)
    selected_expected_sha: dict[str, str] = {}
    selected_count_by_sha: dict[str, int] = {}
    for _, (container, declared_sha) in selected.items():
        selected_count_by_sha[declared_sha] = selected_count_by_sha.get(declared_sha, 0) + 1
        if Path(container).is_file():
            previous = selected_expected_sha.setdefault(container, declared_sha)
            if previous != declared_sha:
                raise ArchiveBuildError(f"container has conflicting declared SHA: {container}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary_output.exists():
        raise ArchiveBuildError(f"temporary archive already exists: {temporary_output}")

    pair_to_member: dict[tuple[str, str, str], str] = {}
    logical_sha_to_member: dict[tuple[str, str], str] = {}
    recovery_seen: set[tuple[str, str, str]] = set()
    counts = {
        "runs": len(runs), "logical_minutes": 0, "physical_records": 0,
        "historical_missing": 0, "carry_forward": 0, "invalid_output_minutes": 0,
        "source_containers": len(specs),
    }
    with tempfile.TemporaryDirectory(
        prefix=".core50_raw_snapshot_",
        dir=output.parent,
    ) as tmp_name:
        tmp = Path(tmp_name)
        runs_csv = tmp / "runs.csv"
        physical_csv = tmp / "physical_records.csv.gz"
        binding_csv = tmp / "logical_minute_bindings.csv.gz"
        unavailable_csv = tmp / "physical_unavailable.csv"
        containers_csv = tmp / "source_containers.csv"
        sums = tmp / "SHA256SUMS"

        with runs_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, lineterminator="\n")
            writer.writeheader()
            for run in sorted(runs.values(), key=lambda item: item.logical_key):
                writer.writerow(run.__dict__)

        physical_handles = _open_gzip_csv(physical_csv, PHYSICAL_FIELDS)
        physical_writer = physical_handles[-1]
        unavailable_handle = unavailable_csv.open("w", encoding="utf-8", newline="")
        unavailable_writer = csv.DictWriter(
            unavailable_handle, fieldnames=UNAVAILABLE_FIELDS, lineterminator="\n"
        )
        unavailable_writer.writeheader()
        containers_handle = containers_csv.open("w", encoding="utf-8", newline="")
        containers_writer = csv.DictWriter(
            containers_handle, fieldnames=CONTAINER_FIELDS, lineterminator="\n"
        )
        containers_writer.writeheader()
        sums_handle = sums.open("w", encoding="utf-8")

        try:
            with temporary_output.open("wb") as compressed_file:
                compressor = zstandard.ZstdCompressor(
                    level=compression_level,
                    threads=compression_threads,
                    write_checksum=True,
                )
                with compressor.stream_writer(compressed_file, closefd=False) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                        for spec in specs:
                            path = spec.path.resolve()
                            if not path.is_file():
                                raise ArchiveBuildError(f"source container missing: {path}")
                            actual_container_sha = sha256_file(path)
                            normalized_container = str(path)
                            expected_sha = selected_expected_sha.get(normalized_container)
                            if expected_sha and actual_container_sha != expected_sha:
                                raise ArchiveBuildError(
                                    f"source container SHA mismatch: {path}: "
                                    f"{actual_container_sha} != {expected_sha}"
                                )
                            containers_writer.writerow(
                                {
                                    "kind": spec.kind,
                                    "path": str(path),
                                    "sha256": actual_container_sha,
                                    "size_bytes": path.stat().st_size,
                                    "historical": str(spec.historical).lower(),
                                    "selected_run_count": container_counts.get(
                                        normalized_container,
                                        selected_count_by_sha.get(actual_container_sha, 0),
                                    ),
                                    "status": "verified",
                                }
                            )
                            if spec.kind == "symbolfit":
                                task_id = path.name.removesuffix(".jsonl.gz")
                                iterator: Iterable[RawObject | UnavailableObject] = (
                                    iter_symbolfit_trajectory(spec, task_id=task_id)
                                )
                            else:
                                reader = READERS.get(spec.kind)
                                if reader is None:
                                    raise ArchiveBuildError(f"unknown source reader: {spec.kind}")
                                iterator = reader(spec)
                            for item in iterator:
                                if isinstance(item, UnavailableObject):
                                    unavailable_writer.writerow(item.__dict__)
                                    counts["historical_missing"] += 1
                                    continue
                                if spec.kind in {"task_freeze", "flat_raw", "symbolfit"}:
                                    logical_key = ordinary_locator.get(
                                        (normalized_container, item.task_id)
                                    )
                                    if logical_key is None:
                                        matching = ordinary_sha_locator.get(
                                            (actual_container_sha, item.task_id), []
                                        )
                                        if len(matching) > 1:
                                            raise ArchiveBuildError(
                                                f"ordinary SHA identity is ambiguous for "
                                                f"{item.task_id}: {matching}"
                                            )
                                        logical_key = matching[0] if matching else None
                                    if logical_key is None:
                                        continue
                                    run = runs[logical_key]
                                else:
                                    # Recovery evidence is selected by logical identity and raw SHA.
                                    logical_key = item.logical_key
                                    if (
                                        logical_key in runs
                                        and runs[logical_key].task_id != item.task_id
                                    ):
                                        raise ArchiveBuildError(
                                            f"recovery logical identity disagrees with task_id: "
                                            f"{logical_key}/{item.task_id}"
                                        )
                                    if logical_key not in runs:
                                        matching = [
                                            candidate
                                            for candidate in logical_keys_by_task.get(item.task_id, [])
                                            if item.sha256 in required_sha_by_logical[candidate]
                                        ]
                                        if len(matching) > 1:
                                            raise ArchiveBuildError(
                                                f"recovery identity is ambiguous for {item.task_id}: "
                                                f"{matching}"
                                            )
                                        logical_key = matching[0] if matching else None
                                    if logical_key is None:
                                        continue
                                    if item.sha256 not in required_sha_by_logical[logical_key]:
                                        continue
                                    run = runs[logical_key]
                                    recovery_key = (
                                        logical_key,
                                        _normal_path(item.source_path, repo_root),
                                        item.sha256,
                                    )
                                    if recovery_key in recovery_seen:
                                        continue
                                    recovery_seen.add(recovery_key)
                                member = _member_for(item, run)
                                _add_bytes(archive, member, item.raw)
                                sums_handle.write(f"{item.sha256}  {member}\n")
                                physical_writer.writerow(
                                    {
                                        "archive_member": member,
                                        "sha256": item.sha256,
                                        "size_bytes": len(item.raw),
                                        "source_path": item.source_path,
                                        "task_id": item.task_id,
                                        "logical_key": run.logical_key,
                                        "condition": run.condition,
                                        "algorithm": run.algorithm,
                                        "dataset_id": run.dataset_id,
                                        "seed": run.seed,
                                        "minute": "" if item.minute is None else item.minute,
                                        "record_type": item.record_type,
                                        "evidence_kind": item.evidence_kind,
                                        "container_path": item.container_path,
                                    }
                                )
                                counts["physical_records"] += 1
                                normalized_source = _normal_path(item.source_path, repo_root)
                                pair = (run.logical_key, normalized_source, item.sha256)
                                if pair in required_pairs:
                                    pair_to_member.setdefault(pair, member)
                                if item.sha256 in required_sha_by_logical[run.logical_key]:
                                    logical_sha_to_member.setdefault(
                                        (run.logical_key, item.sha256), member
                                    )

                        _close_gzip_csv(physical_handles)
                        physical_handles = None
                        unavailable_handle.close()
                        unavailable_handle = None
                        containers_handle.close()
                        containers_handle = None

                        if (
                            expected_historical_missing is not None
                            and counts["historical_missing"] != expected_historical_missing
                        ):
                            raise ArchiveBuildError(
                                f"historical missing={counts['historical_missing']} "
                                f"!= {expected_historical_missing}"
                            )

                        binding_handles = _open_gzip_csv(binding_csv, BINDING_FIELDS)
                        binding_writer = binding_handles[-1]
                        try:
                            for path in authoritative_paths:
                                for _, record in _json_lines(path):
                                    run = runs[record["logical_key"]]
                                    arrays = zip(
                                        record["trajectory_source"],
                                        record["incumbent_source_minute"],
                                        record["valid_output"],
                                        record["source_path"],
                                        record["source_sha256"],
                                    )
                                    for minute, values in enumerate(arrays, start=1):
                                        trajectory_source, incumbent_minute, valid, source_path, source_sha = values
                                        source_sha = "" if source_sha in (None, "") else str(source_sha)
                                        member = ""
                                        if source_sha:
                                            pair = (
                                                run.logical_key,
                                                _normal_path(str(source_path), repo_root),
                                                source_sha,
                                            ) if source_path not in (None, "") else None
                                            if pair is not None:
                                                member = pair_to_member.get(pair, "")
                                            if not member:
                                                member = logical_sha_to_member.get(
                                                    (run.logical_key, source_sha), ""
                                                )
                                            if not member:
                                                raise ArchiveBuildError(
                                                    f"{run.logical_key}: minute {minute}: "
                                                    f"required raw SHA not found: {source_sha}"
                                                )
                                        source_text = str(trajectory_source or "")
                                        if "carry_forward" in source_text:
                                            counts["carry_forward"] += 1
                                        if not bool(valid):
                                            counts["invalid_output_minutes"] += 1
                                        binding_writer.writerow(
                                            {
                                                "logical_key": run.logical_key,
                                                "task_id": run.task_id,
                                                "condition": run.condition,
                                                "algorithm": run.algorithm,
                                                "dataset_id": run.dataset_id,
                                                "seed": run.seed,
                                                "minute": minute,
                                                "trajectory_source": source_text,
                                                "incumbent_source_minute": (
                                                    "" if incumbent_minute is None else incumbent_minute
                                                ),
                                                "valid_output": str(bool(valid)).lower(),
                                                "source_path": "" if source_path is None else source_path,
                                                "source_sha256": source_sha,
                                                "archive_member": member,
                                                "binding_status": "bound" if member else "protocol_no_file",
                                            }
                                        )
                                        counts["logical_minutes"] += 1
                        finally:
                            _close_gzip_csv(binding_handles)

                        expected_minutes = len(runs) * horizon
                        if counts["logical_minutes"] != expected_minutes:
                            raise ArchiveBuildError(
                                f"logical minutes={counts['logical_minutes']} != {expected_minutes}"
                            )

                        report = {
                            "schema_version": "core50_raw_snapshot_archive.v1",
                            "status": "passed",
                            **counts,
                            "horizon_minutes": horizon,
                            "candidate_selection_source": "eff_revision_v3/native_trajectories",
                            "carry_forward_materialized": False,
                            "remote_access_used": False,
                            "compression_threads": compression_threads,
                        }
                        readme = (
                            "# Core-50 raw snapshot archive\n\n"
                            "Physical JSON files are preserved byte-for-byte and checked by SHA256.\n"
                            "`logical_minute_bindings.csv.gz` is the complete logical trajectory.\n"
                            "Carry-forward and protocol-zero rows only bind evidence; they are never\n"
                            "fabricated as later raw snapshot files. Historical physical omissions\n"
                            "are listed separately in `physical_unavailable.csv`.\n"
                        ).encode("utf-8")
                        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
                        for name, path in (
                            ("runs.csv", runs_csv),
                            ("physical_records.csv.gz", physical_csv),
                            ("logical_minute_bindings.csv.gz", binding_csv),
                            ("physical_unavailable.csv", unavailable_csv),
                            ("source_containers.csv", containers_csv),
                        ):
                            sums_handle.write(f"{sha256_file(path)}  {name}\n")
                            _add_path(archive, name, path)
                        sums_handle.write(f"{sha256_bytes(readme)}  README.md\n")
                        sums_handle.write(f"{sha256_bytes(report_bytes)}  report.json\n")
                        sums_handle.close()
                        sums_handle = None
                        _add_bytes(archive, "README.md", readme)
                        _add_bytes(archive, "report.json", report_bytes)
                        _add_path(archive, "SHA256SUMS", sums)
        finally:
            if physical_handles is not None:
                _close_gzip_csv(physical_handles)
            if unavailable_handle is not None:
                unavailable_handle.close()
            if containers_handle is not None:
                containers_handle.close()
            if sums_handle is not None:
                sums_handle.close()

    temporary_output.replace(output)
    archive_sha = sha256_file(output)
    final_report = {
        "schema_version": "core50_raw_snapshot_archive_delivery.v1",
        "status": "passed",
        "archive": str(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": archive_sha,
        **counts,
        "horizon_minutes": horizon,
        "remote_access_used": False,
        "compression_threads": compression_threads,
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    sha_path = output.with_suffix(output.suffix + ".sha256")
    report_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sha_path.write_text(f"{archive_sha}  {output.name}\n", encoding="utf-8")
    return final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("AAAI_experiments/Core50_final_20260914"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("AAAI_experiments/Core50_raw_snapshots_20260914.tar.zst"),
    )
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--compression-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    release_root = args.release_root
    if not release_root.is_absolute():
        release_root = repo_root / release_root
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    report = build_archive(
        repo_root=repo_root,
        release_root=release_root,
        output=output,
        compression_level=args.compression_level,
        compression_threads=args.compression_threads,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
