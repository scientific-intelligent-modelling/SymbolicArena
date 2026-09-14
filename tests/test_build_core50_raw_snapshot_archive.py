from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
import zstandard

SCRIPT = Path(__file__).parents[1] / "check/build_core50_raw_snapshot_archive.py"
SPEC = importlib.util.spec_from_file_location("build_core50_raw_snapshot_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ArchiveBuildError = MODULE.ArchiveBuildError
SourceSpec = MODULE.SourceSpec
build_archive = MODULE.build_archive
iter_flat_raw_bundle = MODULE.iter_flat_raw_bundle
iter_recovery_evidence = MODULE.iter_recovery_evidence
iter_symbolfit_trajectory = MODULE.iter_symbolfit_trajectory
iter_task_freeze = MODULE.iter_task_freeze


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def raw_json(value: str, minute: int) -> bytes:
    return json.dumps({"value": value, "checkpoint_index": minute}, sort_keys=True).encode()


def task_freeze_record(task_id: str, source_dir: Path, values: list[str]) -> dict:
    snapshots = []
    for minute, value in enumerate(values, start=1):
        raw = raw_json(value, minute)
        snapshots.append(
            {
                "minute": minute,
                "status": "ok",
                "selected_path": str(source_dir / f"minute_{minute:04d}.json"),
                "selected_sha256": digest(raw),
                "raw_text": raw.decode(),
            }
        )
    result_raw = b'{"status":"ok"}'
    return {
        "source": {"task_id": task_id},
        "result": {
            "status": "ok",
            "path": str(source_dir / "result.json"),
            "sha256": digest(result_raw),
            "raw_text": result_raw.decode(),
        },
        "snapshots": snapshots,
    }


def authoritative_record(
    *,
    task_id: str,
    algorithm: str,
    bundle: str,
    bundle_sha: str,
    source_paths: list[str | None],
    source_shas: list[str | None],
    trajectory: list[str],
) -> dict:
    return {
        "logical_key": f"{algorithm}::dataset_{task_id}::s520::clean",
        "task_id": task_id,
        "condition": "clean",
        "algorithm": algorithm,
        "dataset_id": f"dataset_{task_id}",
        "seed": 520,
        "host": "fixture",
        "bundle_path": bundle,
        "bundle_sha256": bundle_sha,
        "source_path": source_paths,
        "source_sha256": source_shas,
        "trajectory_source": trajectory,
        "incumbent_source_minute": [1, 1],
        "valid_output": [True, True],
    }


def read_tar_zst(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if member.isfile():
                        extracted = archive.extractfile(member)
                        assert extracted is not None
                        result[member.name] = extracted.read()
    return result


def test_four_readers_preserve_raw_bytes_and_validate_sha(tmp_path: Path) -> None:
    task_path = tmp_path / "task.jsonl.gz"
    write_jsonl_gz(task_path, [task_freeze_record("task_a", tmp_path / "remote_a", ["a1", "a2"])])
    task_objects = list(iter_task_freeze(SourceSpec("task_freeze", task_path)))
    assert [item.record_type for item in task_objects] == ["result", "snapshot", "snapshot"]

    flat_raw = raw_json("b1", 1)
    flat_path = tmp_path / "flat.jsonl.gz"
    write_jsonl_gz(
        flat_path,
        [{
            "task_id": "task_b", "record_type": "snapshot", "minute": 1,
            "source_path": "/remote/b/minute_0001.json", "sha256": digest(flat_raw),
            "raw_text": flat_raw.decode(),
        }],
    )
    assert list(iter_flat_raw_bundle(SourceSpec("flat_raw", flat_path)))[0].raw == flat_raw

    symbol_path = tmp_path / "task_c.jsonl.gz"
    symbol_lines = [json.loads(raw_json("c1", 1)), json.loads(raw_json("c2", 2))]
    write_jsonl_gz(symbol_path, symbol_lines)
    symbol_objects = list(iter_symbolfit_trajectory(SourceSpec("symbolfit", symbol_path)))
    assert [item.minute for item in symbol_objects] == [1, 2]

    progress_raw = b'{"updates":[]}'
    candidate_raw = b'{"best":"x"}'
    recovery_path = tmp_path / "recovery.jsonl.gz"
    write_jsonl_gz(
        recovery_path,
        [{
            "status": "ok", "source": {"task_id": "task_d"},
            "progress_path": "/remote/d/progress.json",
            "progress_sha256": digest(progress_raw), "progress_raw_text": progress_raw.decode(),
            "candidates": [{
                "path": "/remote/d/best_sample_1.json", "sha256": digest(candidate_raw),
                "raw_text": candidate_raw.decode(),
            }],
        }],
    )
    recovery_objects = list(iter_recovery_evidence(SourceSpec("recovery", recovery_path)))
    assert [item.raw for item in recovery_objects] == [progress_raw, candidate_raw]

    bad_path = tmp_path / "bad.jsonl.gz"
    write_jsonl_gz(
        bad_path,
        [{
            "task_id": "bad", "record_type": "snapshot", "minute": 1,
            "source_path": "/bad", "sha256": "0" * 64, "raw_text": "not-zero-sha",
        }],
    )
    with pytest.raises(ArchiveBuildError, match="SHA256 mismatch"):
        list(iter_flat_raw_bundle(SourceSpec("flat_raw", bad_path)))


def test_build_archive_binds_carry_forward_without_fabricating_it(tmp_path: Path) -> None:
    repo = tmp_path
    release = repo / "release"
    release.mkdir()

    task_path = repo / "task_freeze.jsonl.gz"
    task_record = task_freeze_record("task_a", Path("/remote/a"), ["a1", "a2"])
    write_jsonl_gz(task_path, [task_record])
    a1 = task_record["snapshots"][0]

    flat_path = repo / "flat.jsonl.gz"
    b1, b2 = raw_json("b1", 1), raw_json("b2", 2)
    write_jsonl_gz(
        flat_path,
        [
            {"task_id": "task_b", "record_type": "snapshot", "minute": minute,
             "source_path": f"/remote/b/minute_{minute:04d}.json", "sha256": digest(raw),
             "raw_text": raw.decode()}
            for minute, raw in ((1, b1), (2, b2))
        ],
    )

    symbol_path = repo / "task_c.jsonl.gz"
    c1, c2 = raw_json("c1", 1), raw_json("c2", 2)
    write_jsonl_gz(symbol_path, [json.loads(c1), json.loads(c2)])
    # write_jsonl_gz canonicalizes these lines; use their exact line hashes.
    symbol_raw = [
        json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode()
        for raw in (c1, c2)
    ]

    progress_raw, candidate_raw = b'{"updates":[]}', b'{"best":"x"}'
    recovery_path = repo / "recovery.jsonl.gz"
    write_jsonl_gz(
        recovery_path,
        [{
            "status": "ok", "source": {"task_id": "task_d"},
            "progress_path": "/remote/d/progress.json",
            "progress_sha256": digest(progress_raw), "progress_raw_text": progress_raw.decode(),
            "candidates": [{
                "path": "/remote/d/best_sample_1.json", "sha256": digest(candidate_raw),
                "raw_text": candidate_raw.decode(),
            }],
        }],
    )

    authority = repo / "authority.jsonl.gz"
    records = [
        authoritative_record(
            task_id="task_a", algorithm="alg_a", bundle=str(task_path),
            bundle_sha=digest(task_path.read_bytes()),
            source_paths=[a1["selected_path"], a1["selected_path"]],
            source_shas=[a1["selected_sha256"], a1["selected_sha256"]],
            trajectory=["native_incumbent:1", "native_carry_forward:1"],
        ),
        authoritative_record(
            task_id="task_b", algorithm="alg_b", bundle=str(flat_path),
            bundle_sha=digest(flat_path.read_bytes()),
            source_paths=["/remote/b/minute_0001.json", "/remote/b/minute_0002.json"],
            source_shas=[digest(b1), digest(b2)],
            trajectory=["native_incumbent:1", "native_incumbent:2"],
        ),
        authoritative_record(
            task_id="task_c", algorithm="SymbolFit", bundle=str(symbol_path),
            bundle_sha=digest(symbol_path.read_bytes()),
            source_paths=[str(symbol_path), str(symbol_path)],
            source_shas=[digest(symbol_raw[0]), digest(symbol_raw[1])],
            trajectory=["no_candidate", "symbolfit_active_pysr_hall_of_fame"],
        ),
        authoritative_record(
            task_id="task_d", algorithm="llmsr", bundle="/remote/d/progress.json",
            bundle_sha=digest(progress_raw),
            source_paths=["/remote/d/progress.json", "/remote/d/best_sample_1.json"],
            source_shas=[digest(progress_raw), digest(candidate_raw)],
            trajectory=["recovered_native_incumbent:1", "recovered_native_incumbent:2"],
        ),
    ]
    write_jsonl_gz(authority, records)

    output = repo / "archive.tar.zst"
    report = build_archive(
        repo_root=repo,
        release_root=release,
        output=output,
        authoritative_paths=[authority],
        source_specs=[
            SourceSpec("task_freeze", task_path),
            SourceSpec("flat_raw", flat_path),
            SourceSpec("symbolfit", symbol_path),
            SourceSpec("recovery", recovery_path),
        ],
        horizon=2,
        expected_runs=4,
        expected_historical_missing=0,
        compression_level=1,
    )
    assert report["runs"] == 4
    assert report["logical_minutes"] == 8
    assert report["carry_forward"] == 1
    assert report["remote_access_used"] is False

    members = read_tar_zst(output)
    assert "logical_minute_bindings.csv.gz" in members
    bindings = list(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(
        members["logical_minute_bindings.csv.gz"]
    )))))
    a_bindings = [row for row in bindings if row["task_id"] == "task_a"]
    assert a_bindings[0]["archive_member"].endswith("minute_0001.json")
    assert a_bindings[1]["archive_member"] == a_bindings[0]["archive_member"]
    # The genuine minute-2 source is archived, but the carry-forward binding never points to it.
    assert any(name.endswith("task_a/progress/minute_0002.json") for name in members)


def test_missing_required_raw_object_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "historical.jsonl.gz"
    missing_sha = "1" * 64
    write_jsonl_gz(
        source,
        [{
            "source": {"task_id": "task_missing"},
            "result": {"status": "missing"},
            "snapshots": [{
                "minute": 1, "status": "missing", "outer_path": "/remote/missing.json"
            }],
        }],
    )
    authority = tmp_path / "authority.jsonl.gz"
    record = {
        "logical_key": "alg::data::s520::clean", "task_id": "task_missing",
        "condition": "clean", "algorithm": "alg", "dataset_id": "data", "seed": 520,
        "host": "fixture", "bundle_path": str(source),
        "bundle_sha256": digest(source.read_bytes()),
        "source_path": ["/remote/missing.json"], "source_sha256": [missing_sha],
        "trajectory_source": ["native_incumbent:1"], "incumbent_source_minute": [1],
        "valid_output": [True],
    }
    write_jsonl_gz(authority, [record])
    with pytest.raises(ArchiveBuildError, match="required raw SHA not found"):
        build_archive(
            repo_root=tmp_path,
            release_root=tmp_path,
            output=tmp_path / "bad.tar.zst",
            authoritative_paths=[authority],
            source_specs=[SourceSpec("task_freeze", source, historical=True)],
            horizon=1,
            expected_runs=1,
            expected_historical_missing=1,
            compression_level=1,
        )


def test_build_archive_refuses_to_overwrite_existing_delivery(tmp_path: Path) -> None:
    output = tmp_path / "existing.tar.zst"
    output.write_bytes(b"keep")
    with pytest.raises(ArchiveBuildError, match="already exists"):
        build_archive(
            repo_root=tmp_path,
            release_root=tmp_path,
            output=output,
            authoritative_paths=[],
            source_specs=[],
            expected_runs=0,
            expected_historical_missing=0,
            compression_level=1,
        )
    assert output.read_bytes() == b"keep"


def test_same_task_id_in_different_bundles_uses_logical_key_identity(tmp_path: Path) -> None:
    first_bundle = tmp_path / "first.jsonl.gz"
    second_bundle = tmp_path / "second.jsonl.gz"
    first = task_freeze_record("shared_task", Path("/remote/first"), ["first-1", "first-2"])
    second = task_freeze_record("shared_task", Path("/remote/second"), ["second-1", "second-2"])
    write_jsonl_gz(first_bundle, [first])
    write_jsonl_gz(second_bundle, [second])

    first_authority = authoritative_record(
        task_id="shared_task",
        algorithm="first_alg",
        bundle=str(first_bundle),
        bundle_sha=digest(first_bundle.read_bytes()),
        source_paths=[item["selected_path"] for item in first["snapshots"]],
        source_shas=[item["selected_sha256"] for item in first["snapshots"]],
        trajectory=["native_incumbent:1", "native_incumbent:2"],
    )
    second_authority = authoritative_record(
        task_id="shared_task",
        algorithm="second_alg",
        bundle=str(second_bundle),
        bundle_sha=digest(second_bundle.read_bytes()),
        source_paths=[item["selected_path"] for item in second["snapshots"]],
        source_shas=[item["selected_sha256"] for item in second["snapshots"]],
        trajectory=["native_incumbent:1", "native_incumbent:2"],
    )
    authority = tmp_path / "authority.jsonl.gz"
    write_jsonl_gz(authority, [first_authority, second_authority])

    output = tmp_path / "duplicate-task-id.tar.zst"
    report = build_archive(
        repo_root=tmp_path,
        release_root=tmp_path,
        output=output,
        authoritative_paths=[authority],
        source_specs=[
            SourceSpec("task_freeze", first_bundle),
            SourceSpec("task_freeze", second_bundle),
        ],
        horizon=2,
        expected_runs=2,
        expected_historical_missing=0,
        compression_level=1,
    )
    assert report["runs"] == 2
    members = read_tar_zst(output)
    bindings = list(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(
        members["logical_minute_bindings.csv.gz"]
    )))))
    assert {row["logical_key"] for row in bindings} == {
        first_authority["logical_key"], second_authority["logical_key"]
    }
    assert any("/first_alg/" in name for name in members)
    assert any("/second_alg/" in name for name in members)


def test_recovery_prefers_source_logical_key_for_repeated_task_id(tmp_path: Path) -> None:
    shared_raw = b'{"updates":["same-content"]}'
    shared_sha = digest(shared_raw)
    first_key = "llmsr::first_dataset::s520::clean"
    second_key = "llmsr::second_dataset::s520::clean"
    first_path = "/remote/first/progress.json"
    second_path = "/remote/second/progress.json"
    recovery = tmp_path / "recovery.jsonl.gz"
    write_jsonl_gz(
        recovery,
        [
            {
                "status": "ok",
                "source": {"task_id": "shared_task", "logical_key": logical_key},
                "progress_path": source_path,
                "progress_sha256": shared_sha,
                "progress_raw_text": shared_raw.decode(),
                "candidates": [],
            }
            for logical_key, source_path in ((first_key, first_path), (second_key, second_path))
        ],
    )
    authority = tmp_path / "authority.jsonl.gz"
    write_jsonl_gz(
        authority,
        [
            {
                "logical_key": logical_key,
                "task_id": "shared_task",
                "condition": "clean",
                "algorithm": "llmsr",
                "dataset_id": dataset_id,
                "seed": 520,
                "host": "fixture",
                "bundle_path": source_path,
                "bundle_sha256": shared_sha,
                "source_path": [source_path],
                "source_sha256": [shared_sha],
                "trajectory_source": ["recovered_native_incumbent:1"],
                "incumbent_source_minute": [1],
                "valid_output": [True],
            }
            for logical_key, dataset_id, source_path in (
                (first_key, "first_dataset", first_path),
                (second_key, "second_dataset", second_path),
            )
        ],
    )

    output = tmp_path / "recovery-identity.tar.zst"
    report = build_archive(
        repo_root=tmp_path,
        release_root=tmp_path,
        output=output,
        authoritative_paths=[authority],
        source_specs=[SourceSpec("recovery", recovery)],
        horizon=1,
        expected_runs=2,
        expected_historical_missing=0,
        compression_level=1,
    )
    assert report["physical_records"] == 2
    members = read_tar_zst(output)
    assert any("/first_dataset/" in name for name in members)
    assert any("/second_dataset/" in name for name in members)


def test_stable_bundle_copy_can_match_authority_by_sha_and_task_id(tmp_path: Path) -> None:
    stable_copy = tmp_path / "stable-copy.jsonl.gz"
    record = task_freeze_record("copied_task", Path("/remote/copied"), ["one", "two"])
    write_jsonl_gz(stable_copy, [record])
    original_release_path = tmp_path / "older-release" / "original.jsonl.gz"
    authority_record = authoritative_record(
        task_id="copied_task",
        algorithm="copied_alg",
        bundle=str(original_release_path),
        bundle_sha=digest(stable_copy.read_bytes()),
        source_paths=[item["selected_path"] for item in record["snapshots"]],
        source_shas=[item["selected_sha256"] for item in record["snapshots"]],
        trajectory=["native_incumbent:1", "native_incumbent:2"],
    )
    authority = tmp_path / "authority.jsonl.gz"
    write_jsonl_gz(authority, [authority_record])

    output = tmp_path / "stable-copy.tar.zst"
    report = build_archive(
        repo_root=tmp_path,
        release_root=tmp_path,
        output=output,
        authoritative_paths=[authority],
        source_specs=[SourceSpec("task_freeze", stable_copy)],
        horizon=2,
        expected_runs=1,
        expected_historical_missing=0,
        compression_level=1,
    )
    assert report["runs"] == 1
    members = read_tar_zst(output)
    assert any(name.endswith("copied_task/progress/minute_0001.json") for name in members)
    source_manifest = list(csv.DictReader(io.StringIO(
        members["source_containers.csv"].decode()
    )))
    assert source_manifest[0]["path"] == str(stable_copy)
    assert source_manifest[0]["selected_run_count"] == "1"
