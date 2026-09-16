"""Focused tests for the read-only Stage4 terminal collection contract."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_core50_terminal_collection.py")
SPEC = importlib.util.spec_from_file_location("build_core50_terminal_collection", MODULE_PATH)
assert SPEC and SPEC.loader
collection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collection)


def test_overlays_are_explicit_supersessions(tmp_path: Path) -> None:
    first = tmp_path / "clean.jsonl"
    second = tmp_path / "noise.jsonl"
    first.write_text(
        json.dumps({
            "condition": "clean", "algorithm": "E2ESR", "dataset_id": "BPG3",
            "seed": 520, "task_id": "e2esr_s520_clean_g0005",
            "result_path": "/run/new/result.json", "result_sha256": "new-sha",
            "bundle_path": "bundle.jsonl.gz", "bundle_sha256": "bundle-sha",
        }) + "\n", encoding="utf-8",
    )
    second.write_text("", encoding="utf-8")
    key = ("clean", "e2esr", "BPG3", 520)
    result = collection.read_overlays(
        (first, second), {key: {"task_id": "e2esr_s520_clean_g0005"}}
    )
    assert result[key]["selected_result_sha256"] == "new-sha"
    assert result[key]["selection_status"] == "eff_overlay_supersession"


def test_symbolfit_targeted_rerun_overrides_failed_initial_validation(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.json"
    strict_path = tmp_path / "strict.json"
    tasks = []
    trajectories = {}
    for index in range(150):
        task_id = f"symbolfit_s520_clean_g{index:04d}"
        status = "requires_targeted_rerun" if index == 39 else "passed"
        tasks.append({"task_id": task_id, "status": status, "result_sha256": f"old-{index}"})
        trajectories[("clean", "symbolfit", f"d{index}", 520)] = {"task_id": task_id}
    validation_path.write_text(
        json.dumps({"tasks": tasks, "rerun_task_ids": [tasks[39]["task_id"]]}), encoding="utf-8"
    )
    strict_path.write_text(json.dumps({"status": "passed", "result_sha256": "strict-new"}), encoding="utf-8")
    overlays = {}
    collection.add_symbolfit(overlays, validation_path, strict_path, trajectories)
    assert len(overlays) == 150
    assert overlays[("clean", "symbolfit", "d39", 520)]["selected_result_sha256"] == "strict-new"
    assert overlays[("clean", "symbolfit", "d38", 520)]["selected_result_sha256"] == "old-38"


def test_archive_indexes_match_terminal_member_and_result(tmp_path: Path) -> None:
    archive = tmp_path / "snapshots.tar"
    key = {
        "logical_key": "e2esr::BPG3::s520::clean", "task_id": "e2esr_s520_clean_g0005",
        "condition": "clean", "algorithm": "e2esr", "dataset_id": "BPG3", "seed": "520",
    }

    def csv_bytes(rows: list[dict]) -> bytes:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode()

    member = "runs/clean/e2esr/BPG3/seed_520/task/progress/minute_0180.json"
    with tarfile.open(archive, "w") as tar:
        content = {
            "runs.csv": csv_bytes([{**key, "host": "host", "bundle_path": "bundle", "bundle_sha256": "bundle-sha"}]),
            "logical_minute_bindings.csv.gz": gzip.compress(csv_bytes([{
                **key, "minute": "180", "trajectory_source": "native", "incumbent_source_minute": "180",
                "valid_output": "true", "source_path": "source", "source_sha256": "source-sha",
                "archive_member": member, "binding_status": "bound",
            }])),
            "physical_records.csv.gz": gzip.compress(csv_bytes([
                {**key, "archive_member": member, "sha256": "source-sha", "size_bytes": "5",
                 "source_path": "source", "minute": "180", "record_type": "snapshot",
                 "evidence_kind": "snapshot", "container_path": "bundle"},
                {**key, "archive_member": "runs/clean/e2esr/BPG3/seed_520/task/result.json",
                 "sha256": "new-result", "size_bytes": "8", "source_path": "result-source",
                 "minute": "", "record_type": "result", "evidence_kind": "result",
                 "container_path": "bundle"},
            ])),
        }
        for name, data in content.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    runs, bindings, results, terminals = collection.archive_indexes(archive)
    four_key = ("clean", "e2esr", "BPG3", 520)
    assert runs[four_key]["bundle_sha256"] == "bundle-sha"
    assert bindings[four_key]["archive_member"] == member
    assert results[four_key]["sha256"] == "new-result"
    assert terminals[member]["sha256"] == "source-sha"


def test_duplicate_keys_fail_closed() -> None:
    row = {"condition": "clean", "algorithm": "E2ESR", "dataset_id": "BPG3", "seed": 520}
    with pytest.raises(ValueError, match="Duplicate"):
        collection.indexed([row, dict(row)], "test")


def test_missing_archived_result_recovers_only_exact_host_bundle_payload(tmp_path: Path) -> None:
    bundle = tmp_path / "host.jsonl.gz"
    raw_text = '{"feature_names": ["x"], "equation": "x"}'
    result_sha = collection.sha256_text(raw_text)
    with gzip.open(bundle, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "record_type": "result", "task_id": "llmsr_s522_noise005_g0038",
            "source_path": "/remote/result.json", "sha256": result_sha,
            "raw_text": raw_text,
        }) + "\n")
    override = {
        "source_bundle_path": str(bundle),
        "source_bundle_sha256": collection.sha256_file(bundle),
        "selected_result_source_path": "/remote/result.json",
        "selected_result_sha256": result_sha,
    }
    recovered = collection.recover_result_from_host_bundle(override, "llmsr_s522_noise005_g0038")
    assert recovered and recovered["raw_text"] == raw_text
    assert collection.recover_result_from_host_bundle(override, "other-task") is None
