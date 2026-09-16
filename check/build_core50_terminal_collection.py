#!/usr/bin/env python3
"""Collect one auditable Stage4 terminal per run from the frozen native trajectories.

This is a collection step, not a symbolic judgment or a formal metric release.
The selected reruns are explicit: the two EFF overlay manifests and the validated
SymbolFit clean batch supersede the old formal run with the same four-part key.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable


CONDITIONS = ("clean", "noise001", "noise005")
ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
DEFAULT_FORMAL = ROOT / "AAAI_experiments/Core50_final_20260914/results"
DEFAULT_TRAJECTORIES = STAGE5 / "work/final_release_20260913/release_v2/eff_revision_v3"
DEFAULT_ARCHIVE = ROOT / "AAAI_experiments/Core50_raw_snapshots_20260914.tar.zst"
DEFAULT_CLEAN_OVERLAY = (
    STAGE5
    / "work/final_release_20260913/release_v2/eff_native_missing165_20260913/validation/overlay_manifest.jsonl"
)
DEFAULT_NOISE_OVERLAY = (
    STAGE5
    / "work/final_release_20260913/release_v2/eff_native_noise_missing309_20260914/validation/overlay_manifest.jsonl"
)
DEFAULT_SYMBOLFIT_VALIDATION = (
    STAGE5
    / "reruns/symbolfit_internal_progress_v1/reports/full150_validation_strict180.json"
)
DEFAULT_SYMBOLFIT_STRICT = (
    STAGE5
    / "reruns/symbolfit_internal_progress_v1/reports/symbolfit_s521_clean_g0039_strict_rerun_validation.json"
)
DEFAULT_OUTPUT = STAGE5 / "work/core50_terminal_collection_20260916_v3"


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None


def key_of(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row["condition"]),
        str(row.get("algorithm_slug") or row["algorithm"]).lower(),
        str(row["dataset_id"]),
        int(row["seed"]),
    )


def logical_key(key: tuple[str, str, str, int], algorithm: str) -> str:
    condition, _, dataset_id, seed = key
    return f"{algorithm}::{dataset_id}::s{seed}::{condition}"


def indexed(rows: Iterable[dict], label: str) -> dict[tuple[str, str, str, int], dict]:
    result = {}
    for row in rows:
        key = key_of(row)
        if key in result:
            raise ValueError(f"Duplicate {label} key: {key}")
        result[key] = row
    return result


def read_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_tar_csv(archive: Path, member: str) -> Iterable[dict[str, str]]:
    command = ["tar"]
    if archive.name.endswith(".tar.zst"):
        command += ["-I", "zstd"]
    command += ["-xOf", str(archive), member]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        binary = gzip.GzipFile(fileobj=process.stdout) if member.endswith(".gz") else process.stdout
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
            yield from csv.DictReader(text)
    finally:
        stderr = process.stderr.read().decode("utf-8", "replace")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Cannot read {member} from {archive}: {stderr.strip()}")


def read_formal(formal_dir: Path) -> dict:
    records = []
    for condition in CONDITIONS:
        for item in read_jsonl_gz(formal_dir / condition / "raw_results.jsonl.gz"):
            source = item["source"]
            result = item["result"]
            payload = json.loads(result["raw_text"]) if result.get("raw_text") else {}
            records.append(
                {
                    "condition": condition,
                    "algorithm": source["algorithm"],
                    "dataset_id": source["dataset_id"],
                    "seed": source["seed"],
                    "task_id": source["task_id"],
                    "result_source_path": source["path"],
                    "result_sha256": result["sha256"],
                    "result_expression": result.get("payload_summary", {}).get("expression"),
                    "feature_names": payload.get("feature_names"),
                    "target_name": payload.get("target_name"),
                    "dataset_dir": payload.get("dataset_dir"),
                }
            )
    return indexed(records, "formal")


def read_trajectories(trajectory_dir: Path) -> dict:
    records = []
    for condition in CONDITIONS:
        path = trajectory_dir / condition / "native_trajectories.jsonl.gz"
        records.extend(read_jsonl_gz(path))
    return indexed(records, "native trajectory")


def read_overlays(paths: tuple[Path, Path], trajectories: dict) -> dict:
    overlays = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = key_of(item)
                trajectory = trajectories.get(key)
                if trajectory is None or trajectory["task_id"] != item["task_id"]:
                    raise ValueError(f"Overlay does not match a native trajectory: {key}")
                overlays.append(
                    {
                        "condition": key[0],
                        "algorithm": key[1],
                        "dataset_id": key[2],
                        "seed": key[3],
                        "selected_result_source_path": item["result_path"],
                        "selected_result_sha256": item["result_sha256"],
                        "source_bundle_path": item["bundle_path"],
                        "source_bundle_sha256": item["bundle_sha256"],
                        "selection_evidence_path": str(path),
                        "selection_evidence_sha256": sha256_file(path),
                        "selection_status": "eff_overlay_supersession",
                    }
                )
    return indexed(overlays, "EFF overlay")


def add_symbolfit(
    overlays: dict, validation_path: Path, strict_path: Path, trajectories: dict
) -> None:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    by_task_id = {
        trajectory["task_id"]: key
        for key, trajectory in trajectories.items()
        if key[0] == "clean" and key[1] == "symbolfit"
    }
    if len(by_task_id) != 150 or len(validation["tasks"]) != 150:
        raise ValueError("SymbolFit clean task inventory is not 150 unique tasks")
    strict_task_ids = set(validation["rerun_task_ids"])
    if len(strict_task_ids) != 1 or strict["status"] != "passed":
        raise ValueError("SymbolFit targeted rerun validation is incomplete")
    for task in validation["tasks"]:
        task_id = task["task_id"]
        if task_id not in by_task_id:
            raise ValueError(f"SymbolFit task has no native trajectory: {task_id}")
        key = by_task_id[task_id]
        if key in overlays:
            raise ValueError(f"SymbolFit task collides with EFF overlay: {key}")
        if task_id in strict_task_ids:
            if task["status"] != "requires_targeted_rerun":
                raise ValueError(f"Unexpected targeted rerun state: {task_id}")
            result_sha = strict["result_sha256"]
            evidence_path = strict_path
            selection_status = "symbolfit_targeted_rerun_supersession"
        else:
            if task["status"] != "passed":
                raise ValueError(f"Unvalidated SymbolFit task: {task_id}")
            result_sha = task["result_sha256"]
            evidence_path = validation_path
            selection_status = "symbolfit_validated_supersession"
        overlays[key] = {
            "condition": key[0],
            "algorithm": key[1],
            "dataset_id": key[2],
            "seed": key[3],
            "selected_result_source_path": None,
            "selected_result_sha256": result_sha,
            "source_bundle_path": None,
            "source_bundle_sha256": None,
            "selection_evidence_path": str(evidence_path),
            "selection_evidence_sha256": sha256_file(evidence_path),
            "selection_status": selection_status,
        }


def archive_indexes(archive: Path) -> tuple[dict, dict, dict, dict]:
    runs = indexed(read_tar_csv(archive, "runs.csv"), "archive run")
    minute180 = indexed(
        (row for row in read_tar_csv(archive, "logical_minute_bindings.csv.gz") if row["minute"] == "180"),
        "archive minute 180",
    )
    # 只保留需要核对的物理记录，不把百万条快照索引全部载入内存。
    result_records = {}
    terminal_members = {row["archive_member"] for row in minute180.values()}
    terminal_records = {}
    for row in read_tar_csv(archive, "physical_records.csv.gz"):
        if row["record_type"] == "result":
            key = key_of(row)
            if key in result_records:
                raise ValueError(f"Duplicate archived result: {key}")
            result_records[key] = row
        if row["archive_member"] in terminal_members:
            if row["archive_member"] in terminal_records:
                raise ValueError(f"Duplicate terminal archive member: {row['archive_member']}")
            terminal_records[row["archive_member"]] = row
    return runs, minute180, result_records, terminal_records


def recover_result_from_host_bundle(override: dict, task_id: str) -> dict | None:
    bundle_name = override.get("source_bundle_path")
    if not bundle_name:
        return None
    bundle = Path(bundle_name)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    if not bundle.is_file() or sha256_file(bundle) != override["source_bundle_sha256"]:
        return None
    matches = [
        item for item in read_jsonl_gz(bundle)
        if item.get("record_type") == "result"
        and item.get("task_id") == task_id
        and item.get("sha256") == override["selected_result_sha256"]
        and item.get("source_path") == override["selected_result_source_path"]
    ]
    if len(matches) != 1:
        return None
    item = matches[0]
    if sha256_text(item.get("raw_text")) != item["sha256"]:
        return None
    return {
        "source_bundle_path": str(bundle),
        "source_bundle_sha256": override["source_bundle_sha256"],
        "result_source_path": item["source_path"],
        "result_sha256": item["sha256"],
        "raw_text": item["raw_text"],
    }


def build(args: argparse.Namespace) -> dict:
    formal = read_formal(args.formal_dir)
    trajectories = read_trajectories(args.trajectory_dir)
    if len(formal) != args.expected_run_count or set(formal) != set(trajectories):
        raise ValueError(f"Formal/native key mismatch: {len(formal)} versus {len(trajectories)}")
    overlays = read_overlays((args.clean_overlay, args.noise_overlay), trajectories)
    add_symbolfit(overlays, args.symbolfit_validation, args.symbolfit_strict, trajectories)
    if len(overlays) != args.expected_supersession_count:
        raise ValueError(f"Expected {args.expected_supersession_count} supersessions, found {len(overlays)}")
    runs, minute180, result_records, terminal_records = archive_indexes(args.archive)
    if set(runs) != set(formal) or set(minute180) != set(formal):
        raise ValueError("Archive run/minute-180 keys differ from formal run keys")

    selections = []
    terminals = []
    unresolved = []
    supplemental_results = []
    statuses = Counter()
    for key in sorted(formal):
        old = formal[key]
        trajectory = trajectories[key]
        run = runs[key]
        binding = minute180[key]
        archived_result = result_records.get(key)
        override = overlays.get(key)
        selected_sha = override["selected_result_sha256"] if override else old["result_sha256"]
        selected_path = override["selected_result_source_path"] if override else old["result_source_path"]
        evidence_path = override["selection_evidence_path"] if override else str(args.formal_dir / key[0] / "raw_results.jsonl.gz")
        evidence_sha = override["selection_evidence_sha256"] if override else sha256_file(args.formal_dir / key[0] / "raw_results.jsonl.gz")
        archive_result_member = None
        if archived_result:
            archive_result_member = archived_result["archive_member"]
            selected_path = selected_path or archived_result["source_path"]
            if archived_result["sha256"] != selected_sha:
                unresolved.append({"logical_key": trajectory["logical_key"], "reason": "selected_result_sha_mismatch", "detail": f"selected={selected_sha}; archived={archived_result['sha256']}"})
        elif override:
            recovered = recover_result_from_host_bundle(override, trajectory["task_id"])
            if recovered is None:
                unresolved.append({"logical_key": trajectory["logical_key"], "reason": "superseding_result_payload_unavailable", "detail": selected_path or ""})
            else:
                supplemental_results.append({"logical_key": trajectory["logical_key"], **recovered})

        # 新运行合法替代同键旧运行时，旧 task_id 可以不同；只核对选中轨迹与归档。
        if trajectory["task_id"] != run["task_id"] or trajectory["task_id"] != binding["task_id"] or (
            archived_result and trajectory["task_id"] != archived_result["task_id"]
        ):
            unresolved.append({"logical_key": trajectory["logical_key"], "reason": "selected_task_identity_mismatch", "detail": "trajectory/archive task IDs differ"})
        if run["bundle_path"] != trajectory["bundle_path"] or run["bundle_sha256"] != trajectory["bundle_sha256"]:
            unresolved.append({"logical_key": trajectory["logical_key"], "reason": "trajectory_bundle_mismatch", "detail": "runs.csv does not match native trajectory"})

        arrays = ("expression", "source_path", "source_sha256", "valid_output", "id_quality", "ood_quality")
        if any(len(trajectory[name]) != 180 for name in arrays):
            raise ValueError(f"Native trajectory does not have 180 minutes: {key}")
        expression = trajectory["expression"][-1]
        source_path = trajectory["source_path"][-1]
        source_sha = trajectory["source_sha256"][-1]
        valid_output = trajectory["valid_output"][-1]
        member = binding["archive_member"]
        physical = terminal_records.get(member)
        if binding["binding_status"] != "bound" or not physical:
            source_status = "unresolved_terminal_archive_binding"
            unresolved.append({"logical_key": trajectory["logical_key"], "reason": source_status, "detail": member or ""})
        elif binding["source_path"] != source_path or binding["source_sha256"] != source_sha or physical["sha256"] != source_sha:
            source_status = "unresolved_terminal_source_mismatch"
            unresolved.append({"logical_key": trajectory["logical_key"], "reason": source_status, "detail": member})
        elif expression is None and valid_output:
            source_status = "unresolved_valid_terminal_without_expression"
            unresolved.append({"logical_key": trajectory["logical_key"], "reason": source_status, "detail": member})
        elif expression is None:
            source_status = "bound_invalid_no_expression"
        elif not valid_output:
            source_status = "bound_invalid_with_expression"
        else:
            source_status = "bound_valid"
        statuses[source_status] += 1

        selection = {
            "condition": key[0], "algorithm": old["algorithm"], "algorithm_slug": key[1],
            "dataset_id": key[2], "seed": key[3], "task_id": trajectory["task_id"],
            "logical_key": trajectory["logical_key"],
            "old_formal_task_id": old["task_id"],
            "selection_status": override["selection_status"] if override else "formal_raw_retained",
            "selected_result_source_path": selected_path,
            "selected_result_sha256": selected_sha,
            "selected_result_archive_member": archive_result_member,
            "selected_result_payload_locator": (
                f"archive:{archive_result_member}" if archive_result_member else
                f"supplemental_result_payloads.jsonl.gz:{trajectory['logical_key']}" if override and recovered is not None else
                f"formal_raw_results:{trajectory['logical_key']}" if not override else None
            ),
            "old_formal_result_source_path": old["result_source_path"],
            "old_formal_result_sha256": old["result_sha256"],
            "trajectory_bundle_path": trajectory["bundle_path"],
            "trajectory_bundle_sha256": trajectory["bundle_sha256"],
            "selection_evidence_path": evidence_path,
            "selection_evidence_sha256": evidence_sha,
            "superseded": bool(override),
        }
        selections.append(selection)
        terminals.append({
            **selection,
            "terminal_expression": expression,
            "terminal_expression_sha256": sha256_text(expression),
            "terminal_source_path": source_path,
            "terminal_source_sha256": source_sha,
            "terminal_archive_member": member,
            "terminal_trajectory_source": trajectory["trajectory_source"][-1],
            "terminal_incumbent_source_minute": trajectory["incumbent_source_minute"][-1],
            "feature_names": old["feature_names"],
            "feature_names_source": "old_formal_result_payload",
            "target_name": old["target_name"],
            "dataset_dir": old["dataset_dir"],
            "old_formal_expression": old["result_expression"],
            "minute180_valid_output": valid_output,
            "minute180_id_quality": trajectory["id_quality"][-1],
            "minute180_ood_quality": trajectory["ood_quality"][-1],
            "source_status": source_status,
        })

    output = args.output_dir
    expected_files = ["latest_run_selection.csv", "terminal_inputs.jsonl.gz", "supersessions.csv", "unresolved.csv", "supplemental_result_payloads.jsonl.gz", "manifest.json"]
    if any((output / name).exists() for name in expected_files):
        raise FileExistsError(f"Output already exists; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "latest_run_selection.csv", selections)
    write_csv(output / "supersessions.csv", [row for row in selections if row["superseded"]])
    write_csv(output / "unresolved.csv", unresolved, fields=("logical_key", "reason", "detail"))
    with gzip.open(output / "terminal_inputs.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in terminals:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with gzip.open(output / "supplemental_result_payloads.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in supplemental_results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "core50_terminal_collection.v1",
        "formal_ready": False,
        "terminal_policy": "native_trajectory_minute_180",
        "supersession_policy": "explicit_eff_overlays_and_validated_symbolfit_clean",
        "run_count": len(selections),
        "supersession_count": len(overlays),
        "supersessions_by_condition": dict(Counter(row["condition"] for row in selections if row["superseded"])),
        "terminal_source_status": dict(statuses),
        "supplemental_result_count": len(supplemental_results),
        "unresolved_count": len(unresolved),
        "unresolved_by_reason": dict(Counter(row["reason"] for row in unresolved)),
        "inputs": {str(path): sha256_file(path) for path in (
            args.archive, args.clean_overlay, args.noise_overlay,
            args.symbolfit_validation, args.symbolfit_strict,
            *(args.formal_dir / condition / "raw_results.jsonl.gz" for condition in CONDITIONS),
            *(args.trajectory_dir / condition / "native_trajectories.jsonl.gz" for condition in CONDITIONS),
        )},
        "outputs": {name: sha256_file(output / name) for name in expected_files if name != "manifest.json"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty CSV: {path}")
        fields = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--clean-overlay", type=Path, default=DEFAULT_CLEAN_OVERLAY)
    parser.add_argument("--noise-overlay", type=Path, default=DEFAULT_NOISE_OVERLAY)
    parser.add_argument("--symbolfit-validation", type=Path, default=DEFAULT_SYMBOLFIT_VALIDATION)
    parser.add_argument("--symbolfit-strict", type=Path, default=DEFAULT_SYMBOLFIT_STRICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-run-count", type=int, default=6750)
    parser.add_argument("--expected-supersession-count", type=int, default=624)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
