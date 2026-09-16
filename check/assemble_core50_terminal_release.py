"""Assemble a self-contained, terminal-consistent Core-50 evidence release.

This is a collection step, not symbolic scoring. Historical Opus judgments are
packaged only as a cache and never promoted to active terminal judgments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
DEFAULT_COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
DEFAULT_OLD = ROOT / "AAAI_experiments/Core50_final_20260914"
DEFAULT_RAW = ROOT / "AAAI_experiments/Core50_raw_snapshots_20260914.tar.zst"
DEFAULT_EFF = STAGE5 / "work/final_release_20260913/release_v2/eff_revision_v3"
DEFAULT_OUTPUT = ROOT / "AAAI_experiments/Core50_terminal_consistent_20260916"
CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = {520, 521, 522}
RAW_METADATA = (
    "runs.csv",
    "physical_records.csv.gz",
    "logical_minute_bindings.csv.gz",
    "physical_unavailable.csv",
    "source_containers.csv",
    "README.md",
    "report.json",
    "SHA256SUMS",
)


class ReleaseError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReleaseError(f"{path}:{line_number}: invalid JSON") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_collection(
    collection: Path,
    *,
    expected_runs: int = 6750,
    expected_supersessions: int = 624,
    expected_datasets: int = 50,
    conditions: tuple[str, ...] = CONDITIONS,
    seeds: set[int] = SEEDS,
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    manifest = json.loads((collection / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["outputs"].items():
        path = collection / name
        require(path.is_file(), f"collection output missing: {path}")
        require(sha256_file(path) == expected_hash, f"collection output SHA mismatch: {path}")

    selection = read_rows(collection / "latest_run_selection.csv")
    require(len(selection) == expected_runs, f"expected {expected_runs} selected runs, got {len(selection)}")
    selected: dict[str, dict] = {}
    for row in selection:
        key = row["logical_key"]
        require(key not in selected, f"duplicate selection key: {key}")
        require(row["condition"] in conditions, f"unexpected condition: {key}")
        require(int(row["seed"]) in seeds, f"unexpected seed: {key}")
        require(row["selected_result_sha256"], f"selected result SHA missing: {key}")
        selected[key] = row
    require(len({r["dataset_id"] for r in selection}) == expected_datasets, "dataset count mismatch")
    # SymbolFit display casing differs between clean and noisy source batches.
    require(len({r["algorithm_slug"] for r in selection}) == expected_runs // (len(conditions) * expected_datasets * len(seeds)), "algorithm count mismatch")
    require(
        Counter(r["condition"] for r in selection)
        == Counter({c: expected_runs // len(conditions) for c in conditions}),
        "condition run counts mismatch",
    )
    supersessions = read_rows(collection / "supersessions.csv")
    superseded = {key for key, row in selected.items() if row["superseded"].lower() == "true"}
    require(len(superseded) == expected_supersessions, "selection supersession count mismatch")
    require(len(supersessions) == expected_supersessions, "supersession ledger count mismatch")
    require({r["logical_key"] for r in supersessions} == superseded, "supersession ledger keys mismatch")
    require(manifest["run_count"] == expected_runs, "collection manifest run count mismatch")
    require(manifest["supersession_count"] == expected_supersessions, "collection manifest supersession count mismatch")

    terminals: dict[str, dict] = {}
    for row in read_jsonl(collection / "terminal_inputs.jsonl.gz"):
        key = row["logical_key"]
        require(key in selected and key not in terminals, f"unexpected/duplicate terminal key: {key}")
        expression = row["terminal_expression"]
        expected_hash = hashlib.sha256(expression.encode("utf-8")).hexdigest() if expression is not None else None
        require(row["terminal_expression_sha256"] == expected_hash, f"terminal expression SHA mismatch: {key}")
        require(row["selected_result_sha256"] == selected[key]["selected_result_sha256"], f"terminal selected result mismatch: {key}")
        require(row["condition"] == selected[key]["condition"], f"terminal condition mismatch: {key}")
        require(row["dataset_id"] == selected[key]["dataset_id"], f"terminal dataset mismatch: {key}")
        require(int(row["seed"]) == int(selected[key]["seed"]), f"terminal seed mismatch: {key}")
        terminals[key] = row
    require(set(terminals) == set(selected), "terminal input keys differ from selection")
    return manifest, selected, terminals


def validate_payloads(collection: Path, selected: dict[str, dict], terminals: dict[str, dict]) -> dict[str, int]:
    counts = {}
    for kind, filename in (
        ("selected_result", "selected_result_payloads.jsonl.gz"),
        ("terminal_snapshot", "terminal_snapshot_payloads.jsonl.gz"),
    ):
        path = collection / filename
        require(path.is_file(), f"required payload file missing: {path}")
        seen = set()
        for row in read_jsonl(path):
            key = row["logical_key"]
            require(key in selected and key not in seen, f"unexpected/duplicate {kind} payload: {key}")
            seen.add(key)
            terminal = terminals[key]
            require(row["condition"] == terminal["condition"], f"{kind} condition mismatch: {key}")
            require(row["algorithm"] == terminal["algorithm"], f"{kind} algorithm mismatch: {key}")
            require(row["dataset_id"] == terminal["dataset_id"], f"{kind} dataset mismatch: {key}")
            require(int(row["seed"]) == int(terminal["seed"]), f"{kind} seed mismatch: {key}")
            require(row["selected_result_sha256"] == terminal["selected_result_sha256"], f"{kind} result SHA mismatch: {key}")
            require(row["terminal_expression"] == terminal["terminal_expression"], f"{kind} expression mismatch: {key}")
            require(row["terminal_expression_sha256"] == terminal["terminal_expression_sha256"], f"{kind} expression SHA mismatch: {key}")
            raw_text = row.get("raw_text")
            require(isinstance(raw_text, str), f"{kind} raw_text missing: {key}")
            raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if kind == "selected_result":
                require(row["result_sha256"] == raw_sha, f"selected result payload bytes mismatch: {key}")
                require(raw_sha == selected[key]["selected_result_sha256"], f"selected result payload not selected: {key}")
            else:
                require(row["terminal_source_sha256"] == raw_sha, f"terminal snapshot payload bytes mismatch: {key}")
                require(raw_sha == terminal["terminal_source_sha256"], f"terminal snapshot payload not selected: {key}")
        require(seen == set(selected), f"{kind} payload keys differ from selected runs")
        counts[kind] = len(seen)
    return counts


def _same_float(actual, expected, *, key: str, field: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
        f"{field} mismatch at minute 180: {key}",
    )


def validate_numerical(eff: Path, terminals: dict[str, dict], *, horizon: int = 180, conditions: tuple[str, ...] = CONDITIONS) -> dict:
    by_condition = {}
    for condition in conditions:
        keys = {key for key, row in terminals.items() if row["condition"] == condition}
        native_path = eff / condition / "native_trajectories.jsonl.gz"
        minute_path = eff / condition / "id_ood_eff_minute.csv.gz"
        native_seen = set()
        for row in read_jsonl(native_path):
            key = row["logical_key"]
            require(key in keys and key not in native_seen, f"unexpected/duplicate native trajectory: {key}")
            native_seen.add(key)
            for field in ("expression", "id_quality", "ood_quality", "valid_output"):
                require(len(row[field]) == horizon, f"native {field} horizon mismatch: {key}")
            terminal = terminals[key]
            require(row["expression"][-1] == terminal["terminal_expression"], f"native terminal expression mismatch: {key}")
            _same_float(row["id_quality"][-1], terminal["minute180_id_quality"], key=key, field="native ID")
            _same_float(row["ood_quality"][-1], terminal["minute180_ood_quality"], key=key, field="native OOD")
            require(bool(row["valid_output"][-1]) == bool(terminal["minute180_valid_output"]), f"native validity mismatch: {key}")
        require(native_seen == keys, f"native trajectories do not cover {condition}")

        minute_masks = {key: 0 for key in keys}
        line_count = 0
        with gzip.open(minute_path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["logical_key"]
                require(key in keys, f"unexpected numerical minute key: {key}")
                minute = int(row["minute"])
                require(1 <= minute <= horizon, f"invalid minute {minute}: {key}")
                bit = 1 << (minute - 1)
                require(not minute_masks[key] & bit, f"duplicate numerical minute {minute}: {key}")
                minute_masks[key] |= bit
                line_count += 1
                if minute == horizon:
                    terminal = terminals[key]
                    expression = row["expression"] or None
                    require(expression == terminal["terminal_expression"], f"CSV terminal expression mismatch: {key}")
                    _same_float(row["id_quality"], terminal["minute180_id_quality"], key=key, field="CSV ID")
                    _same_float(row["ood_quality"], terminal["minute180_ood_quality"], key=key, field="CSV OOD")
        complete_mask = (1 << horizon) - 1
        require(all(mask == complete_mask for mask in minute_masks.values()), f"numerical minute gap in {condition}")
        require(line_count == len(keys) * horizon, f"numerical minute count mismatch in {condition}")
        by_condition[condition] = {"runs": len(keys), "logical_minutes": line_count, "terminal_matches": len(keys)}
    return by_condition


def validate_ground_truth(old_release: Path, terminals: dict[str, dict], *, expected_datasets: int = 50) -> dict:
    references = read_rows(old_release / "ground_truth/current_references.csv")
    dataset_ids = {row["dataset_id"] for row in references}
    require(len(references) == expected_datasets and len(dataset_ids) == expected_datasets, "Ground Truth reference count mismatch")
    require(dataset_ids == {row["dataset_id"] for row in terminals.values()}, "Ground Truth task identities mismatch")
    metadata_count = sum(1 for _ in (old_release / "datasets").rglob("metadata.yaml"))
    require(metadata_count == expected_datasets, "packaged dataset metadata count mismatch")
    return {"references": len(references), "dataset_directories": metadata_count}


def _copy(source: Path, destination: Path) -> None:
    require(source.is_file(), f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _extract_raw_metadata(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for member in RAW_METADATA:
        with (destination / member).open("wb") as stream:
            completed = subprocess.run(
                ["tar", "--zstd", "-xOf", str(archive), member],
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
            )
        require(completed.returncode == 0, f"raw archive member {member}: {completed.stderr.decode(errors='replace')}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    require(rows, f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _collect_available_source_bundles(selection: dict[str, dict], output: Path) -> dict[str, int]:
    sources: dict[str, str] = {}
    for row in selection.values():
        path = row["trajectory_bundle_path"]
        expected_hash = row["trajectory_bundle_sha256"]
        require(path and expected_hash, f"trajectory source missing: {row['logical_key']}")
        if path in sources:
            require(sources[path] == expected_hash, f"trajectory source SHA conflict: {path}")
        sources[path] = expected_hash
    records = []
    copied_hashes = set()
    for source_path, expected_hash in sorted(sources.items()):
        path = Path(source_path)
        if not path.is_absolute():
            path = ROOT / path
        package_path = ""
        status = "original_path_unavailable"
        if path.is_file():
            require(sha256_file(path) == expected_hash, f"trajectory source SHA mismatch: {source_path}")
            package_path = f"source_bundles/{expected_hash[:2]}/{expected_hash}.blob"
            if expected_hash not in copied_hashes:
                _copy(path, output / package_path)
                copied_hashes.add(expected_hash)
            status = "copied_verified"
        records.append({
            "source_path": source_path,
            "source_sha256": expected_hash,
            "package_path": package_path,
            "status": status,
        })
    _write_csv(output / "source_bundle_index.csv", records)
    return {
        "referenced_unique_paths": len(records),
        "copied_verified_paths": sum(row["status"] == "copied_verified" for row in records),
        "original_paths_unavailable": sum(row["status"] == "original_path_unavailable" for row in records),
        "copied_unique_blobs": len(copied_hashes),
    }


def assemble_release(
    collection: Path,
    old_release: Path,
    raw_archive: Path,
    eff: Path,
    output: Path,
    *,
    zip_path: Path | None = None,
    expected_runs: int = 6750,
    expected_supersessions: int = 624,
    expected_datasets: int = 50,
    horizon: int = 180,
    conditions: tuple[str, ...] = CONDITIONS,
    seeds: set[int] = SEEDS,
) -> dict:
    require(not output.exists(), f"output already exists; refusing overwrite: {output}")
    if zip_path is not None:
        require(not zip_path.exists(), f"ZIP already exists; refusing overwrite: {zip_path}")
    collection_manifest, selected, terminals = validate_collection(
        collection,
        expected_runs=expected_runs,
        expected_supersessions=expected_supersessions,
        expected_datasets=expected_datasets,
        conditions=conditions,
        seeds=seeds,
    )
    payload_counts = validate_payloads(collection, selected, terminals)
    payload_report_path = collection / "selected_payloads_validation.json"
    require(payload_report_path.is_file(), "selected payload validation report missing")
    payload_report = json.loads(payload_report_path.read_text(encoding="utf-8"))
    require(payload_report.get("status") == "passed", "selected payload validation did not pass")
    require(payload_report.get("run_count") == expected_runs, "selected payload report run count mismatch")
    require(payload_report.get("terminal_snapshot_count") == expected_runs, "selected payload report snapshot count mismatch")
    numerical = validate_numerical(eff, terminals, horizon=horizon, conditions=conditions)
    ground_truth = validate_ground_truth(old_release, terminals, expected_datasets=expected_datasets)

    require(raw_archive.is_file(), f"raw archive missing: {raw_archive}")
    expected_raw_hash = collection_manifest.get("inputs", {}).get(str(raw_archive.resolve()))
    if expected_raw_hash:
        require(sha256_file(raw_archive) == expected_raw_hash, "raw archive SHA mismatch")
    for condition in conditions:
        native_path = eff / condition / "native_trajectories.jsonl.gz"
        expected_hash = collection_manifest.get("inputs", {}).get(str(native_path.resolve()))
        require(expected_hash is not None, f"collection manifest omits native trajectory SHA: {condition}")
        require(sha256_file(native_path) == expected_hash, f"native trajectory SHA mismatch: {condition}")
    raw_report = json.loads(Path(str(raw_archive) + ".report.json").read_text(encoding="utf-8"))
    require(raw_report["runs"] == expected_runs, "raw archive run count mismatch")
    require(raw_report["logical_minutes"] == expected_runs * horizon, "raw archive logical minute count mismatch")
    require(raw_report["archive_sha256"] == sha256_file(raw_archive), "raw archive report SHA mismatch")

    output.mkdir(parents=True)
    for name in collection_manifest["outputs"]:
        _copy(collection / name, output / "selection" / name)
    for name in (
        "selected_result_payloads.jsonl.gz",
        "terminal_snapshot_payloads.jsonl.gz",
        "selected_payloads_validation.json",
    ):
        source = collection / name
        if source.exists():
            _copy(source, output / "selection" / name)
    _copy(collection / "manifest.json", output / "selection/collection_manifest.json")
    for source in sorted((collection / "symbolic_preflight").glob("*")):
        if source.is_file():
            _copy(source, output / "symbolic_preflight" / source.name)

    shutil.copytree(old_release / "datasets", output / "datasets")
    shutil.copytree(old_release / "ground_truth", output / "ground_truth")
    bundle_status = _collect_available_source_bundles(selected, output)
    _copy(raw_archive, output / "historical_source_raw_snapshots" / raw_archive.name)
    _copy(Path(str(raw_archive) + ".report.json"), output / "historical_source_raw_snapshots" / "delivery_report.json")
    _extract_raw_metadata(raw_archive, output / "historical_source_raw_snapshots" / "archive_metadata")

    for condition in conditions:
        for name in (
            "id_ood_eff_minute.csv.gz",
            "native_trajectories.jsonl.gz",
            "algorithm_180min.csv",
            "run_status.csv",
            "unavailable.csv",
            "manifest.json",
        ):
            _copy(eff / condition / name, output / "numerical" / condition / name)
        for name in ("opus5_prediction.jsonl", "opus5_equivalence.jsonl", "opus5_structure.jsonl"):
            _copy(old_release / "results" / condition / name, output / "historical_cache" / condition / name)
    for name in ("manifest.json", "validation_report.json", "SHA256SUMS"):
        _copy(eff / name, output / "numerical" / name)

    (output / "historical_cache/README.md").write_text(
        "# Historical Opus judgments\n\nThese records are bound to the 2026-09-14 release, not to the selected terminal expressions. "
        "They are lookup candidates only. No record here is active without a new dependency-binding audit.\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Core-50 terminal-consistent evidence collection\n\n"
        "This release selects one run for each condition/algorithm/dataset/seed key. A superseded run is replaced, "
        "not added. `selection/terminal_inputs.jsonl.gz` and the two payload files bind the actual minute-180 "
        "expression to the selected result and physical terminal snapshot. `numerical/` contains all 180 logical "
        "minutes per run. `historical_source_raw_snapshots/` preserves the older physical archive and its binding "
        "tables for audit, not as the active run selector. `source_bundle_index.csv` records which original "
        "trajectory containers could be copied and verified locally; unavailable original paths are not silently "
        "declared present.\n\n"
        "**Status:** terminal_ready=true, formal_ready=false, minute_symbolic_ready=false, "
        "postprocessing_pending. Historical Opus judgments under `historical_cache/` are NOT active judgments. "
        "The raw archive report records physical gaps; logical carry-forward is not an assertion that a physical "
        "minute snapshot exists. The latest terminal expressions still need their matching symbolic postprocessing "
        "before SYM/MIN/STAB can be finalized.\n",
        encoding="utf-8",
    )
    packaged_files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    release_manifest = {
        "schema_version": "core50_terminal_consistent_release.v1",
        "status": "postprocessing_pending",
        "formal_ready": False,
        "terminal_ready": True,
        "minute_symbolic_ready": False,
        "run_count": len(selected),
        "supersession_count": expected_supersessions,
        "payload_counts": payload_counts,
        "numerical_validation": numerical,
        "ground_truth_validation": ground_truth,
        "source_bundle_status": bundle_status,
        "raw_physical_report": {
            "physical_records": raw_report["physical_records"],
            "logical_minutes": raw_report["logical_minutes"],
            "historical_missing": raw_report["historical_missing"],
        },
        "selection_policy": "one_selected_run_per_condition_algorithm_dataset_seed; terminal=native_minute_180",
        "historical_opus_is_active": False,
        "packaged_files_sha256": packaged_files,
    }
    (output / "manifest.json").write_text(json.dumps(release_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(f"{sha256_file(path)}  {path.relative_to(output)}\n")
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
    return release_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--old-release", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--raw-archive", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--eff-dir", type=Path, default=DEFAULT_EFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", type=Path, default=None)
    args = parser.parse_args()
    result = assemble_release(args.collection_dir, args.old_release, args.raw_archive, args.eff_dir, args.output_dir, zip_path=args.zip)
    print(json.dumps({k: result[k] for k in ("status", "run_count", "supersession_count", "terminal_ready", "formal_ready")}, indent=2))


if __name__ == "__main__":
    main()
