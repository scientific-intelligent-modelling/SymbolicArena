#!/usr/bin/env python3
"""Export raw selected results and terminal snapshots for the Stage4 collection.

The selected four-part run key is authoritative. An old formal result is only
used when that same unchanged run has no archived result payload.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "AAAI_experiments/stage5_metric_calculation_0831/work/core50_terminal_collection_20260916_v3"
ARCHIVE = ROOT / "AAAI_experiments/Core50_raw_snapshots_20260914.tar.zst"
FORMAL = ROOT / "AAAI_experiments/Core50_final_20260914/results"
CONDITIONS = ("clean", "noise001", "noise005")


def sha256_text(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def key_of(row: dict) -> tuple[str, str, str, int]:
    return (row["condition"], row["algorithm_slug"], row["dataset_id"], int(row["seed"]))


def indexed(rows, name: str) -> dict:
    result = {}
    for row in rows:
        key = key_of(row)
        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_inputs(collection: Path, formal: Path):
    with (collection / "latest_run_selection.csv").open(newline="", encoding="utf-8") as handle:
        selections = indexed(csv.DictReader(handle), "selection")
    terminals = indexed(read_jsonl_gz(collection / "terminal_inputs.jsonl.gz"), "terminal")
    if set(selections) != set(terminals):
        raise ValueError("Selection and terminal key sets differ")

    supplements = {}
    for row in read_jsonl_gz(collection / "supplemental_result_payloads.jsonl.gz"):
        logical_key = row["logical_key"]
        if logical_key in supplements:
            raise ValueError(f"Duplicate supplemental logical key: {logical_key}")
        supplements[logical_key] = row

    formal_rows = []
    for condition in CONDITIONS:
        for row in read_jsonl_gz(formal / condition / "raw_results.jsonl.gz"):
            source = row["source"]
            formal_rows.append({
                "condition": condition,
                "algorithm_slug": source["algorithm"].lower(),
                "dataset_id": source["dataset_id"],
                "seed": source["seed"],
                "task_id": source["task_id"],
                "source_path": source["path"],
                "result": row["result"],
            })
    return selections, terminals, supplements, indexed(formal_rows, "formal")


def collect_archive(archive: Path, wanted: dict[str, tuple[str, tuple]]) -> dict[str, str]:
    """Read a zstd tar once; do not resolve members by similar task names."""
    found = {}
    process = None
    if archive.name.endswith(".tar.zst"):
        process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        stream = process.stdout
    else:
        stream = archive.open("rb")
    try:
        with tarfile.open(fileobj=stream, mode="r|") as handle:
            for member in handle:
                if member.name not in wanted:
                    continue
                if member.name in found:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                if not member.isfile():
                    raise ValueError(f"Expected regular archive file: {member.name}")
                content = handle.extractfile(member)
                assert content is not None
                found[member.name] = content.read().decode("utf-8")
    except BaseException:
        if process is not None:
            process.terminate()
        raise
    finally:
        stream.close()
        if process is not None:
            stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            code = process.wait()
            if code and len(found) == len(wanted):
                raise RuntimeError(f"Cannot decompress {archive}: {stderr.strip()}")
    missing = set(wanted) - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing)} archive payloads; first: {sorted(missing)[0]}")
    return found


def checked_raw(raw_text: str, expected_sha: str, key: tuple, kind: str,
                require_identity: bool = True) -> dict:
    actual = sha256_text(raw_text)
    if actual != expected_sha:
        raise ValueError(f"{kind} SHA mismatch for {key}: expected {expected_sha}, got {actual}")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} is not a JSON object for {key}")
    if (require_identity or "dataset" in payload) and payload.get("dataset") != key[2]:
        raise ValueError(f"{kind} dataset/seed mismatch for {key}")
    if (require_identity or "seed" in payload) and int(payload.get("seed", -1)) != key[3]:
        raise ValueError(f"{kind} dataset/seed mismatch for {key}")
    if payload.get("condition") not in (None, key[0]):
        raise ValueError(f"{kind} condition mismatch for {key}")
    return payload


def build(collection: Path, archive: Path, formal: Path, expected_runs: int = 6750,
          expected_supersessions: int = 624) -> dict:
    selections, terminals, supplements, formal_rows = read_inputs(collection, formal)
    if len(selections) != expected_runs or set(selections) != set(formal_rows):
        raise ValueError(f"Expected {expected_runs} selected/formal keys; found {len(selections)}")
    supersessions = sum(str(row["superseded"]).lower() == "true" for row in selections.values())
    if supersessions != expected_supersessions:
        raise ValueError(f"Expected {expected_supersessions} supersessions, found {supersessions}")

    wanted = {}
    for key, selection in selections.items():
        terminal = terminals[key]
        if selection["logical_key"] != terminal["logical_key"] or selection["task_id"] != terminal["task_id"]:
            raise ValueError(f"Selection/terminal identity mismatch: {key}")
        member = selection["selected_result_archive_member"]
        if member:
            if member in wanted:
                raise ValueError(f"Archive result reused by multiple runs: {member}")
            wanted[member] = ("result", key)
        snapshot = terminal["terminal_archive_member"]
        if not snapshot or snapshot in wanted:
            raise ValueError(f"Missing or reused terminal snapshot member: {key}, {snapshot}")
        wanted[snapshot] = ("snapshot", key)
    archived = collect_archive(archive, wanted)

    result_rows = []
    snapshot_rows = []
    source_counts = Counter()
    used_supplements = set()
    for key in sorted(selections):
        selection = selections[key]
        terminal = terminals[key]
        selected_sha = selection["selected_result_sha256"]
        member = selection["selected_result_archive_member"]
        logical_key = selection["logical_key"]
        superseded = str(selection["superseded"]).lower() == "true"
        if member:
            raw_text = archived[member]
            payload_kind = "archive"
            locator = f"archive:{member}"
        elif superseded:
            supplement = supplements.get(logical_key)
            if supplement is None or supplement["result_sha256"] != selected_sha or supplement["result_source_path"] != selection["selected_result_source_path"]:
                raise ValueError(f"Missing/mismatched superseding supplement: {key}")
            raw_text = supplement["raw_text"]
            payload_kind = "supplement"
            locator = f"supplemental_result_payloads.jsonl.gz:{logical_key}"
            used_supplements.add(logical_key)
        else:
            old = formal_rows[key]
            if old["result"]["sha256"] != selected_sha or old["source_path"] != selection["selected_result_source_path"] or old["task_id"] != selection["task_id"]:
                raise ValueError(f"Formal fallback does not match selected run: {key}")
            raw_text = old["result"]["raw_text"]
            payload_kind = "formal_fallback"
            locator = f"formal_raw_results:{logical_key}"
        if locator != selection["selected_result_payload_locator"]:
            raise ValueError(f"Result locator mismatch: {key}: {locator}")
        checked_raw(raw_text, selected_sha, key, "result")
        source_counts[payload_kind] += 1

        common = {
            "condition": key[0], "algorithm": selection["algorithm"],
            "algorithm_slug": key[1], "dataset_id": key[2], "seed": key[3],
            "logical_key": logical_key, "selected_task_id": selection["task_id"],
            "selected_result_sha256": selected_sha,
            "terminal_expression": terminal["terminal_expression"],
            "terminal_expression_sha256": terminal["terminal_expression_sha256"],
        }
        expression = common["terminal_expression"]
        actual_expression_sha = sha256_text(expression) if expression is not None else None
        if actual_expression_sha != common["terminal_expression_sha256"]:
            raise ValueError(f"Terminal expression SHA mismatch: {key}")
        result_rows.append({
            **common, "payload_source_kind": payload_kind, "payload_locator": locator,
            "result_source_path": selection["selected_result_source_path"],
            "result_sha256": selected_sha, "raw_text": raw_text,
        })

        snapshot_member = terminal["terminal_archive_member"]
        snapshot_raw = archived[snapshot_member]
        checked_raw(snapshot_raw, terminal["terminal_source_sha256"], key,
                    "terminal snapshot", require_identity=False)
        snapshot_rows.append({
            **common, "archive_member": snapshot_member,
            "payload_locator": f"archive:{snapshot_member}",
            "terminal_source_path": terminal["terminal_source_path"],
            "terminal_source_sha256": terminal["terminal_source_sha256"],
            "terminal_incumbent_source_minute": terminal["terminal_incumbent_source_minute"],
            "raw_text": snapshot_raw,
        })

    if used_supplements != set(supplements):
        raise ValueError(f"Unused supplemental payloads: {sorted(set(supplements) - used_supplements)}")
    if len(result_rows) != expected_runs or len(snapshot_rows) != expected_runs:
        raise ValueError("Incomplete selected result or terminal snapshot export")

    outputs = {
        "selected_result_payloads.jsonl.gz": result_rows,
        "terminal_snapshot_payloads.jsonl.gz": snapshot_rows,
    }
    for name in outputs:
        if (collection / name).exists():
            raise FileExistsError(f"Refusing to overwrite {collection / name}")
    for name, rows in outputs.items():
        with gzip.open(collection / name, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "core50_selected_result_payloads.v1",
        "run_count": len(result_rows),
        "terminal_snapshot_count": len(snapshot_rows),
        "supersession_count": supersessions,
        "result_payload_source_counts": dict(source_counts),
        "result_sha_mismatch_count": 0,
        "terminal_snapshot_sha_mismatch_count": 0,
        "status": "passed",
    }
    report_path = collection / "selected_payloads_validation.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--formal", type=Path, default=FORMAL)
    parser.add_argument("--expected-runs", type=int, default=6750)
    parser.add_argument("--expected-supersessions", type=int, default=624)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(args.collection, args.archive, args.formal, args.expected_runs,
                           args.expected_supersessions), ensure_ascii=False, indent=2, sort_keys=True))
