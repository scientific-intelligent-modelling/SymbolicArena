"""Freeze selected gplearn native prefix endpoints for terminal-bound symbolic work."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "AAAI_experiments/stage5_metric_calculation_0831/work/core50_terminal_collection_20260916_v3"
STRICT = COLLECTION / "strict_scalar_v1_20260917"
GT = COLLECTION / "opus48_downstream_plan_v1/gt_reference_binding.jsonl"
CONDITIONS = ("clean", "noise001", "noise005")
PROTECTED = re.compile(r"\b(?:div|log|sqrt|inv)\(")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def bind_one(terminal: dict, snapshot_row: dict, selected_row: dict,
             native: dict, gt: dict) -> dict:
    key = terminal["logical_key"]
    require(all(row["logical_key"] == key for row in (snapshot_row, selected_row, native)),
            f"logical key drift: {key}")
    snapshot_text, selected_text = snapshot_row["raw_text"], selected_row["raw_text"]
    require(sha_text(snapshot_text) == terminal["terminal_source_sha256"],
            f"terminal snapshot SHA drift: {key}")
    require(sha_text(selected_text) == terminal["selected_result_sha256"],
            f"selected result SHA drift: {key}")
    snapshot, selected = json.loads(snapshot_text), json.loads(selected_text)
    artifact = snapshot.get("canonical_artifact") or {}
    prefix = snapshot.get("equation")
    require(isinstance(prefix, str) and prefix.strip() and
            artifact.get("raw_equation") == prefix and
            artifact.get("raw_equation_kind") in {"prefix_expression", "plain_expression"},
            f"native prefix source missing: {key}")
    require(terminal["terminal_expression"] == native["expression"][-1] and
            terminal["terminal_source_sha256"] == native["source_sha256"][-1] and
            terminal["minute180_valid_output"] is True and native["valid_output"][-1] is True,
            f"minute180 endpoint mismatch: {key}")
    require(terminal["feature_names"] == snapshot.get("feature_names") and
            gt["dataset_id"] == terminal["dataset_id"],
            f"dataset/variable binding mismatch: {key}")
    require(len(native["expression"]) == 180 and len(native["id_quality"]) == 180 and
            len(native["ood_quality"]) == 180,
            f"180-minute trajectory incomplete: {key}")
    prefix_hash = sha_text(prefix)
    old_prefix = selected.get("equation")
    return {
        "logical_key": key, "condition": terminal["condition"], "algorithm": "gplearn",
        "dataset_id": terminal["dataset_id"], "seed": int(terminal["seed"]),
        "task_id": terminal["task_id"], "feature_names": terminal["feature_names"],
        "variable_mapping": {f"X{i}": name for i, name in enumerate(terminal["feature_names"])},
        "terminal_expression": terminal["terminal_expression"],
        "terminal_expression_sha256": terminal["terminal_expression_sha256"],
        "native_prefix": prefix, "native_prefix_sha256": prefix_hash,
        "native_raw_equation_kind": artifact.get("raw_equation_kind"),
        "snapshot_canonical_artifact_sha256": sha_text(json.dumps(artifact, ensure_ascii=False,
                                                               sort_keys=True, separators=(",", ":"))),
        "terminal_snapshot_sha256": terminal["terminal_source_sha256"],
        "terminal_archive_member": terminal["terminal_archive_member"],
        "selected_result_sha256": terminal["selected_result_sha256"],
        "old_result_prefix_sha256": sha_text(old_prefix) if isinstance(old_prefix, str) else None,
        "old_result_prefix_matches_terminal": old_prefix == prefix,
        "native_source_loss": snapshot.get("source_loss"),
        "incumbent_source_minute": native["incumbent_source_minute"][-1],
        "minute180_id_quality": native["id_quality"][-1],
        "minute180_ood_quality": native["ood_quality"][-1],
        "contains_protected_operator": bool(PROTECTED.search(prefix)),
        "display_normalization_mode": artifact.get("normalization_mode"),
        "display_expression": artifact.get("normalized_expression"),
        "gt_frozen_evaluation_key": gt["gt_frozen_evaluation_key"],
        "gt_reference_expression": gt["fixed_reference_expression"],
        "gt_reference_source_sha256": gt["gt_source_evidence_sha256"],
        "symbolic_semantics_version": "gplearn_native_protected_prefix.v1",
    }


def build(collection: Path, strict: Path, gt_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    terminals = {row["logical_key"]: row for row in rows(strict / "terminal_inputs.jsonl.gz")
                 if row["algorithm_slug"].casefold() == "gplearn"}
    snapshots = {row["logical_key"]: row for row in rows(collection / "terminal_snapshot_payloads.jsonl.gz")
                 if row["algorithm"].casefold() == "gplearn"}
    selected = {row["logical_key"]: row for row in rows(collection / "selected_result_payloads.jsonl.gz")
                if row["algorithm"].casefold() == "gplearn"}
    native = {}
    for condition in CONDITIONS:
        path = strict / "numerical" / condition / "native_trajectories.jsonl.gz"
        for row in rows(path):
            if row["algorithm"].casefold() == "gplearn":
                native[row["logical_key"]] = row
    gt = {row["dataset_id"]: row for row in rows(gt_path)}
    require(len(terminals) == len(snapshots) == len(selected) == len(native) == 450,
            "gplearn 450-run grid incomplete")
    require(set(terminals) == set(snapshots) == set(selected) == set(native),
            "gplearn run keys differ across sources")
    require(len(gt) == 50, "GT reference grid incomplete")
    output.mkdir(parents=True)
    bound = []
    for key in sorted(terminals):
        term = terminals[key]
        bound.append(bind_one(term, snapshots[key], selected[key], native[key], gt[term["dataset_id"]]))
    with (output / "gplearn_terminal_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for row in bound:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    counts = Counter(row["condition"] for row in bound)
    require(counts == {condition: 150 for condition in CONDITIONS},
            f"gplearn condition counts wrong: {counts}")
    report = {
        "schema_version": "gplearn_terminal_inventory.v1", "run_count": len(bound),
        "conditions": dict(counts),
        "old_result_prefix_mismatch": sum(not row["old_result_prefix_matches_terminal"] for row in bound),
        "protected_operator_runs": sum(row["contains_protected_operator"] for row in bound),
        "display_nan_but_prefix_present": sum(row["display_expression"] == "nan" for row in bound),
        "source_sha256": {
            "terminal_inputs": sha_file(strict / "terminal_inputs.jsonl.gz"),
            "terminal_snapshot_payloads": sha_file(collection / "terminal_snapshot_payloads.jsonl.gz"),
            "selected_result_payloads": sha_file(collection / "selected_result_payloads.jsonl.gz"),
            "gt_references": sha_file(gt_path),
            **{f"native_{condition}": sha_file(strict / "numerical" / condition /
                                               "native_trajectories.jsonl.gz") for condition in CONDITIONS},
        },
        "output_sha256": sha_file(output / "gplearn_terminal_inventory.jsonl"),
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    parser.add_argument("--strict", type=Path, default=STRICT)
    parser.add_argument("--gt", type=Path, default=GT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.collection, args.strict, args.gt, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
