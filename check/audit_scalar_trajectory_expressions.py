"""Audit non-pointwise feature indexing and non-finite formulas in native trajectories."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re


CONDITIONS = ("clean", "noise001", "noise005")
FEATURE = re.compile(r"x\d+$")
NONFINITE = {"nan", "inf", "infinity"}


@lru_cache(maxsize=100_000)
def classify(expression: str | None) -> tuple[str, ...]:
    if expression is None or not expression.strip():
        return ()
    text = expression.strip()
    flags: set[str] = set()
    if text.lower() in NONFINITE | {"-inf", "-infinity"}:
        flags.add("nonfinite_literal")
    if "[" not in text and not flags and not any(name in text.lower() for name in NONFINITE):
        return ()
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        # Syntax errors in a Python-function payload are a separate replay issue.
        return tuple(sorted(flags | {"unparseable_suspect"}))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if FEATURE.fullmatch(node.value.id):
                flags.add("indexed_feature_column")
        elif isinstance(node, ast.Name) and node.id.lower() in NONFINITE:
            flags.add("nonfinite_literal")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in {"np", "numpy", "math"} and node.attr.lower() in NONFINITE:
                flags.add("nonfinite_literal")
    return tuple(sorted(flags))


def audit(numerical_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    minute_rows = []
    by_run: dict[tuple[str, str, str, int], dict] = {}
    source_hashes = {}
    for condition in CONDITIONS:
        path = numerical_dir / condition / "native_trajectories.jsonl.gz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[condition] = digest
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                run = json.loads(line)
                key = (condition, str(run["algorithm"]).casefold(), run["dataset_id"], int(run["seed"]))
                for minute, expression in enumerate(run["expression"], 1):
                    flags = classify(expression)
                    if not flags:
                        continue
                    id_quality = float(run["id_quality"][minute - 1])
                    ood_quality = float(run["ood_quality"][minute - 1])
                    row = {"condition": condition, "algorithm": run["algorithm"],
                           "dataset_id": run["dataset_id"], "seed": int(run["seed"]),
                           "minute": minute, "flags": ";".join(flags),
                           "id_quality": id_quality, "ood_quality": ood_quality,
                           "valid_output": bool(run["valid_output"][minute - 1]),
                           "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                           "expression": expression}
                    minute_rows.append(row)
                    record = by_run.setdefault(key, {
                        "condition": condition, "algorithm": run["algorithm"],
                        "dataset_id": run["dataset_id"], "seed": int(run["seed"]),
                        "affected_minutes": 0, "nonzero_quality_minutes": 0,
                        "terminal_affected": False, "flags": set(),
                    })
                    record["affected_minutes"] += 1
                    record["nonzero_quality_minutes"] += int(id_quality > 0 or ood_quality > 0)
                    record["terminal_affected"] |= minute == 180
                    record["flags"].update(flags)
    run_rows = [{**row, "flags": ";".join(sorted(row["flags"]))}
                for row in sorted(by_run.values(), key=lambda x: (
                    x["condition"], str(x["algorithm"]).casefold(), x["dataset_id"], x["seed"]))]
    for name, rows in (("affected_minutes.csv", minute_rows), ("affected_runs.csv", run_rows)):
        path = output / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    counts = Counter(flag for row in minute_rows for flag in row["flags"].split(";"))
    summary = {
        "schema_version": "scalar_trajectory_expression_audit.v1",
        "source_sha256": source_hashes, "affected_minutes": len(minute_rows),
        "affected_runs": len(run_rows), "terminal_affected_runs": sum(row["terminal_affected"] for row in run_rows),
        "nonzero_quality_minutes": sum(row["nonzero_quality_minutes"] for row in run_rows),
        "flag_counts": dict(counts),
        "note": "Read-only audit. gplearn display formulas may use protected-operator semantics, so a nonfinite display string is not by itself an invalid native artifact. No numerical value or validity flag has been changed.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.numerical_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
