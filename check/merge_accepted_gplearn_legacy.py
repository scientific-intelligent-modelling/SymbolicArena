"""Fill current six-axis table with user-accepted historical gplearn symbolic scores.

The numerical axes remain bound to the latest selected terminal trajectories.
Historical gplearn symbolic scores are explicitly marked as an exception,
because many old formula inputs differ from current native terminal formulas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "AAAI_experiments/stage5_metric_calculation_0831/work/core50_terminal_collection_20260916_v3/current_terminal_six_axis_strict_v2"
OLD = ROOT / "AAAI_experiments/Core50_final_20260914/results"
CONDITIONS = ("clean", "noise001", "noise005")
AXES = ("ID", "OOD", "SYM", "MIN", "EFF", "STAB")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_row(current: dict[str, str], old: dict[str, str], *, old_sha: str) -> dict[str, str]:
    if current["algorithm"].casefold() != "gplearn" or old["algorithm"].casefold() != "gplearn":
        raise ValueError("legacy merge accepts only gplearn")
    if current["condition"] != old["condition"] or old["run_count"] != "150" or old["task_count"] != "50":
        raise ValueError("legacy gplearn condition/cohort mismatch")
    if not math.isclose(float(current["EFF"]), float(old["EFF"]), rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"gplearn EFF drift: {current['condition']}")
    if any(current[axis] for axis in ("SYM", "MIN", "STAB")):
        raise ValueError("current gplearn symbolic axes unexpectedly populated")
    if any(not old.get(axis) for axis in ("SYM", "MIN", "STAB")):
        raise ValueError("historical gplearn symbolic score missing")
    result = dict(current)
    for axis in ("SYM", "MIN", "STAB"):
        result[axis] = old[axis]
    result.update(
        score_complete="True", symbolic_source="accepted_legacy_gplearn_20260914",
        accepted_legacy_gplearn="True", legacy_symbolic_source_sha256=old_sha,
        terminal_binding_status="historical_formula_not_current_terminal",
    )
    return result


def merge(*, strict: Path, historical: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    rows = _read_csv(strict / "algorithm_six_axis.csv")
    if len(rows) != 45:
        raise ValueError(f"expected 45 algorithm-condition rows, found {len(rows)}")
    old_rows = {}
    old_sha = {}
    for condition in CONDITIONS:
        path = historical / condition / "six_axis.csv"
        old_sha[condition] = sha(path)
        matches = [row for row in _read_csv(path) if row["algorithm"].casefold() == "gplearn"]
        if len(matches) != 1 or matches[0]["condition"] != condition:
            raise ValueError(f"historical gplearn score missing: {condition}")
        old_rows[condition] = matches[0]
    merged = []
    for row in rows:
        if row["algorithm"].casefold() == "gplearn":
            merged.append(merge_row(row, old_rows[row["condition"]], old_sha=old_sha[row["condition"]]))
        else:
            if row["formal_ready"] != "True" or any(not row.get(axis) for axis in AXES):
                raise ValueError(f"non-gplearn six-axis row incomplete: {row['condition']}/{row['algorithm']}")
            merged.append({**row, "score_complete": "True", "symbolic_source": "current_terminal_frozen_judgments",
                           "accepted_legacy_gplearn": "False", "legacy_symbolic_source_sha256": "",
                           "terminal_binding_status": "current_terminal_bound"})
    output.mkdir(parents=True)
    path = output / "algorithm_six_axis_15alg.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    legacy_rows = []
    for condition in CONDITIONS:
        old = old_rows[condition]
        current = next(row for row in rows if row["condition"] == condition and row["algorithm"].casefold() == "gplearn")
        legacy_rows.append({
            "condition": condition, "source_six_axis_sha256": old_sha[condition],
            "old_ID": old["ID"], "old_OOD": old["OOD"], "old_EFF": old["EFF"],
            "current_ID": current["ID"], "current_OOD": current["OOD"], "current_EFF": current["EFF"],
            "accepted_SYM": old["SYM"], "accepted_MIN": old["MIN"], "accepted_STAB": old["STAB"],
            "historical_formula_mismatch_count": {"clean": 101, "noise001": 111, "noise005": 117}[condition],
        })
    with (output / "gplearn_legacy_acceptance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(legacy_rows[0]))
        writer.writeheader()
        writer.writerows(legacy_rows)
    manifest = {
        "schema_version": "current_six_axis_with_accepted_gplearn_legacy.v1",
        "algorithm_condition_rows": 45, "all_six_axis_numbers_present": True,
        "strict_current_terminal_formal_rows": 42, "user_accepted_legacy_rows": 3,
        "strict_all_15_formal_ready": False,
        "numerical_axes": "current_strict_scalar_native_replay",
        "non_gplearn_symbolic_axes": "current_terminal_frozen_judgments",
        "gplearn_symbolic_axes": "2026-09-14 historical release, accepted by user; 329/450 formulas differ from current terminal",
        "input_sha256": {"strict_algorithm_six_axis": sha(strict / "algorithm_six_axis.csv"),
                         **{f"legacy_{condition}": old_sha[condition] for condition in CONDITIONS}},
        "output_sha256": {path.name: sha(path),
                          "gplearn_legacy_acceptance.csv": sha(output / "gplearn_legacy_acceptance.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path, default=STRICT)
    parser.add_argument("--historical", type=Path, default=OLD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(strict=args.strict, historical=args.historical, output=args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
