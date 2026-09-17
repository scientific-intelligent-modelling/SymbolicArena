"""Versioned postprocessing for non-scalar feature indexing and nonfinite formulas.

Native incumbents and their snapshot identities are not changed. Only evaluation
validity and ID/OOD/EFF quality are corrected; gplearn's protected native
semantics are explicitly left untouched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from check.audit_scalar_trajectory_expressions import classify


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
NUMERICAL = STAGE5 / "work/final_release_20260913/release_v2/eff_revision_v3"
JUDGMENTS = COLLECTION / "current_terminal_judgments_v1"
CONDITIONS = ("clean", "noise001", "noise005")
INVALID_FLAGS = {"indexed_feature_column", "nonfinite_literal"}


class StrictScalarError(ValueError):
    pass


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise StrictScalarError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def correct_trajectory(run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expressions = run["expression"]
    horizon = len(expressions)
    for field in ("id_quality", "ood_quality", "quality", "valid_output"):
        require(len(run[field]) == horizon, f"{run['logical_key']} {field} length mismatch")
    if str(run["algorithm"]).casefold() == "gplearn":
        return run, []
    corrected = dict(run)
    for field in ("id_quality", "ood_quality", "quality", "valid_output"):
        corrected[field] = list(run[field])
    changes = []
    for index, expression in enumerate(expressions):
        flags = set(classify(expression))
        if "unparseable_suspect" in flags:
            raise StrictScalarError(f"{run['logical_key']} minute {index + 1}: unparseable suspect")
        if not flags.intersection(INVALID_FLAGS):
            continue
        old_id, old_ood = float(run["id_quality"][index]), float(run["ood_quality"][index])
        old_valid = bool(run["valid_output"][index])
        corrected["id_quality"][index] = 0.0
        corrected["ood_quality"][index] = 0.0
        corrected["quality"][index] = 0.0
        corrected["valid_output"][index] = False
        changes.append({
            "logical_key": run["logical_key"], "condition": run["condition"],
            "algorithm": run["algorithm"], "dataset_id": run["dataset_id"],
            "seed": int(run["seed"]), "minute": index + 1,
            "reason": ";".join(sorted(flags.intersection(INVALID_FLAGS))),
            "expression": expression,
            "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
            "source_sha256": run["source_sha256"][index],
            "native_valid_output": old_valid, "native_id_quality": old_id,
            "native_ood_quality": old_ood,
            "strict_valid_output": False, "strict_id_quality": 0.0,
            "strict_ood_quality": 0.0,
        })
    if changes:
        corrected["q_star"] = max(corrected["quality"], default=0.0)
        best = corrected["q_star"]
        corrected["m_eff"] = (sum(float(q) / best for q in corrected["quality"]) / horizon
                              if best > 0 else 0.0)
        for minute in range(horizon):
            combined = (float(corrected["id_quality"][minute]) +
                        float(corrected["ood_quality"][minute])) / 2
            require(math.isclose(combined, float(corrected["quality"][minute]),
                                 rel_tol=0, abs_tol=1e-12),
                    f"{run['logical_key']} minute {minute + 1}: quality mismatch")
            if not corrected["valid_output"][minute]:
                require(combined == 0.0, f"{run['logical_key']} invalid minute has nonzero quality")
    return corrected, changes


def _curve(run: dict[str, Any]) -> tuple[list[float], list[float]]:
    best = float(run["q_star"])
    relative = [float(q) / best if best > 0 else 0.0 for q in run["quality"]]
    cumulative = []
    total = 0.0
    for minute, value in enumerate(relative, 1):
        total += value
        cumulative.append(total / minute)
    return relative, cumulative


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_numerical_condition(source: Path, output: Path, condition: str):
    source_dir, target = source / condition, output / "numerical" / condition
    target.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, dict[str, Any]] = {}
    changes: list[dict[str, Any]] = []
    original_count = 0
    with gzip.open(target / "native_trajectories.jsonl.gz", "wt", encoding="utf-8") as handle:
        for run in read_jsonl(source_dir / "native_trajectories.jsonl.gz"):
            original_count += 1
            corrected, items = correct_trajectory(run)
            if items:
                overrides[run["logical_key"]] = corrected
                changes.extend(items)
            handle.write(json.dumps(corrected, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    require(original_count == 2250, f"{condition}: expected 2250 native runs")
    affected_algorithms = {str(run["algorithm"]).casefold() for run in overrides.values()}
    sums: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, int], int] = defaultdict(int)
    seen = defaultdict(set)
    source_minute = source_dir / "id_ood_eff_minute.csv.gz"
    with gzip.open(source_minute, "rt", encoding="utf-8", newline="") as original, gzip.open(
        target / "id_ood_eff_minute.csv.gz", "wt", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(original)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            key, minute = row["logical_key"], int(row["minute"])
            if key in overrides:
                run = overrides[key]
                index = minute - 1
                require(row["expression"] == (run["expression"][index] or "") and
                        row["source_sha256"] == run["source_sha256"][index],
                        f"minute source/formula drift: {key}/{minute}")
                relative, cumulative = _curve(run)
                row.update(id_quality=str(run["id_quality"][index]),
                           ood_quality=str(run["ood_quality"][index]),
                           quality=str(run["quality"][index]),
                           valid_output="true" if run["valid_output"][index] else "false",
                           relative_progress=str(relative[index]),
                           cumulative_eff=str(cumulative[index]))
                seen[key].add(minute)
            algorithm = row["algorithm"].casefold()
            if algorithm in affected_algorithms:
                bucket = (algorithm, minute)
                counts[bucket] += 1
                for field in ("id_quality", "ood_quality", "quality", "relative_progress", "cumulative_eff"):
                    sums[bucket][field] += float(row[field])
            writer.writerow(row)
    require(all(len(seen[key]) == 180 for key in overrides),
            f"{condition}: affected minute grid is incomplete")
    for name in ("unavailable.csv",):
        shutil.copy2(source_dir / name, target / name)
    statuses = _read_csv(source_dir / "run_status.csv")
    for row in statuses:
        run = overrides.get(row["logical_key"])
        if run is not None:
            row.update(q_star=str(run["q_star"]), m_eff=str(run["m_eff"]),
                       eff_score=str(100 * run["m_eff"]))
    _write_csv(target / "run_status.csv", statuses)
    status_by_alg: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in statuses:
        status_by_alg[row["algorithm"].casefold()].append(row)
    summary_rows = _read_csv(source_dir / "algorithm_summary.csv")
    for row in summary_rows:
        algorithm = row["algorithm"].casefold()
        if algorithm in affected_algorithms:
            runs = status_by_alg[algorithm]
            require(len(runs) == 150, f"{condition}/{algorithm}: run count != 150")
            value = 100 * sum(float(run["m_eff"]) for run in runs) / 150
            row["EFF"] = str(value)
            row["diagnostic_available_run_eff"] = str(value)
    _write_csv(target / "algorithm_summary.csv", summary_rows)
    curve_rows = _read_csv(source_dir / "algorithm_180min.csv")
    for row in curve_rows:
        algorithm, minute = row["algorithm"].casefold(), int(row["minute"])
        if algorithm not in affected_algorithms:
            continue
        bucket = (algorithm, minute)
        require(counts[bucket] == 150, f"{condition}/{algorithm}/{minute}: minute count != 150")
        for output_field, source_field, scale in (
            ("mean_id_quality", "id_quality", 1),
            ("mean_ood_quality", "ood_quality", 1),
            ("mean_quality", "quality", 1),
            ("mean_relative_progress", "relative_progress", 1),
            ("cumulative_eff_score", "cumulative_eff", 100),
            ("diagnostic_available_mean_id_quality", "id_quality", 1),
            ("diagnostic_available_mean_ood_quality", "ood_quality", 1),
            ("diagnostic_available_mean_quality", "quality", 1),
            ("diagnostic_available_mean_relative_progress", "relative_progress", 1),
            ("diagnostic_available_cumulative_eff_score", "cumulative_eff", 100),
        ):
            row[output_field] = str(sums[bucket][source_field] * scale / 150)
    _write_csv(target / "algorithm_180min.csv", curve_rows)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "strict_scalar_native_eff_overlay.v1"
    manifest["upstream_manifest_sha256"] = sha(source_dir / "manifest.json")
    manifest["strict_scalar_override"] = {
        "affected_runs": len(overrides), "affected_minutes": len(changes),
        "algorithms": sorted(affected_algorithms), "gplearn_unchanged": True,
    }
    for name in ("native_trajectories.jsonl.gz", "id_ood_eff_minute.csv.gz",
                 "run_status.csv", "algorithm_summary.csv", "algorithm_180min.csv", "unavailable.csv"):
        path = target / name
        manifest["artifacts"][name] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha(path), "size_bytes": path.stat().st_size,
        }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return overrides, changes


def apply(*, source: Path, collection: Path, judgments: Path, output: Path) -> dict[str, Any]:
    source, collection, judgments, output = (
        source.resolve(), collection.resolve(), judgments.resolve(), output.resolve()
    )
    require(not output.exists(), f"output already exists: {output}")
    output.mkdir(parents=True)
    all_overrides: dict[str, dict[str, Any]] = {}
    all_changes: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        overrides, changes = _build_numerical_condition(source, output, condition)
        require(not set(overrides).intersection(all_overrides), "duplicate run override")
        all_overrides.update(overrides)
        all_changes.extend(changes)
    require(len(all_changes) == 373 and len(all_overrides) == 9,
            f"unexpected strict-scalar inventory: {len(all_changes)} minutes/{len(all_overrides)} runs")
    terminal_path = collection / "terminal_inputs.jsonl.gz"
    terminal_changed: set[tuple[str, str, str, int]] = set()
    terminal_rows = []
    for row in read_jsonl(terminal_path):
        run = all_overrides.get(row["logical_key"])
        if run is not None:
            require(row["terminal_expression"] == run["expression"][-1] and
                    row["terminal_source_sha256"] == run["source_sha256"][-1],
                    f"terminal formula/source drift: {row['logical_key']}")
            if (bool(row["minute180_valid_output"]) != bool(run["valid_output"][-1]) or
                not math.isclose(float(row["minute180_id_quality"]), float(run["id_quality"][-1]), abs_tol=1e-12) or
                not math.isclose(float(row["minute180_ood_quality"]), float(run["ood_quality"][-1]), abs_tol=1e-12)):
                row["native_minute180_valid_output"] = row["minute180_valid_output"]
                row["native_minute180_id_quality"] = row["minute180_id_quality"]
                row["native_minute180_ood_quality"] = row["minute180_ood_quality"]
                row["minute180_valid_output"] = bool(run["valid_output"][-1])
                row["minute180_id_quality"] = float(run["id_quality"][-1])
                row["minute180_ood_quality"] = float(run["ood_quality"][-1])
                row["evaluation_override"] = "strict_scalar_invalid_expression.v1"
                terminal_changed.add((row["condition"], row["algorithm_slug"].casefold(),
                                      row["dataset_id"], int(row["seed"])))
        terminal_rows.append(row)
    require(len(terminal_rows) == 6750 and len(terminal_changed) == 2,
            f"unexpected terminal override count: {len(terminal_changed)}")
    write_jsonl(output / "terminal_inputs.jsonl.gz", terminal_rows)
    bindings = []
    for row in read_jsonl(collection / "opus48_downstream_plan_v1/active_prediction_binding.jsonl"):
        key = (row["condition"], row["algorithm"].casefold(), row["dataset_id"], int(row["seed"]))
        if key in terminal_changed:
            require(row["valid_output"] is True, f"active binding was already invalid: {key}")
            row["native_valid_output"] = True
            row["valid_output"] = False
            row["processing_status"] = "invalid_final_output"
            row["evaluation_override"] = "strict_scalar_invalid_expression.v1"
        bindings.append(row)
    write_jsonl(output / "active_prediction_binding.jsonl", bindings)
    kept = {}
    omitted = {}
    for phase in ("equivalence", "structure"):
        kept_rows, omitted_rows = [], []
        for row in read_jsonl(judgments / f"{phase}_effective_index.jsonl"):
            prefix = (row["condition"], row["algorithm"].casefold(), row["dataset_id"])
            seeds = (int(row["seed"]),) if phase == "equivalence" else (
                int(row["seed_left"]), int(row["seed_right"]))
            (omitted_rows if any((*prefix, seed) in terminal_changed for seed in seeds)
             else kept_rows).append(row)
        kept[phase], omitted[phase] = len(kept_rows), len(omitted_rows)
        write_jsonl(output / f"{phase}_effective_index.jsonl", kept_rows)
        write_jsonl(output / f"{phase}_superseded_by_invalid_output.jsonl", omitted_rows)
    write_jsonl(output / "override_minutes.jsonl.gz", all_changes)
    write_jsonl(output / "invalid_terminal_keys.jsonl", (
        {"condition": c, "algorithm": a, "dataset_id": d, "seed": s,
         "reason": "strict_scalar_invalid_expression.v1"}
        for c, a, d, s in sorted(terminal_changed)))
    manifest = {
        "schema_version": "strict_scalar_terminal_policy.v1",
        "source_numerical_manifest_sha256": sha(source / "manifest.json"),
        "source_terminal_inputs_sha256": sha(terminal_path),
        "source_judgment_manifest_sha256": sha(judgments / "manifest.json"),
        "affected_runs": len(all_overrides), "affected_minutes": len(all_changes),
        "invalid_terminals": len(terminal_changed),
        "invalid_terminal_keys": [list(key) for key in sorted(terminal_changed)],
        "active_judgment_counts": kept, "superseded_judgment_counts": omitted,
        "gplearn_policy": "preserve_native_protected_operator_replay",
        "selection_policy": "preserve_native_incumbent_expression_and_source; strict_invalid_quality_zero",
        "outputs_sha256": {str(path.relative_to(output)): sha(path) for path in output.rglob("*")
                           if path.is_file() and path.name != "manifest.json"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical", type=Path, default=NUMERICAL)
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    parser.add_argument("--judgments", type=Path, default=JUDGMENTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply(source=args.numerical, collection=args.collection,
                           judgments=args.judgments, output=args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
