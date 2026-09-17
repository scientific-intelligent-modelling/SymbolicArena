"""Publish strict terminal scores beside immutable raw evidence and legacy provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
PENDING = ROOT / "AAAI_experiments/Core50_terminal_evidence_20260917_audit_pending"
STRICT = COLLECTION / "strict_scalar_v1_20260917"
SIX_AXIS = COLLECTION / "current_terminal_six_axis_strict_v2"
MERGED = COLLECTION / "final_six_axis_15alg_strict_v2"
HISTORICAL = ROOT / "AAAI_experiments/Core50_final_20260914/results"
CONDITIONS = ("clean", "noise001", "noise005")
AXES = ("ID", "OOD", "SYM", "MIN", "EFF", "STAB")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def copy(source: Path, target: Path) -> None:
    require(source.is_file(), f"missing source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finalize(*, pending: Path, strict: Path, six_axis: Path, merged: Path,
             historical: Path, output: Path, zip_path: Path | None = None) -> dict:
    require(not output.exists(), f"output already exists: {output}")
    if zip_path is not None:
        require(not zip_path.exists(), f"ZIP already exists: {zip_path}")
    pending_manifest = json.loads((pending / "current_manifest.json").read_text(encoding="utf-8"))
    strict_manifest = json.loads((strict / "manifest.json").read_text(encoding="utf-8"))
    score_manifest = json.loads((six_axis / "manifest.json").read_text(encoding="utf-8"))
    merged_manifest = json.loads((merged / "manifest.json").read_text(encoding="utf-8"))
    require(pending_manifest["run_count"] == 6750 and pending_manifest["status"] == "symbolic_pending",
            "pending package identity mismatch")
    require(strict_manifest["invalid_terminals"] == 2 and strict_manifest["affected_minutes"] == 373,
            "strict scalar correction inventory mismatch")
    for name, expected in strict_manifest["outputs_sha256"].items():
        require(sha(strict / name) == expected, f"strict output SHA drift: {name}")
    require(score_manifest["unresolved_count"] == 1500 and score_manifest["run_counts"] ==
            {condition: 2250 for condition in CONDITIONS}, "strict score inventory mismatch")
    strict_input_paths = {
        "terminal_inputs": strict / "terminal_inputs.jsonl.gz",
        "prediction_bindings": strict / "active_prediction_binding.jsonl",
        "equivalence_bindings": strict / "equivalence_effective_index.jsonl",
        "structure_bindings": strict / "structure_effective_index.jsonl",
        "gt_bindings": COLLECTION / "opus48_downstream_plan_v1/gt_reference_binding.jsonl",
    }
    strict_input_paths.update({f"minute_{condition}": strict / "numerical" / condition /
                               "id_ood_eff_minute.csv.gz" for condition in CONDITIONS})
    for name, path in strict_input_paths.items():
        require(score_manifest["input_sha256"][name] == sha(path), f"score input drift: {name}")
    require(merged_manifest["all_six_axis_numbers_present"] is True and
            merged_manifest["strict_current_terminal_formal_rows"] == 42 and
            merged_manifest["user_accepted_legacy_rows"] == 3,
            "merged six-axis status mismatch")
    require(merged_manifest["input_sha256"]["strict_algorithm_six_axis"] ==
            sha(six_axis / "algorithm_six_axis.csv"), "merged strict score source drift")
    rows = _read_csv(merged / "algorithm_six_axis_15alg.csv")
    require(len(rows) == 45 and len({(r["condition"], r["algorithm"].casefold()) for r in rows}) == 45,
            "final score grid incomplete")
    require(all(r["score_complete"] == "True" and all(r[axis] for axis in AXES) for r in rows),
            "final score contains blanks")
    for condition in CONDITIONS:
        require(sum(r["condition"] == condition for r in rows) == 15,
                f"{condition}: expected 15 algorithm scores")

    shutil.copytree(pending, output)
    provenance = output / "provenance/pending_candidate"
    provenance.mkdir(parents=True)
    copy(output / "current_manifest.json", provenance / "current_manifest.json")
    copy(output / "CURRENT_STATUS.md", provenance / "CURRENT_STATUS.md")
    copy(output / "CURRENT_SHA256SUMS", provenance / "CURRENT_SHA256SUMS")
    # Old checksums describe the pending candidate; the final release gets new checksums.

    destination = output / "evaluation_strict"
    for name in ("terminal_inputs.jsonl.gz", "active_prediction_binding.jsonl",
                 "equivalence_effective_index.jsonl", "structure_effective_index.jsonl",
                 "invalid_terminal_keys.jsonl", "override_minutes.jsonl.gz",
                 "equivalence_superseded_by_invalid_output.jsonl",
                 "structure_superseded_by_invalid_output.jsonl", "manifest.json"):
        copy(strict / name, destination / name)
    shutil.copytree(strict / "numerical", destination / "numerical")
    for name in ("run_metrics.csv", "task_stability.csv", "algorithm_six_axis.csv",
                 "unresolved.csv", "manifest.json"):
        copy(six_axis / name, destination / "six_axis_current_terminal" / name)
    for name in ("algorithm_six_axis_15alg.csv", "gplearn_legacy_acceptance.csv", "manifest.json"):
        copy(merged / name, destination / "six_axis_with_accepted_legacy" / name)
    copy(merged / "algorithm_six_axis_15alg.csv", output / "algorithm_six_axis_15alg.csv")
    for condition in CONDITIONS:
        for name in ("six_axis.csv", "run_final.csv", "task_stability.csv"):
            copy(historical / condition / name,
                 destination / "gplearn_historical_source" / condition / name)
    for name in ("apply_strict_scalar_terminal_policy.py", "merge_accepted_gplearn_legacy.py",
                 "audit_scalar_trajectory_expressions.py"):
        copy(ROOT / "check" / name, output / "code" / name)

    manifest = {
        "schema_version": "core50_strict_terminal_with_accepted_gplearn_legacy.v1",
        "status": "six_axis_scores_complete_with_accepted_gplearn_legacy_exception",
        "run_count": 6750, "algorithms": 15, "conditions": list(CONDITIONS),
        "six_axis_score_rows": 45, "all_six_axis_numbers_present": True,
        "strict_current_terminal_formal_rows": 42, "accepted_historical_gplearn_rows": 3,
        "all_15_current_terminal_formal_ready": False,
        "minute_symbolic_required": False, "minute_symbolic_ready": False,
        "minute_numerical_ready": True,
        "invalid_terminal_count": 2, "strict_scalar_affected_minutes": 373,
        "gplearn_native_numerical_unchanged": True,
        "latest_run_selection": "selection/latest_run_selection.csv",
        "raw_snapshots": "historical_source_raw_snapshots/Core50_raw_snapshots_20260914.tar.zst",
        "strict_terminal_inputs": "evaluation_strict/terminal_inputs.jsonl.gz",
        "strict_numerical_root": "evaluation_strict/numerical",
        "current_symbolic_bindings": "evaluation_strict",
        "final_score_table": "algorithm_six_axis_15alg.csv",
        "input_manifest_sha256": {
            "pending_candidate": sha(pending / "current_manifest.json"),
            "strict_policy": sha(strict / "manifest.json"),
            "strict_scores": sha(six_axis / "manifest.json"),
            "accepted_legacy_merge": sha(merged / "manifest.json"),
        },
        "final_score_table_sha256": sha(output / "algorithm_six_axis_15alg.csv"),
    }
    (output / "current_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    (output / "CURRENT_STATUS.md").write_text(
        "# Core-50 final score status\n\n"
        "The 45 algorithm-condition rows contain all six final-axis numbers. ID/OOD/EFF "
        "and their 180-minute curves use the strict scalar-validity replay. A non-scalar "
        "indexed feature column or nonfinite formula is invalid and scores zero at that "
        "minute; the native incumbent expression and source hash are preserved. The two "
        "affected terminal runs score SYM/MIN=0, and seed pairs involving them are "
        "structurally inconsistent. gplearn native numerical replay is unchanged.\n\n"
        "For gplearn only, SYM/MIN/STAB are the user-accepted 2026-09-14 historical "
        "scores. These are not bound to the latest native terminal expression for 329 "
        "of 450 runs. The final table labels those three rows and preserves their "
        "historical source files; this is a declared exception, not a current-terminal "
        "formal validation. All other 42 rows are current-terminal formal scores.\n\n"
        "Minute-level SYM/MIN/STAB are intentionally not computed. The original raw "
        "selection, snapshots, and pre-correction evaluations remain in the package "
        "for audit. Use `evaluation_strict/` for current scoring and numerical curves.\n",
        encoding="utf-8",
    )
    with (output / "CURRENT_SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != output / "CURRENT_SHA256SUMS":
                handle.write(f"{sha(path)}  {path.relative_to(output)}\n")
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", type=Path, default=PENDING)
    parser.add_argument("--strict", type=Path, default=STRICT)
    parser.add_argument("--six-axis", type=Path, default=SIX_AXIS)
    parser.add_argument("--merged", type=Path, default=MERGED)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(pending=args.pending, strict=args.strict, six_axis=args.six_axis,
                              merged=args.merged, historical=args.historical,
                              output=args.output, zip_path=args.zip),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
