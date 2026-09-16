"""Add current-terminal symbolic evidence to the verified Stage4 data collection.

The output is a new directory. The input data-only release is never modified.
Incomplete symbolic judgments are explicitly marked pending, not scored as zero.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
BASE = ROOT / "AAAI_experiments/Core50_terminal_consistent_20260916"
PLAN = COLLECTION / "opus48_downstream_plan_v1"
JUDGMENTS = COLLECTION / "current_terminal_judgments_v1"
SCORES = COLLECTION / "current_terminal_six_axis_v1"
PROMPTS = STAGE5 / "config/prompts"
SCHEMAS = STAGE5 / "config/schemas"
OLD_STAGE = STAGE5 / "work/final_release_20260913/release_v2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _source_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    require(path.is_relative_to(ROOT), f"response source outside repository: {path}")
    return path


def _copy(source: Path, destination: Path) -> None:
    require(source.is_file(), f"missing source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _gzip_plan(source: Path, destination: Path, selected_keys: set[str] | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("rb") as original, gzip.open(destination, "wb", compresslevel=6) as packed:
        for line in original:
            if not line.strip():
                continue
            if selected_keys is not None:
                record = json.loads(line)
                if record["evaluation_key"] not in selected_keys:
                    continue
            packed.write(line)
            count += 1
    return {"source_sha256": sha(source), "packaged_sha256": sha(destination), "row_count": count}


def assemble(*, base: Path, plan: Path, judgments: Path, scores: Path,
             output: Path, zip_path: Path | None = None) -> dict:
    require(not output.exists(), f"output exists: {output}")
    if zip_path is not None:
        require(not zip_path.exists(), f"ZIP exists: {zip_path}")
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    judgment_manifest = json.loads((judgments / "manifest.json").read_text(encoding="utf-8"))
    score_manifest = json.loads((scores / "manifest.json").read_text(encoding="utf-8"))
    require(base_manifest["terminal_ready"] is True and base_manifest["run_count"] == 6750,
            "base release not terminal-ready")
    for name, expected in judgment_manifest["output_sha256"].items():
        require(sha(judgments / name) == expected, f"judgment output hash drift: {name}")
    require(score_manifest["input_sha256"]["equivalence_bindings"] ==
            sha(judgments / "equivalence_effective_index.jsonl"), "score/Eq index differs")
    require(score_manifest["input_sha256"]["structure_bindings"] ==
            sha(judgments / "structure_effective_index.jsonl"), "score/structure index differs")
    require(score_manifest["input_sha256"]["prediction_bindings"] ==
            sha(plan / "active_prediction_binding.jsonl"), "score/prediction binding differs")
    require(score_manifest["input_sha256"]["gt_bindings"] ==
            sha(plan / "gt_reference_binding.jsonl"), "score/GT binding differs")

    # Keep the verified data-only package intact; add symbolic material in a new version.
    shutil.copytree(base, output)
    symbolic = output / "symbolic"
    symbolic.mkdir()
    for source, destination in (
        (plan / "active_prediction_binding.jsonl", symbolic / "active_prediction_binding.jsonl"),
        (plan / "gt_reference_binding.jsonl", symbolic / "gt_reference_binding.jsonl"),
        (judgments / "equivalence_effective_index.jsonl", symbolic / "equivalence_effective_index.jsonl"),
        (judgments / "structure_effective_index.jsonl", symbolic / "structure_effective_index.jsonl"),
        (judgments / "unresolved.jsonl", symbolic / "unresolved.jsonl"),
        (judgments / "manifest.json", symbolic / "judgment_manifest.json"),
    ):
        _copy(source, destination)
    for name in ("run_metrics.csv", "task_stability.csv", "algorithm_six_axis.csv",
                 "unresolved.csv", "manifest.json"):
        _copy(scores / name, output / "six_axis" / name)
    audit = COLLECTION / "scalar_trajectory_audit_v1"
    if audit.is_dir():
        shutil.copytree(audit, output / "audits/scalar_trajectory_v1")
    corrections = STAGE5 / "audits/formula_quality_1000_0903_v2/corrections_v2"
    if corrections.is_dir():
        shutil.copytree(corrections, output / "audits/historical_equivalence_overlays")
    shutil.copytree(PROMPTS, output / "contracts/prompts")
    shutil.copytree(SCHEMAS, output / "contracts/schemas")
    for name in ("metrics.py", "symbolic_evidence.py", "aggregate_current_terminal_six_axis.py",
                 "freeze_current_terminal_judgments.py", "prepare_current_terminal_opus48_downstream.py"):
        _copy(STAGE5 / "pipeline" / name, output / "code" / name)

    refs: list[dict[str, str]] = []
    prediction = list(rows(plan / "active_prediction_binding.jsonl"))
    eq = list(rows(judgments / "equivalence_effective_index.jsonl"))
    structure = list(rows(judgments / "structure_effective_index.jsonl"))
    gt = list(rows(plan / "gt_reference_binding.jsonl"))
    require(len(prediction) == 6750 and len(gt) == 50,
            "active prediction/GT inventory incomplete")

    def add_response(kind: str, key: str, source_value: str, expected_hash: str) -> None:
        path = _source_path(source_value)
        require(path.is_file() and sha(path) == expected_hash,
                f"response SHA mismatch: {kind}/{key}")
        relative = Path("symbolic/response_blobs") / expected_hash[:2] / f"{expected_hash}.json"
        destination = output / relative
        if not destination.exists():
            _copy(path, destination)
        refs.append({"kind": kind, "evaluation_key": key, "response_sha256": expected_hash,
                     "original_path": str(path.relative_to(ROOT)), "packaged_path": str(relative)})

    for row in prediction:
        if row["processing_status"] == "ready":
            add_response("prediction", row["prediction_frozen_evaluation_key"],
                         row["prediction_response_path"], row["prediction_response_sha256"])
    for phase, items in (("equivalence", eq), ("structure", structure)):
        for row in items:
            add_response(phase, row["evaluation_key"], row["response_path"], row["response_sha256"])
    gt_index = {row["evaluation_key"]: row for row in rows(
        OLD_STAGE / "simplify_full/gt_frozen_index.jsonl")}
    for row in gt:
        index = gt_index[row["gt_frozen_evaluation_key"]]
        require(index["result_sha256"] == row["gt_result_sha256"],
                f"GT result SHA drift: {row['dataset_id']}")
        add_response("ground_truth", row["gt_frozen_evaluation_key"],
                     index["result_path"], index["result_sha256"])
    with (symbolic / "response_source_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(refs[0]))
        writer.writeheader()
        writer.writerows(refs)

    plan_report = {}
    plan_sources = {
        "new_equivalence": plan / "equivalence_plan.jsonl",
        "new_structure": plan / "structure_plan.jsonl",
        "reuse_equivalence": plan / "equivalence_reuse_candidates.jsonl",
        "reuse_structure": plan / "structure_reuse_candidates.jsonl",
        "new_prediction": COLLECTION / "opus_prediction_plan_v2/pred_simplify_plan.jsonl",
        "retry_prediction": COLLECTION / "opus48_prediction_retry_v2_plan/pred_simplify_retry_plan.jsonl",
        "extra_equivalence": COLLECTION / "opus48_undetermined_selected_v1/equivalence_plan.jsonl",
        "extra_structure": COLLECTION / "opus48_undetermined_selected_v1/structure_plan.jsonl",
        "qlattice_noise005_repair": COLLECTION / "opus48_structure_noise005_json_repair_v2_plan/structure_retry_plan.jsonl",
        "gt": OLD_STAGE / "inputs/gt_plan_full.jsonl",
        "gt_frozen": OLD_STAGE / "simplify_full/gt_frozen_index.jsonl",
    }
    for name, source in plan_sources.items():
        plan_report[name] = _gzip_plan(source, symbolic / "plans" / f"{name}.jsonl.gz")
    old_keys = {
        "equivalence": {row["evaluation_key"] for row in eq if row["source"] == "verified_opus5_reuse"},
        "structure": {row["evaluation_key"] for row in structure if row["source"] == "verified_opus5_reuse"},
        "prediction": {row["prediction_plan_evaluation_key"] for row in prediction
                       if row["processing_status"] == "ready" and row["source"].startswith("candidate_reuse")},
    }
    for phase in ("equivalence", "structure"):
        total = 0
        for condition in ("clean", "noise001", "noise005"):
            source = OLD_STAGE / "downstream" / f"{condition}_{phase}_full_plan.jsonl"
            name = f"historical_{phase}_{condition}"
            report = _gzip_plan(source, symbolic / "plans" / f"{name}.jsonl.gz", old_keys[phase])
            plan_report[name] = report
            total += report["row_count"]
        require(total == len(old_keys[phase]), f"historical {phase} plan coverage mismatch")
    old_pred_total = 0
    for condition in ("clean", "noise001", "noise005"):
        source = OLD_STAGE / "inputs" / f"pred_plan_full_{condition}_effective.jsonl"
        name = f"historical_prediction_{condition}"
        report = _gzip_plan(source, symbolic / "plans" / f"{name}.jsonl.gz", old_keys["prediction"])
        plan_report[name] = report
        old_pred_total += report["row_count"]
    require(old_pred_total == len(old_keys["prediction"]), "historical prediction plan coverage mismatch")

    with (symbolic / "plan_source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(plan_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output / "six_axis/algorithm_six_axis.csv").open(newline="", encoding="utf-8") as handle:
        algorithm_rows = list(csv.DictReader(handle))
    formal_cells = sum(row["formal_ready"] == "True" for row in algorithm_rows)
    status = "non_gplearn_formal_ready" if formal_cells == 42 else "symbolic_pending"
    manifest = {
        "schema_version": "core50_terminal_symbolic_evidence.v1", "status": status,
        "run_count": 6750, "algorithm_condition_cells": 45,
        "formal_algorithm_condition_cells": formal_cells,
        "gplearn_deferred": True, "all_15_formal_ready": False,
        "current_terminal_inputs_sha256": score_manifest["input_sha256"]["terminal_inputs"],
        "selected_numerical_sha256": {
            condition: score_manifest["input_sha256"][f"minute_{condition}"]
            for condition in ("clean", "noise001", "noise005")},
        "symbolic_unresolved_count": judgment_manifest["unresolved_count"],
        "score_unresolved_count": score_manifest["unresolved_count"],
        "response_references": len(refs), "unique_response_blobs": len({row["response_sha256"] for row in refs}),
        "response_index_sha256": sha(symbolic / "response_source_index.csv"),
        "plans_sha256": sha(symbolic / "plan_source_manifest.json"),
        "base_release_manifest_sha256": sha(base / "manifest.json"),
    }
    (output / "current_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    (output / "CURRENT_STATUS.md").write_text(
        "# Current terminal symbolic evidence\n\n"
        f"Status: `{status}`. This package includes the selected 6,750 runs, raw snapshots, "
        "current terminal expressions, frozen GT/prediction/equivalence/structure evidence, "
        "deterministic six-axis components, and provenance hashes. The data-only base README "
        "describes its original pre-symbolic state; `current_manifest.json` is authoritative "
        "for this augmented package.\n\n"
        "Missing judgments remain unresolved, not zero. gplearn symbolic metrics are deferred. "
        "`audits/scalar_trajectory_v1/` records suspected non-pointwise expressions; its "
        "read-only findings have not changed the numerical trajectories.\n",
        encoding="utf-8",
    )
    with (output / "CURRENT_SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "CURRENT_SHA256SUMS":
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
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--judgments", type=Path, default=JUDGMENTS)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(base=args.base, plan=args.plan, judgments=args.judgments,
                              scores=args.scores, output=args.output, zip_path=args.zip),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
