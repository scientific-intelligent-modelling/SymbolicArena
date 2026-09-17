"""Package all 15 Core-50 algorithms with current-terminal gplearn symbolism."""

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
BASE = ROOT / "AAAI_experiments/Core50_final_strict_scalar_20260917"
SCORES = COLLECTION / "gplearn_current_terminal_six_axis_v1"
CONDITIONS = ("clean", "noise001", "noise005")
AXES = ("ID", "OOD", "SYM", "MIN", "EFF", "STAB")
EVIDENCE_DIRS = (
    "gplearn_terminal_inventory_v2",
    "gplearn_opus48_prediction_plan_v4",
    "gplearn_opus48_exhausted_retry_plan_v1",
    "gplearn_protected_downstream_full_v2",
    "gplearn_opus48_equivalence_json_retry_plan_v1",
    "gplearn_opus48_structure_json_retry_plan_v1",
    "gplearn_active_equivalence_v1",
    "gplearn_active_structure_v1",
    "gplearn_opus48_prediction_run_v2",
    "gplearn_opus48_exhausted_retry_run_v1",
    "gplearn_opus48_equivalence_run_v1",
    "gplearn_opus48_equivalence_json_retry_run_v1",
    "gplearn_opus48_structure_run_v1",
    "gplearn_opus48_structure_json_retry_run_v1",
)
PHASE_RUNS = (
    ("gplearn_opus48_prediction_run_v2", "gplearn_opus48_exhausted_retry_run_v1"),
    ("gplearn_opus48_equivalence_run_v1", "gplearn_opus48_equivalence_json_retry_run_v1"),
    ("gplearn_opus48_structure_run_v1", "gplearn_opus48_structure_json_retry_run_v1"),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def copy(source: Path, destination: Path) -> None:
    require(source.is_file(), f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_phase_summaries(base: dict, retry: dict) -> dict:
    require(base["plan_tasks"] == 450 and base["frozen"] + base["remaining"] == 450,
            "base Opus phase inventory mismatch")
    require(retry["plan_tasks"] == base["remaining"] and retry["remaining"] == 0
            and retry["frozen"] == retry["plan_tasks"],
            "versioned Opus retry does not complete the base phase")
    return {
        "logical_tasks_complete": base["frozen"] + retry["frozen"],
        "physical_attempts": base["physical_attempts"] + retry["physical_attempts"],
        "estimated_cost_cny_assumed_tariff": (
            base["estimated_cost_cny_assumed_tariff"] + retry["estimated_cost_cny_assumed_tariff"]),
    }


def finalize(*, base: Path, collection: Path, scores: Path,
             output: Path, zip_path: Path | None = None) -> dict:
    if output.exists() or (zip_path is not None and zip_path.exists()):
        raise ValueError("target output or ZIP already exists")
    parent = json.loads((base / "current_manifest.json").read_text(encoding="utf-8"))
    score_manifest = json.loads((scores / "manifest.json").read_text(encoding="utf-8"))
    require(parent["run_count"] == 6750 and parent["strict_scalar_affected_minutes"] == 373,
            "base strict-scalar release identity mismatch")
    require(score_manifest["formal_ready"] is True and score_manifest["run_count"] == 450
            and score_manifest["task_count"] == 150 and score_manifest["combined_algorithm_count"] == 45
            and score_manifest["unresolved_count"] == 0,
            "gplearn current-terminal six-axis input is not complete")
    for label, source in (
        ("inventory", collection / "gplearn_terminal_inventory_v2/gplearn_terminal_inventory.jsonl"),
        ("gt_binding", collection / "opus48_downstream_plan_v1/gt_reference_binding.jsonl"),
        ("strict_run_metrics", collection / "current_terminal_six_axis_strict_v2/run_metrics.csv"),
        ("strict_algorithm_metrics", collection / "current_terminal_six_axis_strict_v2/algorithm_six_axis.csv"),
        ("equivalence_plan", collection / "gplearn_active_equivalence_v1/equivalence_active_plan.jsonl"),
        ("structure_plan", collection / "gplearn_active_structure_v1/stab_structure_active_plan.jsonl"),
    ):
        require(score_manifest["input_sha256"][label] == sha(source), f"score source SHA drift: {label}")
    final_rows = _read_csv(scores / "algorithm_six_axis_15alg.csv")
    require(len(final_rows) == 45 and len({(r["condition"], r["algorithm"].casefold()) for r in final_rows}) == 45,
            "final six-axis cohort incomplete")
    require(all(r["formal_ready"] == "True" and all(r[axis] for axis in AXES) for r in final_rows),
            "final six-axis table contains incomplete row")
    for condition in CONDITIONS:
        require(sum(r["condition"] == condition for r in final_rows) == 15,
                f"{condition}: expected 15 algorithms")
    require(len(_read_csv(scores / "gplearn_run_metrics.csv")) == 450 and
            len(_read_csv(scores / "gplearn_task_stability.csv")) == 150,
            "gplearn run/task result count mismatch")

    api_physical = 0
    api_cost = 0.0
    for base_name, retry_name in PHASE_RUNS:
        base_summary = json.loads((collection / base_name / "summary.json").read_text(encoding="utf-8"))
        retry_summary = json.loads((collection / retry_name / "summary.json").read_text(encoding="utf-8"))
        phase = validate_phase_summaries(base_summary, retry_summary)
        api_physical += phase["physical_attempts"]
        api_cost += phase["estimated_cost_cny_assumed_tariff"]
    require(api_physical <= 2025, "API physical attempts exceed 50 percent retry ceiling")

    shutil.copytree(base, output)
    provenance = output / "provenance/legacy_gplearn_exception_20260917"
    provenance.mkdir(parents=True)
    for name in ("current_manifest.json", "CURRENT_STATUS.md", "CURRENT_SHA256SUMS",
                 "algorithm_six_axis_15alg.csv"):
        copy(output / name, provenance / name)
    destination = output / "evaluation_strict/gplearn_current_terminal"
    for name in ("gplearn_run_metrics.csv", "gplearn_task_stability.csv",
                 "gplearn_algorithm_six_axis.csv", "algorithm_six_axis_15alg.csv",
                 "unresolved.csv", "manifest.json"):
        copy(scores / name, destination / name)
    copy(scores / "algorithm_six_axis_15alg.csv", output / "algorithm_six_axis_15alg.csv")
    evidence_root = destination / "evidence"
    for name in EVIDENCE_DIRS:
        source = collection / name
        require(source.is_dir(), f"gplearn evidence directory missing: {source}")
        shutil.copytree(source, evidence_root / name)
    for name in ("gplearn_native_prefix_evidence.py", "gplearn_current_metrics.py",
                 "build_gplearn_protected_downstream_plan.py", "aggregate_gplearn_current_terminal.py"):
        copy(STAGE5 / "pipeline" / name, output / "code" / name)
    for name in ("prepare_gplearn_terminal_inventory.py", "build_gplearn_opus48_prediction_plan.py",
                 "build_gplearn_exhausted_retry_plan.py", "build_gplearn_schema_retry_plan.py",
                 "merge_gplearn_active_pair_plans.py", "validate_gplearn_opus48_response.py",
                 "validate_gplearn_opus48_pair.py", "run_opus48_terminal_plan.py"):
        copy(ROOT / "check" / name, output / "code" / name)
    for name in ("gplearn_protected_simplify.v1.txt", "gplearn_protected_simplify_exhausted.v2.txt",
                 "gplearn_protected_equivalence.v1.txt", "gplearn_protected_equivalence_json_repair.v2.txt",
                 "gplearn_protected_structure.v1.txt", "gplearn_protected_structure_json_repair.v2.txt"):
        copy(STAGE5 / "config/prompts" / name, output / "contracts/prompts" / name)

    manifest = {
        "schema_version": "core50_all_current_terminal_six_axis.v1",
        "status": "formal_ready_current_terminal_with_protected_semantics_extension",
        "run_count": 6750, "algorithm_condition_rows": 45,
        "all_15_current_terminal_formal_ready": True,
        "gplearn_current_terminal_runs": 450, "gplearn_seed_pair_tasks": 450,
        "gplearn_symbolic_basis": score_manifest["gplearn_symbolic_basis"],
        "gplearn_tree_basis_note": score_manifest["gplearn_tree_basis_note"],
        "minute_symbolic_required": False, "minute_numerical_ready": True,
        "strict_scalar_invalid_terminal_count": 2,
        "gplearn_numerical_basis": "unchanged_strict_native_replay",
        "api_model": "claude-opus-4-8", "api_physical_attempts": api_physical,
        "estimated_api_cost_cny_assumed_tariff": round(api_cost, 6),
        "api_cost_note": "Uses the user's earlier assumed CNY/M-token tariff, not a verified channel invoice.",
        "latest_run_selection": "selection/latest_run_selection.csv",
        "raw_snapshots": "historical_source_raw_snapshots/Core50_raw_snapshots_20260914.tar.zst",
        "strict_terminal_inputs": "evaluation_strict/terminal_inputs.jsonl.gz",
        "strict_numerical_root": "evaluation_strict/numerical",
        "gplearn_current_evidence": "evaluation_strict/gplearn_current_terminal/evidence",
        "final_score_table": "algorithm_six_axis_15alg.csv",
        "final_score_table_sha256": sha(output / "algorithm_six_axis_15alg.csv"),
        "input_manifest_sha256": {
            "previous_legacy_exception_release": sha(base / "current_manifest.json"),
            "gplearn_current_terminal_scores": sha(scores / "manifest.json"),
        },
    }
    (output / "current_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    (output / "CURRENT_STATUS.md").write_text(
        "# Core-50 current-terminal six-axis release\n\n"
        "All 15 algorithms and all three conditions now have six final-axis scores bound "
        "to the selected minute-180 native terminal incumbent. The original 6,750 run "
        "selection, raw snapshots, and 180-minute numerical curves remain unchanged. "
        "The two earlier non-scalar/nonfinite terminal runs retain their strict invalid "
        "status and zero quality where applicable. Minute-level SYM/MIN/STAB are not required.\n\n"
        "gplearn uses its raw native prefix and protected div/log/sqrt semantics, not the "
        "older result.json expression. Opus4.8 simplification, GT equivalence, and seed-pair "
        "structure responses are versioned and bound by native prefix, snapshot, model key, "
        "prompt, GT, and source hashes. A model 'unable' simplification falls back to the "
        "original typed prefix; an undetermined relation remains labeled undetermined "
        "while scoring as not established under the accepted gplearn rule.\n\n"
        "The gplearn SYM/MIN structural representation is a protected typed prefix, "
        "while the fixed GT reference uses the existing SymPy canonical tree. The same "
        "normalized tree-edit cost is used, but the operator vocabulary differs. This "
        "native-semantics extension must be stated when reporting symbolic scores. "
        "The prior legacy-gplearn table remains only under provenance.\n",
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
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(base=args.base, collection=args.collection,
                              scores=args.scores, output=args.output, zip_path=args.zip),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
