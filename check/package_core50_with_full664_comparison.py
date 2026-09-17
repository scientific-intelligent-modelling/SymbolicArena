"""Package current Core-50 results beside the frozen matched Full-664 comparison."""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "AAAI_experiments/Core50_final_current_selected_only_20260917_v2"
COMPARISON = ROOT / "A_Neurips_experiments/rebuttal/02_core50_full664_rank_correlation_9algs"
OUTPUT = ROOT / "AAAI_experiments/Core50_final_current_with_Full664_9alg_20260917.zip"
PACKAGE_ROOT = OUTPUT.stem
NINE = {"udsr", "imcts", "pysr", "symbolfit", "dso", "fepysr", "jaxsr", "llmsr", "pyoperon"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_bytes(data: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
    return buffer.getvalue().encode("utf-8")


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    current_manifest = json.loads((CURRENT / "manifest.json").read_text(encoding="utf-8"))
    comparison_summary = json.loads(
        (COMPARISON / "full664_run_level_9alg_summary.json").read_text(encoding="utf-8")
    )
    if current_manifest["run_count"] != 6750 or current_manifest["superseded_old_runs_included"] != 0:
        raise ValueError("current release is not the selected-only 6,750-run release")
    selection = rows(CURRENT / "selection/current_run_selection.csv")
    if len(selection) != 6750 or len({r["logical_key"].casefold() for r in selection}) != 6750:
        raise ValueError("current run selection is incomplete or duplicated")
    current_tasks = {str(p.parent.relative_to(CURRENT / "datasets"))
                     for p in (CURRENT / "datasets").rglob("metadata.yaml")}
    frozen_tasks = {r["dataset_dir"].removeprefix("sim-datasets-data/")
                    for r in rows(COMPARISON / "core50_task_ids.csv")}
    if len(current_tasks) != 50 or current_tasks != frozen_tasks:
        raise ValueError("the current and frozen comparisons do not use the same 50 tasks")
    run_rows = rows(COMPARISON / "full664_run_level_9alg.csv")
    if len(run_rows) != comparison_summary["coverage"]["run_rows"] or len(run_rows) != 15272:
        raise ValueError("Full-664 run-level inventory changed")
    if {r["algorithm_key"] for r in run_rows} != NINE:
        raise ValueError("Full-664 algorithm panel changed")
    if len({r["dataset_dir"] for r in run_rows}) != 664:
        raise ValueError("Full-664 dataset panel changed")
    core_rows = [r for r in run_rows if r["is_core50"] == "True"]
    if len(core_rows) != 1150 or {r["dataset_dir"].removeprefix("sim-datasets-data/")
                                   for r in core_rows} != frozen_tasks:
        raise ValueError("matched Core-50 rows changed")
    if Counter(r["algorithm_key"] for r in run_rows) != comparison_summary["coverage"]["runs_by_algorithm"]:
        raise ValueError("Full-664 run counts by algorithm changed")
    scores = rows(CURRENT / "metrics/algorithm_six_axis_15alg.csv")
    latest_nine = [r for r in scores if r["algorithm"].casefold() in NINE]
    if len(scores) != 45 or len(latest_nine) != 27:
        raise ValueError("current six-axis score table is incomplete")
    for condition in ("clean", "noise001", "noise005"):
        if {r["algorithm"].casefold() for r in latest_nine if r["condition"] == condition} != NINE:
            raise ValueError(f"missing nine-algorithm Stage4 scores: {condition}")
    if len(rows(COMPARISON / "algorithm_scores_and_ranks.csv")) != 9:
        raise ValueError("frozen comparison ranks are incomplete")

    source_files = sorted(p for p in CURRENT.rglob("*") if p.is_file())
    comparison_files = sorted(p for p in COMPARISON.iterdir() if p.is_file())
    manifest = {
        "schema_version": "core50_current_plus_frozen_full664_matched.v1",
        "current_stage4": {
            "status": current_manifest["status"],
            "conditions": ["clean", "noise001", "noise005"],
            "selected_runs": 6750,
            "algorithms": 15,
            "budget_hours": 3,
            "manifest_sha256": sha256(CURRENT / "manifest.json"),
        },
        "matched_full664_comparison": {
            "run_rows": 15272,
            "algorithms": 9,
            "full_tasks": 664,
            "core_tasks": 50,
            "budget_hours": 1,
            "frozen_summary_sha256": sha256(COMPARISON / "full664_run_level_9alg_summary.json"),
            "run_level_sha256": sha256(COMPARISON / "full664_run_level_9alg.csv"),
            "matched_task_set_equals_current_stage4": True,
            "is_recomputed_from_current_stage4": False,
        },
        "metric_warning": "The matched Full-664 comparison uses penalized log10(NMSE), lower is better. "
                          "Current Stage4 six-axis ID/OOD uses 0-100 quality, higher is better. "
                          "Different runs, budgets, seed panels and metric scales must not be pooled.",
    }
    readme = (
        "# Current Core-50 and matched Full-664 comparison\n\n"
        "`current_core50/` is the selected-only Stage4 release: 15 algorithms, "
        "50 tasks, three seeds, clean/noise001/noise005, three-hour budget. "
        "Only the 6,750 selected runs are included; the 624 superseded runs are not.\n\n"
        "`full664_matched_9alg/` is a separate, frozen one-hour nine-algorithm "
        "comparison. Its Core-50 column is recomputed by restricting the *same* "
        "Full-664 runs to the frozen 50 tasks, not from the updated Stage4 runs. "
        "The 50 task directories match the current Core-50 task set. Its original "
        "rank and correlation statistics are unchanged by Stage4 run replacements.\n\n"
        "`stage4_current_9alg_six_axis.csv` is a convenience slice of the latest "
        "Stage4 scores for the same nine algorithm names and all three conditions. "
        "It is not a paired Full-664 comparison. The frozen comparison's penalized "
        "log10(NMSE) is lower-is-better; current Stage4 ID/OOD quality is "
        "0-100 and higher-is-better. Do not directly compare or pool their values.\n"
    )

    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        prefix = Path(PACKAGE_ROOT)
        for path in source_files:
            archive.write(path, arcname=str(prefix / "current_core50" / path.relative_to(CURRENT)))
        for path in comparison_files:
            archive.write(path, arcname=str(prefix / "full664_matched_9alg" / path.name))
        archive.writestr(str(prefix / "stage4_current_9alg_six_axis.csv"), csv_bytes(latest_nine))
        archive.writestr(str(prefix / "README.md"), readme.encode("utf-8"))
        archive.writestr(str(prefix / "package_manifest.json"),
                         (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    os.replace(partial, OUTPUT)
    return {"output": str(OUTPUT), "zip_sha256": sha256(OUTPUT),
            "source_files": len(source_files), "comparison_files": len(comparison_files),
            "stage4_nine_score_rows": len(latest_nine), **manifest}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
