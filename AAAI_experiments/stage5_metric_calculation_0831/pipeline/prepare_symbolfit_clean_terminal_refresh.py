"""Freeze SymbolFit clean minute-180 endpoints for symbolic post-processing."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .clean_task_builder import build_clean_task_plan


TASK_RE = re.compile(r"symbolfit_s(520|521|522)_clean_g(\d{4})")
RETRY_TASK_ID = "symbolfit_s521_clean_g0039"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"{label}: expected one file, found {len(paths)}")
    return paths[0]


def _terminal_identity_matches(terminal: dict[str, Any], *, seed: int, index: int) -> bool:
    explicit_index = terminal.get("task_global_index")
    return terminal.get("seed") == seed and (explicit_index is None or explicit_index == index)


def _stage4_sources(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["task_id"]: row
            for row in csv.DictReader(handle)
            if row["algorithm"].lower() == "symbolfit" and row["noise_tag"] == "clean"
        }
    if len(rows) != 150:
        raise ValueError(f"Stage4 SymbolFit clean grid is {len(rows)}, expected 150")
    return rows


def collect_endpoints(batch_root: Path, stage4_csv: Path) -> list[dict[str, Any]]:
    report = json.loads((batch_root / "reports/full150_validation_strict180.json").read_text())
    retry_report = json.loads(
        (batch_root / "reports/symbolfit_s521_clean_g0039_strict_rerun_validation.json").read_text()
    )
    if report["passed_task_count"] != 149 or report["rerun_task_ids"] != [RETRY_TASK_ID]:
        raise ValueError("full150 validation no longer matches the audited 149+1 selection")
    if retry_report["status"] != "passed":
        raise ValueError("g0039/s521 strict rerun did not pass")

    sources = _stage4_sources(stage4_csv)
    tasks = report["tasks"]
    if len(tasks) != 150 or {item["task_id"] for item in tasks} != set(sources):
        raise ValueError("validation task identities differ from the Stage4 clean grid")

    selected: list[dict[str, Any]] = []
    for item in sorted(tasks, key=lambda row: row["task_id"]):
        task_id = item["task_id"]
        match = TASK_RE.fullmatch(task_id)
        if match is None:
            raise ValueError(f"invalid SymbolFit task ID: {task_id}")
        seed, index = int(match.group(1)), int(match.group(2))
        if task_id == RETRY_TASK_ID:
            snapshot = _one(
                list((batch_root / "strict_rerun_g0039/collected/symbolfit").glob(
                    "g0039_*/progress/minute_0180.json"
                )),
                task_id,
            )
            validation_sha = retry_report["result_sha256"]
            validation_source = "strict_rerun_g0039"
        else:
            if item["status"] != "passed":
                raise ValueError(f"{task_id}: validation did not pass")
            snapshot = _one(
                list((batch_root / "collected/full150/symbolfit").glob(
                    f"seed{seed}/tasks/{task_id}/iaaccn*/symbolfit/g{index:04d}_*/progress/minute_0180.json"
                )),
                task_id,
            )
            validation_sha = item["result_sha256"]
            validation_source = "full150"

        result_path = snapshot.parent.parent / "result.json"
        snapshot_bytes = snapshot.read_bytes()
        result_bytes = result_path.read_bytes()
        if _sha(result_bytes) != validation_sha:
            raise ValueError(f"{task_id}: validated result SHA drift")
        terminal = json.loads(snapshot_bytes)
        result = json.loads(result_bytes)
        artifact = terminal.get("canonical_artifact") or {}
        expression = artifact.get("instantiated_expression") or terminal.get("equation")
        if (
            terminal.get("checkpoint_index") != 180
            or terminal.get("record_type") != "budget_end_internal_best"
            or terminal.get("status") != "ok"
            or artifact.get("artifact_valid") is not True
            or not isinstance(expression, str)
            or not expression.strip()
        ):
            raise ValueError(f"{task_id}: invalid terminal symbolic candidate")
        if not _terminal_identity_matches(terminal, seed=seed, index=index):
            raise ValueError(f"{task_id}: terminal identity mismatch")
        if terminal.get("dataset") != sources[task_id]["dataset_id"]:
            raise ValueError(f"{task_id}: dataset identity mismatch")
        if terminal.get("feature_names") != result.get("feature_names"):
            raise ValueError(f"{task_id}: feature-name mapping changed")

        source = {
            "algorithm": "symbolfit",
            "batch": (
                "symbolfit_internal_progress_v1_strict_rerun_g0039"
                if validation_source == "strict_rerun_g0039"
                else "symbolfit_internal_progress_v1_full"
            ),
            "dataset_id": terminal["dataset"],
            "host": next((part for part in snapshot.parts if part.startswith("iaaccn")), None),
            "noise_tag": "clean",
            "path": str(snapshot.resolve()),
            "seed": str(seed),
            "task_id": task_id,
            "selected_result_path": str(result_path.resolve()),
            "selected_result_sha256": validation_sha,
            "terminal_expression_sha256": _sha(expression.encode("utf-8")),
            "terminal_snapshot_sha256": _sha(snapshot_bytes),
        }
        source["source_row_sha256"] = _sha(_json_bytes(source))
        result_expression = (result.get("canonical_artifact") or {}).get("instantiated_expression")
        selected.append(
            {
                "freeze": {"source": source, "result": {"raw_text": snapshot_bytes.decode("utf-8"), "sha256": _sha(snapshot_bytes)}},
                "inventory": {
                    "task_id": task_id,
                    "dataset_id": terminal["dataset"],
                    "seed": seed,
                    "validation_source": validation_source,
                    "snapshot_path": str(snapshot.resolve()),
                    "snapshot_sha256": _sha(snapshot_bytes),
                    "result_path": str(result_path.resolve()),
                    "result_sha256": validation_sha,
                    "terminal_expression": expression,
                    "terminal_expression_sha256": _sha(expression.encode("utf-8")),
                    "terminal_matches_result": expression == result_expression,
                },
            }
        )
    return selected


def prepare(*, batch_root: Path, stage4_csv: Path, old_prediction: Path, output: Path, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 6:
        raise ValueError("local worker count must be between 1 and 6")
    selected = collect_endpoints(batch_root, stage4_csv)
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "symbolfit_clean_terminal_freeze.jsonl.gz"
    with freeze_path.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", filename="", mtime=0) as compressed:
            for item in selected:
                compressed.write(_json_bytes(item["freeze"]) + b"\n")

    inventory_path = output / "terminal_inventory.csv"
    fields = list(selected[0]["inventory"])
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item["inventory"] for item in selected)

    tasks, plan_report = build_clean_task_plan(
        phase="pred",
        freeze_glob=str(freeze_path.resolve()),
        expected_pred_count=150,
        repo_root=Path(__file__).resolve().parents[3],
        build_workers=workers,
    )
    if len(tasks) != 150 or plan_report["planning_counts"]["pred_no_call_count"]:
        raise ValueError("SymbolFit terminal prediction plan is incomplete")
    plan_path = output / "pred_simplify_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(_json_bytes(task.to_json_record()).decode("utf-8") + "\n")

    old = {}
    with old_prediction.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["logical_id"].startswith("pred_simplify::symbolfit::"):
                old["::".join(record["logical_id"].split("::")[:5])] = record
    if len(old) != 150:
        raise ValueError(f"old SymbolFit prediction index has {len(old)} entries")
    matches = []
    for task in tasks:
        prior = old[task.logical_id]
        matches.append(
            {
                "logical_id": task.logical_id,
                "current_evaluation_key": task.evaluation_key,
                "old_evaluation_key": prior["evaluation_key"],
                "evaluation_key_matches": task.evaluation_key == prior["evaluation_key"],
                "input_expression_matches": task.request["original_expression"] == prior.get("input_expression"),
            }
        )
    match_path = output / "prediction_cache_match.csv"
    with match_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
        writer.writeheader()
        writer.writerows(matches)

    manifest = {
        "schema_version": "symbolfit_clean_terminal_symbolic_preflight.v1",
        "status": "ready_for_prediction_simplification",
        "api_invoked": False,
        "formal_ready": False,
        "terminal_count": len(selected),
        "strict_rerun_count": sum(item["inventory"]["validation_source"] == "strict_rerun_g0039" for item in selected),
        "terminal_differs_from_result_count": sum(not item["inventory"]["terminal_matches_result"] for item in selected),
        "prediction_plan_count": len(tasks),
        "exact_cache_hit_count": sum(item["evaluation_key_matches"] for item in matches),
        "same_input_expression_count": sum(item["input_expression_matches"] for item in matches),
        "next_steps": ["run_pred_simplify", "build_equivalence_and_structure", "aggregate_symbolic_metrics"],
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha(path.read_bytes())}
            for name, path in (
                ("terminal_freeze", freeze_path),
                ("terminal_inventory", inventory_path),
                ("pred_simplify_plan", plan_path),
                ("prediction_cache_match", match_path),
            )
        },
        "prompt": plan_report["contract"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    stage = repo / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=stage / "reruns/symbolfit_internal_progress_v1")
    parser.add_argument("--stage4-csv", type=Path, default=repo / "AAAI_experiments/stage4_ssr50_15algs_3seeds_3noise_3h/selected_runs_with_fepysr_rerun.csv")
    parser.add_argument("--old-prediction", type=Path, default=repo / "AAAI_experiments/Core50_final_20260914/results/clean/opus5_prediction.jsonl")
    parser.add_argument("--output", type=Path, default=stage / "work/symbolfit_clean_terminal_refresh_20260916")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(batch_root=args.batch_root, stage4_csv=args.stage4_csv, old_prediction=args.old_prediction, output=args.output, workers=args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
