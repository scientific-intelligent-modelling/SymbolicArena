"""Publish one current run per Core-50 key without superseded run bundles."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AAAI_experiments/Core50_final_current_terminal_20260917"
CONDITIONS = ("clean", "noise001", "noise005")
AXES = ("ID", "OOD", "SYM", "MIN", "EFF", "STAB")
SELECTION_FIELDS = (
    "condition", "algorithm", "algorithm_slug", "dataset_id", "seed", "task_id",
    "logical_key", "selection_status", "selected_result_sha256",
    "selected_result_archive_member", "selected_result_payload_locator",
    "trajectory_bundle_sha256", "terminal_expression_sha256", "terminal_source_sha256",
    "terminal_archive_member",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def copy(source: Path, destination: Path) -> None:
    require(source.is_file(), f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _key(value: str) -> str:
    return value.casefold()


def current_selection_row(row: dict, terminal: dict) -> dict:
    return {
        **{field: row.get(field, "") for field in SELECTION_FIELDS if field in row},
        "terminal_expression_sha256": terminal["terminal_expression_sha256"],
        "terminal_source_sha256": terminal["terminal_source_sha256"],
        "terminal_archive_member": terminal["terminal_archive_member"],
    }


def current_response_reference(row: dict) -> dict:
    result = dict(row)
    result["source_package_path"] = row["packaged_path"]
    result["packaged_path"] = str(
        Path("symbolic_current/response_blobs") / row["response_sha256"][:2] /
        (row["response_sha256"] + ".json")
    )
    return result


def validate_selected_archive(source: Path) -> tuple[list[dict], dict[str, dict], dict]:
    selection = list(read_csv(source / "selection/latest_run_selection.csv"))
    selected = {_key(row["logical_key"]): row for row in selection}
    require(len(selection) == len(selected) == 6750, "selected run grid is incomplete or duplicated")
    require(Counter(row["condition"] for row in selection) == {c: 2250 for c in CONDITIONS},
            "selected run condition counts differ")
    superseded = sum(row["superseded"].lower() == "true" for row in selection)
    require(superseded == 624, "expected 624 replaced historical runs")
    terminals = {_key(row["logical_key"]): row for row in read_jsonl(
        source / "evaluation_strict/terminal_inputs.jsonl.gz")}
    require(set(selected) == set(terminals), "terminal grid differs from selected run grid")
    archive_dir = source / "historical_source_raw_snapshots"
    meta = archive_dir / "archive_metadata"
    archive = archive_dir / "Core50_raw_snapshots_20260914.tar.zst"
    delivery = json.loads((archive_dir / "delivery_report.json").read_text(encoding="utf-8"))
    require(delivery["archive_sha256"] == sha(archive), "selected snapshot archive SHA drift")
    archive_runs = {_key(row["logical_key"]): row for row in read_csv(meta / "runs.csv")}
    require(set(archive_runs) == set(selected), "raw snapshot archive contains unselected run key")
    for key, row in selected.items():
        require(archive_runs[key]["bundle_sha256"] == row["trajectory_bundle_sha256"],
                f"archive contains a different run bundle: {key}")
        terminal = terminals[key]
        require(terminal["selected_result_sha256"] == row["selected_result_sha256"] and
                terminal["terminal_expression_sha256"] ==
                (hashlib.sha256(terminal["terminal_expression"].encode()).hexdigest()
                 if terminal["terminal_expression"] is not None else None),
                f"selected result/terminal expression drift: {key}")
    physical_members: set[str] = set()
    result_members = 0
    recovery_members: set[str] = set()
    for row in read_csv(meta / "physical_records.csv.gz"):
        key = _key(row["logical_key"])
        require(key in selected, f"physical archive contains unselected key: {key}")
        member = row["archive_member"]
        require(member not in physical_members, f"duplicate physical member: {member}")
        physical_members.add(member)
        if row["record_type"] == "result":
            result_members += 1
            require(row["sha256"] == selected[key]["selected_result_sha256"],
                    f"old result leaked into selected archive: {key}")
        if row["record_type"] == "recovery":
            recovery_members.add(member)
    require(result_members == 5986 and len(physical_members) == 1_140_921,
            "physical archive inventory differs from frozen delivery")
    minute_masks = {key: 0 for key in selected}
    minute_count = 0
    used_recovery: set[str] = set()
    for row in read_csv(meta / "logical_minute_bindings.csv.gz"):
        key = _key(row["logical_key"])
        require(key in selected and row["archive_member"] in physical_members and
                row["binding_status"] == "bound", f"minute binding lacks selected physical source: {key}")
        minute = int(row["minute"])
        require(1 <= minute <= 180 and not minute_masks[key] & (1 << (minute - 1)),
                f"duplicate or invalid logical minute: {key}/{minute}")
        minute_masks[key] |= 1 << (minute - 1)
        minute_count += 1
        if row["archive_member"] in recovery_members:
            used_recovery.add(row["archive_member"])
        if minute == 180:
            require(row["source_sha256"] == terminals[key]["terminal_source_sha256"],
                    f"minute180 source not the selected terminal: {key}")
    require(minute_count == 1_215_000 and all(mask == (1 << 180) - 1 for mask in minute_masks.values()),
            "selected minute grid incomplete")
    require(used_recovery == recovery_members,
            "raw recovery evidence includes unused or stale source")
    return selection, terminals, {
        "selected_runs": len(selected), "superseded_old_runs_excluded": superseded,
        "physical_records": len(physical_members), "logical_minutes": minute_count,
        "selected_result_members": result_members,
        "selected_recovery_members": len(recovery_members),
        "archive_sha256": sha(archive),
    }


def _unified_metrics(source: Path, output: Path) -> tuple[int, int]:
    prior = source / "evaluation_strict/six_axis_current_terminal"
    gplearn = source / "evaluation_strict/gplearn_current_terminal"
    run_rows = []
    for row in read_csv(prior / "run_metrics.csv"):
        if row["algorithm"].casefold() == "gplearn":
            continue
        run_rows.append({
            "condition": row["condition"], "algorithm": row["algorithm"],
            "dataset_id": row["dataset_id"], "seed": row["seed"],
            "logical_key": f"{row['algorithm'].casefold()}::{row['dataset_id']}::s{row['seed']}::{row['condition']}",
            "terminal_expression_sha256": row["terminal_expression_sha256"],
            "terminal_source_sha256": row["terminal_source_sha256"],
            "selected_result_sha256": row["selected_result_sha256"],
            "valid_output": row["valid_output"], "id_quality": row["id_quality"],
            "ood_quality": row["ood_quality"], "m_eff": row["m_eff"],
            "m_sym": row["m_sym"], "m_min": row["m_min"],
            "formal_ready": row["formal_ready"], "symbolic_basis": "current_terminal_frozen_judgments",
        })
    for row in read_csv(gplearn / "gplearn_run_metrics.csv"):
        run_rows.append({
            "condition": row["condition"], "algorithm": "gplearn",
            "dataset_id": row["dataset_id"], "seed": row["seed"],
            "logical_key": row["logical_key"],
            "terminal_expression_sha256": row["terminal_expression_sha256"],
            "terminal_source_sha256": row["terminal_snapshot_sha256"],
            "selected_result_sha256": row["selected_result_sha256"],
            "valid_output": "True", "id_quality": row["id_quality"],
            "ood_quality": row["ood_quality"], "m_eff": row["m_eff"],
            "m_sym": row["m_sym"], "m_min": row["m_min"],
            "formal_ready": row["formal_ready"],
            "symbolic_basis": "gplearn_native_protected_typed_prefix",
        })
    require(len(run_rows) == 6750 and len({_key(r["logical_key"]) for r in run_rows}) == 6750,
            "unified run metrics incomplete")
    require(all(r["formal_ready"] == "True" and all(r[k] for k in
                ("id_quality", "ood_quality", "m_eff", "m_sym", "m_min")) for r in run_rows),
            "unified run metrics contain unresolved axis")
    run_rows.sort(key=lambda r: (r["condition"], r["algorithm"].casefold(), r["dataset_id"], int(r["seed"])))
    write_csv(output / "metrics/run_six_axis_components.csv", run_rows, list(run_rows[0]))

    task_rows = []
    for row in read_csv(prior / "task_stability.csv"):
        if row["algorithm"].casefold() == "gplearn":
            continue
        task_rows.append({"condition": row["condition"], "algorithm": row["algorithm"],
                          "dataset_id": row["dataset_id"], "N": row["n"], "V": row["v"],
                          "C": row["c"], "m_stab": row["m_stab"],
                          "formal_ready": row["formal_ready"],
                          "symbolic_basis": "current_terminal_frozen_judgments"})
    for row in read_csv(gplearn / "gplearn_task_stability.csv"):
        task_rows.append({"condition": row["condition"], "algorithm": "gplearn",
                          "dataset_id": row["dataset_id"], "N": row["N"], "V": row["V"],
                          "C": row["C"], "m_stab": row["m_stab"],
                          "formal_ready": row["formal_ready"],
                          "symbolic_basis": "gplearn_native_protected_typed_prefix"})
    require(len(task_rows) == 2250 and len({(r["condition"],r["algorithm"].casefold(),r["dataset_id"])
                                             for r in task_rows}) == 2250,
            "unified task STAB incomplete")
    require(all(r["formal_ready"] == "True" and r["m_stab"] for r in task_rows),
            "unified task STAB contains unresolved score")
    task_rows.sort(key=lambda r: (r["condition"],r["algorithm"].casefold(),r["dataset_id"]))
    write_csv(output / "metrics/task_stability.csv", task_rows, list(task_rows[0]))
    return len(run_rows), len(task_rows)


def _copy_active_symbolic(source: Path, output: Path) -> int:
    strict = source / "evaluation_strict"
    symbolic = source / "symbolic"
    target = output / "symbolic_current"
    target.mkdir(parents=True, exist_ok=True)
    pred = [row for row in read_jsonl(strict / "active_prediction_binding.jsonl")
            if row["algorithm"].casefold() != "gplearn"]
    require(len(pred) == 6300, "expected 6300 non-gplearn prediction bindings")
    with (target / "non_gplearn_prediction_binding.jsonl").open("w", encoding="utf-8") as handle:
        for row in pred:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    for source_path, name in (
        (strict / "equivalence_effective_index.jsonl", "equivalence_effective_index.jsonl"),
        (strict / "structure_effective_index.jsonl", "structure_effective_index.jsonl"),
        (symbolic / "gt_reference_binding.jsonl", "gt_reference_binding.jsonl"),
    ):
        copy(source_path, target / name)
    active = {
        "prediction": {row["prediction_frozen_evaluation_key"] for row in pred
                       if row["processing_status"] == "ready" and row["valid_output"] is True},
        "equivalence": {row["evaluation_key"] for row in read_jsonl(
            target / "equivalence_effective_index.jsonl")},
        "structure": {row["evaluation_key"] for row in read_jsonl(
            target / "structure_effective_index.jsonl")},
        "ground_truth": {row["gt_frozen_evaluation_key"] for row in read_jsonl(
            target / "gt_reference_binding.jsonl")},
    }
    filtered = [row for row in read_csv(symbolic / "response_source_index.csv")
                if row["evaluation_key"] in active.get(row["kind"], set())]
    expected = sum(map(len, active.values()))
    require(len(filtered) == expected and len({(r["kind"],r["evaluation_key"]) for r in filtered}) == expected,
            f"active response source index incomplete: {len(filtered)}/{expected}")
    for row in filtered:
        source_blob = source / row["packaged_path"]
        require(sha(source_blob) == row["response_sha256"],
                f"active response blob SHA mismatch: {row['evaluation_key']}")
        relative = Path("symbolic_current/response_blobs") / row["response_sha256"][:2] / (
            row["response_sha256"] + ".json")
        destination = output / relative
        if not destination.exists():
            copy(source_blob, destination)
        row.update(current_response_reference(row))
    write_csv(target / "response_source_index.csv", filtered, list(filtered[0]))
    return expected


def build(source: Path, output: Path, zip_path: Path | None = None) -> dict:
    if output.exists() or (zip_path is not None and zip_path.exists()):
        raise ValueError("selected-only output or ZIP already exists")
    current = json.loads((source / "current_manifest.json").read_text(encoding="utf-8"))
    source_bundle_status = json.loads((source / "manifest.json").read_text(encoding="utf-8"))[
        "source_bundle_status"]
    require(current["all_15_current_terminal_formal_ready"] is True and current["run_count"] == 6750,
            "source is not the final current-terminal release")
    selection, terminals, archive_report = validate_selected_archive(source)
    output.mkdir(parents=True)
    shutil.copytree(source / "datasets", output / "datasets")
    shutil.copytree(source / "ground_truth", output / "ground_truth")
    shutil.copytree(source / "code", output / "code")
    shutil.copytree(source / "contracts", output / "contracts")
    shutil.copytree(source / "audits", output / "audits")
    copy(source / "algorithm_six_axis_15alg.csv", output / "metrics/algorithm_six_axis_15alg.csv")
    score_rows=list(read_csv(output / "metrics/algorithm_six_axis_15alg.csv"))
    require(len(score_rows)==45 and all(r["formal_ready"]=="True" and all(r[k] for k in AXES)
                                        for r in score_rows), "latest six-axis table incomplete")
    run_count, task_count = _unified_metrics(source, output)
    slim_selection=[]
    for row in selection:
        term=terminals[_key(row["logical_key"])]
        slim_selection.append(current_selection_row(row, term))
    write_csv(output / "selection/current_run_selection.csv",slim_selection,SELECTION_FIELDS)
    for name in ("selected_result_payloads.jsonl.gz","terminal_snapshot_payloads.jsonl.gz",
                 "selected_payloads_validation.json"):
        copy(source / "selection" / name, output / "selection" / name)
    copy(source / "evaluation_strict/terminal_inputs.jsonl.gz",
         output / "selection/terminal_inputs.jsonl.gz")
    shutil.copytree(source / "evaluation_strict/numerical", output / "numerical")
    shutil.copytree(source / "evaluation_strict/gplearn_current_terminal",
                    output / "symbolic_current/gplearn_current_terminal")
    response_count=_copy_active_symbolic(source,output)
    archive_source=source / "historical_source_raw_snapshots"
    copy(archive_source / "Core50_raw_snapshots_20260914.tar.zst",
         output / "snapshots/selected_run_physical_snapshots.tar.zst")
    copy(archive_source / "delivery_report.json", output / "snapshots/delivery_report.json")
    shutil.copytree(archive_source / "archive_metadata", output / "snapshots/archive_metadata")
    (output / "snapshots/README.md").write_text(
        "# Selected run snapshots\n\nThis is a byte-identical copy of the verified physical "
        "snapshot archive. Its frozen runs.csv contains exactly the 6,750 current selected "
        "run keys and matching trajectory-bundle SHA256 values. Every result member matches "
        "the selected result SHA256. All 1,215,000 logical minute bindings resolve to "
        "members in this archive, including recovery evidence used by the selected runs. "
        "The old archive filename reflected its build date, not superseded run content.\n",
        encoding="utf-8",
    )
    manifest={
        "schema_version":"core50_selected_only_current_terminal.v1",
        "status":"formal_ready_selected_runs_only",
        "run_count":run_count,"task_count":task_count,"algorithm_condition_rows":45,
        "conditions":list(CONDITIONS),"algorithms":15,"seeds":[520,521,522],
        "superseded_old_runs_included":0,
        "superseded_old_runs_excluded":archive_report["superseded_old_runs_excluded"],
        "selection":"selection/current_run_selection.csv",
        "terminal_inputs":"selection/terminal_inputs.jsonl.gz",
        "numerical_root":"numerical",
        "final_score_table":"metrics/algorithm_six_axis_15alg.csv",
        "run_metric_table":"metrics/run_six_axis_components.csv",
        "task_metric_table":"metrics/task_stability.csv",
        "selected_raw_snapshots":"snapshots/selected_run_physical_snapshots.tar.zst",
        "snapshot_audit":archive_report,
        "physical_source_gaps_carried_forward":17,
        "selected_source_bundle_copies_omitted":source_bundle_status["copied_unique_blobs"],
        "source_bundle_omission_reason":"Selected physical members and all logical minute bindings are retained in the verified archive; byte-for-byte original trajectory-container copies are not included.",
        "current_symbolic_response_count":response_count,
        "gplearn_symbolic_basis":current["gplearn_symbolic_basis"],
        "gplearn_tree_basis_note":current["gplearn_tree_basis_note"],
        "minute_symbolic_required":False,"minute_numerical_ready":True,
        "input_release_manifest_sha256":sha(source / "current_manifest.json"),
        "output_score_sha256":sha(output / "metrics/algorithm_six_axis_15alg.csv"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",
                                          encoding="utf-8")
    (output / "README.md").write_text(
        "# Core-50 selected-only release\n\nExactly one current run is kept for every "
        "condition/algorithm/dataset/seed key. The 624 superseded runs and all historical "
        "leaderboards and caches are omitted. Copies of 333 selected trajectory source "
        "containers are also omitted as redundant: their selected physical members and "
        "minute bindings remain in the verified snapshot archive, but byte-for-byte "
        "reinspection of those original containers requires the larger provenance release. "
        "The 6,750 selected "
        "result and terminal payloads, current 180-minute ID/OOD/EFF trajectories, final "
        "six-axis components, datasets, fixed Ground Truth, active model judgments, and "
        "physical selected-run snapshots remain. gplearn uses current native protected "
        "prefix evidence; its typed-vs-GT tree vocabulary difference is recorded in manifest. "
        "No minute-level SYM/MIN/STAB is claimed.\n",
        encoding="utf-8",
    )
    with (output / "SHA256SUMS").open("w",encoding="utf-8") as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path!=output / "SHA256SUMS":
                handle.write(f"{sha(path)}  {path.relative_to(output)}\n")
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zip_path,"x",compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1,allowZip64=True) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path,arcname=str(Path(output.name)/path.relative_to(output)))
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--zip",type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output,args.zip),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
