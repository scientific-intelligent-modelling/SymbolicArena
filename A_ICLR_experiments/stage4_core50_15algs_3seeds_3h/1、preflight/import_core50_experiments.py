from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = HERE.parent
OLD = ROOT / "A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h"
EXPERIMENTS = STAGE / "2.2 core50 experiments"
HISTORY = ROOT / ".agent/work/EXP-001"
SUPPORT = HERE / "reuse_support"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def copy_verified(source, destination):
    if source.is_symlink():
        raise ValueError(f"源文件包含符号链接: {source}")
    data = source.read_bytes()
    digest = sha_bytes(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha_bytes(destination.read_bytes()) == digest, destination
        return digest, len(data)
    with destination.open("xb") as handle:
        handle.write(data)
    assert sha_bytes(destination.read_bytes()) == digest
    return digest, len(data)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def opus_index():
    database = HISTORY / "opus_postprocess/state/opus_pred_simplify.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        records = {key: {"evaluation_key": key, "result_path": path, "result_sha256": digest,
                         "source_index": str(database)} for key, path, digest in connection.execute(
                             "SELECT evaluation_key,result_path,result_sha256 FROM frozen_results")}
    indices = []
    for path in [HISTORY / "oversample/opus/selected.json", HISTORY / "followup/base/selected.json"]:
        data = path.read_bytes()
        indices.append({"path": str(path), "sha256": sha_bytes(data)})
        for key, record in json.loads(data).items():
            if record.get("state") != "frozen":
                continue
            records[key] = dict(record, source_index=str(path))
    return records, indices


def copy_run(row):
    from experiment_paths import relocated_path

    destination = relocated_path(row["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    if row["training_state"] == "pending":
        return row
    source = Path(row["source"])
    binding_path = destination / "import_binding.json"
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        assert binding["source_directory"] == str(source)
        assert relocated_path(binding["destination_directory"]) == destination
        assert binding["result_sha256"] == row["result_sha256"]
        assert sha_bytes((destination / "result.json").read_bytes()) == row["result_sha256"]
        paths = [Path(current) / name for current, _, names in os.walk(source) for name in names]
        assert len(paths) == binding["copied_file_count"]
        row.update(copied_files=len(paths), copied_bytes=sum(path.stat().st_size for path in paths),
                   tree_sha256=binding["verified_tree_sha256"],
                   progress_files=sum(path.parent.name == "progress" and path.name.startswith("minute_") for path in paths))
        return row
    subprocess.run(["cp", "-a", "--reflink=auto", str(source) + "/.", str(destination)], check=True)
    paths = [Path(current) / name for current, _, names in os.walk(source) for name in names]
    count = len(paths)
    size = sum(path.stat().st_size for path in paths)
    progress_count = sum(path.parent.name == "progress" and path.name.startswith("minute_") for path in paths)
    copied = [Path(current) / name for current, _, names in os.walk(destination) for name in names]
    assert len(copied) == count and sum(path.stat().st_size for path in copied) == size
    assert sha_bytes((destination / "result.json").read_bytes()) == row["result_sha256"]
    row.update(copied_files=count, copied_bytes=size, tree_sha256="", progress_files=progress_count)
    write_json(destination / "import_binding.json", {
        "source_directory": str(source), "destination_directory": str(destination),
        "result_sha256": row["result_sha256"], "copied_file_count": count,
        "verified_tree_sha256": row["tree_sha256"],
        "verification": "cp返回成功、文件数量和总字节数一致、result.json SHA256一致",
        "preserve_original_artifacts": True,
        "historical_local_prefix": str(STAGE / "2、experiments"),
        "archived_source_prefix": str(OLD / "2、experiments"),
        "reference_policy": "原始JSON和来源路径保留；当前位置由本文件及reuse_manifest.csv索引"})
    return row


def main():
    assert not (HERE / "reuse_summary.json").exists(), "导入已经完成"
    archive = HERE / "core50_selection_outputs.zip"
    with ZipFile(archive) as source:
        content = source.read("core50.csv")
    datasets = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    by_id = {row["dataset_id"]: row for row in datasets}
    by_name = {row["dataset_name"]: row for row in datasets}
    assert len(datasets) == len(by_id) == len(by_name) == 50
    config_path = OLD / "experiment_config.json"
    config_data = config_path.read_bytes()
    config = json.loads(config_data)
    previous = {row["dataset_id"]: row for row in config["dataset_selection"]["datasets"]}
    common = set(by_id) & set(previous)
    new = set(by_id)-set(previous)
    assert len(common) == 35 and len(new) == 15
    algorithms = list(config["algorithms"])
    canonical = {name.lower(): name for name in algorithms}
    assert len(algorithms) == len(canonical) == 15
    seeds = config["experiment"]["seeds"]
    conditions = config["experiment"]["conditions"]
    assert seeds == [520, 521, 522]
    assert [c["name"] for c in conditions] == ["clean", "noise001", "noise005"]
    rows = []
    for condition in conditions:
        for algorithm in algorithms:
            for dataset in datasets:
                identity, name = dataset["dataset_id"], dataset["dataset_name"]
                assert Path(name).name == name and name not in {".", ".."}
                directory = previous[identity]["directory_name"] if identity in common else name
                if identity in common:
                    assert dataset["dataset_rel"] == previous[identity]["dataset_rel"]
                for seed in seeds:
                    source = OLD / "2、experiments" / algorithm / directory / condition["name"] / str(seed)
                    destination = EXPERIMENTS / condition["name"] / algorithm / name / str(seed)
                    record = {"noise": condition["name"], "noise_sigma": condition["sigma"],
                              "algorithm": algorithm, "dataset_id": identity, "dataset_name": name,
                              "dataset_rel": dataset["dataset_rel"], "seed": seed,
                              "training_budget_seconds": config["experiment"]["training_budget_seconds"],
                              "params_reference": str(config_path), "params_key": f"algorithms.{algorithm}",
                              "source": str(source) if identity in common else "", "destination": str(destination),
                              "training_state": "pending", "pending_reason": "new_dataset",
                              "result_sha256": "", "execution_status": "", "numeric_available": False,
                              "formula_available": False, "opus_status": "awaiting_training",
                              "opus_artifact_count": 0, "copied_files": 0, "copied_bytes": 0,
                              "tree_sha256": "", "progress_files": 0}
                    if identity in common:
                        result_path = source / "result.json"
                        record["pending_reason"] = "missing_result"
                        if result_path.exists():
                            data = result_path.read_bytes()
                            payload = json.loads(data)
                            assert int(payload["task_global_index"]) == int(identity[1:])
                            assert payload["expected_dataset_rel"] == dataset["dataset_rel"]
                            assert payload["tool"].lower() == algorithm.lower() and payload["seed"] == seed
                            assert payload["dataset_identity_check"]["match"] is True
                            provenance = json.loads((source / "provenance.json").read_text())
                            assert provenance["condition"] == condition["name"] and provenance["seed"] == seed
                            assert provenance["global_index"] == int(identity[1:])
                            formula = payload.get("equation") or (payload.get("canonical_artifact") or {}).get("instantiated_expression")
                            numeric = all(isinstance(payload.get(split), dict)
                                          and isinstance(payload[split].get("nmse"), (int, float))
                                          and math.isfinite(payload[split]["nmse"]) for split in ["id_test", "ood_test"])
                            record.update(training_state="copied", pending_reason="", result_sha256=sha_bytes(data),
                                          execution_status=payload["status"], formula_available=bool(formula),
                                          numeric_available=numeric, opus_status="missing")
                    rows.append(record)
    assert len(rows) == 6750
    print(json.dumps({"overlap_datasets": len(common), "new_datasets": len(new),
                      "existing_results": sum(r["training_state"] == "copied" for r in rows),
                      "pending_training": sum(r["training_state"] == "pending" for r in rows)}), flush=True)
    EXPERIMENTS.mkdir(exist_ok=True)
    SUPPORT.mkdir(exist_ok=True)
    if (SUPPORT / "core50.csv").exists():
        assert (SUPPORT / "core50.csv").read_bytes() == content
    else:
        with (SUPPORT / "core50.csv").open("xb") as handle:
            handle.write(content)
    copy_verified(config_path, SUPPORT / "source_core80_config.json")
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(copy_run, rows))
    print("training_files_copied", flush=True)
    by_key = {(r["noise"], r["algorithm"].lower(), r["dataset_id"], r["seed"]): r for r in rows}
    records, indices = opus_index()
    evidence, rejected = [], Counter()
    stale = Counter()
    ground_truth = defaultdict(list)
    for key, binding in records.items():
        logical = binding.get("logical_id", "")
        if logical and not logical.startswith(("pred_simplify::", "gt_simplify::")):
            continue
        path = Path(binding["result_path"])
        data = path.read_bytes()
        assert sha_bytes(data) == binding["result_sha256"], path
        frozen = json.loads(data)
        kind = frozen["task_type"]
        assert frozen["evaluation_key"] == key
        if kind not in {"pred_simplify", "gt_simplify"}:
            continue
        request = frozen["request"]
        if kind == "gt_simplify":
            name = request["dataset_id"]
            if name not in by_name:
                continue
            if frozen.get("validation", {}).get("ok") is not True:
                rejected["gt_validation_failed"] += 1
                continue
            destination = SUPPORT / "ground_truth" / by_name[name]["dataset_id"] / f"{key}.json"
            ground_truth[name].append(str(destination))
            row = None
        else:
            run_key = (request["noise_tag"], request["algorithm_slug"].lower(),
                       request["dataset_index"], int(request["seed"]))
            if run_key not in by_key:
                continue
            row = by_key[run_key]
            if row["training_state"] != "copied":
                continue
            if frozen.get("validation", {}).get("ok") is not True:
                rejected["pred_validation_failed"] += 1
                continue
            expected_hash = request["ast_source_evidence"]["result_raw_sha256"]
            if expected_hash != row["result_sha256"]:
                stale[run_key] += 1
                rejected["source_result_changed"] += 1
                continue
            assert request["dataset_id"] == row["dataset_name"]
            destination = Path(row["destination"]) / "opus/pred_simplify" / f"{key}.json"
            row["opus_status"] = "validated_bound"
            row["opus_artifact_count"] += 1
        copy_verified(path, destination)
        evidence.append({"task_type": kind, "evaluation_key": key, "source": str(path),
                         "destination": str(destination), "sha256": binding["result_sha256"],
                         "source_index": binding["source_index"], "validation_ok": True,
                         "logical_id": frozen["logical_id"], "outcome": frozen["structured_output"]["outcome"],
                         "result_binding_verified": row is not None,
                         "historical_result_path": request.get("ast_source_evidence", {}).get("result_path", "")})
    for key, row in by_key.items():
        if row["training_state"] != "copied":
            continue
        if row["opus_status"] == "missing":
            row["opus_status"] = "source_changed" if stale[key] else "missing"
            if not row["formula_available"]:
                row["opus_status"] = "unavailable_expression"
    pending = [r for r in rows if r["training_state"] == "pending"]
    missing_opus = [r for r in rows if r["training_state"] == "copied" and r["opus_status"] != "validated_bound"]
    fields = list(rows[0])
    write_csv(HERE / "reuse_manifest.csv", rows, fields)
    write_csv(HERE / "pending_training.csv", pending, fields)
    write_csv(HERE / "pending_opus.csv", missing_opus, fields)
    write_csv(HERE / "pending_datasets.csv", [by_id[i] for i in sorted(new)],
              ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup"])
    write_json(SUPPORT / "opus_bindings.json", evidence)
    write_json(SUPPORT / "ground_truth_index.json", dict(ground_truth))
    for row in rows:
        destination = Path(row["destination"])
        assert destination.is_dir()
        if row["training_state"] == "copied":
            assert sha_bytes((destination / "result.json").read_bytes()) == row["result_sha256"]
        else:
            assert not (destination / "result.json").exists()
    report = {"schema": "core50_reuse.v1", "layout": "noise/algorithm/dataset_name/seed",
              "source_selection": str(archive), "selection_archive_sha256": sha_bytes(archive.read_bytes()),
              "selection_csv_sha256": sha_bytes(content), "source_config_sha256": sha_bytes(config_data),
              "script_sha256": sha_bytes(Path(__file__).read_bytes()), "algorithm_count": len(algorithms),
              "dataset_count": 50, "overlap_datasets": len(common), "new_datasets": len(new),
              "total_runs": len(rows), "copied_runs": sum(r["training_state"] == "copied" for r in rows),
              "pending_training_runs": len(pending), "pending_training_reasons": dict(Counter(r["pending_reason"] for r in pending)),
              "copied_files": sum(r["copied_files"] for r in rows), "copied_bytes": sum(r["copied_bytes"] for r in rows),
              "copied_progress_files": sum(r["progress_files"] for r in rows),
              "imported_runs_with_180_progress_files": sum(r["progress_files"] == 180 for r in rows if r["training_state"] == "copied"),
              "numeric_available_runs": sum(r["numeric_available"] for r in rows),
              "opus_status_counts": dict(Counter(r["opus_status"] for r in rows)),
              "opus_pending_existing_runs": len(missing_opus), "opus_evidence_files": len(evidence),
              "ground_truth_datasets": len(ground_truth), "rejected_opus_bindings": dict(rejected),
              "opus_indices": indices, "source_unchanged": True, "terminal_and_opus_sha256_verified": True,
              "copied_file_counts_and_sizes_verified": True,
              "training_launched": False, "api_called": False, "minute_level_opus_complete": False}
    write_json(HERE / "reuse_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
