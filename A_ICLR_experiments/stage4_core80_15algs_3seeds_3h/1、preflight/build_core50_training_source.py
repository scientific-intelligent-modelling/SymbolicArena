import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


PREFLIGHT = Path(__file__).resolve().parent
ROOT = PREFLIGHT.parents[2]
ARCHIVE = PREFLIGHT / "core50_selection_outputs.zip"
SUMMARY = PREFLIGHT / "reuse_summary.json"
PENDING = PREFLIGHT / "pending_datasets.csv"
FROZEN_CONFIG = ROOT / "A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/experiment_config.json"
PARAMS = ROOT / ".agent/work/EXP-001/controller/params"
SOURCE_CSV = PREFLIGHT / "core50_new15_training_source.csv"
INPUT_MANIFEST = PREFLIGHT / "core50_new15_input_manifest.json"
REQUIRED = ("metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not SOURCE_CSV.exists() and not INPUT_MANIFEST.exists()
    summary_bytes = SUMMARY.read_bytes()
    summary = json.loads(summary_bytes)
    frozen_bytes = FROZEN_CONFIG.read_bytes()
    frozen = json.loads(frozen_bytes)
    assert summary["copied_runs"] == 4725 and summary["pending_training_runs"] == 2025
    with ZipFile(ARCHIVE) as archive:
        selected_data = archive.read("selection_scores_664.csv")
    selected_rows = list(csv.DictReader(io.StringIO(selected_data.decode("utf-8-sig"))))
    with PENDING.open(encoding="utf-8", newline="") as handle:
        pending = list(csv.DictReader(handle))
    pending_ids = {row["dataset_id"] for row in pending}
    by_id = {row["dataset_id"]: row for row in selected_rows}
    assert len(by_id) == 664 and len(pending) == len(pending_ids) == 15
    assert set(pending_ids).isdisjoint(
        {row["dataset_id"] for row in frozen["dataset_selection"]["datasets"]})
    source_rows, file_records = [], []
    algorithms = list(frozen["algorithms"])
    assert len(algorithms) == 15
    conditions = {condition["name"]: condition["sigma"] for condition in frozen["experiment"]["conditions"]}
    for dataset_id in sorted(pending_ids, key=lambda value: int(value[1:])):
        row = by_id[dataset_id]
        assert row["selected_core50"].lower() == "true"
        assert row["dataset_rel"] == next(item["dataset_rel"] for item in pending
                                           if item["dataset_id"] == dataset_id)
        record = dict(row)
        record["dataset_dir"] = record["dataset_rel"]
        assert int(record["global_index"]) == int(dataset_id[1:])
        path = ROOT / record["dataset_rel"]
        fingerprints = {}
        for filename in REQUIRED:
            file_path = path / filename
            data = file_path.read_bytes()
            assert data and not data.startswith(b"version https://git-lfs.github.com/spec/v1"), file_path
            fingerprints[filename] = {"size": len(data), "sha256": sha256(data)}
        source_rows.append(record)
        file_records.append({"dataset_id": dataset_id, "dataset_name": record["dataset_name"],
                             "dataset_rel": record["dataset_rel"], "global_index": int(record["global_index"]),
                             "target_name": record["target_name"], "n_features": int(record["feature_count"]),
                             "files": fingerprints})
    parameter_hashes = {}
    for algorithm in algorithms:
        algorithm_key = algorithm.lower()
        for condition in conditions:
            path = PARAMS / f"{algorithm_key}__{condition}.json"
            payload = json.loads(path.read_text())
            assert int(payload["timeout_in_seconds"]) == frozen["experiment"]["training_budget_seconds"]
            assert int(payload["progress_snapshot_interval_seconds"]) == frozen["experiment"]["snapshot_interval_seconds"]
            parameter_hashes[f"{algorithm_key}__{condition}"] = sha256(path.read_bytes())
    fields = list(source_rows[0])
    assert len(fields) == len(set(fields))
    with SOURCE_CSV.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(source_rows)
    manifest = {"schema": "core50_new15_training_inputs.v1",
                "source_selection_archive": str(ARCHIVE),
                "source_selection_archive_sha256": sha256(ARCHIVE.read_bytes()),
                "selected_core50_sha256": sha256(selected_data),
                "frozen_core80_config": str(FROZEN_CONFIG),
                "frozen_core80_config_sha256": sha256(frozen_bytes),
                "reuse_summary_sha256": sha256(summary_bytes),
                "training_source_csv": str(SOURCE_CSV),
                "training_source_csv_sha256": sha256(SOURCE_CSV.read_bytes()),
                "dataset_count": len(file_records), "datasets": file_records,
                "algorithms": algorithms, "parameter_sha256": parameter_hashes,
                "seeds": frozen["experiment"]["seeds"], "conditions": conditions,
                "per_run_budget_seconds": frozen["experiment"]["training_budget_seconds"],
                "snapshot_interval_seconds": frozen["experiment"]["snapshot_interval_seconds"],
                "expected_runs": len(file_records)*len(algorithms)*len(conditions)*len(frozen["experiment"]["seeds"]),
                "llm_training_algorithms": ["llmsr", "drsr"],
                "llm_training_dispatch_authorized": False}
    assert manifest["expected_runs"] == 2025
    with INPUT_MANIFEST.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"datasets": len(file_records), "algorithms": len(algorithms),
                      "conditions": len(conditions), "seeds": len(frozen["experiment"]["seeds"]),
                      "expected_runs": manifest["expected_runs"],
                      "source_sha256": manifest["training_source_csv_sha256"],
                      "inputs_sha256": sha256(INPUT_MANIFEST.read_bytes())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
