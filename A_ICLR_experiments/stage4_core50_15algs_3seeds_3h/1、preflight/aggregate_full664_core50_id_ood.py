import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METRIC_ROOT = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
sys.path.insert(0, str(METRIC_ROOT))
from pipeline.metrics import aggregate_quality, phi_nmse


OUTPUT = HERE.parent / "3、metrics/full664_core50_10m"
ALGORITHMS = ("drsr", "dso", "e2esr", "fepysr", "gplearn", "imcts", "jaxsr", "llmsr",
              "pyoperon", "pysr", "qlattice", "ragsr", "symbolfit", "tpsr", "udsr")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest_path = HERE / "full664_10m/collection.csv"
    population_path = HERE / "full664_10m/datasets.csv"
    core_path = HERE / "core50_selection_outputs.zip"
    manifest_bytes = manifest_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"))))
    with population_path.open(newline="") as handle:
        population = {row["dataset_id"]: row["dataset_rel"] for row in csv.DictReader(handle)}
    with zipfile.ZipFile(core_path) as archive:
        names = [name for name in archive.namelist() if name == "core50.csv"]
        assert len(names) == 1
        core_bytes = archive.read(names[0])
        core = list(csv.DictReader(io.StringIO(core_bytes.decode("utf-8-sig"))))
    core_ids = {row["dataset_id"] for row in core}
    assert len(population) == 664 and len(core) == len(core_ids) == 50
    assert all(population[row["dataset_id"]] == row["dataset_rel"] for row in core)
    assert len(rows) == 9960
    expected = {(algorithm, task) for algorithm in ALGORITHMS for task in population}
    seen, inputs = set(), []
    values = {algorithm: {} for algorithm in ALGORITHMS}
    invalid = {algorithm: {axis: {"full664": 0, "core50": 0} for axis in ("ID", "OOD")} for algorithm in ALGORITHMS}
    for row in rows:
        identity = (row["algorithm"], row["dataset_id"])
        assert identity in expected and identity not in seen
        seen.add(identity)
        assert row["dataset_rel"] == population[identity[1]]
        assert row["collection_state"] == "collected" and row["noise"] == "clean"
        assert int(row["seed"]) == 1314 and int(row["training_budget_seconds"]) == 600
        path = Path(row["destination"]) / "result.json"
        data = path.read_bytes()
        assert sha(data) == row["result_sha256"], str(path)
        result = json.loads(data)
        assert result["tool"].lower() == identity[0] and int(result["task_global_index"]) == int(identity[1][1:])
        assert result["expected_dataset_rel"] == row["dataset_rel"] and result["dataset_identity_check"]["match"] is True
        assert result["seed"] == 1314 and result["params"]["timeout_in_seconds"] == 600
        item = {"algorithm": identity[0], "dataset_id": identity[1], "dataset_rel": row["dataset_rel"],
                "is_core50": identity[1] in core_ids, "result_sha256": row["result_sha256"]}
        raw = {}
        for axis, split in (("ID", "id_test"), ("OOD", "ood_test")):
            metrics = result.get(split)
            nmse = metrics.get("nmse") if isinstance(metrics, dict) else None
            score = 100 * phi_nmse(nmse)
            assert math.isfinite(score) and 0 <= score <= 100
            valid = isinstance(nmse, (int, float)) and not isinstance(nmse, bool) and math.isfinite(nmse) and nmse >= 0
            if nmse is not None and not isinstance(nmse, (int, float)):
                raise ValueError(f"NMSE类型无法确认: {identity}, {axis}, {type(nmse)}")
            if not valid:
                assert score == 0
                invalid[identity[0]][axis]["full664"] += 1
                invalid[identity[0]][axis]["core50"] += int(item["is_core50"])
            raw[axis] = nmse
            item[f"{axis}_nmse"] = "" if nmse is None else nmse
            item[f"{axis}_score"] = score
        values[identity[0]][identity[1]] = (item, raw)
        inputs.append(item)
    assert seen == expected
    summaries = {}
    for axis in ("ID", "OOD"):
        summaries[axis] = []
        for algorithm in ALGORITHMS:
            all_tasks = list(values[algorithm])
            selected = [task for task in all_tasks if task in core_ids]
            assert len(all_tasks) == 664 and len(selected) == 50
            means = {}
            for name, tasks in (("full664", all_tasks), ("core50", selected)):
                scores = [values[algorithm][task][0][f"{axis}_score"] for task in tasks]
                mean = math.fsum(scores) / len(tasks)
                reference = aggregate_quality(values[algorithm][task][1][axis] for task in tasks)
                assert math.isclose(mean, reference, abs_tol=1e-10, rel_tol=0)
                means[f"{name}_mean"] = mean
            summaries[axis].append({"algorithm": algorithm, **means})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not any((OUTPUT / name).exists() for name in ("id_mean.csv", "ood_mean.csv", "score_inputs.csv", "manifest.json"))
    for axis in ("ID", "OOD"):
        write_csv(OUTPUT / f"{axis.lower()}_mean.csv", ("algorithm", "full664_mean", "core50_mean"), summaries[axis])
    write_csv(OUTPUT / "score_inputs.csv", tuple(inputs[0]), sorted(inputs, key=lambda row: (row["algorithm"], row["dataset_id"])))
    evidence = {"condition": "clean", "seed": 1314, "budget_seconds": 600, "scale": "0-100", "higher_is_better": True,
        "metric": "100 * phi_nmse(result.id_test.nmse or result.ood_test.nmse)",
        "formula": "100 * (1 - (clip(log10(max(nmse, 1e-12)), -12, 2) + 12) / 14)",
        "aggregation": "equal-weight arithmetic mean of per-task scores; invalid numeric results score zero",
        "denominators": {"full664": 664, "core50": 50}, "core50_from_same_runs": True,
        "core50_ids": sorted(core_ids), "algorithms": list(ALGORITHMS), "verified_run_count": len(seen),
        "invalid_nmse_counts": invalid,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in
                         (manifest_path, population_path, core_path, METRIC_ROOT / "pipeline/metrics.py")},
        "core50_csv_inside_zip_sha256": sha(core_bytes), "script_sha256": sha(Path(__file__).read_bytes()),
        "output_sha256": {path.name: sha(path.read_bytes()) for path in OUTPUT.glob("*.csv")}}
    (OUTPUT / "manifest.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "rows_per_summary": 15, "verified_runs": len(seen),
                      "full664_tasks": 664, "core50_tasks": 50, "scale": "0-100"}, ensure_ascii=False))
    for axis in ("ID", "OOD"):
        print(axis)
        print((OUTPUT / f"{axis.lower()}_mean.csv").read_text(), end="")


if __name__ == "__main__":
    main()
