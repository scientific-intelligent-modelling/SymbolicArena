from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import evaluate_core50 as scoring


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preflight = Path(__file__).parent
    previous = json.loads((preflight / "core50_mae.json").read_text())
    assert sha(scoring.SOURCE) == previous["source_sha256"]
    assert sha(Path(scoring.__file__)) == previous["script_sha256"]
    archive = scoring.ROOT / "A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h"
    config_path = archive / "experiment_config.json"
    config = json.loads(config_path.read_text())
    selection = config["dataset_selection"]
    reference_path = archive / Path(selection["reference"]["path"]).relative_to(
        "A_ICLR_experiments/stage4_core80_15algs_3seeds_3h")
    assert sha(reference_path) == selection["reference"]["sha256"]
    identifiers = [row["dataset_id"] for row in selection["datasets"]]
    assert len(identifiers) == len(set(identifiers)) == 80
    with scoring.SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7968 and {row["method"] for row in rows} == set(scoring.PROBES)
    paths = {row["dataset_id"]: row["dataset_rel"] for row in rows}
    assert all(paths[row["dataset_id"]] == row["dataset_rel"] for row in selection["datasets"])
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["dataset_id"]].append(row)
    reports = []
    for probe in scoring.PROBES:
        scores = {dataset: scoring.response(group) for (method, dataset), group in groups.items() if method == probe}
        assert len(scores) == 664 and set(identifiers).issubset(scores)
        full_mean = mean(scores.values())
        core_mean = mean(scores[dataset] for dataset in identifiers)
        reports.append({"probe": probe, "full664_mean_response": full_mean,
                        "core80_mean_response": core_mean, "signed_error": core_mean - full_mean,
                        "absolute_error": abs(core_mean - full_mean)})
    report = {"dataset_count": 80, "full_dataset_count": 664, "probe_count": 4,
              "dataset_ids": identifiers, "score_version": previous["score_version"],
              "score_parameters": previous["score_parameters"],
              "source": previous["source"], "source_sha256": previous["source_sha256"],
              "selection_configuration": str(config_path.relative_to(scoring.ROOT)),
              "selection_configuration_sha256": sha(config_path),
              "selection_reference_sha256": sha(reference_path),
              "script_sha256": sha(Path(__file__)), "scoring_helper_sha256": sha(Path(scoring.__file__)),
              "historical_native_replay_verified": False, "role": "archived_core80_probe_diagnostic",
              "probe_results": reports, "mae": mean(row["absolute_error"] for row in reports)}
    with (preflight / "core80_mae.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"mae": report["mae"], "probe_results": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
