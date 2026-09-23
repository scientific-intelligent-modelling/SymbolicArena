from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "A_ICLR_experiments/stage3_664dats_4probes_3seeds_1h/probe4_current_run_level_raw_digest_7968.csv"
PROBES = ("dso", "imcts", "pyoperon", "udsr")
DATASET_IDS = (
    "g0620", "g0625", "g0649", "g0651", "g0018", "g0020", "g0024", "g0025", "g0027", "g0028",
    "g0052", "g0053", "g0066", "g0105", "g0130", "g0151", "g0168", "g0184", "g0208", "g0233",
    "g0636", "g0641", "g0644", "g0496", "g0524", "g0542", "g0578", "g0594", "g0595", "g0601",
    "g0602", "g0604", "g0608", "g0611", "g0266", "g0270", "g0277", "g0290", "g0292", "g0320",
    "g0358", "g0365", "g0371", "g0390", "g0399", "g0406", "g0413", "g0436", "g0461", "g0660",
)


def response(rows):
    assert len(rows) == 3 and {int(row["seed"]) for row in rows} == {520, 521, 522}
    logs = []
    for row in rows:
        assert row["state"] in {"done", "failed"} and row["result_dataset_identity_match"] == "1"
        assert row["result_valid_output"] in {"0", "1"}
        if row["result_valid_output"] == "0":
            continue
        values = [float(row[field]) for field in ("result_id_test_nmse", "result_ood_test_nmse")]
        assert all(math.isfinite(value) and value >= 0 for value in values)
        logs.append([min(12.0, max(-12.0, math.log10(max(value, 1e-12)))) for value in values])
    if not logs:
        return -12.0
    mean_id = mean(pair[0] for pair in logs)
    mean_ood = mean(pair[1] for pair in logs)
    return max(-12.0, -(mean_id + mean_ood) / 2 - 2 * (1 - len(logs) / 3))


def main():
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_sha == "1a588501a0b31b5b392592b05c3105489aee18ab19e3cffb190102e6bee5ddcb"
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7968 and len(set(DATASET_IDS)) == 50
    assert {row["method"] for row in rows} == set(PROBES)
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["dataset_id"]].append(row)
    reports = []
    for probe in PROBES:
        scores = {dataset: response(group) for (method, dataset), group in groups.items() if method == probe}
        assert len(scores) == 664 and set(DATASET_IDS).issubset(scores)
        full_mean = mean(scores.values())
        core_mean = mean(scores[dataset] for dataset in DATASET_IDS)
        reports.append({"probe": probe, "full664_mean_response": full_mean,
                        "core50_mean_response": core_mean, "signed_error": core_mean - full_mean,
                        "absolute_error": abs(core_mean - full_mean)})
    report = {"dataset_count": 50, "full_dataset_count": 664, "probe_count": 4,
              "dataset_ids": list(DATASET_IDS), "score_version": "probe_consensus_response_v1",
              "score_parameters": {"nmse_floor": 1e-12, "log_clip": [-12, 12],
                  "seed_aggregation": "arithmetic_mean_valid_paired_seeds", "expected_seeds": [520, 521, 522],
                  "failure_penalty": 2, "response_floor": -12, "all_invalid_response": -12},
              "source": str(SOURCE.relative_to(ROOT)), "source_sha256": source_sha,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "historical_native_replay_verified": False, "role": "fixed_user_selection_probe_diagnostic",
              "probe_results": reports, "mae": mean(row["absolute_error"] for row in reports)}
    path = Path(__file__).with_name("core50_mae.json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"mae": report["mae"], "probe_results": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
