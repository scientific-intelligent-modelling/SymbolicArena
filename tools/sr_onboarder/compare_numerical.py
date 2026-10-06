import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality_score(nmse):
    if not isinstance(nmse, (int, float)) or not math.isfinite(nmse) or nmse < 0:
        raise ValueError("The run does not have a finite nonnegative NMSE")
    return 100 * (2 - min(2, max(-12, math.log10(max(nmse, 1e-12))))) / 14


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-root", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--paper", default="docs/paper-results.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    local = Path(args.collected_root).resolve()
    remote = Path(args.remote_root)
    acceptance_path = local / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    paper_path = Path(args.paper)
    paper = json.loads(paper_path.read_text())
    if not acceptance["passed"] or len(acceptance["core50_runs"]) != 50:
        raise ValueError("The complete accepted Core50 run set is required")
    if len({run["dataset_id"] for run in acceptance["core50_runs"]}) != 50:
        raise ValueError("The run set contains duplicate dataset IDs")
    tasks = []
    durations = []
    configurations = []
    for run in acceptance["core50_runs"]:
        result_path = local / Path(run["result_path"]).relative_to(remote)
        if digest(result_path) != run["result_sha256"]:
            raise ValueError("The collected result checksum differs")
        result = json.loads(result_path.read_text())
        if result["status"] != "ok" or result["condition"] != "clean":
            raise ValueError("Comparison requires usable clean-condition results")
        scores = {axis: quality_score(result[split]["nmse"]) for axis, split in (("ID", "id_test"), ("OOD", "ood_test"))}
        for axis in scores:
            recorded = result["six_axis_evidence"][axis]["score"]
            if not math.isclose(scores[axis], recorded, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Recomputed score differs from the recorded quality")
        tasks.append({"dataset_id": run["dataset_id"], "seed": result["seed"], "ID": scores["ID"], "OOD": scores["OOD"], "id_nmse": result["id_test"]["nmse"], "ood_nmse": result["ood_test"]["nmse"], "result_sha256": run["result_sha256"]})
        durations.append(result["seconds"])
        configurations.append({key: result["params"][key] for key in ("niterations", "epochs", "samples_first", "diversification_steps", "n_jobs", "timeout_in_seconds")})
    if any(config != configurations[0] for config in configurations):
        raise ValueError("Run configurations are inconsistent")
    means = {axis: math.fsum(task[axis] for task in tasks) / len(tasks) for axis in ("ID", "OOD")}
    rows = [{"algorithm": item["name"], "ID": item["clean"][0], "OOD": item["clean"][1], "experiment": "paper_final"} for item in paper["algorithms"]]
    rows.append({"algorithm": acceptance["tool_name"].upper(), **means, "experiment": "integration_acceptance"})
    for row in rows:
        for axis in ("ID", "OOD"):
            row[axis + "_rank"] = 1 + sum(other[axis] > row[axis] for other in rows)
    rows.sort(key=lambda row: row["ID"], reverse=True)
    report = {
        "schema": "symbolicarena-numerical-reference-comparison-v1",
        "condition": "clean",
        "quality_protocol": {"epsilon": 1e-12, "log_min": -12, "log_max": 2, "display_scale": 100},
        "acceptance_sha256": digest(acceptance_path),
        "paper_sha256": digest(paper_path),
        "paper_source": paper["source"],
        "code_revision": acceptance["code_revision"],
        "DGP_scores": means,
        "DGP_parameters": configurations[0],
        "DGP_seeds": sorted({task["seed"] for task in tasks}),
        "DGP_seconds": {"min": min(durations), "mean": math.fsum(durations) / len(durations), "max": max(durations)},
        "paper_protocol": {"tasks": 50, "seeds": [520, 521, 522], "budget_seconds": 10800},
        "formal_comparison": False,
        "rankings": rows,
        "per_task": sorted(tasks, key=lambda task: task["dataset_id"]),
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "rankings.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["algorithm", "ID", "ID_rank", "OOD", "OOD_rank", "experiment"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ("DGP_scores", "DGP_parameters", "DGP_seconds", "DGP_seeds", "formal_comparison")}, indent=2))


if __name__ == "__main__":
    main()
