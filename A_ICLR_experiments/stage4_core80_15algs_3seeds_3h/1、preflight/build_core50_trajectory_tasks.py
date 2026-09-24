import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


CONDITIONS = ("clean", "noise001", "noise005")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield row


def build_tasks(freeze_root, output_root):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"trajectory task output is not empty: {output_root}")

    per_condition = {}
    task_counts = {}
    inner_counts = {}
    for condition in CONDITIONS:
        freeze_path = freeze_root / f"{condition}_runs.jsonl.gz"
        if not freeze_path.is_file():
            raise FileNotFoundError(freeze_path)
        runs_by_host = defaultdict(list)
        seen = set()
        count = 0
        for row in read_rows(freeze_path):
            source = row["source"]
            if source["noise_tag"] != condition or source["condition"] != condition:
                raise ValueError(f"{source.get('task_id')}: condition mismatch")
            key = f"{source['algorithm']}::{source['dataset_id']}::s{int(source['seed'])}::{condition}"
            if key in seen:
                raise ValueError(f"duplicate logical key: {key}")
            seen.add(key)

            result_path = Path(source["path"])
            expected_sha = str(source["result_file_sha256"])
            if sha256_file(result_path) != expected_sha or row["result"]["sha256"] != expected_sha:
                raise ValueError(f"{key}: result SHA256 mismatch")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            dataset_index = str(source["dataset_index"])
            checks = (
                str(payload["tool"]).lower() == str(source["algorithm"]).lower(),
                payload["dataset"] == source["dataset_id"],
                int(payload["seed"]) == int(source["seed"]),
                payload["expected_dataset_rel"] == source["dataset_rel"],
                payload["dataset_identity_check"]["match"] is True,
                int(payload["task_global_index"]) == int(dataset_index[1:]),
            )
            if not all(checks):
                raise ValueError(f"{key}: result identity mismatch")

            binding_path = result_path.parent / "import_binding.json"
            provenance_path = result_path.parent / "provenance.json"
            binding = json.loads(binding_path.read_text(encoding="utf-8")) if binding_path.is_file() else {}
            if binding and binding.get("result_sha256") not in (None, expected_sha):
                raise ValueError(f"{key}: import binding SHA256 mismatch")
            if binding.get("controller_task_id"):
                host = str(binding.get("source_host") or source.get("host") or "")
                task_id = source["task_id"]
                batch = source["batch"]
                binding_source = "import_binding.result_sha256"
            elif provenance_path.is_file():
                binding = json.loads(provenance_path.read_text(encoding="utf-8"))
                if binding["source_result_sha256"] != expected_sha:
                    raise ValueError(f"{key}: provenance SHA256 mismatch")
                if (binding["algorithm"].lower() != source["algorithm"].lower()
                        or binding["dataset_name"] != source["dataset_id"]
                        or binding["condition"] != condition
                        or int(binding["seed"]) != int(source["seed"])):
                    raise ValueError(f"{key}: provenance task identity mismatch")
                host = str(binding["source_host"])
                task_id = binding["task_id"]
                batch = binding["batch"]
                binding_source = "provenance.source_result_sha256"
            else:
                raise FileNotFoundError(f"{key}: import_binding.json and provenance.json are missing")
            experiment_dir = payload.get("experiment_dir")
            if not isinstance(experiment_dir, str) or not experiment_dir.strip():
                raise ValueError(f"{key}: experiment_dir is missing")
            if not host:
                path_hosts = [part for part in Path(experiment_dir).parts if re.fullmatch(r"iaaccn\d+", part)]
                if len(path_hosts) != 1:
                    raise ValueError(f"{key}: cannot derive a unique source host from experiment_dir")
                host = path_hosts[0]
            if not host:
                raise ValueError(f"{key}: source host is missing")
            if source.get("host") and source["host"] != host:
                raise ValueError(f"{key}: source host mismatch")

            inner_progress = result_path.parent / "experiments" / Path(experiment_dir).name / "progress"
            inner_progress_available = inner_progress.is_dir()
            runs_by_host[host].append({
                "algorithm": source["algorithm"],
                "batch": batch,
                "binding_source": binding_source,
                "dataset_id": source["dataset_id"],
                "dataset_dir": source["dataset_rel"],
                "expected_dataset_rel": source["dataset_rel"],
                "expected_dataset_dir": source["dataset_rel"],
                "global_index": int(dataset_index[1:]),
                "host": host,
                "inner_progress_dir": str(inner_progress.resolve()) if inner_progress_available else None,
                "inner_progress_local_available": inner_progress_available,
                "noise_tag": condition,
                "path": str(result_path.resolve()),
                "seed": int(source["seed"]),
                "task_id": task_id,
                "task_global_index": int(dataset_index[1:]),
            })
            count += 1
        per_condition[condition] = runs_by_host
        task_counts[condition] = count
        inner_counts[condition] = sum(
            task["inner_progress_local_available"]
            for host_tasks in runs_by_host.values()
            for task in host_tasks
        )

    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(f"trajectory task staging path exists: {staging}")
    staging.mkdir(parents=True)
    manifest = {
        "schema": "core50.trajectory_tasks.v1",
        "freeze_root": str(freeze_root.resolve()),
        "conditions": {},
    }
    for condition in CONDITIONS:
        condition_dir = staging / condition
        condition_dir.mkdir()
        files = {}
        for host, tasks in sorted(per_condition[condition].items()):
            tasks.sort(key=lambda item: (item["algorithm"].lower(), item["dataset_id"], item["seed"]))
            path = condition_dir / f"{host}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for task in tasks:
                    handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
            final_path = output_root / condition / path.name
            files[host] = {"path": str(final_path), "rows": len(tasks), "sha256": sha256_file(path)}
        manifest["conditions"][condition] = {
            "run_count": task_counts[condition],
            "inner_progress_local_count": inner_counts[condition],
            "inner_progress_not_imported_count": task_counts[condition] - inner_counts[condition],
            "freeze_path": str((freeze_root / f"{condition}_runs.jsonl.gz").resolve()),
            "freeze_sha256": sha256_file(freeze_root / f"{condition}_runs.jsonl.gz"),
            "hosts": files,
        }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(output_root)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_tasks(args.freeze_root.resolve(), args.output_root.resolve())
    print(json.dumps({
        condition: {"runs": details["run_count"], "hosts": len(details["hosts"])}
        for condition, details in report["conditions"].items()
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
