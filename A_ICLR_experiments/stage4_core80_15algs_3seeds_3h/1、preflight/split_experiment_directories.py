import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

from experiment_paths import CORE_ROOT, FULL_ROOT, LEGACY_ROOT, CONDITIONS, relocated_path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "experiment_layout_split"
PLAN = EVIDENCE / "moves.json"
INDEXES = [HERE / name for name in (
    "reuse_manifest.csv", "pending_training.csv", "pending_opus.csv",
    "core50_training_collection.csv", "full664_10m/collection.csv",
)]


def signature(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def plan():
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    assert LEGACY_ROOT.is_dir() and not CORE_ROOT.exists() and not FULL_ROOT.exists()
    entries, counts, primary_counts = [], Counter(), Counter()
    for condition in sorted(LEGACY_ROOT.iterdir()):
        assert condition.is_dir() and condition.name in CONDITIONS, condition
        for algorithm in sorted(condition.iterdir()):
            assert algorithm.is_dir() and not algorithm.is_symlink(), algorithm
            for dataset in sorted(algorithm.iterdir()):
                assert dataset.is_dir() and not dataset.is_symlink(), dataset
                for seed in sorted(dataset.iterdir()):
                    assert seed.is_dir() and not seed.is_symlink() and seed.name in {"520", "521", "522", "1314"}, seed
                    target = relocated_path(seed)
                    assert not target.exists(), target
                    result = seed / "result.json"
                    entries.append({"old": str(seed), "new": str(target), "relative": str(seed.relative_to(LEGACY_ROOT)),
                                    "seed": seed.name, "condition": condition.name, "algorithm": algorithm.name,
                                    "inode": [seed.stat().st_dev, seed.stat().st_ino],
                                    "result_signature": signature(result) if result.is_file() else None})
                    counts[seed.name] += int(result.is_file())
                    if algorithm.name not in {"imcts", "qlattice"}:
                        primary_counts[seed.name] += int(result.is_file())
    assert len({entry["new"] for entry in entries}) == len(entries)
    assert primary_counts == {"1314": 9960, "520": 2250, "521": 2250, "522": 2250}, primary_counts
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = {"layout": "condition/algorithm/dataset/seed", "results_per_seed": dict(counts),
               "primary_results_per_seed": dict(primary_counts), "moves": entries}
    write_json(PLAN, payload)
    with (EVIDENCE / "path_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["old", "new", "seed"], extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)
    print(json.dumps({"run_directories": len(entries), "results_per_seed": dict(counts)}), flush=True)
    return payload


def update_indexes():
    changed = []
    for path in INDEXES:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        count = 0
        for row in rows:
            for field in fields:
                value = row[field]
                if not value or not value.startswith(str(LEGACY_ROOT) + "/"):
                    continue
                updated = str(relocated_path(value))
                if updated != value:
                    row[field] = updated
                    count += 1
        if not count:
            continue
        before = sha(path)
        backup = EVIDENCE / "previous_indexes" / path.relative_to(HERE)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            assert sha(backup) == before
        else:
            shutil.copy2(path, backup)
        temporary = path.with_suffix(path.suffix + ".pending")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        changed.append({"path": str(path), "updated_fields": count, "previous_sha256": before, "sha256": sha(path), "previous_index": str(backup)})
    for source, summary in ((HERE / "full664_10m/collection.csv", HERE / "full664_10m/collection_summary.json"),
                            (HERE / "core50_training_collection.csv", HERE / "core50_training_collection_summary.json")):
        if summary.exists():
            value = json.loads(summary.read_text())
            value["run_collection_manifest_sha256"] = sha(source)
            value["path_mapping"] = str(EVIDENCE / "path_mapping.csv")
            write_json(summary, value)
    return changed


def migrate(payload):
    if LEGACY_ROOT.exists():
        assert not CORE_ROOT.exists()
        LEGACY_ROOT.rename(CORE_ROOT)
    assert CORE_ROOT.is_dir()
    FULL_ROOT.mkdir(exist_ok=True)
    for entry in payload["moves"]:
        target = Path(entry["new"])
        source = CORE_ROOT / entry["relative"]
        if entry["seed"] == "1314" and source.exists():
            assert not target.exists(), target
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        assert [target.stat().st_dev, target.stat().st_ino] == entry["inode"], target
        if entry["result_signature"] is not None:
            assert signature(target / "result.json") == entry["result_signature"], target
    # 仅移除1314迁出后为空的容器目录。
    for entry in payload["moves"]:
        if entry["seed"] != "1314":
            continue
        parent = (CORE_ROOT / entry["relative"]).parent
        for _ in range(3):
            if parent == CORE_ROOT or not parent.exists() or any(parent.iterdir()):
                break
            parent.rmdir()
            parent = parent.parent
    indexes = update_indexes()
    counts = {}
    for label, root in (("full", FULL_ROOT), ("core50", CORE_ROOT)):
        counts[label] = dict(Counter(path.parent.name for path in root.glob("*/*/*/*/result.json")))
    assert counts["full"] == {"1314": 9960}
    assert counts["core50"] == {seed: count for seed, count in payload["results_per_seed"].items() if seed != "1314"}
    report = {"status": "complete", "roots": {"full": str(FULL_ROOT), "core50": str(CORE_ROOT)},
              "results_per_seed": counts, "run_directories": len(payload["moves"]),
              "directory_inodes_preserved": True, "result_file_signatures_preserved": True,
              "raw_result_contents_modified": False, "updated_indexes": indexes}
    write_json(EVIDENCE / "verification.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "updated_indexes"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    payload = plan()
    if args.apply:
        migrate(payload)
