import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "A_ICLR_experiments/stage4_core80_15algs_3seeds_3h"
PREFLIGHT = STAGE / "1、preflight"
EXPERIMENTS = STAGE / "2、experiments"
WORK = ROOT / ".agent/work/GOAL-CORE50"
CORE50 = PREFLIGHT / "reuse_support/core50.csv"
FROZEN_CONFIG = PREFLIGHT / "reuse_support/source_core80_config.json"
REUSE_MANIFEST = PREFLIGHT / "reuse_manifest.csv"
COLLECTION_MANIFEST = PREFLIGHT / "core50_training_collection.csv"
PENDING_TRAINING = PREFLIGHT / "pending_training.csv"
TRAINING_SOURCE = PREFLIGHT / "core50_new15_training_source.csv"
INPUT_MANIFEST = PREFLIGHT / "core50_new15_input_manifest.json"
GROUND_TRUTH = ROOT / "AAAI_experiments/stage5_metric_calculation_0831/reports/ground_truth_extract.jsonl"
DATASET_PROBES = ROOT / "AAAI_experiments/stage5_metric_calculation_0831/reports/dataset_probes.jsonl"
CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = (520, 521, 522)
DATASET_ID_PATTERN = re.compile(r"g\d{4}\Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def algorithm_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON 行必须为 object")
            rows.append(row)
    return rows


def run_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["noise"]),
        algorithm_slug(str(row["algorithm"])),
        str(row["dataset_id"]),
        int(row["seed"]),
    )


def load_dataset_contract() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    datasets = read_csv(CORE50)
    if len(datasets) != 50:
        raise ValueError(f"Core50 行数必须为50，实际为{len(datasets)}")
    by_id = {row["dataset_id"]: row for row in datasets}
    if len(by_id) != 50:
        raise ValueError("Core50 存在重复 dataset_id")
    if len({row["dataset_name"] for row in datasets}) != 50:
        raise ValueError("Core50 存在重复 dataset_name")
    if any(DATASET_ID_PATTERN.fullmatch(dataset_id) is None for dataset_id in by_id):
        raise ValueError("Core50 dataset_id 必须使用 gNNNN 格式")

    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    algorithms = {algorithm_slug(key) for key in config["algorithms"]}
    if len(algorithms) != 15:
        raise ValueError(f"冻结配置算法数必须为15，实际为{len(algorithms)}")
    expected = {
        (condition, algorithm, dataset_id, seed)
        for condition in CONDITIONS
        for algorithm in algorithms
        for dataset_id in by_id
        for seed in SEEDS
    }
    if len(expected) != 6750:
        raise ValueError(f"期望任务键数量必须为6750，实际为{len(expected)}")
    return by_id, {"algorithms": sorted(algorithms), "expected_keys": expected}


def verify_dataset_sources(datasets: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    gt_rows = read_jsonl(GROUND_TRUTH)
    probe_rows = read_jsonl(DATASET_PROBES)
    gt_by_name = {str(row["dataset_id"]): row for row in gt_rows}
    probes_by_name = {str(row["dataset_name"]): row for row in probe_rows}
    names = {row["dataset_name"] for row in datasets.values()}
    if len(gt_by_name) != len(gt_rows) or set(gt_by_name) != names:
        raise ValueError("Stage5 Ground Truth 数据集名称与当前Core50不一致")
    if len(probes_by_name) != len(probe_rows) or set(probes_by_name) != names:
        raise ValueError("Stage5 dataset probes 数据集名称与当前Core50不一致")

    checked = []
    filenames = {
        "metadata_yaml_sha256": "metadata.yaml",
        "formula_py_sha256": "formula.py",
        "train_csv_sha256": "train.csv",
        "valid_csv_sha256": "valid.csv",
        "id_test_csv_sha256": "id_test.csv",
        "ood_test_csv_sha256": "ood_test.csv",
    }
    for dataset_id, dataset in sorted(datasets.items()):
        name = dataset["dataset_name"]
        relative = Path(dataset["dataset_rel"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{dataset_id}: dataset_rel 路径非法")
        directory = ROOT / relative
        gt = gt_by_name[name]
        probe = probes_by_name[name]
        if Path(str(probe["dataset_dir"])) != relative:
            raise ValueError(f"{dataset_id}: dataset probe 路径与Core50清单不一致")
        if list(gt["ordered_variables"]) != list(probe["variables"]):
            raise ValueError(f"{dataset_id}: Ground Truth 与 dataset probe 变量顺序不一致")
        if gt["target"] != probe["target_name"]:
            raise ValueError(f"{dataset_id}: Ground Truth 与 dataset probe target 不一致")

        source_checksums = gt["source_checksums"]
        checked_files = {}
        for checksum_key, filename in filenames.items():
            path = directory / filename
            actual = sha256_file(path)
            expected = source_checksums[checksum_key]
            if actual != expected:
                raise ValueError(f"{dataset_id}: {filename} SHA256 与冻结 Ground Truth 不一致")
            checked_files[filename] = actual

        probe_checksums = probe["source_sha256"]
        for probe_key, checksum_key in (
            ("metadata_yaml", "metadata_yaml_sha256"),
            ("id_test_csv", "id_test_csv_sha256"),
            ("ood_test_csv", "ood_test_csv_sha256"),
        ):
            if probe_checksums[probe_key] != source_checksums[checksum_key]:
                raise ValueError(f"{dataset_id}: dataset probe 来源哈希不一致")
        checked.append({
            "dataset_id": dataset_id,
            "dataset_name": name,
            "dataset_rel": str(relative),
            "source_checksums": checked_files,
            "ground_truth_evidence_sha256": gt["evidence_sha256"],
            "dataset_probe_evidence_sha256": probe["evidence_sha256"],
        })
    return checked


def read_reuse_records(
    expected: set[tuple[str, str, str, int]],
) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], set[tuple[str, str, str, int]]]:
    records = {}
    manifest_pending = set()
    seen = set()
    for row in read_csv(REUSE_MANIFEST):
        key = run_key(row)
        if key not in expected:
            raise ValueError(f"reuse manifest 含非预期任务: {key}")
        if key in seen:
            raise ValueError(f"reuse manifest 重复任务: {key}")
        seen.add(key)
        if row["training_state"] == "pending":
            manifest_pending.add(key)
            continue
        if row["training_state"] != "copied":
            raise ValueError(f"reuse manifest 任务状态异常: {key} {row['training_state']}")
        if row["execution_status"] not in {"ok", "error", "timed_out"}:
            raise ValueError(f"reuse manifest 执行状态无法识别: {key} {row['execution_status']}")
        records[key] = {"row": row, "origin": "reused_core80"}
    if seen != expected:
        raise ValueError(f"reuse manifest 未闭合期望任务集合: {len(seen)}/{len(expected)}")
    return records, manifest_pending


def merge_collection_records(
    records: dict[tuple[str, str, str, int], dict[str, Any]],
    expected: set[tuple[str, str, str, int]],
) -> tuple[set[tuple[str, str, str, int]], Counter[str]]:
    reused_keys = set(records)
    collection_rows = read_csv(COLLECTION_MANIFEST)
    collection_keys = set()
    states: Counter[str] = Counter()
    for row in collection_rows:
        key = run_key(row)
        if key not in expected:
            raise ValueError(f"collection manifest 含非预期任务: {key}")
        if key in collection_keys:
            raise ValueError(f"collection manifest 重复任务: {key}")
        collection_keys.add(key)
        state = row["collection_state"]
        states[state] += 1
        if key in reused_keys:
            raise ValueError(f"reuse 与 new15 任务键重叠: {key}")
        records[key] = {"row": row, "origin": "new_core50"}

    if len(collection_keys) != 2025:
        raise ValueError(f"new15 collection 行数必须为2025，实际为{len(collection_keys)}")
    pending_rows = read_csv(PENDING_TRAINING)
    pending_keys = {run_key(row) for row in pending_rows}
    if len(pending_rows) != 2025 or len(pending_keys) != 2025:
        raise ValueError(f"pending_training 必须包含2025个唯一任务，实际为{len(pending_keys)}")
    if collection_keys != pending_keys:
        raise ValueError("collection 与 pending_training 任务集合不一致")
    expected_new = expected - reused_keys
    if collection_keys != expected_new:
        raise ValueError("new15 collection 任务集合未匹配 Core50 预期差集")
    return collection_keys, states


def source_and_payload(
    key: tuple[str, str, str, int],
    record: dict[str, Any],
    datasets: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    noise, algorithm, dataset_id, seed = key
    row = record["row"]
    if record["origin"] == "new_core50" and row["collection_state"] != "collected":
        return None

    dataset = datasets[dataset_id]
    run_directory = Path(row["destination"])
    result_path = run_directory / "result.json"
    result_bytes = result_path.read_bytes()
    result_sha = sha256_bytes(result_bytes)
    expected_sha = row["result_sha256"]
    if not expected_sha or result_sha != expected_sha:
        raise ValueError(f"{key}: result.json SHA256 与来源 manifest 不一致")
    raw_text = result_bytes.decode("utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"{key}: result.json 顶层必须为 object")

    identity = {
        "algorithm": algorithm,
        "batch": row.get("queue_batch") or record["origin"],
        "condition": noise,
        "dataset_id": dataset["dataset_name"],
        "dataset_index": dataset_id,
        "dataset_rel": dataset["dataset_rel"],
        "host": row.get("assigned_host") or None,
        "noise_tag": noise,
        "path": str(result_path.resolve()),
        "result_file_sha256": result_sha,
        "seed": seed,
        "source_run_dir": row.get("source") or row.get("source_run_dir") or None,
        "task_id": f"{algorithm}_s{seed}_{noise}_{dataset_id}",
    }

    checks = {
        "tool": algorithm_slug(str(payload["tool"])),
        "dataset": payload["dataset"],
        "task_label": payload["task_label"],
        "task_global_index": int(payload["task_global_index"]),
        "seed": int(payload["seed"]),
        "condition": payload["condition"],
        "expected_dataset_rel": payload["expected_dataset_rel"],
    }
    expected_checks = {
        "tool": algorithm,
        "dataset": dataset["dataset_name"],
        "task_label": f"{dataset_id}_{dataset['dataset_name']}",
        "task_global_index": int(dataset_id[1:]),
        "seed": seed,
        "condition": noise,
        "expected_dataset_rel": dataset["dataset_rel"],
    }
    if checks != expected_checks:
        raise ValueError(f"{key}: result.json 任务身份不匹配: {checks}")

    source = dict(identity)
    source["source_row_sha256"] = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return {
        "source": source,
        "result": {"raw_text": raw_text, "sha256": result_sha},
    }


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((canonical_json(row) + "\n").encode("utf-8"))
                count += 1
    return count, sha256_file(path), path.stat().st_size


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.candidate")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_report(
    *,
    write_freezes: bool,
    write_available_freezes: bool,
    output_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    datasets, contract = load_dataset_contract()
    dataset_checks = verify_dataset_sources(datasets)
    expected = contract["expected_keys"]
    records, manifest_pending = read_reuse_records(expected)
    reuse_keys = set(records)
    collection_keys, collection_states = merge_collection_records(records, expected)
    if reuse_keys & collection_keys:
        raise ValueError("reuse 与 collection 任务集合重叠")
    if manifest_pending != collection_keys:
        raise ValueError("reuse manifest pending 键与 collection 键不一致")
    if reuse_keys | collection_keys != expected:
        raise ValueError("reuse 与 collection 清单未覆盖期望 Core50 笛卡尔积")

    available: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    for key in sorted(records):
        frozen = source_and_payload(key, records[key], datasets)
        if frozen is not None:
            available[key] = frozen
            statuses[str(json.loads(frozen["result"]["raw_text"]).get("status"))] += 1
    missing = sorted(expected - set(available))
    available_by_condition = Counter(key[0] for key in available)
    missing_by_condition = Counter(key[0] for key in missing)

    report: dict[str, Any] = {
        "schema": "core50.sixaxis_input_audit.v1",
        "status": "ready" if not missing else "incomplete",
        "expected_runs": len(expected),
        "available_runs": len(available),
        "available_runs_by_condition": dict(sorted(available_by_condition.items())),
        "missing_runs": len(missing),
        "missing_runs_by_condition": dict(sorted(missing_by_condition.items())),
        "reuse_runs": len(reuse_keys),
        "new_core50_runs": len(collection_keys),
        "new_collection_states": dict(collection_states),
        "available_result_statuses": dict(statuses),
        "missing_sample": [
            {"noise": key[0], "algorithm": key[1], "dataset_id": key[2], "seed": key[3]}
            for key in missing[:20]
        ],
        "dataset_source_count": len(dataset_checks),
        "dataset_sources": dataset_checks,
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                CORE50,
                FROZEN_CONFIG,
                REUSE_MANIFEST,
                COLLECTION_MANIFEST,
                PENDING_TRAINING,
                TRAINING_SOURCE,
                INPUT_MANIFEST,
                GROUND_TRUTH,
                DATASET_PROBES,
            )
        },
    }

    if write_freezes or write_available_freezes:
        if write_freezes and missing:
            raise ValueError(f"仍缺少 {len(missing)} 项训练结果，拒绝生成完整 source freeze")
        suffix = "_runs.jsonl.gz" if write_freezes else "_runs_available.jsonl.gz"
        if any(path.exists() for path in output_root.glob(f"*{suffix}")):
            raise FileExistsError(f"source freeze 已存在，拒绝覆盖: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for condition in CONDITIONS:
            rows = (
                available[key]
                for key in sorted(available)
                if key[0] == condition
            )
            path = output_root / f"{condition}{suffix}"
            count, digest, size = write_gzip_jsonl(path, rows)
            if count == 0:
                raise ValueError(f"{condition}: source freeze 没有可用任务")
            if write_freezes and count != 2250:
                raise ValueError(f"{condition}: source freeze 行数必须为2250，实际为{count}")
            outputs[condition] = {"path": str(path), "rows": count, "sha256": digest, "size_bytes": size}
        report["source_freezes"] = outputs
        report["source_freeze_kind"] = "full" if write_freezes else "available_partial_candidate"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--write-freezes", action="store_true")
    output_mode.add_argument("--write-available-freezes", action="store_true")
    parser.add_argument("--output-root", type=Path, default=WORK / "source_freezes")
    parser.add_argument("--report", type=Path, default=WORK / "sixaxis_input_audit.json")
    args = parser.parse_args()
    report = build_report(
        write_freezes=args.write_freezes,
        write_available_freezes=args.write_available_freezes,
        output_root=args.output_root,
        report_path=args.report,
    )
    print(json.dumps({key: report[key] for key in ("status", "expected_runs", "available_runs", "missing_runs", "reuse_runs", "new_core50_runs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
