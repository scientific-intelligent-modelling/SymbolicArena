#!/usr/bin/env python3
"""把两个噪声条件的结果索引转换为按主机分片的轨迹扫描任务。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STAGE5_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATHS = {
    "noise001": STAGE5_ROOT / "source_snapshot/result_index_noise001.csv",
    "noise005": STAGE5_ROOT / "source_snapshot/result_index_noise005.csv",
}
DEFAULT_OUTPUT_DIR = STAGE5_ROOT / "work/noise_trajectory_freeze_v1/tasks"
DEFAULT_REPORT = STAGE5_ROOT / "work/noise_trajectory_freeze_v1/task_manifest.json"
EXPECTED_HOSTS = tuple(f"iaaccn{suffix}" for suffix in range(22, 30))
EXPECTED_ALGORITHMS = (
    "QLattice",
    "drsr",
    "dso",
    "e2esr",
    "fepysr",
    "gplearn",
    "iMCTS",
    "jaxsr",
    "llmsr",
    "pyoperon",
    "pysr",
    "ragsr",
    "symbolfit",
    "tpsr",
    "udsr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(path: Path, expected_condition: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "logical_key",
            "batch",
            "noise_tag",
            "algorithm",
            "dataset_id",
            "seed",
            "task_id",
            "host",
            "remote_result_path",
            "status",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} 缺少字段: {missing}")
        rows = list(reader)
    wrong_conditions = sorted(
        {row["noise_tag"] for row in rows if row["noise_tag"] != expected_condition}
    )
    if wrong_conditions:
        raise ValueError(
            f"{path} 包含错误 noise_tag: {wrong_conditions}; 期望 {expected_condition}"
        )
    return rows


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    staging = path.with_suffix(path.suffix + ".staging")
    staging.write_text(text, encoding="utf-8")
    staging.replace(path)


def build_noise_trajectory_tasks(
    *,
    index_paths: Mapping[str, Path],
    output_dir: Path,
    expected_hosts: Sequence[str] = EXPECTED_HOSTS,
    expected_algorithms: Sequence[str] = EXPECTED_ALGORITHMS,
    expected_rows_per_condition: int | None = 2250,
    expected_runs_per_algorithm_condition: int | None = 150,
) -> dict[str, object]:
    if set(index_paths) != {"noise001", "noise005"}:
        raise ValueError("index_paths 必须且只能包含 noise001 和 noise005")

    records: list[dict[str, object]] = []
    input_files: dict[str, dict[str, object]] = {}
    for condition in ("noise001", "noise005"):
        path = Path(index_paths[condition]).resolve()
        rows = _read_index(path, condition)
        if expected_rows_per_condition is not None and len(rows) != expected_rows_per_condition:
            raise ValueError(
                f"{condition} 索引应有 {expected_rows_per_condition} 行，实际 {len(rows)}"
            )
        input_files[condition] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": _sha256(path),
        }
        for row in rows:
            if row["status"] != "ok":
                raise ValueError(f"{row['task_id']} status 不是 ok: {row['status']!r}")
            result_path = Path(row["remote_result_path"])
            if not result_path.is_absolute():
                raise ValueError(f"{row['task_id']} 远端 result 路径不是绝对路径")
            try:
                seed = int(row["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{row['task_id']} seed 非法: {row['seed']!r}") from exc
            records.append(
                {
                    "algorithm": row["algorithm"],
                    "batch": row["batch"],
                    "dataset_id": row["dataset_id"],
                    "host": row["host"],
                    "logical_key": row["logical_key"],
                    "noise_tag": row["noise_tag"],
                    "path": str(result_path),
                    "seed": seed,
                    "task_id": row["task_id"],
                }
            )

    logical_keys = [str(record["logical_key"]) for record in records]
    duplicates = sorted(key for key, count in Counter(logical_keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"logical_key 存在重复: {duplicates[:10]}")

    actual_hosts = {str(record["host"]) for record in records}
    allowed_hosts = set(expected_hosts)
    unexpected_hosts = sorted(actual_hosts - allowed_hosts)
    if unexpected_hosts:
        raise ValueError(f"出现未允许主机: {unexpected_hosts}")
    actual_algorithms = {str(record["algorithm"]) for record in records}
    if actual_algorithms != set(expected_algorithms):
        raise ValueError(
            "算法集合不匹配: "
            f"missing={sorted(set(expected_algorithms) - actual_algorithms)}, "
            f"unexpected={sorted(actual_algorithms - set(expected_algorithms))}"
        )

    algorithm_condition_counts = Counter(
        (str(record["algorithm"]), str(record["noise_tag"])) for record in records
    )
    if expected_runs_per_algorithm_condition is not None:
        bad = {
            f"{algorithm}/{condition}": algorithm_condition_counts[(algorithm, condition)]
            for algorithm in expected_algorithms
            for condition in ("noise001", "noise005")
            if algorithm_condition_counts[(algorithm, condition)]
            != expected_runs_per_algorithm_condition
        }
        if bad:
            raise ValueError(f"算法-条件运行数不满足契约: {bad}")

    by_host: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_host[str(record["host"])].append(record)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, dict[str, object]] = {}
    for host in expected_hosts:
        host_records = sorted(
            by_host.get(host, []),
            key=lambda item: (
                str(item["algorithm"]).lower(),
                str(item["noise_tag"]),
                str(item["dataset_id"]),
                int(item["seed"]),
            ),
        )
        path = output_dir / f"{host}.jsonl"
        _write_jsonl(path, host_records)
        output_files[host] = {
            "path": str(path),
            "rows": len(host_records),
            "sha256": _sha256(path),
        }

    condition_counts = Counter(str(record["noise_tag"]) for record in records)
    host_counts = Counter(str(record["host"]) for record in records)
    report: dict[str, object] = {
        "status": "ok",
        "schema_version": "stage5.noise_trajectory_scan_tasks.v1",
        "total_tasks": len(records),
        "unique_logical_keys": len(set(logical_keys)),
        "condition_counts": dict(sorted(condition_counts.items())),
        "host_counts": {host: host_counts[host] for host in expected_hosts},
        "algorithm_condition_counts": {
            f"{algorithm}/{condition}": algorithm_condition_counts[(algorithm, condition)]
            for algorithm in expected_algorithms
            for condition in ("noise001", "noise005")
        },
        "inputs": input_files,
        "outputs": output_files,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise001-index", type=Path, default=DEFAULT_INDEX_PATHS["noise001"])
    parser.add_argument("--noise005-index", type=Path, default=DEFAULT_INDEX_PATHS["noise005"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = build_noise_trajectory_tasks(
        index_paths={"noise001": args.noise001_index, "noise005": args.noise005_index},
        output_dir=args.output_dir,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
