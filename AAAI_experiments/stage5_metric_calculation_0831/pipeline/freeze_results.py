from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_runs(source_runs_csv: Path, *, noise_tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["noise_tag"] != noise_tag:
                continue
            rows.append(
                {
                    "logical_key": row["logical_key"],
                    "batch": row["batch"],
                    "algorithm": row["algorithm"],
                    "dataset_id": row["dataset_id"],
                    "seed": int(row["seed"]),
                    "noise_tag": row["noise_tag"],
                    "task_id": row["task_id"],
                    "host": row["host"],
                    "status": row["status"],
                    "id_nmse_source": float(row["id_nmse"]),
                    "ood_nmse_source": float(row["ood_nmse"]),
                    "remote_result_path": row["path"],
                }
            )
    return rows


def _local_result_path(snapshot_root: Path, row: dict[str, Any]) -> Path:
    return (
        snapshot_root
        / "results"
        / row["noise_tag"]
        / row["algorithm"]
        / row["task_id"]
        / "result.json"
    )


def _copy_remote_file(host: str, remote_path: str, local_path: Path, *, max_attempts: int = 3) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(".tmp")
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        cmd = [
            "timeout",
            "120",
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            f"{host}:{remote_path}",
            str(tmp_path),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            tmp_path.replace(local_path)
            return
        last_error = completed.stderr.strip() or completed.stdout.strip() or f"rc={completed.returncode}"
        if attempt < max_attempts:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"SCP 失败 {host}:{remote_path}: {last_error}")


def _nmse_from_payload(payload: dict[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    value = block.get("nmse")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _approx_equal(left: float | None, right: float | None, *, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= (atol + rtol * abs(right))


def _build_index_row(row: dict[str, Any], local_result_path: Path) -> dict[str, Any]:
    payload = json.loads(local_result_path.read_text(encoding="utf-8"))
    id_nmse_result = _nmse_from_payload(payload, "id_test")
    ood_nmse_result = _nmse_from_payload(payload, "ood_test")
    metrics_match = _approx_equal(row["id_nmse_source"], id_nmse_result) and _approx_equal(
        row["ood_nmse_source"], ood_nmse_result
    )
    equation = payload.get("equation")
    canonical_artifact = payload.get("canonical_artifact")
    return {
        "logical_key": row["logical_key"],
        "batch": row["batch"],
        "noise_tag": row["noise_tag"],
        "algorithm": row["algorithm"],
        "dataset_id": row["dataset_id"],
        "seed": row["seed"],
        "task_id": row["task_id"],
        "host": row["host"],
        "remote_result_path": row["remote_result_path"],
        "local_result_path": str(local_result_path),
        "local_sha256": _sha256_file(local_result_path),
        "status": payload.get("status"),
        "equation_present": bool(isinstance(equation, str) and equation.strip()),
        "canonical_artifact_present": isinstance(canonical_artifact, dict),
        "id_nmse_source": f"{row['id_nmse_source']:.17g}",
        "id_nmse_result": "" if id_nmse_result is None else f"{id_nmse_result:.17g}",
        "ood_nmse_source": f"{row['ood_nmse_source']:.17g}",
        "ood_nmse_result": "" if ood_nmse_result is None else f"{ood_nmse_result:.17g}",
        "metrics_match": metrics_match,
    }


def freeze_results(
    source_runs_csv: Path,
    *,
    snapshot_root: Path,
    noise_tag: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = load_source_runs(source_runs_csv, noise_tag=noise_tag)
    if limit is not None:
        rows = rows[:limit]
    index_rows: list[dict[str, Any]] = []
    for row in rows:
        local_result_path = _local_result_path(snapshot_root, row)
        if not local_result_path.exists():
            _copy_remote_file(row["host"], row["remote_result_path"], local_result_path)
        index_rows.append(_build_index_row(row, local_result_path))
    return index_rows


def write_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("result index 不能为空")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_index(index_csv: Path) -> dict[str, dict[str, Any]]:
    if not index_csv.exists():
        return {}
    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["logical_key"]: dict(row) for row in reader}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="冻结 Stage5 clean/noise 的 result.json 到本地快照")
    parser.add_argument(
        "--source-runs-csv",
        type=Path,
        default=repo_root / "AAAI_experiments/stage5_metric_calculation_0831/manifests/source_runs.csv",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=repo_root / "AAAI_experiments/stage5_metric_calculation_0831/source_snapshot",
    )
    parser.add_argument("--noise-tag", choices=("clean", "noise001", "noise005"), default="clean")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=repo_root / "AAAI_experiments/stage5_metric_calculation_0831/source_snapshot/result_index_clean.csv",
    )
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.force_refresh and args.snapshot_root.exists():
        target_dir = args.snapshot_root / "results" / args.noise_tag
        if target_dir.exists():
            shutil.rmtree(target_dir)
    source_rows = load_source_runs(args.source_runs_csv.resolve(), noise_tag=args.noise_tag)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    existing_by_key = load_existing_index(args.index_csv.resolve())
    result_rows: list[dict[str, Any]] = []
    for row in source_rows:
        local_result_path = _local_result_path(args.snapshot_root.resolve(), row)
        if not local_result_path.exists():
            _copy_remote_file(row["host"], row["remote_result_path"], local_result_path)
        index_row = _build_index_row(row, local_result_path)
        if not index_row["metrics_match"]:
            raise RuntimeError(f"发现 NMSE 对不上源 manifest 的结果: {row['logical_key']}")
        existing_by_key[row["logical_key"]] = index_row
        ordered_keys = [item["logical_key"] for item in source_rows if item["logical_key"] in existing_by_key]
        result_rows = [existing_by_key[key] for key in ordered_keys]
        write_index_csv(args.index_csv, result_rows)
    summary = {
        "noise_tag": args.noise_tag,
        "row_count": len(result_rows),
        "equation_present": sum(1 for row in result_rows if row["equation_present"]),
        "canonical_artifact_present": sum(1 for row in result_rows if row["canonical_artifact_present"]),
        "metrics_match": sum(1 for row in result_rows if row["metrics_match"]),
        "index_csv": str(args.index_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
