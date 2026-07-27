#!/usr/bin/env python3
"""Extract the one-hour checkpoint from the AAAI three-hour runs.

This script is intentionally host-local. The selected-run manifest records the
machine that owns each result. Run ``extract-host`` on every owner machine, then
run ``merge`` locally to validate and combine the eight host parts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ALGORITHMS = {"fepysr", "jaxsr", "symbolfit"}
NOISE_SIGMA = {"clean": 0.0, "noise001": 0.01, "noise005": 0.05}
EXPECTED_HOST_ROWS = {
    "iaaccn22": 163,
    "iaaccn23": 167,
    "iaaccn24": 183,
    "iaaccn25": 184,
    "iaaccn26": 160,
    "iaaccn27": 173,
    "iaaccn28": 158,
    "iaaccn29": 162,
}
EXPECTED_TOTAL = 1350
FIELDNAMES = [
    "algorithm",
    "gid",
    "dataset",
    "seed",
    "noise_tag",
    "noise_sigma",
    "host",
    "batch",
    "task_id",
    "snapshot_path",
    "snapshot_sha256",
    "checkpoint_index",
    "elapsed_seconds",
    "status",
    "valid_output",
    "metric_complete",
    "id_test_nmse",
    "ood_test_nmse",
    "seconds",
    "equation",
    "expression_canonical",
    "artifact_valid",
    "ast_node_count",
    "tree_depth",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def split_nmse(payload: dict[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    return finite_nonnegative(block.get("nmse"))


def canonical_expression(payload: dict[str, Any]) -> str:
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, dict):
        artifact = {}
    for key in (
        "instantiated_expression",
        "normalized_expression",
        "return_expression_source",
    ):
        value = artifact.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    equation = payload.get("equation")
    return equation.strip() if isinstance(equation, str) else ""


def snapshot_path(result_path: str) -> Path:
    path = Path(result_path)
    if path.name != "result.json":
        raise ValueError(f"unexpected result path: {result_path}")
    return path.parent / "progress" / "minute_0060.json"


def gid_from_task_id(task_id: str) -> str:
    match = re.search(r"_g(\d{4})$", task_id)
    if match is None:
        raise ValueError(f"cannot parse gid from task_id={task_id!r}")
    return f"g{match.group(1)}"


def load_selected_rows(path: Path, host: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("algorithm", "").lower() in ALGORITHMS
        and row.get("host") == host
    ]
    expected = EXPECTED_HOST_ROWS.get(host)
    if expected is None:
        raise ValueError(f"unsupported host: {host}")
    if len(selected) != expected:
        raise ValueError(
            f"{host}: selected rows={len(selected)}, expected={expected}"
        )
    return selected


def extract_row(manifest_row: dict[str, str]) -> dict[str, Any]:
    path = snapshot_path(manifest_row["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    algorithm = manifest_row["algorithm"].lower()
    dataset = manifest_row["dataset_id"]
    seed = int(manifest_row["seed"])
    noise_tag = manifest_row["noise_tag"]
    checkpoint = int(payload.get("checkpoint_index", -1))
    elapsed = finite_nonnegative(payload.get("elapsed_seconds"))
    if payload.get("tool", "").lower() != algorithm:
        raise ValueError(f"{path}: tool mismatch")
    if payload.get("dataset") != dataset:
        raise ValueError(f"{path}: dataset mismatch")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"{path}: seed mismatch")
    if checkpoint != 60:
        raise ValueError(f"{path}: checkpoint_index={checkpoint}, expected=60")
    if elapsed is None or not 3300 <= elapsed <= 3900:
        raise ValueError(f"{path}: elapsed_seconds={elapsed}")

    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, dict):
        artifact = {}
    artifact_valid = artifact.get("artifact_valid")
    id_nmse = split_nmse(payload, "id_test")
    ood_nmse = split_nmse(payload, "ood_test")
    equation = payload.get("equation")
    metric_complete = id_nmse is not None and ood_nmse is not None
    valid_output = bool(
        isinstance(equation, str)
        and equation.strip()
        and artifact_valid is not False
        and metric_complete
    )

    return {
        "algorithm": algorithm,
        "gid": gid_from_task_id(manifest_row["task_id"]),
        "dataset": dataset,
        "seed": seed,
        "noise_tag": noise_tag,
        "noise_sigma": NOISE_SIGMA[noise_tag],
        "host": manifest_row["host"],
        "batch": manifest_row["batch"],
        "task_id": manifest_row["task_id"],
        "snapshot_path": str(path),
        "snapshot_sha256": sha256(path),
        "checkpoint_index": checkpoint,
        "elapsed_seconds": elapsed,
        "status": payload.get("status"),
        "valid_output": valid_output,
        "metric_complete": metric_complete,
        "id_test_nmse": id_nmse,
        "ood_test_nmse": ood_nmse,
        "seconds": finite_nonnegative(payload.get("seconds")),
        "equation": equation,
        "expression_canonical": canonical_expression(payload),
        "artifact_valid": artifact_valid,
        "ast_node_count": artifact.get("ast_node_count"),
        "tree_depth": artifact.get("tree_depth"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_host(args: argparse.Namespace) -> None:
    selected = load_selected_rows(Path(args.selected_runs_csv), args.host)
    rows = [extract_row(row) for row in selected]
    rows.sort(
        key=lambda row: (
            row["algorithm"],
            row["seed"],
            row["noise_tag"],
            row["gid"],
        )
    )
    write_csv(Path(args.output), rows)
    print(
        json.dumps(
            {
                "host": args.host,
                "rows": len(rows),
                "valid_outputs": sum(bool(row["valid_output"]) for row in rows),
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def merge_parts(args: argparse.Namespace) -> None:
    parts_dir = Path(args.parts_dir)
    files = sorted(parts_dir.glob("iaaccn*.csv"))
    if len(files) != len(EXPECTED_HOST_ROWS):
        raise ValueError(f"part files={len(files)}, expected=8")

    rows: list[dict[str, str]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            part = list(csv.DictReader(handle))
        host = path.stem
        expected = EXPECTED_HOST_ROWS[host]
        if len(part) != expected:
            raise ValueError(f"{host}: rows={len(part)}, expected={expected}")
        rows.extend(part)

    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"merged rows={len(rows)}, expected={EXPECTED_TOTAL}")
    keys = [
        (
            row["algorithm"],
            row["gid"],
            int(row["seed"]),
            row["noise_tag"],
        )
        for row in rows
    ]
    if len(set(keys)) != EXPECTED_TOTAL:
        raise ValueError("duplicate algorithm/gid/seed/noise keys")
    expected_grid = {
        (algorithm, f"g{gid:04d}", seed, noise)
        for algorithm in sorted(ALGORITHMS)
        for gid in range(1, 51)
        for seed in (520, 521, 522)
        for noise in NOISE_SIGMA
    }
    missing = expected_grid - set(keys)
    extra = set(keys) - expected_grid
    if missing or extra:
        raise ValueError(
            f"grid mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    rows.sort(
        key=lambda row: (
            row["algorithm"],
            int(row["seed"]),
            row["noise_tag"],
            row["gid"],
        )
    )
    write_csv(Path(args.output), rows)
    valid = sum(bool_value(row["valid_output"]) for row in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "algorithms": sorted(ALGORITHMS),
                "datasets": 50,
                "seeds": [520, 521, 522],
                "noise_tags": list(NOISE_SIGMA),
                "valid_outputs": valid,
                "output": str(Path(args.output).resolve()),
                "sha256": sha256(Path(args.output)),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-host")
    extract.add_argument("--selected-runs-csv", required=True)
    extract.add_argument("--host", required=True)
    extract.add_argument("--output", required=True)
    extract.set_defaults(func=extract_host)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--parts-dir", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=merge_parts)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
