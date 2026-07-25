#!/usr/bin/env python3
"""严格合并 Full664 性能榜单与 formal symbolic-fidelity 指标。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "01_new3algs_full664_3seeds_clean_1h"
)
DEFAULT_PERFORMANCE_CSV = (
    DEFAULT_BATCH_DIR / "analysis" / "full664_7alg_leaderboard.csv"
)
DEFAULT_SYMBOLIC_CSV = (
    DEFAULT_BATCH_DIR
    / "symf"
    / "full664_7alg"
    / "symbolic_metrics_formal_algorithm_summary.csv"
)
DEFAULT_OUTPUT_CSV = (
    DEFAULT_BATCH_DIR
    / "analysis"
    / "full664_7alg_leaderboard_with_symf.csv"
)

SYMBOLIC_FIELDS = {
    "algorithm",
    "datasets",
    "SYM_F_formal",
    "exact_equiv_rate",
    "cas_equiv_rate",
    "numeric_equiv_rate",
    "pred_parse_rate",
    "mean_tree_similarity",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 不是数值: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 不是有限数值: {value!r}")
    return number


def _unique_by(
    rows: list[dict[str, str]],
    field: str,
    source: str,
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(field) or "").strip().lower()
        if not key:
            raise ValueError(f"{source} 存在空 {field}")
        if key in out:
            raise ValueError(f"{source} 存在重复 {field}: {key}")
        out[key] = row
    return out


def merge_metrics(
    *,
    performance_csv: Path,
    symbolic_csv: Path,
    output_csv: Path,
    expected_algorithms: int = 7,
) -> dict[str, Any]:
    performance_fields, performance_rows = _read_csv(performance_csv)
    symbolic_fields, symbolic_rows = _read_csv(symbolic_csv)
    if "Algorithm key" not in performance_fields:
        raise ValueError("性能榜单缺少 Algorithm key")
    missing_symbolic = SYMBOLIC_FIELDS - set(symbolic_fields)
    if missing_symbolic:
        raise ValueError(
            f"符号指标缺少字段: {sorted(missing_symbolic)}"
        )
    if len(performance_rows) != expected_algorithms:
        raise ValueError(
            f"性能榜单算法数错误: actual={len(performance_rows)}, "
            f"expected={expected_algorithms}"
        )
    if len(symbolic_rows) != expected_algorithms:
        raise ValueError(
            f"符号指标算法数错误: actual={len(symbolic_rows)}, "
            f"expected={expected_algorithms}"
        )

    performance_by_key = _unique_by(
        performance_rows,
        "Algorithm key",
        "性能榜单",
    )
    symbolic_by_key = _unique_by(
        symbolic_rows,
        "algorithm",
        "符号指标",
    )
    if set(performance_by_key) != set(symbolic_by_key):
        raise ValueError(
            "性能榜单与符号指标算法集合不一致: "
            f"performance={sorted(performance_by_key)}, "
            f"symbolic={sorted(symbolic_by_key)}"
        )

    output_rows: list[dict[str, Any]] = []
    for performance in performance_rows:
        key = str(performance["Algorithm key"]).strip().lower()
        symbolic = symbolic_by_key[key]
        output_rows.append(
            {
                **performance,
                "SYM-F": _float(symbolic["SYM_F_formal"], "SYM_F_formal"),
                "Exact equiv %": 100
                * _float(
                    symbolic["exact_equiv_rate"],
                    "exact_equiv_rate",
                ),
                "TreeSim": _float(
                    symbolic["mean_tree_similarity"],
                    "mean_tree_similarity",
                ),
                "CAS equiv %": 100
                * _float(symbolic["cas_equiv_rate"], "cas_equiv_rate"),
                "Numeric equiv %": 100
                * _float(
                    symbolic["numeric_equiv_rate"],
                    "numeric_equiv_rate",
                ),
                "Pred parse rate": _float(
                    symbolic["pred_parse_rate"],
                    "pred_parse_rate",
                ),
                "SYM-F datasets": int(
                    _float(symbolic["datasets"], "datasets")
                ),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [
        *performance_fields,
        "SYM-F",
        "Exact equiv %",
        "TreeSim",
        "CAS equiv %",
        "Numeric equiv %",
        "Pred parse rate",
        "SYM-F datasets",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "performance_csv": performance_csv.name,
        "performance_sha256": _file_sha256(performance_csv),
        "symbolic_csv": symbolic_csv.name,
        "symbolic_sha256": _file_sha256(symbolic_csv),
        "output_csv": output_csv.name,
        "output_sha256": _file_sha256(output_csv),
        "algorithms": len(output_rows),
        "algorithm_keys": [
            str(row["Algorithm key"]).strip().lower()
            for row in output_rows
        ],
        "rank_semantics": "penalized mean OOD log NMSE ascending",
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance-csv",
        type=Path,
        default=DEFAULT_PERFORMANCE_CSV,
    )
    parser.add_argument(
        "--symbolic-csv",
        type=Path,
        default=DEFAULT_SYMBOLIC_CSV,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument("--expected-algorithms", type=int, default=7)
    args = parser.parse_args()
    summary = merge_metrics(
        performance_csv=args.performance_csv,
        symbolic_csv=args.symbolic_csv,
        output_csv=args.output_csv,
        expected_algorithms=args.expected_algorithms,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
