#!/usr/bin/env python3
"""筛查双探针实验数据集是否可被统一 runner 正常读取。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from scientific_intelligent_modelling.benchmarks.runner import load_canonical_dataset


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_invalid(invalid_rows: list[dict[str, str]]) -> str:
    family_counter = Counter(row["family"] for row in invalid_rows)
    kind_counter = Counter(row["error_type"] for row in invalid_rows)

    lines = []
    lines.append("# 可运行性筛查摘要")
    lines.append("")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 异常数据集数：{len(invalid_rows)}")
    lines.append("")
    lines.append("## 按 family 统计")
    lines.append("")
    for family, count in sorted(family_counter.items()):
        lines.append(f"- `{family}`: {count}")
    lines.append("")
    lines.append("## 按错误类型统计")
    lines.append("")
    for error_type, count in sorted(kind_counter.items()):
        lines.append(f"- `{error_type}`: {count}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _error_type(exc: Exception) -> str:
    name = exc.__class__.__name__
    msg = str(exc)
    if "缺少目标列" in msg:
        return "missing_target_column"
    if name == "FileNotFoundError":
        return "missing_file_or_dir"
    if name == "ValueError":
        return "value_error"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="筛查双探针实验数据集的可运行性。")
    parser.add_argument(
        "--input-csv",
        default="exp-planning/01.双探针实验/datasets_to_run.csv",
        help="待筛查的数据集清单",
    )
    parser.add_argument(
        "--runnable-csv",
        default="exp-planning/01.双探针实验/datasets_runnable.csv",
        help="输出可运行白名单 CSV",
    )
    parser.add_argument(
        "--invalid-csv",
        default="exp-planning/01.双探针实验/datasets_invalid.csv",
        help="输出异常清单 CSV",
    )
    parser.add_argument(
        "--summary-md",
        default="exp-planning/01.双探针实验/dataset_runnable_summary.md",
        help="输出 Markdown 摘要",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    runnable_csv = Path(args.runnable_csv).resolve()
    invalid_csv = Path(args.invalid_csv).resolve()
    summary_md = Path(args.summary_md).resolve()

    rows = _load_rows(input_csv)
    runnable_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []

    for row in rows:
        dataset_dir = Path(row["dataset_dir"]).resolve()
        try:
            dataset = load_canonical_dataset(dataset_dir)
        except Exception as exc:
            invalid_rows.append(
                {
                    **row,
                    "resolved_dataset_dir": str(dataset_dir),
                    "error_type": _error_type(exc),
                    "error": repr(exc),
                }
            )
            continue

        runnable_rows.append(
            {
                **row,
                "resolved_dataset_dir": str(dataset_dir),
                "target_name": dataset.target_name,
                "feature_count": str(len(dataset.feature_names)),
                "train_rows_checked": str(dataset.train.rows),
                "valid_rows_checked": str(dataset.valid.rows if dataset.valid else 0),
                "id_test_rows_checked": str(dataset.id_test.rows if dataset.id_test else 0),
                "ood_test_rows_checked": str(dataset.ood_test.rows if dataset.ood_test else 0),
            }
        )

    runnable_fields = list(rows[0].keys()) + [
        "resolved_dataset_dir",
        "target_name",
        "feature_count",
        "train_rows_checked",
        "valid_rows_checked",
        "id_test_rows_checked",
        "ood_test_rows_checked",
    ]
    invalid_fields = list(rows[0].keys()) + [
        "resolved_dataset_dir",
        "error_type",
        "error",
    ]

    _write_csv(runnable_csv, runnable_rows, runnable_fields)
    _write_csv(invalid_csv, invalid_rows, invalid_fields)
    summary_md.write_text(_summarize_invalid(invalid_rows), encoding="utf-8")

    report = {
        "input_total": len(rows),
        "runnable_total": len(runnable_rows),
        "invalid_total": len(invalid_rows),
        "runnable_csv": str(runnable_csv),
        "invalid_csv": str(invalid_csv),
        "summary_md": str(summary_md),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
