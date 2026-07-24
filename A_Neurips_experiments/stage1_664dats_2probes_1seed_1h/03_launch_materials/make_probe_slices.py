#!/usr/bin/env python3
"""按顺序切分双探针实验任务清单。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PYSR_MACHINES = ["iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25"]
LLMSR_MACHINES = ["iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29"]


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"任务清单为空: {csv_path}")
    return rows


def _contiguous_slices(total: int, num_parts: int) -> list[tuple[int, int]]:
    base, remainder = divmod(total, num_parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(num_parts):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _write_slice(
    *,
    tool_name: str,
    machine: str,
    rows: list[dict[str, str]],
    start: int,
    end: int,
    output_dir: Path,
) -> Path:
    slice_rows = rows[start:end]
    target_dir = output_dir / tool_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{machine}.csv"

    fieldnames = ["tool", "machine", "global_index", "slice_index"] + list(rows[0].keys())
    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for slice_index, row in enumerate(slice_rows, start=1):
            writer.writerow(
                {
                    "tool": tool_name,
                    "machine": machine,
                    "global_index": start + slice_index,
                    "slice_index": slice_index,
                    **row,
                }
            )
    return target_path


def _emit_tool_slices(tool_name: str, machines: list[str], rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for machine, (start, end) in zip(machines, _contiguous_slices(len(rows), len(machines)), strict=True):
        paths.append(
            _write_slice(
                tool_name=tool_name,
                machine=machine,
                rows=rows,
                start=start,
                end=end,
                output_dir=output_dir,
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="生成双探针实验的静态顺序切片 CSV。")
    parser.add_argument(
        "--datasets-csv",
        default="exp-planning/01.双探针实验/datasets_to_run.csv",
        help="全量数据集清单 CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="exp-planning/01.双探针实验/slices",
        help="切片输出目录",
    )
    args = parser.parse_args()

    datasets_csv = Path(args.datasets_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = _load_rows(datasets_csv)

    pysr_paths = _emit_tool_slices("pysr", PYSR_MACHINES, rows, output_dir)
    llmsr_paths = _emit_tool_slices("llmsr", LLMSR_MACHINES, rows, output_dir)

    print(f"数据集总数: {len(rows)}")
    print("PySR 切片:")
    for path in pysr_paths:
        print(f"  - {path}")
    print("LLMSR 切片:")
    for path in llmsr_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
