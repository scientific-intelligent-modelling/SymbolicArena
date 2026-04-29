#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/"
    "e1_12_dataset_algorithm_nmse_table.csv"
)
LOG_FLOOR = 1e-12
LOG_CLIP_MIN = -12.0
LOG_CLIP_MAX = 12.0
EXPLOSION_THRESHOLD = 100.0

DERIVED_COLUMNS = [
    "finite_train",
    "finite_id",
    "finite_ood",
    "finite_id_ood",
    "finite_train_id_ood",
    "log_train_nmse_clipped",
    "log_id_nmse_clipped",
    "log_ood_nmse_clipped",
    "combined_log_id_ood_nmse",
    "gap_log_ood_minus_id",
    "delta_id_minus_train",
    "delta_ood_minus_id",
    "id_ood_explosion_gt_100",
    "train_id_ood_explosion_gt_100",
]
STRIP_COLUMNS = set(DERIVED_COLUMNS) | {
    "finite_valid",
    "log_valid_nmse_clipped",
    "delta_valid_minus_train",
    "delta_id_minus_valid",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _nmse(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0:
        return None
    return out


def _log_nmse(value: float | None) -> float | None:
    if value is None:
        return None
    out = math.log10(max(value, LOG_FLOOR))
    return min(LOG_CLIP_MAX, max(LOG_CLIP_MIN, out))


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _mean2(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return 0.5 * a + 0.5 * b


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.17g}"


def _augment_row(row: dict[str, str]) -> dict[str, str]:
    train = _nmse(row.get("train_nmse"))
    id_test = _nmse(row.get("id_nmse"))
    ood = _nmse(row.get("ood_nmse"))

    log_train = _log_nmse(train)
    log_id = _log_nmse(id_test)
    log_ood = _log_nmse(ood)

    finite_train = train is not None
    finite_id = id_test is not None
    finite_ood = ood is not None
    finite_id_ood = finite_id and finite_ood
    finite_train_id_ood = finite_train and finite_id_ood

    id_ood_explosion = any(
        value is not None and value > EXPLOSION_THRESHOLD
        for value in (id_test, ood)
    )
    train_id_ood_explosion = any(
        value is not None and value > EXPLOSION_THRESHOLD
        for value in (train, id_test, ood)
    )

    derived = {
        "finite_train": _flag(finite_train),
        "finite_id": _flag(finite_id),
        "finite_ood": _flag(finite_ood),
        "finite_id_ood": _flag(finite_id_ood),
        "finite_train_id_ood": _flag(finite_train_id_ood),
        "log_train_nmse_clipped": _fmt(log_train),
        "log_id_nmse_clipped": _fmt(log_id),
        "log_ood_nmse_clipped": _fmt(log_ood),
        "combined_log_id_ood_nmse": _fmt(_mean2(log_id, log_ood)),
        "gap_log_ood_minus_id": _fmt(_sub(log_ood, log_id)),
        "delta_id_minus_train": _fmt(_sub(log_id, log_train)),
        "delta_ood_minus_id": _fmt(_sub(log_ood, log_id)),
        "id_ood_explosion_gt_100": _flag(id_ood_explosion),
        "train_id_ood_explosion_gt_100": _flag(train_id_ood_explosion),
    }
    out = dict(row)
    out.update(derived)
    return out


def augment(path: Path, output: Path | None = None) -> None:
    fields, rows = _read_csv(path)
    base_fields = [field for field in fields if field not in STRIP_COLUMNS]
    out_fields = base_fields + DERIVED_COLUMNS
    out_rows = []
    for row in rows:
        base = {field: row.get(field, "") for field in base_fields}
        out_rows.append(_augment_row(base))
    _write_csv(output or path, out_fields, out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    augment(args.input, args.output)
    print({"input": str(args.input), "output": str(args.output or args.input)})


if __name__ == "__main__":
    main()
