from __future__ import annotations

import csv
from pathlib import Path


RERUN_FIELDS = ["task_id", "algorithm", "dataset_id", "seed", "failure_class"]


def write_rerun_queue(*, batch_dir: Path, round_id: int) -> Path:
    failures_path = batch_dir / "audit" / "failure_cases.csv"
    repair_dir = batch_dir / "repair" / f"round_{round_id:03d}"
    repair_dir.mkdir(parents=True, exist_ok=True)
    output = repair_dir / "rerun_tasks.csv"
    rows = _read_failures(failures_path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RERUN_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RERUN_FIELDS})
    return output


def _read_failures(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
