"""Select only new unresolved judgments absent from the frozen base plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _identity(row: dict) -> tuple:
    request = row["request"]
    prefix = (row["condition"], request["algorithm_slug"], request["dataset_id"])
    return prefix + ((request["seed"],) if row["task_type"] == "equivalence" else
                     (request["seed_a"], request["seed_b"]))


def select(base: Path, replanned: Path, output: Path) -> dict:
    base_rows, retry_rows = _rows(base), _rows(replanned)
    existing = {_identity(row) for row in base_rows}
    if len(existing) != len(base_rows):
        raise ValueError("duplicate base task identity")
    keys = [_identity(row) for row in retry_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate replan task identity")
    selected = [row for row in retry_rows if _identity(row) not in existing]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "base_count": len(base_rows), "replan_count": len(retry_rows),
        "selected_count": len(selected), "already_planned_count": len(retry_rows) - len(selected),
        "selected_source_keys": [row["evaluation_key"] for row in selected],
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (output.parent / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.base, args.replan, args.output), indent=2))


if __name__ == "__main__":
    main()
