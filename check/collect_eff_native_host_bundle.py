#!/usr/bin/env python3
"""在单台实验机本地冻结 result + 180 snapshots 为 gzip JSONL。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    records = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for task in request["tasks"]:
            experiment_root = Path(task["experiment_root"])
            task_id = task["task_id"]
            matches = list(experiment_root.glob(f"**/tasks/{task_id}/**/result.json"))
            matches = [
                path
                for path in matches
                if "/experiments/" not in str(path).split(f"/tasks/{task_id}/", 1)[-1]
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{task_id}: result count={len(matches)}")
            result_path = matches[0]
            progress_dir = result_path.parent / "progress"
            snapshots = [progress_dir / f"minute_{minute:04d}.json" for minute in range(1, 181)]
            missing = [str(path) for path in snapshots if not path.is_file()]
            if missing:
                raise RuntimeError(f"{task_id}: missing snapshots={missing[:3]}")
            for record_type, path, minute in [
                ("result", result_path, None),
                *(("snapshot", path, index) for index, path in enumerate(snapshots, start=1)),
            ]:
                raw = path.read_bytes()
                payload = {
                    "schema_version": "eff_native_raw_record.v1",
                    "record_type": record_type,
                    "task_id": task_id,
                    "tool": task["tool"],
                    "condition": task["noise_tag"],
                    "seed": task["seed"],
                    "minute": minute,
                    "source_host": request["host"],
                    "source_path": str(path),
                    "sha256": sha(raw),
                    "raw_text": raw.decode("utf-8"),
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                records += 1
    temporary.replace(args.output)
    expected = len(request["tasks"]) * 181
    if records != expected:
        raise RuntimeError(f"records={records} != {expected}")
    report = {
        "host": request["host"],
        "task_count": len(request["tasks"]),
        "record_count": records,
        "output": str(args.output),
        "output_sha256": sha(args.output.read_bytes()),
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
