#!/usr/bin/env python3
"""收集 Probe4 smoke-test 状态。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _tail(path: Path, n: int = 30) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [f"<read error: {exc!r}>"]
    return lines[-n:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: collect_probe4_smoke_status.py <batch-name>")

    batch = sys.argv[1]
    root = Path("experiments") / batch
    print(f"BATCH {batch}")
    print(f"ROOT {root.resolve()}")
    print(f"ROOT_EXISTS {root.exists()}")

    status_files = sorted(root.glob("*/**/__launcher__/task_status.jsonl"))
    print(f"STATUS_FILES {len(status_files)}")
    for status_file in status_files:
        print(f"### STATUS {status_file}")
        for line in _tail(status_file, 5):
            print(line)

    report_files = sorted(root.glob("*/**/__launcher__/logs/*.report.json"))
    print(f"REPORT_FILES {len(report_files)}")
    for report_file in report_files:
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"REPORT_BAD {report_file} {exc!r}")
            continue
        print(
            "REPORT "
            f"{report_file} "
            f"status={data.get('status')} "
            f"timeout_type={data.get('timeout_type')} "
            f"seconds={data.get('seconds')} "
            f"error={data.get('error')}"
        )

    smoke_logs = sorted(root.glob("__smoke_logs/*/*.log"))
    print(f"SMOKE_LOGS {len(smoke_logs)}")
    for log in smoke_logs:
        print(f"### LOG {log}")
        for line in _tail(log, 20):
            print(line)


if __name__ == "__main__":
    main()
