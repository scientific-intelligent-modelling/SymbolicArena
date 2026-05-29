from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from audit import audit_batch
from heartbeat import write_heartbeat
from rerun import write_rerun_queue


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--write-heartbeat", action="store_true")
    parser.add_argument("--write-rerun", action="store_true")
    parser.add_argument("--round-id", type=int, default=1)
    args = parser.parse_args()
    batch_dir = _resolve_repo_path(args.batch_dir)
    summary = audit_batch(batch_dir=batch_dir)
    latest_rerun_queue = ""
    if args.write_rerun and summary["failed"] > 0:
        rerun_path = write_rerun_queue(batch_dir=batch_dir, round_id=args.round_id)
        latest_rerun_queue = rerun_path.relative_to(batch_dir).as_posix()
    if args.write_heartbeat:
        write_heartbeat(
            batch_dir=batch_dir,
            phase="repair" if summary["failed"] > 0 else "done",
            latest_rerun_queue=latest_rerun_queue,
        )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
