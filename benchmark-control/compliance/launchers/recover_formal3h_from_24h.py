from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from snapshot_recovery import recover_snapshots


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--source-batch", required=True)
    parser.add_argument("--snapshot-name", default="minute_0180.json")
    parser.add_argument("--source-root", action="append", required=True)
    args = parser.parse_args()

    summary = recover_snapshots(
        batch_dir=_resolve_repo_path(args.batch_dir),
        source_roots=[_resolve_repo_path(value) for value in args.source_root],
        source_batch=args.source_batch,
        snapshot_name=args.snapshot_name,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
