from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from harvest import harvest_batch


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--experiment-root", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = harvest_batch(
        batch_dir=_resolve_repo_path(args.batch_dir),
        experiment_roots=[_resolve_repo_path(path) for path in args.experiment_root],
        dry_run=args.dry_run,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
