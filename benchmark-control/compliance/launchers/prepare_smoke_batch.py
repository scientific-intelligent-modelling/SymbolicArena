from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from smoke import prepare_smoke_batch


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default="benchmark-runs/compliance/latest")
    args = parser.parse_args()
    print(prepare_smoke_batch(batch_dir=_resolve_repo_path(args.batch_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
