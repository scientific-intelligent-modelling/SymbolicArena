from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from models import FORMAL3H_SPEC, FULL24H_SPEC, STAGE1_SPEC
from readiness import check_readiness


PROFILE_CHOICES = ("stage1_1h", STAGE1_SPEC.name, FULL24H_SPEC.name, FORMAL3H_SPEC.name)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="stage1_1h")
    parser.add_argument("--batch-dir", default="benchmark-runs/compliance/latest")
    args = parser.parse_args()
    summary = check_readiness(batch_dir=_resolve_repo_path(args.batch_dir), profile=args.profile)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
