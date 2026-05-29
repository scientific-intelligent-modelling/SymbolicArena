from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from audit_gate import check_audit_success


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--expected-total-tasks", type=int, default=None)
    args = parser.parse_args()
    summary = check_audit_success(
        batch_dir=_resolve_repo_path(args.batch_dir),
        expected_total_tasks=args.expected_total_tasks,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
