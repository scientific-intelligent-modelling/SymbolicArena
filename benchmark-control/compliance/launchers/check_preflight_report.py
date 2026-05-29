from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from preflight_gate import check_preflight_report


DEFAULT_HOSTS = [
    "iaaccn22",
    "iaaccn23",
    "iaaccn24",
    "iaaccn25",
    "iaaccn26",
    "iaaccn27",
    "iaaccn28",
    "iaaccn29",
]


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default="benchmark-runs/compliance/latest")
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-hosts", nargs="+", default=DEFAULT_HOSTS)
    args = parser.parse_args()
    summary = check_preflight_report(
        report_path=_resolve_repo_path(args.report),
        expected_hosts=args.expected_hosts,
        batch_dir=_resolve_repo_path(args.batch_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready_for_smoke"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
