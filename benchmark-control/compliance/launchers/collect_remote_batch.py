from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from remote_collect import DEFAULT_HOSTS, DEFAULT_REMOTE_ROOT, collect_remote_batch


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--hosts", nargs="+", default=DEFAULT_HOSTS)
    parser.add_argument("--remote-root", default=str(DEFAULT_REMOTE_ROOT))
    parser.add_argument("--controller-host", default="iaaccn22")
    parser.add_argument("--use-internal-ips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-task-logs", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    summary = collect_remote_batch(
        batch_dir=_resolve_repo_path(args.batch_dir),
        batch_id=args.batch_id,
        hosts=args.hosts,
        remote_root=Path(args.remote_root).expanduser(),
        controller_host=args.controller_host,
        use_internal_ips=args.use_internal_ips,
        include_task_logs=args.include_task_logs,
        timeout=args.timeout,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
