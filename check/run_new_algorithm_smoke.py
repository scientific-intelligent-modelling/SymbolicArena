"""Smoke runner for newly integrated SR algorithms.

This script is intentionally small and host-local.  It is used both by
developers and by remote tmux smoke jobs on iaaccn22~29.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path


CHECK_MODULES = {
    "fepysr": "check.check_fepysr",
    "jaxsr": "check.check_jaxsr",
    "symbolfit": "check.check_symbolfit",
}


def run_one(tool: str) -> dict:
    started = time.time()
    payload = {
        "tool": tool,
        "status": "ok",
        "seconds": None,
        "error": None,
    }
    try:
        module = importlib.import_module(CHECK_MODULES[tool])
        module.run()
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = repr(exc)
    payload["seconds"] = round(time.time() - started, 3)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tools",
        default="fepysr,jaxsr,symbolfit",
        help="逗号分隔工具名，默认跑三个新增算法",
    )
    parser.add_argument(
        "--output",
        default="new_algorithm_smoke_summary.json",
        help="summary JSON 输出路径",
    )
    args = parser.parse_args()

    tools = [item.strip() for item in args.tools.split(",") if item.strip()]
    unknown = sorted(set(tools) - set(CHECK_MODULES))
    if unknown:
        raise ValueError(f"未知工具: {unknown}")

    rows = [run_one(tool) for tool in tools]
    summary = {
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "error",
        "rows": rows,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
