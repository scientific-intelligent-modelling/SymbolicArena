#!/usr/bin/env python3
"""Run a 5-minute Core50 smoke for all integrated algorithms on one host."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


TOOLS: dict[str, dict[str, str]] = {
    "gplearn": {"tool_arg": "gplearn", "params": "gplearn", "env": "sim_base"},
    "pysr": {"tool_arg": "pysr", "params": "pysr", "env": "sim_base"},
    "pyoperon": {"tool_arg": "pyoperon", "params": "pyoperon", "env": "sim_base"},
    "llmsr": {"tool_arg": "llmsr", "params": "llmsr", "env": "sim_llm"},
    "drsr": {"tool_arg": "drsr", "params": "drsr", "env": "sim_llm"},
    "dso": {"tool_arg": "dso", "params": "dso", "env": "sim_dso"},
    "udsr": {"tool_arg": "udsr", "params": "udsr", "env": "sim_dso"},
    "tpsr": {"tool_arg": "tpsr", "params": "tpsr", "env": "sim_tpsr"},
    "e2esr": {"tool_arg": "e2esr", "params": "e2esr", "env": "sim_e2esr"},
    "qlattice": {"tool_arg": "QLattice", "params": "qlattice", "env": "sim_qLattice"},
    "imcts": {"tool_arg": "iMCTS", "params": "imcts", "env": "sim_iMCTS"},
    "ragsr": {"tool_arg": "ragsr", "params": "ragsr", "env": "sim_ragsr"},
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_dataset_row(root: Path, data_root: Path, dataset_index: int) -> dict[str, str]:
    core50_csv = root / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
    with core50_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["core50_index"]) == dataset_index:
            out = dict(row)
            rel = Path(out["dataset_dir"])
            if rel.parts and rel.parts[0] == "sim-datasets-data":
                dataset_dir = data_root.joinpath(*rel.parts[1:])
            else:
                dataset_dir = data_root / rel
            out["dataset_dir"] = str(dataset_dir)
            out["dataset_rel"] = row["dataset_dir"]
            out["global_index"] = row["core50_index"]
            return out
    raise ValueError(f"core50_index not found: {dataset_index}")


def write_slice(out_root: Path, row: dict[str, str]) -> Path:
    path = out_root / "__smoke_assets" / "single_dataset_slice.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return path


def write_params(root: Path, out_root: Path, tool: str, timeout_seconds: int) -> Path:
    cfg = TOOLS[tool]
    src = root / "exp-planning/02.E1选择验证/generated/params" / f"{cfg['params']}.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["timeout_in_seconds"] = timeout_seconds
    payload["progress_snapshot_interval_seconds"] = 60
    if tool in {"llmsr", "drsr"}:
        payload["inject_prompt_semantics"] = True
        payload["canonical_prompt_variables"] = True
        payload["llm_config_path"] = str(
            root / "exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config"
        )
    path = out_root / "__smoke_assets" / "params" / f"{tool}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last


def load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def progress_count(result: dict[str, Any] | None) -> int:
    if not result:
        return 0
    result_path = result.get("result_path")
    if result_path:
        root = Path(result_path).parent
        return len(list(root.rglob("minute_*.json")))
    experiment_dir = result.get("experiment_dir")
    if experiment_dir:
        return len(list(Path(experiment_dir).rglob("minute_*.json")))
    return 0


def run_tool(
    *,
    root: Path,
    out_root: Path,
    slice_csv: Path,
    tool: str,
    params_json: Path,
    seed: int,
    outer_timeout_seconds: int,
) -> dict[str, Any]:
    cfg = TOOLS[tool]
    tool_out = out_root / tool
    tool_out.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "__smoke_assets" / "host_logs" / f"{tool}.launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_prefix = (
        "PYTHONPATH=. "
        "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 "
    )
    pysr_cache = Path.home() / "pyjuliapkg_pysr"
    if tool == "pysr" and pysr_cache.exists():
        env_prefix = f"PYTHON_JULIAPKG_PROJECT={pysr_cache} " + env_prefix
    cmd = (
        f"cd {root} && "
        f"{env_prefix}"
        f"conda run -n {cfg['env']} python check/launch_e1_benchmark.py run "
        f"--tool {cfg['tool_arg']} "
        f"--slice-csv {slice_csv} "
        f"--params-json {params_json} "
        f"--output-root {tool_out} "
        f"--seed {seed} "
        f"--workers 1 "
        f"--retry-failed"
    )
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["timeout", "--kill-after=30s", str(outer_timeout_seconds), "bash", "-lc", cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - start
    status = read_last_jsonl(tool_out / "__launcher__" / "task_status.jsonl")
    result = load_json(status.get("result_path") if status else None)
    metrics: dict[str, Any] = {}
    if result:
        for key in (
            "status",
            "seconds",
            "budget_exhausted",
            "timeout_type",
            "termination_reason",
            "recovered_from_timeout",
            "equation",
        ):
            metrics[f"result_{key}"] = result.get(key)
        for split in ("train", "valid", "id_test", "ood_test"):
            value = (result.get(split) or {}).get("nmse") if isinstance(result.get(split), dict) else None
            metrics[f"{split}_nmse"] = value
    return {
        "tool": tool,
        "env": cfg["env"],
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "launcher_status": status.get("status") if status else None,
        "launcher_error": status.get("error") if status else None,
        "result_path": status.get("result_path") if status else None,
        "log_path": str(log_path),
        "progress_snapshots": progress_count(status),
        **metrics,
    }


def write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
    summary = {
        "time": now(),
        "out_root": str(out_root),
        "total": len(rows),
        "launcher_status_counts": {},
        "returncode_counts": {},
        "rows": rows,
    }
    for row in rows:
        summary["launcher_status_counts"][str(row.get("launcher_status"))] = (
            summary["launcher_status_counts"].get(str(row.get("launcher_status")), 0) + 1
        )
        summary["returncode_counts"][str(row.get("returncode"))] = (
            summary["returncode_counts"].get(str(row.get("returncode")), 0) + 1
        )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--dataset-index", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--outer-timeout-seconds", type=int, default=480)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--tools", nargs="+", default=list(TOOLS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = root / "experiments" / args.batch_name / args.host_label
    row = load_dataset_row(root, data_root, args.dataset_index)
    slice_csv = write_slice(out_root, row)
    tools = [tool for tool in args.tools if tool in TOOLS]
    params = {tool: write_params(root, out_root, tool, args.timeout_seconds) for tool in tools}
    manifest = {
        "time": now(),
        "host_label": args.host_label,
        "root": str(root),
        "data_root": str(data_root),
        "dataset": row,
        "timeout_seconds": args.timeout_seconds,
        "outer_timeout_seconds": args.outer_timeout_seconds,
        "tools": tools,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "start", **manifest}, ensure_ascii=False), flush=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        future_map = {
            executor.submit(
                run_tool,
                root=root,
                out_root=out_root,
                slice_csv=slice_csv,
                tool=tool,
                params_json=params[tool],
                seed=args.seed,
                outer_timeout_seconds=args.outer_timeout_seconds,
            ): tool
            for tool in tools
        }
        for future in as_completed(future_map):
            tool = future_map[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {"tool": tool, "returncode": -1, "launcher_status": "smoke_exception", "launcher_error": repr(exc)}
            rows.append(item)
            write_summary(out_root, sorted(rows, key=lambda r: str(r.get("tool"))))
            print(json.dumps({"event": "tool_done", **item}, ensure_ascii=False), flush=True)

    write_summary(out_root, sorted(rows, key=lambda r: str(r.get("tool"))))
    print(json.dumps({"event": "done", "summary": str(out_root / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
