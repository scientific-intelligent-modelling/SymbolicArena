#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
EXP = BASE / "experiments"

BATCHES = [
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260606-015625",
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658",
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658_fepysr_rerun",
    "formal3h_tpsr_ragsr_rerun_20260624-115748",
    "llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442",
    "drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-161400",
    "drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355",
    "compliance_15alg_ssr50_seed520_1h_20260529-234937",
]

HOSTS = [
    ("iaaccn22", None),
    ("iaaccn23", "10.10.100.23"),
    ("iaaccn24", "10.10.100.24"),
    ("iaaccn25", "10.10.100.25"),
    ("iaaccn26", "10.10.100.26"),
    ("iaaccn27", "10.10.100.27"),
    ("iaaccn28", "10.10.100.28"),
    ("iaaccn29", "10.10.100.29"),
]


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return None


def split_metric(payload: dict[str, Any], split: str, key: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    return finite_float(block.get(key))


def parse_path(batch: str, path: Path) -> tuple[str | None, int | None, str | None, str | None]:
    rel = path.relative_to(EXP / batch)
    parts = rel.parts
    algorithm = None
    seed = None
    task_id = None
    dataset_id = None

    if "tasks" in parts:
        idx = parts.index("tasks")
        if idx > 0:
            # common layout: <algorithm>/seed<seed>/tasks/<task_id>/<host>/<algorithm>/...
            algorithm = parts[idx - 2] if idx >= 2 else None
            m_seed = re.match(r"seed(\d+)$", parts[idx - 1])
            seed = int(m_seed.group(1)) if m_seed else None
        if idx + 1 < len(parts):
            task_id = parts[idx + 1]
            m_task = re.match(r"(?P<alg>[^_]+)_s(?P<seed>\d+)_(?P<noise>clean|noise001|noise005)_g(?P<gid>\d+)", task_id)
            if m_task:
                algorithm = algorithm or m_task.group("alg")
                seed = seed or int(m_task.group("seed"))
        # dataset directory is usually after .../<host>/<algorithm>/
        for p in parts[idx + 2:]:
            if re.match(r"g\d{4}_", p):
                dataset_id = p.split("_", 1)[1]
                break
    else:
        # compliance layout: <algorithm>/seed<seed>/<noise>/<dataset>/result.json
        if len(parts) >= 5:
            algorithm = parts[0]
            m_seed = re.match(r"seed(\d+)$", parts[1])
            seed = int(m_seed.group(1)) if m_seed else None
            dataset_id = parts[3]

    if task_id is None:
        m = re.search(r"([a-zA-Z0-9]+)_s(\d+)_(clean|noise001|noise005)_g(\d+)", str(path))
        if m:
            task_id = m.group(0)
    return algorithm, seed, task_id, dataset_id


def parse_noise(payload: dict[str, Any], task_id: str | None, path: Path) -> str | None:
    for key in ("noise_tag", "noise"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    for source in (task_id or "", str(path)):
        m = re.search(r"_(clean|noise001|noise005)_", source)
        if m:
            return m.group(1)
        m = re.search(r"/(clean|noise001|noise005)/", source)
        if m:
            return m.group(1)
    return None


def iter_result_paths(batch: str) -> list[Path]:
    root = EXP / batch
    if not root.exists():
        return []
    # Avoid inner algorithm experiment result.json files; keep task-level result.json.
    out: list[Path] = []
    for p in root.glob("**/result.json"):
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            rel_parts = p.parts
        if "experiments" in rel_parts:
            continue
        out.append(p)
    return out


def scan_local(host_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in BATCHES:
        for p in iter_result_paths(batch):
            try:
                payload = json.loads(p.read_text())
            except Exception as exc:
                out.append({"host": host_label, "batch": batch, "path": str(p), "read_error": repr(exc)})
                continue
            alg, seed, task_id, dataset_id = parse_path(batch, p)
            alg = payload.get("tool") or payload.get("algorithm") or alg
            task_id = payload.get("task_id") or task_id
            row = {
                "host": host_label,
                "batch": batch,
                "path": str(p),
                "algorithm": alg,
                "seed": seed,
                "task_id": task_id,
                "dataset_id": payload.get("dataset_id") or dataset_id,
                "noise_tag": parse_noise(payload, task_id, p),
                "status": payload.get("status"),
                "seconds": finite_float(payload.get("seconds")),
                "id_nmse": split_metric(payload, "id_test", "nmse"),
                "ood_nmse": split_metric(payload, "ood_test", "nmse"),
                "valid_nmse": split_metric(payload, "valid", "nmse"),
                "train_nmse": split_metric(payload, "train", "nmse"),
                "id_r2": split_metric(payload, "id_test", "r2"),
                "ood_r2": split_metric(payload, "ood_test", "r2"),
                "id_acc": split_metric(payload, "id_test", "acc_0_1"),
                "ood_acc": split_metric(payload, "ood_test", "acc_0_1"),
                "recovered_from_timeout": bool(payload.get("recovered_from_timeout")),
            }
            out.append(row)
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_batch = Counter(r.get("batch") for r in rows)
    by_alg = Counter(r.get("algorithm") for r in rows)
    return {"rows": len(rows), "by_batch": dict(by_batch), "by_algorithm": dict(by_alg)}


def remote_collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    self_path = Path(__file__)
    for host_label, addr in HOSTS:
        if addr is None:
            try:
                rows.extend(scan_local(host_label))
            except Exception as exc:
                errors.append({"host": host_label, "error": repr(exc)})
            continue
        cmd = [
            "timeout",
            "90",
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            addr,
            f"python3 {self_path} --local {host_label}",
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            errors.append({"host": host_label, "error": proc.stderr.strip() or f"rc={proc.returncode}"})
            continue
        try:
            payload = json.loads(proc.stdout)
            rows.extend(payload.get("rows", []))
            for e in payload.get("errors", []):
                errors.append(e)
        except Exception as exc:
            errors.append({"host": host_label, "error": f"bad json: {exc!r}; stdout={proc.stdout[:200]!r}"})
    return rows, errors


def _median(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(statistics.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(statistics.fmean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def clipped_log_nmse(v: float | None, clip: float = 12.0) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return min(max(math.log10(max(v, 1e-12)), -12.0), clip)


def phi_nmse(v: float | None) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    # Same monotone shape used in several local analysis scripts: higher is better.
    return 1.0 / (1.0 + max(v, 0.0))


def dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("batch"), r.get("algorithm"), r.get("seed"), r.get("noise_tag"), r.get("task_id") or r.get("dataset_id"))
        grouped[key].append(r)
    chosen = []
    dups = []
    for key, items in grouped.items():
        if len(items) > 1:
            dups.extend(items)
        def rank(r: dict[str, Any]) -> tuple[int, int, float]:
            ok = 1 if r.get("status") == "ok" else 0
            finite = 1 if r.get("id_nmse") is not None and r.get("ood_nmse") is not None else 0
            sec = r.get("seconds")
            return (ok, finite, float(sec) if isinstance(sec, (int, float)) else -1.0)
        chosen.append(sorted(items, key=rank, reverse=True)[0])
    return chosen, dups


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_batch_alg: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("algorithm"):
            by_batch_alg[(r["batch"], r["algorithm"])].append(r)
    tables = []
    for (batch, alg), items in sorted(by_batch_alg.items()):
        id_vals = [r["id_nmse"] for r in items if r.get("id_nmse") is not None]
        ood_vals = [r["ood_nmse"] for r in items if r.get("ood_nmse") is not None]
        id_logs = [clipped_log_nmse(r.get("id_nmse")) for r in items if clipped_log_nmse(r.get("id_nmse")) is not None]
        ood_logs = [clipped_log_nmse(r.get("ood_nmse")) for r in items if clipped_log_nmse(r.get("ood_nmse")) is not None]
        quality = []
        for r in items:
            qi = phi_nmse(r.get("id_nmse"))
            qo = phi_nmse(r.get("ood_nmse"))
            if qi is not None and qo is not None:
                quality.append(0.5 * (qi + qo))
        row = {
            "batch": batch,
            "algorithm": alg,
            "n": len(items),
            "ok": sum(1 for r in items if r.get("status") == "ok"),
            "status_counts": dict(Counter(r.get("status") for r in items)),
            "finite_id_ood": sum(1 for r in items if r.get("id_nmse") is not None and r.get("ood_nmse") is not None),
            "median_id_nmse": _median(id_vals),
            "median_ood_nmse": _median(ood_vals),
            "mean_log10_id_nmse": _mean(id_logs),
            "mean_log10_ood_nmse": _mean(ood_logs),
            "median_quality": _median(quality),
            "p25_quality": _quantile(quality, 0.25),
            "p75_quality": _quantile(quality, 0.75),
            "median_id_r2": _median([r["id_r2"] for r in items if r.get("id_r2") is not None]),
            "median_ood_r2": _median([r["ood_r2"] for r in items if r.get("ood_r2") is not None]),
            "median_id_acc": _median([r["id_acc"] for r in items if r.get("id_acc") is not None]),
            "median_ood_acc": _median([r["ood_acc"] for r in items if r.get("ood_acc") is not None]),
        }
        tables.append(row)
    return {"by_batch_algorithm": tables}


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--local":
        host_label = sys.argv[2]
        rows = scan_local(host_label)
        print(json.dumps({"rows": rows, "errors": [], "summary": summarize_rows(rows)}, ensure_ascii=False))
        return 0

    rows, errors = remote_collect()
    unique, dups = dedupe(rows)
    payload = {
        "raw_summary": summarize_rows(rows),
        "unique_summary": summarize_rows(unique),
        "duplicate_count": len(dups),
        "errors": errors,
        "aggregates": aggregate(unique),
        "rows": unique,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
