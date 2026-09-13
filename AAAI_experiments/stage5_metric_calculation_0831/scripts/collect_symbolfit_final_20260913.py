#!/usr/bin/env python3
"""冻结并验收 SymbolFit 最新 noise300 最终结果与分钟轨迹。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


TASK_RE = re.compile(
    r"symbolfit_s(?P<seed>520|521|522)_(?P<condition>noise001|noise005)_g(?P<index>\d{4})$"
)
EXPECTED_NOISE = {"noise001": 0.01, "noise005": 0.05}
EXPECTED_MINUTES = tuple(range(1, 181))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是 object: {path}")
    return value


def read_source_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {int(row["global_index"]): row for row in rows}
    if len(rows) != 50 or len(indexed) != 50 or set(indexed) != set(range(1, 51)):
        raise ValueError("ssr50_source.csv 必须包含唯一的 global_index 1..50")
    return indexed


def read_state(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("batch_name") != "symbolfit_noise_internal_progress_v1_20260910_full":
        raise ValueError(f"批次名不匹配: {payload.get('batch_name')!r}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 300:
        raise ValueError(f"state 必须包含 300 个任务，实际 {len(tasks or {})}")
    if set(tasks) != {str(item.get("task_id")) for item in tasks.values()}:
        raise ValueError("state task key 与 task_id 不一致或重复")

    counts: dict[tuple[str, int], int] = {}
    for task_id, task in tasks.items():
        match = TASK_RE.fullmatch(task_id)
        if match is None:
            raise ValueError(f"任务 ID 不符合 noise300 契约: {task_id}")
        seed = int(match.group("seed"))
        condition = match.group("condition")
        index = int(match.group("index"))
        if task.get("state") != "done" or task.get("status_counts") != {"ok": 1}:
            raise ValueError(f"任务未以唯一 ok 结束: {task_id}")
        if int(task.get("attempts", 0)) != 1:
            raise ValueError(f"任务 attempts 不是 1: {task_id}")
        if int(task.get("seed")) != seed or int(task.get("task_index")) != index:
            raise ValueError(f"state 身份字段与 task_id 不一致: {task_id}")
        if task.get("noise_tag") != condition:
            raise ValueError(f"state condition 与 task_id 不一致: {task_id}")
        if not math.isclose(float(task.get("noise_sigma")), EXPECTED_NOISE[condition]):
            raise ValueError(f"state noise_sigma 不匹配: {task_id}")
        host = str(task.get("assigned_host") or "")
        if host not in {"iaaccn22", "iaaccn23", "iaaccn24", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29"}:
            raise ValueError(f"assigned_host 不合法: {task_id} -> {host!r}")
        counts[(condition, seed)] = counts.get((condition, seed), 0) + 1
    expected_counts = {(condition, seed): 50 for condition in EXPECTED_NOISE for seed in (520, 521, 522)}
    if counts != expected_counts:
        raise ValueError(f"condition/seed 网格不完整: {counts}")
    return tasks


def read_last_jsonl(path: Path) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    last = value
    if last is None:
        raise ValueError(f"JSONL 没有 object 记录: {path}")
    return last


def require_finite_metric(result: dict[str, Any], split: str, task_id: str) -> float:
    metrics = result.get(split)
    if not isinstance(metrics, dict):
        raise ValueError(f"{task_id}: 缺少 {split} 指标")
    value = float(metrics.get("nmse"))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{task_id}: {split}.nmse 非有限非负数")
    return value


def validate_result(
    result: dict[str, Any], task_id: str, source: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    match = TASK_RE.fullmatch(task_id)
    assert match is not None
    seed = int(match.group("seed"))
    condition = match.group("condition")
    index = int(match.group("index"))
    expected_dataset = source["dataset_id"]
    checks = {
        "tool": result.get("tool") == "symbolfit",
        "status": result.get("status") == "ok",
        "seed": int(result.get("seed", -1)) == seed,
        "condition": result.get("condition") == condition,
        "global_index": int(result.get("task_global_index", -1)) == index,
        "dataset": result.get("dataset") == expected_dataset,
        "dataset_identity": result.get("dataset_identity_check", {}).get("match") is True,
    }
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{task_id}: final result 契约失败: {','.join(failed)}")

    equation = result.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        raise ValueError(f"{task_id}: final equation 为空")
    artifact = result.get("canonical_artifact")
    if not isinstance(artifact, dict) or artifact.get("artifact_valid") is not True:
        raise ValueError(f"{task_id}: canonical_artifact 无效")
    if artifact.get("raw_equation") != equation or artifact.get("tool_name") != "symbolfit":
        raise ValueError(f"{task_id}: canonical_artifact 未绑定 final equation")
    if result.get("canonical_artifact_error") is not None:
        raise ValueError(f"{task_id}: canonical_artifact_error 非空")

    params = result.get("params")
    required_params = {
        "timeout_in_seconds": 10800,
        "niterations": 1000000,
        "maxsize": 25,
        "max_complexity": 25,
        "procs": 1,
        "parallelism": "serial",
        "deterministic": True,
    }
    if not isinstance(params, dict) or any(params.get(key) != value for key, value in required_params.items()):
        raise ValueError(f"{task_id}: 固定参数契约不匹配")
    noise = result.get("train_label_noise")
    if not isinstance(noise, dict) or noise.get("enabled") is not True:
        raise ValueError(f"{task_id}: train_label_noise 未启用")
    if not math.isclose(float(noise.get("sigma")), EXPECTED_NOISE[condition]):
        raise ValueError(f"{task_id}: final noise sigma 不匹配")
    protocol = str(noise.get("protocol") or "")
    if "clean labels are used for evaluation" not in protocol:
        raise ValueError(f"{task_id}: 噪声训练/干净评估协议缺失")
    id_nmse = require_finite_metric(result, "id_test", task_id)
    ood_nmse = require_finite_metric(result, "ood_test", task_id)
    return equation, {
        "id_nmse": id_nmse,
        "ood_nmse": ood_nmse,
        "artifact": artifact,
        "params": params,
    }


def matching_candidate(history_path: Path, endpoint: dict[str, Any]) -> dict[str, Any]:
    target_loss = float(endpoint.get("source_internal_loss"))
    target_attempt = int(endpoint.get("candidate_attempt"))
    target_complexity = int(endpoint.get("source_complexity"))
    matches: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                isinstance(row, dict)
                and int(row.get("attempt", -1)) == target_attempt
                and int(row.get("complexity", -1)) == target_complexity
                and math.isclose(float(row.get("internal_loss")), target_loss, rel_tol=1e-12, abs_tol=1e-15)
                and row.get("source") == endpoint.get("candidate_source")
                and row.get("coordinate_transform") == endpoint.get("candidate_coordinate_transform")
            ):
                matches.append(row)
    if not matches:
        raise ValueError("minute_0180 在内部候选历史中没有匹配记录")
    identities = {
        (
            row.get("candidate_key"),
            row.get("equation"),
            row.get("first_discovered_attempt"),
            row.get("first_discovered_minute"),
            row.get("first_discovered_elapsed_seconds"),
        )
        for row in matches
    }
    if len(identities) != 1:
        raise ValueError(f"minute_0180 匹配到 {len(identities)} 个不同内部候选")
    row = matches[0]
    if row.get("first_discovered_attempt") != endpoint.get("candidate_first_discovered_attempt"):
        raise ValueError("minute_0180 首次发现 attempt 与内部历史不一致")
    if row.get("first_discovered_minute") != endpoint.get("candidate_first_discovered_minute"):
        raise ValueError("minute_0180 首次发现 minute 与内部历史不一致")
    return {**row, "matching_history_row_count": len(matches)}


def validate_snapshots(progress_dir: Path, task_id: str, *, write_to: Path | None) -> dict[str, Any]:
    paths = sorted(progress_dir.glob("minute_*.json"))
    minute_map: dict[int, Path] = {}
    for path in paths:
        match = re.fullmatch(r"minute_(\d{4})\.json", path.name)
        if match:
            minute_map[int(match.group(1))] = path
    missing = sorted(set(EXPECTED_MINUTES).difference(minute_map))
    extra = sorted(set(minute_map).difference(EXPECTED_MINUTES))
    if missing:
        raise ValueError(f"{task_id}: 正式 1..180 分钟快照不完整 missing={missing}")

    candidate_minutes = 0
    last_candidate_minute: int | None = None
    writer = gzip.open(write_to, "wt", encoding="utf-8") if write_to else None
    try:
        for minute in EXPECTED_MINUTES:
            payload = load_json(minute_map[minute])
            if int(payload.get("checkpoint_index", -1)) != minute:
                raise ValueError(f"{task_id}: minute_{minute:04d} checkpoint_index 不匹配")
            available = payload.get("candidate_available") is True
            if available:
                candidate_minutes += 1
                last_candidate_minute = minute
                if payload.get("candidate_source") != "symbolfit_active_pysr_hall_of_fame":
                    raise ValueError(f"{task_id}: minute_{minute:04d} candidate_source 不合法")
                if not math.isfinite(float(payload.get("source_internal_loss"))):
                    raise ValueError(f"{task_id}: minute_{minute:04d} source_internal_loss 非有限")
            if writer:
                writer.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                writer.write("\n")
    finally:
        if writer:
            writer.close()
    return {
        "snapshot_count": 180,
        "candidate_snapshot_count": candidate_minutes,
        "last_candidate_minute": last_candidate_minute,
        "excluded_post_budget_snapshot_count": len(extra),
        "excluded_post_budget_minutes": ";".join(str(minute) for minute in extra),
    }


def locate_task_files(batch_root: Path, task_id: str, host: str) -> dict[str, Path]:
    match = TASK_RE.fullmatch(task_id)
    assert match is not None
    seed = int(match.group("seed"))
    index = int(match.group("index"))
    task_root = batch_root / "symbolfit" / f"seed{seed}" / "tasks" / task_id / host
    run_roots = list((task_root / "symbolfit").glob(f"g{index:04d}_*"))
    if len(run_roots) != 1:
        raise ValueError(f"{task_id}: 外层 run 目录数量为 {len(run_roots)}")
    run_root = run_roots[0]
    result_path = run_root / "result.json"
    progress_dir = run_root / "progress"
    launcher_path = task_root / "__launcher__" / "task_status.jsonl"
    if not result_path.is_file() or not progress_dir.is_dir() or not launcher_path.is_file():
        raise ValueError(f"{task_id}: final/progress/launcher 文件不完整")
    result = load_json(result_path)
    experiment_dir = Path(str(result.get("experiment_dir") or ""))
    history_path = experiment_dir / ".symbolfit_pysr_candidates.jsonl"
    if not history_path.is_file():
        raise ValueError(f"{task_id}: 内部候选历史不存在: {history_path}")
    return {
        "task_root": task_root,
        "run_root": run_root,
        "result": result_path,
        "progress": progress_dir,
        "launcher": launcher_path,
        "history": history_path,
    }


MANIFEST_FIELDS = (
    "task_id",
    "logical_key",
    "condition",
    "algorithm",
    "dataset_id",
    "dataset_global_index",
    "seed",
    "host",
    "source_result_path",
    "source_result_sha256",
    "source_launcher_path",
    "source_launcher_sha256",
    "frozen_result_path",
    "frozen_result_sha256",
    "final_equation",
    "final_equation_sha256",
    "canonical_artifact_sha256",
    "id_nmse",
    "ood_nmse",
    "endpoint_path",
    "endpoint_sha256",
    "endpoint_candidate_source",
    "endpoint_internal_loss",
    "endpoint_matches_final",
    "endpoint_candidate_evidence_path",
    "endpoint_candidate_evidence_sha256",
    "trajectory_path",
    "trajectory_sha256",
    "snapshot_count",
    "candidate_snapshot_count",
    "last_candidate_minute",
    "excluded_post_budget_snapshot_count",
    "excluded_post_budget_minutes",
    "validation_status",
    "replacement_authorization",
)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_host(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    source_csv = args.source_csv.resolve()
    batch_root = args.batch_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = read_state(state_path)
    sources = read_source_rows(source_csv)
    selected = [(task_id, task) for task_id, task in sorted(tasks.items()) if task["assigned_host"] == args.host]
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []

    for task_id, task in selected:
        try:
            match = TASK_RE.fullmatch(task_id)
            assert match is not None
            condition = match.group("condition")
            seed = int(match.group("seed"))
            index = int(match.group("index"))
            source = sources[index]
            files = locate_task_files(batch_root, task_id, args.host)
            result = load_json(files["result"])
            equation, validated = validate_result(result, task_id, source)
            launcher = read_last_jsonl(files["launcher"])
            if launcher.get("status") != "ok" or int(launcher.get("task_global_index", -1)) != index:
                raise ValueError(f"{task_id}: launcher 最终记录不合法")
            endpoint_path = files["progress"] / "minute_0180.json"
            endpoint = load_json(endpoint_path)
            if endpoint.get("record_type") != "budget_end_internal_best":
                raise ValueError(f"{task_id}: minute_0180 不是预算终点内部最优")
            if endpoint.get("candidate_available") is not True:
                raise ValueError(f"{task_id}: minute_0180 没有内部候选")
            endpoint_matches_final = (
                endpoint.get("equation") == equation
                and endpoint.get("canonical_artifact") == validated["artifact"]
            )
            candidate = matching_candidate(files["history"], endpoint)

            frozen_result = output_root / "final_results" / f"{task_id}.result.json"
            frozen_launcher = output_root / "launcher_status" / f"{task_id}.task_status.jsonl"
            frozen_endpoint = output_root / "endpoints" / f"{task_id}.minute_0180.json"
            frozen_evidence = output_root / "endpoint_candidate_evidence" / f"{task_id}.json"
            for destination in (frozen_result, frozen_launcher, frozen_endpoint, frozen_evidence):
                destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files["result"], frozen_result)
            shutil.copyfile(files["launcher"], frozen_launcher)
            shutil.copyfile(endpoint_path, frozen_endpoint)
            frozen_evidence.write_text(
                json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            trajectory_path: Path | None = None
            trajectory_stats: dict[str, Any] = {
                "snapshot_count": "",
                "candidate_snapshot_count": "",
                "last_candidate_minute": "",
                "excluded_post_budget_snapshot_count": "",
                "excluded_post_budget_minutes": "",
            }
            if args.include_trajectories:
                trajectory_path = output_root / "trajectories" / f"{task_id}.jsonl.gz"
                trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                trajectory_stats = validate_snapshots(files["progress"], task_id, write_to=trajectory_path)

            artifact_text = json.dumps(validated["artifact"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            row = {
                "task_id": task_id,
                "logical_key": f"SymbolFit::{source['dataset_id']}::s{seed}::{condition}",
                "condition": condition,
                "algorithm": "SymbolFit",
                "dataset_id": source["dataset_id"],
                "dataset_global_index": index,
                "seed": seed,
                "host": args.host,
                "source_result_path": str(files["result"]),
                "source_result_sha256": sha256_file(files["result"]),
                "source_launcher_path": str(files["launcher"]),
                "source_launcher_sha256": sha256_file(files["launcher"]),
                "frozen_result_path": str(frozen_result.relative_to(output_root)),
                "frozen_result_sha256": sha256_file(frozen_result),
                "final_equation": equation,
                "final_equation_sha256": sha256_text(equation),
                "canonical_artifact_sha256": sha256_text(artifact_text),
                "id_nmse": validated["id_nmse"],
                "ood_nmse": validated["ood_nmse"],
                "endpoint_path": str(frozen_endpoint.relative_to(output_root)),
                "endpoint_sha256": sha256_file(frozen_endpoint),
                "endpoint_candidate_source": endpoint["candidate_source"],
                "endpoint_internal_loss": endpoint["source_internal_loss"],
                "endpoint_matches_final": str(endpoint_matches_final).lower(),
                "endpoint_candidate_evidence_path": str(frozen_evidence.relative_to(output_root)),
                "endpoint_candidate_evidence_sha256": sha256_file(frozen_evidence),
                "trajectory_path": str(trajectory_path.relative_to(output_root)) if trajectory_path else "",
                "trajectory_sha256": sha256_file(trajectory_path) if trajectory_path else "",
                **trajectory_stats,
                "validation_status": "passed",
                "replacement_authorization": "not_authorized_collected_candidate_only",
            }
            rows.append(row)
        except Exception as exc:
            unresolved.append({"task_id": task_id, "host": args.host, "error": f"{type(exc).__name__}: {exc}"})

    write_csv(output_root / "manifest.csv", rows, MANIFEST_FIELDS)
    (output_root / "unresolved.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in unresolved),
        encoding="utf-8",
    )
    report = {
        "schema_version": "symbolfit_noise_latest_host_freeze_v1",
        "host": args.host,
        "phase": "trajectory" if args.include_trajectories else "final",
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "batch_root": str(batch_root),
        "expected_task_count": len(selected),
        "validated_task_count": len(rows),
        "unresolved_count": len(unresolved),
        "status": "passed" if not unresolved and len(rows) == len(selected) else "failed",
        "replacement_authorization": "not_authorized_collected_candidate_only",
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


def merge_hosts(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    tasks = read_state(state_path)
    input_root = args.input_root.resolve()
    manifests = sorted((input_root / "hosts").glob("iaaccn*/manifest.csv"))
    rows: list[dict[str, str]] = []
    unresolved: list[dict[str, Any]] = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
        unresolved_path = manifest.with_name("unresolved.jsonl")
        if unresolved_path.is_file():
            for line in unresolved_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    unresolved.append(json.loads(line))
    seen = [row["task_id"] for row in rows]
    duplicate = sorted({task_id for task_id in seen if seen.count(task_id) > 1})
    missing = sorted(set(tasks).difference(seen))
    extra = sorted(set(seen).difference(tasks))
    if duplicate:
        unresolved.append({"error": "duplicate_task_ids", "task_ids": duplicate})
    if missing:
        unresolved.append({"error": "missing_task_ids", "task_ids": missing})
    if extra:
        unresolved.append({"error": "extra_task_ids", "task_ids": extra})

    rows.sort(key=lambda row: (row["condition"], int(row["seed"]), int(row["dataset_global_index"])))
    manifest_dir = input_root / "manifests"
    write_csv(manifest_dir / "symbolfit_noise_latest_runs.csv", rows, MANIFEST_FIELDS)
    (manifest_dir / "unresolved.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in unresolved),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['condition']}::s{row['seed']}"
        counts[key] = counts.get(key, 0) + 1
    report = {
        "schema_version": "symbolfit_noise_latest_freeze_v1",
        "status": "passed" if len(rows) == 300 and not unresolved else "failed",
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "host_manifest_count": len(manifests),
        "expected_task_count": 300,
        "validated_task_count": len(rows),
        "unresolved_count": len(unresolved),
        "condition_seed_counts": counts,
        "all_results_have_canonical_artifact": all(row["canonical_artifact_sha256"] for row in rows),
        "all_endpoints_bound_to_internal_history": all(row["endpoint_candidate_evidence_sha256"] for row in rows),
        "endpoint_matches_final_count": sum(row["endpoint_matches_final"] == "true" for row in rows),
        "endpoint_differs_from_final_count": sum(row["endpoint_matches_final"] == "false" for row in rows),
        "all_trajectories_complete": len(rows) == 300
        and all(row["snapshot_count"] == "180" and row["trajectory_sha256"] for row in rows),
        "replacement_authorization": "not_authorized_collected_candidate_only",
    }
    report_path = input_root / "reports" / "symbolfit_noise_latest_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = [path for path in input_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    with (input_root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(checksum_paths):
            handle.write(f"{sha256_file(path)}  {path.relative_to(input_root)}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect-host")
    collect.add_argument("--state", required=True, type=Path)
    collect.add_argument("--source-csv", required=True, type=Path)
    collect.add_argument("--batch-root", required=True, type=Path)
    collect.add_argument("--host", required=True)
    collect.add_argument("--output-root", required=True, type=Path)
    collect.add_argument("--include-trajectories", action="store_true")
    collect.set_defaults(func=collect_host)

    merge = subparsers.add_parser("merge-hosts")
    merge.add_argument("--state", required=True, type=Path)
    merge.add_argument("--input-root", required=True, type=Path)
    merge.set_defaults(func=merge_hosts)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
