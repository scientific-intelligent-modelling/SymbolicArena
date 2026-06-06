from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HARVEST_FIELDS = [
    "task_id",
    "algorithm",
    "dataset_id",
    "seed",
    "noise_tag",
    "noise_sigma",
    "status",
    "source_result",
    "target_dir",
    "reason",
]


@dataclass(frozen=True)
class HarvestCandidate:
    result_path: Path
    source_host: str
    task_dir: Path


def harvest_batch(
    *,
    batch_dir: Path,
    experiment_roots: list[Path],
    dry_run: bool = False,
) -> dict[str, int]:
    tasks = _read_csv(batch_dir / "manifest" / "tasks.csv")
    queue_index = _read_queue_index(batch_dir / "queues")
    assigned_hosts = _read_assigned_hosts(batch_dir / "queues")
    harvest_dir = batch_dir / "harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    harvested = 0
    for task in tasks:
        row = _harvest_task(
            batch_dir=batch_dir,
            task=task,
            queue_index=queue_index,
            assigned_hosts=assigned_hosts,
            experiment_roots=experiment_roots,
            dry_run=dry_run,
        )
        rows.append(row)
        if row["status"] == "harvested":
            harvested += 1

    _write_csv(harvest_dir / "harvested_tasks.csv", rows)
    summary = {
        "total_tasks": len(tasks),
        "harvested": harvested,
        "missing": len(tasks) - harvested,
    }
    (harvest_dir / "harvest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _harvest_task(
    *,
    batch_dir: Path,
    task: dict[str, str],
    queue_index: dict[str, dict[str, str]],
    assigned_hosts: dict[str, str],
    experiment_roots: list[Path],
    dry_run: bool,
) -> dict[str, Any]:
    dataset_id = task["dataset_id"]
    seed = task["seed"]
    queue_row = queue_index.get(dataset_id)
    if queue_row is None:
        return _harvest_row(task, status="missing", reason="dataset missing from queue csv")

    global_index = int(queue_row["global_index"])
    tool_key = _scheduler_tool_key(task["algorithm"])
    noise_tag = task.get("noise_tag") or "clean"
    use_noise_dimension = _task_uses_noise_dimension(task)
    if use_noise_dimension:
        scheduler_task_id = f"{tool_key}_s{seed}_{noise_tag}_g{global_index:04d}"
    else:
        scheduler_task_id = f"{tool_key}_s{seed}_g{global_index:04d}"
    target_dir = _target_dir(batch_dir, task, use_noise_dimension=use_noise_dimension)
    candidates = _find_candidates(
        experiment_roots=experiment_roots,
        tool_key=tool_key,
        seed=seed,
        scheduler_task_id=scheduler_task_id,
    )
    assigned_host = assigned_hosts.get(scheduler_task_id)
    if assigned_host:
        candidates = [candidate for candidate in candidates if candidate.source_host == assigned_host]
    if not candidates:
        recovered_result = target_dir / "result.json"
        if _has_recovered_result(recovered_result):
            return _harvest_row(
                task,
                status="harvested",
                source_result=str(recovered_result),
                target_dir=str(target_dir),
            )
        return _harvest_row(
            task,
            status="missing",
            reason=f"result not found for {scheduler_task_id}",
        )

    candidate = _choose_candidate(candidates)
    if not dry_run:
        _copy_candidate(candidate, target_dir)
    return _harvest_row(
        task,
        status="harvested",
        source_result=str(candidate.result_path),
        target_dir=str(target_dir),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_queue_index(queue_dir: Path) -> dict[str, dict[str, str]]:
    rows: list[dict[str, str]] = []
    preferred = [queue_dir / "ssr50_source.csv", queue_dir / "smoke_2datasets_source.csv"]
    seen_paths: set[Path] = set()
    for path in preferred:
        if path.exists():
            rows.extend(_read_csv(path))
            seen_paths.add(path.resolve())
    if queue_dir.exists():
        for path in sorted(queue_dir.glob("*.csv")):
            if path.resolve() in seen_paths:
                continue
            rows.extend(_read_csv(path))

    index: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset_id = row.get("dataset_id") or row.get("dataset_name")
        global_index = row.get("global_index") or row.get("core50_index")
        if not dataset_id or not global_index:
            continue
        index.setdefault(
            dataset_id,
            {
                **row,
                "global_index": str(int(global_index)),
            },
        )
    return index


def _read_assigned_hosts(queue_dir: Path) -> dict[str, str]:
    state_dir = queue_dir / "load_queue_full" / "state"
    if not state_dir.exists():
        return {}
    assigned_hosts: dict[str, str] = {}
    for path in sorted(state_dir.glob("*.state.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict):
            continue
        for task_id, task in tasks.items():
            if not isinstance(task, dict):
                continue
            assigned_host = str(task.get("assigned_host") or "").strip()
            if assigned_host:
                assigned_hosts[str(task_id)] = assigned_host
    return assigned_hosts


def _scheduler_tool_key(algorithm: str) -> str:
    return {
        "QLattice": "qlattice",
        "iMCTS": "imcts",
    }.get(algorithm, algorithm.lower())


def _task_uses_noise_dimension(task: dict[str, str]) -> bool:
    noise_tag = task.get("noise_tag") or ""
    return bool(noise_tag and f"__{noise_tag}__" in task.get("task_id", ""))


def _target_dir(batch_dir: Path, task: dict[str, str], *, use_noise_dimension: bool) -> Path:
    seed = task["seed"]
    dataset_id = task["dataset_id"]
    if use_noise_dimension:
        noise_tag = task.get("noise_tag") or "clean"
        return batch_dir / "runs" / task["algorithm"] / f"seed{seed}" / noise_tag / dataset_id
    return batch_dir / "runs" / task["algorithm"] / f"seed{seed}" / dataset_id


def _find_candidates(
    *,
    experiment_roots: list[Path],
    tool_key: str,
    seed: str,
    scheduler_task_id: str,
) -> list[HarvestCandidate]:
    candidates: list[HarvestCandidate] = []
    for root in experiment_roots:
        source_host = root.name
        direct_task_dir = root / tool_key / f"seed{seed}" / "tasks" / scheduler_task_id
        task_dirs = [direct_task_dir] if direct_task_dir.exists() else []
        task_dirs.extend(
            path
            for path in root.glob(f"**/{tool_key}/seed{seed}/tasks/{scheduler_task_id}")
            if path not in task_dirs
        )
        for task_dir in task_dirs:
            for result_path in task_dir.glob("**/result.json"):
                candidates.append(
                    HarvestCandidate(result_path=result_path, source_host=source_host, task_dir=task_dir)
                )
    return candidates


def _choose_candidate(candidates: list[HarvestCandidate]) -> HarvestCandidate:
    return max(candidates, key=lambda candidate: candidate.result_path.stat().st_mtime)


def _copy_candidate(candidate: HarvestCandidate, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_result = target_dir / "result.json"
    if _has_recovered_result(target_result) and not _candidate_has_usable_result(candidate.result_path):
        return
    report_path = _find_launcher_report(candidate.task_dir)
    _copy_result_with_report_metadata(
        result_path=candidate.result_path,
        report_path=report_path,
        target_path=target_result,
    )
    if report_path is not None:
        shutil.copy2(report_path, target_dir / "launcher_report.json")
    progress_dir = candidate.result_path.parent / "progress"
    if progress_dir.exists():
        shutil.copytree(progress_dir, target_dir / "progress", dirs_exist_ok=True)
    source_payload = {
        "source_result": str(candidate.result_path),
        "source_run_dir": str(candidate.result_path.parent),
        "source_report": str(report_path) if report_path is not None else "",
    }
    (target_dir / "harvest_source.json").write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _has_recovered_result(path: Path) -> bool:
    payload = _read_json_object(path)
    return bool(payload and payload.get("recovered_from_24h"))


def _candidate_has_usable_result(path: Path) -> bool:
    payload = _read_json_object(path)
    if payload is None:
        return False
    status = str(payload.get("status", "")).strip().lower()
    if status == "error":
        return False
    if status == "timed_out" and not (payload.get("recovered_from_timeout") or payload.get("timeout_type") == "budget_exhausted_with_output"):
        return False
    return _has_metrics(payload) and _has_artifact(payload)


def _has_metrics(payload: dict[str, Any]) -> bool:
    for split in ("valid", "id_test", "ood_test"):
        metrics = payload.get(split)
        if not isinstance(metrics, dict):
            return False
        if metrics.get("nmse") is None:
            return False
    return True


def _has_artifact(payload: dict[str, Any]) -> bool:
    artifact = payload.get("canonical_artifact")
    if isinstance(artifact, dict) and any(value for value in artifact.values()):
        return True
    equation = payload.get("equation")
    return isinstance(equation, str) and bool(equation.strip())


def _find_launcher_report(task_dir: Path) -> Path | None:
    reports = sorted((task_dir / "__launcher__" / "logs").glob("*.report.json"))
    reports.extend(
        path
        for path in sorted(task_dir.glob("**/__launcher__/logs/*.report.json"))
        if path not in reports
    )
    if not reports:
        return None
    return max(reports, key=lambda path: path.stat().st_mtime)


def _copy_result_with_report_metadata(
    *,
    result_path: Path,
    report_path: Path | None,
    target_path: Path,
) -> None:
    result_payload = _read_json_object(result_path)
    if result_payload is None:
        shutil.copy2(result_path, target_path)
        return

    if report_path is not None:
        report_payload = _read_json_object(report_path)
        if report_payload is not None:
            _merge_launcher_report_metadata(result_payload, report_payload)

    target_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _merge_launcher_report_metadata(result: dict[str, Any], report: dict[str, Any]) -> None:
    seconds = report.get("seconds")
    if "runtime_seconds" not in result and seconds not in (None, ""):
        result["runtime_seconds"] = seconds
    if "seconds" not in result and seconds not in (None, ""):
        result["seconds"] = seconds
    for key in (
        "budget_exhausted",
        "recovered_from_timeout",
        "termination_reason",
        "timeout_type",
    ):
        if key not in result and key in report:
            result[key] = report[key]


def _harvest_row(
    task: dict[str, str],
    *,
    status: str,
    source_result: str = "",
    target_dir: str = "",
    reason: str = "",
) -> dict[str, str]:
    return {
        "task_id": task["task_id"],
        "algorithm": task["algorithm"],
        "dataset_id": task["dataset_id"],
        "seed": task["seed"],
        "noise_tag": task.get("noise_tag", "clean"),
        "noise_sigma": task.get("noise_sigma", "0"),
        "status": status,
        "source_result": source_result,
        "target_dir": target_dir,
        "reason": reason,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HARVEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
