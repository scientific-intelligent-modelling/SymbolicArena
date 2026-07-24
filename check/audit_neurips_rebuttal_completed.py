#!/usr/bin/env python3
"""校验指定主机上已完成 rebuttal 任务的结果与预算契约。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _outer_result_paths(
    experiment_root: Path,
    *,
    tool: str,
    seed: int,
    task_id: str,
) -> list[Path]:
    task_root = (
        experiment_root
        / tool
        / f"seed{seed}"
        / "tasks"
        / task_id
    )
    return sorted(
        path
        for path in task_root.glob("**/result.json")
        if "experiments" not in path.relative_to(task_root).parts
    )


def _result_checks(
    payload: dict[str, Any],
    *,
    min_runtime: float,
) -> tuple[dict[str, bool], float | None]:
    runtime = payload.get(
        "runtime_seconds",
        payload.get("seconds"),
    )
    runtime_number = (
        float(runtime) if _finite_nonnegative(runtime) else None
    )
    artifact = payload.get("canonical_artifact")
    identity = payload.get("dataset_identity_check")
    id_test = payload.get("id_test")
    ood_test = payload.get("ood_test")
    checks = {
        "status_ok": str(payload.get("status") or "") == "ok",
        "runtime_compliant": (
            runtime_number is not None
            and runtime_number >= min_runtime
        ),
        "identity_match": (
            isinstance(identity, dict)
            and identity.get("match") is True
        ),
        "id_nmse_valid": (
            isinstance(id_test, dict)
            and _finite_nonnegative(id_test.get("nmse"))
        ),
        "ood_nmse_valid": (
            isinstance(ood_test, dict)
            and _finite_nonnegative(ood_test.get("nmse"))
        ),
        "artifact_valid": (
            isinstance(artifact, dict)
            and artifact.get("artifact_valid") is True
        ),
        "has_equation": bool(payload.get("equation")),
    }
    return checks, runtime_number


def audit_completed(
    *,
    state: dict[str, Any],
    experiment_root: Path,
    host: str,
    min_runtime: float,
) -> dict[str, Any]:
    """返回单主机已完成任务的可聚合审计报告。"""
    tasks_block = state.get("tasks")
    if not isinstance(tasks_block, dict):
        raise ValueError("state 缺少 tasks 对象")
    done = [
        task
        for task in tasks_block.values()
        if isinstance(task, dict)
        and task.get("state") == "done"
        and task.get("assigned_host") == host
    ]
    issues: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    runtime_values: list[float] = []
    result_paths: list[str] = []

    for task in done:
        task_id = str(task.get("task_id") or "")
        tool = str(task.get("tool") or "")
        try:
            seed = int(task["seed"])
        except (KeyError, TypeError, ValueError):
            issues.append(
                {
                    "task_id": task_id,
                    "issue": "invalid_task_identity",
                }
            )
            continue
        candidates = _outer_result_paths(
            experiment_root,
            tool=tool,
            seed=seed,
            task_id=task_id,
        )
        if len(candidates) != 1:
            issues.append(
                {
                    "task_id": task_id,
                    "issue": "result_path_count",
                    "count": len(candidates),
                }
            )
            continue
        result_path = candidates[0]
        try:
            payload = json.loads(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "task_id": task_id,
                    "issue": "result_unreadable",
                    "error": repr(exc),
                }
            )
            continue
        if not isinstance(payload, dict):
            issues.append(
                {
                    "task_id": task_id,
                    "issue": "result_not_object",
                }
            )
            continue

        result_paths.append(str(result_path))
        status = str(payload.get("status") or "")
        status_counts[status] += 1
        algorithm_counts[tool] += 1
        checks, runtime = _result_checks(
            payload,
            min_runtime=min_runtime,
        )
        if runtime is not None:
            runtime_values.append(runtime)
        failed_checks = [
            name for name, passed in checks.items() if not passed
        ]
        if failed_checks:
            issues.append(
                {
                    "task_id": task_id,
                    "issue": "contract_failed",
                    "result_path": str(result_path),
                    "failed_checks": failed_checks,
                    "status": status,
                    "runtime": runtime,
                }
            )

    validated_results = sum(algorithm_counts.values())
    return {
        "host": host,
        "done_tasks": len(done),
        "validated_results": validated_results,
        "algorithm_counts": dict(sorted(algorithm_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "runtime_min": min(runtime_values) if runtime_values else None,
        "runtime_max": max(runtime_values) if runtime_values else None,
        "result_paths": result_paths,
        "issues": issues,
        "passed": not issues and validated_results == len(done),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--min-runtime", type=float, default=3300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    report = audit_completed(
        state=state,
        experiment_root=args.experiment_root,
        host=args.host,
        min_runtime=args.min_runtime,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
