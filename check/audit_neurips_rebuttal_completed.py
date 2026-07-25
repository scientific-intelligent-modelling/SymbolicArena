#!/usr/bin/env python3
"""校验指定主机上已完成 rebuttal 任务的结果与预算契约。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


CONSUMED_CONTROL_PARAMS = {
    "progress_snapshot_interval_seconds",
    "train_label_noise_enabled",
    "train_label_noise_sigma",
}
DYNAMIC_DATASET_PARAMS = {
    "exp_name",
    "exp_path",
    "feature_names",
    "n_features",
    "target_name",
}


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _finite_zero(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number == 0


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    return None


def _algorithm_params_match(
    actual: Any,
    expected: dict[str, Any] | None,
) -> bool:
    if expected is None:
        return True
    if not isinstance(actual, dict):
        return False
    expected_static = {
        key: value
        for key, value in expected.items()
        if key not in CONSUMED_CONTROL_PARAMS
    }
    actual_static = {
        key: value
        for key, value in actual.items()
        if key not in DYNAMIC_DATASET_PARAMS
    }
    return actual_static == expected_static


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
    expected_params: dict[str, Any] | None = None,
    expected_tool: str | None = None,
    expected_seed: int | None = None,
    expected_task_index: int | None = None,
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
    train_label_noise = payload.get("train_label_noise")
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
        "tool_match": (
            expected_tool is None
            or payload.get("tool") == expected_tool
        ),
        "seed_match": (
            expected_seed is None
            or _integer_value(payload.get("seed")) == expected_seed
        ),
        "task_global_index_match": (
            expected_task_index is None
            or _integer_value(payload.get("task_global_index"))
            == expected_task_index
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
        "train_label_noise_clean": (
            isinstance(train_label_noise, dict)
            and train_label_noise.get("enabled") is False
            and train_label_noise.get("requested") is False
            and _finite_zero(train_label_noise.get("sigma"))
            and _finite_zero(train_label_noise.get("scale"))
        ),
        "algorithm_params_match": _algorithm_params_match(
            payload.get("params"),
            expected_params,
        ),
    }
    return checks, runtime_number


def audit_completed(
    *,
    state: dict[str, Any],
    experiment_root: Path,
    host: str,
    min_runtime: float,
    expected_params_by_tool: dict[str, dict[str, Any]] | None = None,
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
        task_index = _integer_value(task.get("task_index"))
        if not tool or task_index is None:
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
            expected_params=(
                None
                if expected_params_by_tool is None
                else expected_params_by_tool.get(tool, {})
            ),
            expected_tool=tool,
            expected_seed=seed,
            expected_task_index=task_index,
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
    parser.add_argument("--params-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    tasks_block = state.get("tasks")
    if not isinstance(tasks_block, dict):
        raise ValueError("state 缺少 tasks 对象")
    tools = sorted(
        {
            str(task.get("tool") or "")
            for task in tasks_block.values()
            if isinstance(task, dict) and task.get("tool")
        }
    )
    expected_params_by_tool: dict[str, dict[str, Any]] = {}
    for tool in tools:
        params_path = args.params_root / f"{tool}__clean.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise ValueError(f"参数文件不是 JSON 对象: {params_path}")
        expected_params_by_tool[tool] = params
    report = audit_completed(
        state=state,
        experiment_root=args.experiment_root,
        host=args.host,
        min_runtime=args.min_runtime,
        expected_params_by_tool=expected_params_by_tool,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
