from pathlib import Path
from types import SimpleNamespace

import pytest

from check import run_e1_candidate200_12alg_load_queue as scheduler


def _write_params(root: Path, *names: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / f"{name}.json").write_text("{}", encoding="utf-8")


def _scheduler_args(**overrides):
    base = {
        "tools": ["llmsr", "drsr", "gplearn"],
        "round_robin_tools": True,
        "prioritize_llm": True,
        "default_max_running_per_tool": 0,
        "llm_model_bucket_limits_parsed": {"base": 1, "turbo": 1},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_tasks_stable_half_uses_llm_bucket_params(tmp_path):
    params_root = tmp_path / "params"
    _write_params(
        params_root,
        "llmsr_base",
        "llmsr_turbo",
        "drsr_base",
        "drsr_turbo",
        "gplearn",
    )
    rows = [
        {
            "global_index": str(i),
            "dataset_dir": f"sim-datasets-data/demo/dataset_{i}",
            "dataset_name": f"dataset_{i}",
        }
        for i in range(1, 7)
    ]

    tasks = scheduler._build_tasks(
        rows,
        tools=["llmsr", "drsr", "gplearn"],
        seeds=[0],
        queue_root=tmp_path / "queue",
        params_root=params_root,
        llm_model_assignment="stable-half",
        llm_model_buckets=["base", "turbo"],
    )

    llm_tasks = [task for task in tasks if task.tool in {"llmsr", "drsr"}]
    assert {task.llm_model_bucket for task in llm_tasks} <= {"base", "turbo"}
    assert {task.params_name for task in llm_tasks} <= {
        "llmsr_base",
        "llmsr_turbo",
        "drsr_base",
        "drsr_turbo",
    }
    assert all(task.params_name == "gplearn" and task.llm_model_bucket is None for task in tasks if task.tool == "gplearn")


def test_build_tasks_stable_half_requires_variant_params(tmp_path):
    params_root = tmp_path / "params"
    _write_params(params_root, "llmsr_base")
    rows = [{"global_index": "1", "dataset_dir": "sim-datasets-data/demo/dataset", "dataset_name": "dataset"}]

    with pytest.raises(FileNotFoundError):
        scheduler._build_tasks(
            rows,
            tools=["llmsr"],
            seeds=[0],
            queue_root=tmp_path / "queue",
            params_root=params_root,
            llm_model_assignment="stable-half",
            llm_model_buckets=["base", "turbo"],
        )


def test_next_pending_prioritizes_llm_then_falls_back_when_buckets_full():
    state = {
        "tasks": {
            "running_base": {"state": "running", "tool": "llmsr", "llm_model_bucket": "base"},
            "running_turbo": {"state": "running", "tool": "drsr", "llm_model_bucket": "turbo"},
            "pending_base": {"state": "pending", "tool": "llmsr", "llm_model_bucket": "base"},
            "pending_turbo": {"state": "pending", "tool": "drsr", "llm_model_bucket": "turbo"},
            "pending_non_llm": {"state": "pending", "tool": "gplearn", "llm_model_bucket": None},
        },
        "round_robin_cursor": 0,
        "llm_round_robin_cursor": 0,
    }

    picked = scheduler._next_pending_task_id(state, _scheduler_args())

    assert picked == "pending_non_llm"


def test_next_pending_uses_available_llm_bucket_before_non_llm():
    state = {
        "tasks": {
            "running_base": {"state": "running", "tool": "llmsr", "llm_model_bucket": "base"},
            "pending_base": {"state": "pending", "tool": "llmsr", "llm_model_bucket": "base"},
            "pending_turbo": {"state": "pending", "tool": "drsr", "llm_model_bucket": "turbo"},
            "pending_non_llm": {"state": "pending", "tool": "gplearn", "llm_model_bucket": None},
        },
        "round_robin_cursor": 0,
        "llm_round_robin_cursor": 0,
    }

    picked = scheduler._next_pending_task_id(state, _scheduler_args())

    assert picked == "pending_turbo"
