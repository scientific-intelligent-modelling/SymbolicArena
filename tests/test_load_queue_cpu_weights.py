from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_queue_module():
    path = Path("check/run_e1_candidate200_12alg_load_queue.py")
    spec = importlib.util.spec_from_file_location("load_queue_cpu_weights", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scheduler_args(*, max_cpu_used_ratio=None):
    return SimpleNamespace(
        max_jobs_per_host=100,
        max_load_ratio=0.80,
        max_memory_used_ratio=0.80,
        min_free_mem_gb=0.0,
        max_cpu_used_ratio=max_cpu_used_ratio,
        default_max_running_per_tool=0,
        llm_model_bucket_limits_parsed={},
        seed_dispatch_mode="mixed",
        seeds=[520],
        prioritize_llm=False,
        round_robin_tools=True,
        tools=["qlattice", "gplearn"],
    )


def test_tool_config_declares_positive_cpu_weights() -> None:
    queue = _load_queue_module()

    assert set(queue.TOOL_CONFIG) == set(queue.DEFAULT_TOOLS)
    assert all(int(config["cpu_weight"]) > 0 for config in queue.TOOL_CONFIG.values())
    assert queue.TOOL_CONFIG["qlattice"]["cpu_weight"] == 4
    assert queue.TOOL_CONFIG["dso"]["cpu_weight"] == 4
    assert queue.TOOL_CONFIG["symbolfit"]["cpu_weight"] == 1


def test_host_probe_collects_active_session_tools_and_cpu_weight(monkeypatch) -> None:
    queue = _load_queue_module()
    observed_commands: list[str] = []

    def fake_ssh(host, command, **kwargs):
        observed_commands.append(command)
        return subprocess.CompletedProcess(
            ["ssh", host],
            0,
            json.dumps(
                {
                    "load1": 4.0,
                    "load5": 4.0,
                    "load15": 4.0,
                    "cpu_count": 256,
                    "load_ratio": 4.0 / 256.0,
                    "mem_total_gb": 512.0,
                    "mem_available_gb": 500.0,
                    "mem_used_ratio": 0.02,
                    "active_sessions": [
                        {
                            "session": "all_qlattice_s520_clean_g0001",
                            "task_id": "qlattice_s520_clean_g0001",
                            "tool": "qlattice",
                            "cpu_weight": 4,
                        },
                        {
                            "session": "all_gplearn_s520_clean_g0002",
                            "task_id": "gplearn_s520_clean_g0002",
                            "tool": "gplearn",
                            "cpu_weight": 1,
                        },
                    ],
                    "queue_sessions": 2,
                    "active_cpu_weight": 5,
                }
            )
            + "\n",
            "",
        )

    monkeypatch.setattr(queue, "_ssh", fake_ssh)
    state = queue._probe_host(
        "iaaccn23",
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="current_",
        host_session_count_prefix="all_",
    )

    assert "active_sessions" in observed_commands[0]
    assert state["queue_sessions"] == 2
    assert state["active_cpu_weight"] == 5
    assert [item["tool"] for item in state["active_sessions"]] == ["qlattice", "gplearn"]


def test_cpu_budget_admission_blocks_a_full_host_but_old_mode_is_unchanged() -> None:
    queue = _load_queue_module()
    full_host = {
        "ok": True,
        "queue_sessions": 10,
        "load_ratio": 0.1,
        "mem_used_ratio": 0.1,
        "mem_available_gb": 100.0,
        "cpu_count": 256,
        "active_cpu_weight": 192,
    }

    allowed, reason = queue._host_can_accept(full_host, _scheduler_args(max_cpu_used_ratio=0.75))
    assert allowed is False
    assert "cpu" in reason

    legacy_allowed, legacy_reason = queue._host_can_accept(
        {**full_host, "active_cpu_weight": 999},
        _scheduler_args(max_cpu_used_ratio=None),
    )
    assert legacy_allowed is True
    assert legacy_reason == "ok"


def test_pending_selection_accumulates_cpu_weight_within_one_dispatch_round() -> None:
    queue = _load_queue_module()
    args = _scheduler_args(max_cpu_used_ratio=0.75)
    state = {
        "tasks": {
            "qlattice_s520_clean_g0001": {
                "task_id": "qlattice_s520_clean_g0001",
                "tool": "qlattice",
                "seed": 520,
                "state": "pending",
                "cpu_weight": 4,
                "llm_model_bucket": None,
            },
            "gplearn_s520_clean_g0002": {
                "task_id": "gplearn_s520_clean_g0002",
                "tool": "gplearn",
                "seed": 520,
                "state": "pending",
                "cpu_weight": 1,
                "llm_model_bucket": None,
            },
        },
        "round_robin_cursor": 0,
    }

    first = queue._next_pending_task_id(state, args, max_cpu_weight=4)
    assert first == "qlattice_s520_clean_g0001"
    state["tasks"][first]["state"] = "dispatching"

    # The first task consumes the whole four-core allowance for this round.
    second = queue._next_pending_task_id(state, args, max_cpu_weight=0)
    assert second is None


def test_initial_state_records_cpu_weight() -> None:
    queue = _load_queue_module()
    task = queue.QueueTask(
        task_id="qlattice_s520_clean_g0001",
        tool="qlattice",
        seed=520,
        noise_tag="clean",
        noise_sigma=0.0,
        task_index=1,
        rows=[{"global_index": "1", "dataset_dir": "sim-datasets-data/ssr50/x"}],
        slice_path=Path("check/queue_slice.csv"),
        params_name="qlattice",
    )

    state = queue._initial_state("cpu_test", [task])
    assert state["tasks"][task.task_id]["cpu_weight"] == 4


def test_submit_logs_are_not_written_to_tmp() -> None:
    queue = _load_queue_module()
    source = inspect.getsource(queue._task_submit_line)

    assert "Path(\"/tmp\")" not in source
    assert "queue_root_path" in source
