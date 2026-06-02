from pathlib import Path
import json
import subprocess
from types import SimpleNamespace

import pytest

from check import run_e1_candidate200_12alg_load_queue as scheduler


def test_tool_config_supports_current_15_toolbox_algorithms():
    expected_tools = {
        "qlattice",
        "drsr",
        "dso",
        "e2esr",
        "fepysr",
        "gplearn",
        "imcts",
        "jaxsr",
        "llmsr",
        "pyoperon",
        "pysr",
        "ragsr",
        "symbolfit",
        "tpsr",
        "udsr",
    }

    assert set(scheduler.TOOL_CONFIG) == expected_tools
    assert scheduler.TOOL_CONFIG["fepysr"]["tool_arg"] == "fepysr"
    assert scheduler.TOOL_CONFIG["jaxsr"]["env"] == "sim_jaxsr"
    assert scheduler.TOOL_CONFIG["symbolfit"]["params"] == "symbolfit"
    

def test_preflight_script_import_checks_include_new_algorithm_envs(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "QUEUE_ROOT", tmp_path / "queue")

    script = scheduler._write_preflight_script()
    content = script.read_text(encoding="utf-8")

    assert "candidate200_unified.csv" not in content
    assert '"sim_fepysr"' in content
    assert "scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper" in content
    assert '"sim_jaxsr"' in content
    assert "scientific_intelligent_modelling.algorithms.jaxsr_wrapper.wrapper" in content
    assert '"sim_symbolfit"' in content
    assert "scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper" in content


def test_preflight_local_files_use_requested_source_csv(tmp_path):
    source_csv = scheduler.REPO_ROOT / "benchmark-runs" / "compliance" / "latest" / "queues" / "smoke_2datasets_source.csv"

    local_files = scheduler._preflight_local_files(source_csv)

    assert local_files["source_csv"] == "benchmark-runs/compliance/latest/queues/smoke_2datasets_source.csv"
    assert "candidate200" not in local_files


def test_preflight_local_files_include_all_algorithm_wrappers(tmp_path):
    source_csv = scheduler.REPO_ROOT / "benchmark-runs" / "compliance" / "latest" / "queues" / "smoke_2datasets_source.csv"

    local_files = scheduler._preflight_local_files(source_csv)

    expected_wrapper_paths = {
        "qlattice_wrapper": "scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py",
        "drsr_wrapper": "scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py",
        "dso_wrapper": "scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py",
        "e2esr_wrapper": "scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py",
        "fepysr_wrapper": "scientific_intelligent_modelling/algorithms/fepysr_wrapper/wrapper.py",
        "gplearn_wrapper": "scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py",
        "imcts_wrapper": "scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py",
        "jaxsr_wrapper": "scientific_intelligent_modelling/algorithms/jaxsr_wrapper/wrapper.py",
        "llmsr_wrapper": "scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py",
        "pyoperon_wrapper": "scientific_intelligent_modelling/algorithms/pyoperon_wrapper/wrapper.py",
        "pysr_wrapper": "scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py",
        "ragsr_wrapper": "scientific_intelligent_modelling/algorithms/ragsr_wrapper/wrapper.py",
        "symbolfit_wrapper": "scientific_intelligent_modelling/algorithms/symbolfit_wrapper/wrapper.py",
        "tpsr_wrapper": "scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py",
        "udsr_wrapper": "scientific_intelligent_modelling/algorithms/udsr_wrapper/wrapper.py",
    }
    for label, rel_path in expected_wrapper_paths.items():
        assert local_files[label] == rel_path


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
        "seed_dispatch_mode": "mixed",
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


def test_build_tasks_includes_noise_dimension_in_task_ids(tmp_path: Path) -> None:
    rows = [{"global_index": "1", "dataset_dir": "sim-datasets-data/ssr50/d0", "dataset_name": "d0"}]
    tasks = scheduler._build_tasks(
        rows,
        tools=["pysr"],
        seeds=[520, 521],
        noise_sigmas=[0.0, 0.01],
        queue_root=tmp_path / "queue",
        params_root=tmp_path / "params",
    )

    assert [task.task_id for task in tasks] == [
        "pysr_s520_clean_g0001",
        "pysr_s520_noise001_g0001",
        "pysr_s521_clean_g0001",
        "pysr_s521_noise001_g0001",
    ]
    assert [task.params_name for task in tasks] == [
        "pysr__clean",
        "pysr__noise001",
        "pysr__clean",
        "pysr__noise001",
    ]
    assert [task.noise_tag for task in tasks] == ["clean", "noise001", "clean", "noise001"]


def test_preflight_uses_requested_source_csv(tmp_path, monkeypatch):
    source_csv = tmp_path / "smoke.csv"
    source_csv.write_text(
        "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel\n"
        f"1,d1,d1,{tmp_path / 'd1'},{tmp_path / 'd1'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scheduler, "SOURCE_CSV", tmp_path / "missing_candidate200.csv")
    monkeypatch.setattr(scheduler, "_preflight_local_files", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(scheduler, "_local_git_head", lambda: "head")
    monkeypatch.setattr(scheduler, "_local_file_hashes", lambda *_args, **_kwargs: {})
    scp_sources = []

    def _fake_scp(local_path, *args, **kwargs):
        scp_sources.append(Path(local_path))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(scheduler, "_scp", _fake_scp)
    monkeypatch.setattr(
        scheduler,
        "_ssh",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            json.dumps({"host": "iaaccn22", "ok": True}),
            "",
        ),
    )

    args = SimpleNamespace(
        source_csv_path=source_csv,
        expected_rows_value=1,
        queue_root_path=tmp_path / "queue",
        preflight_report=tmp_path / "preflight.json",
        tools=["gplearn"],
        hosts=["iaaccn22"],
        controller_host="iaaccn22",
        use_internal_ips=True,
        preflight_host_timeout=5,
    )

    summary = scheduler._run_preflight(args)

    assert summary["hosts"] == [{"host": "iaaccn22", "ok": True}]
    assert scp_sources[0].parent == args.queue_root_path / "preflight"
    assert scp_sources[1].parent == args.queue_root_path / "preflight"


def test_preflight_dataset_resolution_prefers_home_data_root_for_sim_datasets(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    home_root = tmp_path / "home"
    repo_dataset = repo_root / "sim-datasets-data" / "ssr50" / "datasets" / "demo"
    home_dataset = home_root / "sim-datasets-data" / "ssr50" / "datasets" / "demo"
    repo_dataset.mkdir(parents=True)
    home_dataset.mkdir(parents=True)
    monkeypatch.setattr(scheduler, "REPO_ROOT", repo_root)
    monkeypatch.setattr(Path, "home", lambda: home_root)

    resolved = scheduler._resolve_local_dataset_dir(
        {"dataset_rel": "sim-datasets-data/ssr50/datasets/demo"}
    )

    assert resolved == home_dataset


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


def test_list_queue_sessions_does_not_mask_tmux_ls_timeout(monkeypatch):
    def fake_ssh(_host, command, **_kwargs):
        if "|| true" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 124, "", "timeout")

    monkeypatch.setattr(scheduler, "_ssh", fake_ssh)

    sessions = scheduler._list_queue_sessions(
        "iaaccn25",
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="formal24h_full_",
    )

    assert sessions is None


def test_update_running_tasks_keeps_running_when_tmux_ls_unavailable(tmp_path, monkeypatch):
    state = {
        "tasks": {
            "gplearn_s520_clean_g0004": {
                "state": "running",
                "assigned_host": "iaaccn25",
                "session": "formal24h_full_gplearn_s520_clean_g0004",
                "expected": 1,
            }
        }
    }
    args = SimpleNamespace(
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="formal24h_full_",
        batch_name="formal24h",
        queue_root_path=tmp_path / "queue",
        retry_limit=3,
        remote_root_path=tmp_path / "remote",
        host_remote_root_overrides_parsed={},
    )

    monkeypatch.setattr(scheduler, "_list_queue_sessions", lambda *args, **kwargs: None)

    def fail_if_session_running_called(*_args, **_kwargs):
        raise AssertionError("host-level tmux ls failure should not trigger per-task SSH fallback")

    monkeypatch.setattr(scheduler, "_session_running", fail_if_session_running_called)

    def fail_if_status_read(*_args, **_kwargs):
        raise AssertionError("running tmux session should not be treated as finished")

    monkeypatch.setattr(scheduler, "_read_task_statuses_bulk", fail_if_status_read)

    scheduler._update_running_tasks(state, args)

    assert state["tasks"]["gplearn_s520_clean_g0004"]["state"] == "running"


def test_update_running_tasks_requeues_unavailable_host_after_budget_grace(tmp_path, monkeypatch):
    params_root = tmp_path / "params"
    params_root.mkdir()
    (params_root / "gplearn__clean.json").write_text(
        json.dumps({"timeout_in_seconds": 3600}),
        encoding="utf-8",
    )
    state = {
        "tasks": {
            "gplearn_s520_clean_g0004": {
                "state": "running",
                "assigned_host": "iaaccn29",
                "session": "formal24h_full_gplearn_s520_clean_g0004",
                "expected": 1,
                "started_at": "2026-06-01T00:00:00",
                "host_unavailable_since": "2026-06-01T01:05:00",
                "params_name": "gplearn__clean",
                "attempts": 1,
            }
        }
    }
    args = SimpleNamespace(
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="formal24h_full_",
        batch_name="formal24h",
        queue_root_path=tmp_path / "queue",
        retry_limit=3,
        remote_root_path=tmp_path / "remote",
        host_remote_root_overrides_parsed={},
        params_root_path=params_root,
        host_unavailable_grace_seconds=600,
    )

    monkeypatch.setattr(scheduler, "_now", lambda: "2026-06-01T01:20:00")
    monkeypatch.setattr(scheduler, "_list_queue_sessions", lambda *args, **kwargs: None)

    def fail_if_status_read(*_args, **_kwargs):
        raise AssertionError("unavailable host should be requeued without reading task status")

    monkeypatch.setattr(scheduler, "_read_task_statuses_bulk", fail_if_status_read)

    scheduler._update_running_tasks(state, args)

    task = state["tasks"]["gplearn_s520_clean_g0004"]
    assert task["state"] == "pending"
    assert task["assigned_host"] is None
    assert task["session"] is None
    assert task["started_at"] is None
    assert task["ended_at"] is None
    assert task["error"] is None
    assert "host unavailable" in task["last_requeue_reason"]
    assert task["last_requeued_at"] == "2026-06-01T01:20:00"
    assert task["last_unavailable_host"] == "iaaccn29"


def test_update_running_tasks_rechecks_session_when_tmux_ls_omits_live_session(tmp_path, monkeypatch):
    state = {
        "tasks": {
            "gplearn_s520_clean_g0004": {
                "state": "running",
                "assigned_host": "iaaccn25",
                "session": "formal24h_full_gplearn_s520_clean_g0004",
                "expected": 1,
            }
        }
    }
    args = SimpleNamespace(
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="formal24h_full_",
        batch_name="formal24h",
        queue_root_path=tmp_path / "queue",
        retry_limit=3,
        remote_root_path=tmp_path / "remote",
        host_remote_root_overrides_parsed={},
    )

    monkeypatch.setattr(scheduler, "_list_queue_sessions", lambda *args, **kwargs: set())
    monkeypatch.setattr(scheduler, "_session_running", lambda *args, **kwargs: True)

    def fail_if_status_read(*_args, **_kwargs):
        raise AssertionError("live tmux session omitted from tmux ls should not be treated as finished")

    monkeypatch.setattr(scheduler, "_read_task_statuses_bulk", fail_if_status_read)

    scheduler._update_running_tasks(state, args)

    assert state["tasks"]["gplearn_s520_clean_g0004"]["state"] == "running"


def test_update_running_tasks_marks_done_when_session_absent_after_precise_recheck(tmp_path, monkeypatch):
    state = {
        "tasks": {
            "gplearn_s520_clean_g0004": {
                "state": "running",
                "assigned_host": "iaaccn25",
                "session": "formal24h_full_gplearn_s520_clean_g0004",
                "expected": 1,
            }
        }
    }
    args = SimpleNamespace(
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="formal24h_full_",
        batch_name="formal24h",
        queue_root_path=tmp_path / "queue",
        retry_limit=3,
        remote_root_path=tmp_path / "remote",
        host_remote_root_overrides_parsed={},
    )

    monkeypatch.setattr(scheduler, "_list_queue_sessions", lambda *args, **kwargs: set())
    monkeypatch.setattr(scheduler, "_session_running", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scheduler,
        "_read_task_statuses_bulk",
        lambda *args, **kwargs: {
            "gplearn_s520_clean_g0004": {
                "read_error": None,
                "seen": 1,
                "done": 1,
                "errors": 0,
                "counts": {"ok": 1},
            }
        },
    )

    scheduler._update_running_tasks(state, args)

    assert state["tasks"]["gplearn_s520_clean_g0004"]["state"] == "done"


def test_update_running_tasks_keeps_running_when_finished_status_read_fails(tmp_path, monkeypatch):
    state = {
        "tasks": {
            "gplearn_s520_clean_g0004": {
                "state": "running",
                "assigned_host": "iaaccn25",
                "session": "formal24h_full_gplearn_s520_clean_g0004",
                "expected": 1,
                "started_at": "2026-06-01T00:00:00",
                "attempts": 1,
            }
        }
    }
    args = SimpleNamespace(
        controller_host="iaaccn22",
        use_internal_ips=True,
        session_prefix="formal24h_full_",
        batch_name="formal24h",
        queue_root_path=tmp_path / "queue",
        retry_limit=3,
        remote_root_path=tmp_path / "remote",
        host_remote_root_overrides_parsed={},
    )

    monkeypatch.setattr(scheduler, "_now", lambda: "2026-06-01T01:00:00")
    monkeypatch.setattr(scheduler, "_list_queue_sessions", lambda *args, **kwargs: set())
    monkeypatch.setattr(scheduler, "_session_running", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scheduler,
        "_read_task_statuses_bulk",
        lambda *args, **kwargs: {
            "gplearn_s520_clean_g0004": {
                "read_error": "transient ssh error",
                "seen": 0,
                "done": 0,
                "errors": 0,
                "counts": {},
            }
        },
    )

    scheduler._update_running_tasks(state, args)

    task = state["tasks"]["gplearn_s520_clean_g0004"]
    assert task["state"] == "running"
    assert task["assigned_host"] == "iaaccn25"
    assert task["session"] == "formal24h_full_gplearn_s520_clean_g0004"
    assert task["started_at"] == "2026-06-01T00:00:00"
    assert task["last_status_read_error"] == "transient ssh error"
    assert task["last_status_read_error_at"] == "2026-06-01T01:00:00"
    assert task["status_read_error_since"] == "2026-06-01T01:00:00"
    assert "last_requeue_reason" not in task
