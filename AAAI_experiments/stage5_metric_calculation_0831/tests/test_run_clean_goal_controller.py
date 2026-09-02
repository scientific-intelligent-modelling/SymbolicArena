from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_PATH = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/scripts/run_clean_goal_controller.py"
)


def _load_controller():
    spec = importlib.util.spec_from_file_location("stage5_clean_goal_controller", CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _terminal_counts(*, pending: int) -> dict[str, int]:
    return {
        "pending": pending,
        "running": 0,
        "retry_wait": 0,
        "frozen": 0,
        "non_applicable": 0,
        "exhausted": 0,
    }


def test_controller_uses_dual_api_runner_and_cpu_headroom() -> None:
    controller = _load_controller()

    assert controller.GT_PLAN.name == "clean_gt_simplify_tasks_v2.jsonl"
    assert controller.PRED_PLAN.name == "clean_pred_simplify_tasks_active_v5.jsonl"
    assert controller.PRED_INDEX.name == "clean_pred_simplify_frozen_index_active_v5.jsonl"
    assert controller.GT_INDEX.name == "clean_gt_simplify_frozen_index_v2.jsonl"
    assert controller.RUN_API_MODULE.endswith("run_anthropic_api_plan")
    assert controller.AUDIT_API_FROZEN_MODULE.endswith("audit_api_frozen_simplifications")
    assert controller.WORKERS == 64
    assert controller.SEMANTIC_WORKERS == 4
    assert controller.MAX_TOKENS == 6144
    assert controller.RESOURCE_PREFIX == [
        "nice",
        "-n",
        "5",
        "taskset",
        "-c",
        "0-3,6-9",
    ]


def test_latest_revisioned_plan_prefers_highest_active_version(tmp_path: Path) -> None:
    controller = _load_controller()
    base = tmp_path / "clean_equivalence_tasks.jsonl"
    base.write_text("base\n", encoding="utf-8")
    (tmp_path / "clean_equivalence_tasks_active_v2.jsonl").write_text(
        "v2\n", encoding="utf-8"
    )
    (tmp_path / "clean_equivalence_tasks_active_v10.jsonl").write_text(
        "v10\n", encoding="utf-8"
    )

    selected = controller._latest_revisioned_plan(base)

    assert selected.name == "clean_equivalence_tasks_active_v10.jsonl"


def test_drive_plan_batches_applies_resource_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = _load_controller()
    counts = iter([_terminal_counts(pending=1), _terminal_counts(pending=0)])
    commands: list[list[str]] = []

    monkeypatch.setattr(controller, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(controller, "_task_counts", lambda *_args, **_kwargs: next(counts))
    monkeypatch.setattr(controller, "_active_plan_pids", lambda _path: [])
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)

    def fake_run(cmd, **_kwargs):
        commands.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    controller._drive_plan_batches(
        plan_path=tmp_path / "plan.jsonl",
        task_type="equivalence",
        report_prefix="clean_equivalence_w8",
    )

    assert len(commands) == 1
    assert commands[0][: len(controller.RESOURCE_PREFIX)] == controller.RESOURCE_PREFIX
    assert controller.RUN_API_MODULE in commands[0]
    workers_index = commands[0].index("--workers")
    assert commands[0][workers_index + 1] == "64"
    channel_limit_index = commands[0].index("--per-channel-concurrency")
    assert commands[0][channel_limit_index + 1] == "32"
    max_tokens_index = commands[0].index("--max-tokens")
    assert commands[0][max_tokens_index + 1] == "6144"


def test_pred_frozen_audit_uses_api_auditor_and_combined_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    verified: list[bool] = []
    monkeypatch.setattr(
        controller,
        "PRED_COMBINED_AUDIT_REPORT",
        tmp_path / "missing-combined.json",
    )
    monkeypatch.setattr(controller, "_run", lambda cmd, **_kwargs: commands.append(list(cmd)))
    monkeypatch.setattr(controller, "_verify_mixed_pred_audit", lambda: verified.append(True))

    controller._audit_pred_frozen()

    assert len(commands) == 1
    assert controller.AUDIT_API_FROZEN_MODULE in commands[0]
    assert commands[0][commands[0].index("--expected-api-count") + 1] == "22"
    assert verified == [True]


def test_pred_frozen_audit_reuses_valid_combined_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = _load_controller()
    combined_report = tmp_path / "combined.json"
    combined_report.write_text("{}\n", encoding="utf-8")
    verified: list[bool] = []
    monkeypatch.setattr(controller, "PRED_COMBINED_AUDIT_REPORT", combined_report)
    monkeypatch.setattr(controller, "_verify_mixed_pred_audit", lambda: verified.append(True))
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应重跑审计")),
    )

    controller._audit_pred_frozen()

    assert verified == [True]


def test_materialize_recovered_passes_state_db_once(monkeypatch) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    monkeypatch.setattr(controller, "_run", lambda cmd, **_kwargs: commands.append(list(cmd)))

    controller._materialize_recovered_attempts(controller.PRED_PLAN, controller.PRED_RECOVERED_REPORT)

    state_flag = commands[0].index("--state-db")
    assert commands[0][state_flag + 1] == controller._rel(controller.STATE_DB)
    assert commands[0].count(controller._rel(controller.STATE_DB)) == 1


def test_build_pred_index_declares_exhausted_state_contract(monkeypatch) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    monkeypatch.setattr(controller, "_run", lambda cmd, **_kwargs: commands.append(list(cmd)))
    monkeypatch.setattr(
        controller,
        "_read_json",
        lambda _path: {"state_counts": {"frozen": 2250, "non_applicable": 0, "exhausted": 0}},
    )

    controller._build_pred_index()

    assert len(commands) == 1
    assert "--allow-exhausted" in commands[0]
    attempts_flag = commands[0].index("--attempts-dir")
    assert commands[0][attempts_flag + 1] == controller._rel(controller.API_ATTEMPTS_DIR)
