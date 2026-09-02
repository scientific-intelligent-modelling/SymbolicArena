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


def test_controller_uses_active_v2_inputs_and_cpu_headroom() -> None:
    controller = _load_controller()

    assert controller.GT_PLAN.name == "clean_gt_simplify_tasks_v2.jsonl"
    assert controller.PRED_PLAN.name == "clean_pred_simplify_tasks_active_v2.jsonl"
    assert controller.GT_INDEX.name == "clean_gt_simplify_frozen_index_v2.jsonl"
    assert controller.WORKERS == 8
    assert controller.CLAUDE_RESOURCE_PREFIX == [
        "nice",
        "-n",
        "5",
        "taskset",
        "-c",
        "0-3,6-9",
    ]


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
    assert commands[0][: len(controller.CLAUDE_RESOURCE_PREFIX)] == controller.CLAUDE_RESOURCE_PREFIX
    workers_index = commands[0].index("--workers")
    assert commands[0][workers_index + 1] == "8"


def test_materialize_recovered_passes_state_db_once(monkeypatch) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    monkeypatch.setattr(controller, "_run", lambda cmd, **_kwargs: commands.append(list(cmd)))

    controller._materialize_recovered_attempts(controller.PRED_PLAN, controller.PRED_RECOVERED_REPORT)

    state_flag = commands[0].index("--state-db")
    assert commands[0][state_flag + 1] == controller._rel(controller.STATE_DB)
    assert commands[0].count(controller._rel(controller.STATE_DB)) == 1
