#!/usr/bin/env python3
"""串行驱动 Stage5 clean 六轴正式计算闭环。

说明：
1. 该控制器不会重复启动已在运行中的 pred simplify batch，而是等待其收口；
2. pred simplify 收口后，会自动执行 exhausted 审计/补冻、symbolic 任务构建、
   equivalence 与 structure 批处理、evidence 物化，以及 clean 聚合；
3. 该脚本可重复执行，尽量保持幂等；已完成阶段会被复用。
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
REPORTS_DIR = STAGE_ROOT / "reports"
RESULTS_DIR = STAGE_ROOT / "results"
STATE_DB = STAGE_ROOT / "llm/control/state_v2.sqlite3"
ATTEMPTS_DIR = STAGE_ROOT / "llm/attempts_v2"
FROZEN_DIR = STAGE_ROOT / "llm/frozen_v2"
PREDECESSOR_MANIFEST = STAGE_ROOT / "manifests/predecessor_attempts_v1.json"

GT_PLAN = REPORTS_DIR / "clean_gt_simplify_tasks_v2.jsonl"
PRED_PLAN = REPORTS_DIR / "clean_pred_simplify_tasks_active_v3.jsonl"
GT_INDEX = RESULTS_DIR / "clean_gt_simplify_frozen_index_v2.jsonl"
GT_SUMMARY = REPORTS_DIR / "clean_gt_simplify_frozen_index_v2_summary.json"
PRED_INDEX = RESULTS_DIR / "clean_pred_simplify_frozen_index_active_v3.jsonl"
PRED_SUMMARY = REPORTS_DIR / "clean_pred_simplify_frozen_index_active_v3_summary.json"
EQ_PLAN = REPORTS_DIR / "clean_equivalence_tasks.jsonl"
EQ_NON_APPLICABLE = REPORTS_DIR / "clean_equivalence_non_applicable.jsonl"
EQ_FULL_PLAN = REPORTS_DIR / "clean_equivalence_full_plan.jsonl"
EQ_REGISTER_REPORT = REPORTS_DIR / "clean_equivalence_register_report.json"
EQ_INDEX = REPORTS_DIR / "clean_equivalence_frozen_index.jsonl"
EQ_SUMMARY = REPORTS_DIR / "clean_equivalence_frozen_index_summary.json"
STRUCT_PLAN = REPORTS_DIR / "clean_structure_tasks.jsonl"
STRUCT_NON_APPLICABLE = REPORTS_DIR / "clean_structure_non_applicable.jsonl"
STRUCT_FULL_PLAN = REPORTS_DIR / "clean_structure_full_plan.jsonl"
STRUCT_REGISTER_REPORT = REPORTS_DIR / "clean_structure_register_report.json"
STRUCT_INDEX = REPORTS_DIR / "clean_structure_frozen_index.jsonl"
STRUCT_SUMMARY = REPORTS_DIR / "clean_structure_frozen_index_summary.json"
SYMBOLIC_PLAN_REPORT = REPORTS_DIR / "clean_symbolic_task_plan.json"
PRED_RECOVERED_REPORT = REPORTS_DIR / "clean_pred_active_v3_recovered_attempts.json"
PRED_EXHAUSTED_AUDIT_JSONL = REPORTS_DIR / "clean_pred_active_v3_exhausted_audit.jsonl"
PRED_EXHAUSTED_AUDIT_REPORT = REPORTS_DIR / "clean_pred_active_v3_exhausted_audit_report.json"
PRED_FROZEN_AUDIT_JSONL = REPORTS_DIR / "clean_pred_active_v3_frozen_audit.jsonl"
PRED_FROZEN_AUDIT_REPORT = REPORTS_DIR / "clean_pred_active_v3_frozen_audit_report.json"
EQ_RECOVERED_REPORT = REPORTS_DIR / "clean_equivalence_recovered_attempts.json"
STRUCT_RECOVERED_REPORT = REPORTS_DIR / "clean_structure_recovered_attempts.json"
EVIDENCE_JSONL = RESULTS_DIR / "clean_pred_vs_gt_evidence.jsonl"
EVIDENCE_REPORT = REPORTS_DIR / "clean_pred_vs_gt_evidence_report.json"
AGGREGATE_REPORT = REPORTS_DIR / "aggregate_clean_metrics.json"

RUN_CLAUDE_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan"
MATERIALIZE_RECOVERED_MODULE = (
    "AAAI_experiments.stage5_metric_calculation_0831.pipeline.materialize_recovered_attempts"
)
AUDIT_EXHAUSTED_MODULE = (
    "AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_exhausted_simplifications"
)
AUDIT_FROZEN_MODULE = (
    "AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_frozen_simplifications"
)
PROMOTE_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.promote_revalidated_attempt"
FROZEN_INDEX_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index"
SYMBOLIC_BUILDER_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder"
REGISTER_SYMBOLIC_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.register_symbolic_plan"
EVIDENCE_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.materialize_pred_vs_gt_evidence"
AGGREGATE_MODULE = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_clean_metrics"

BATCH_LIMIT = 256
WORKERS = 8
POLL_SECONDS = 30
CLAUDE_RESOURCE_PREFIX = ["nice", "-n", "5", "taskset", "-c", "0-3,6-9"]


class GoalControllerError(RuntimeError):
    """控制器阶段失败。"""


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(event: str, **payload: Any) -> None:
    message = {"ts": _timestamp(), "event": event, **payload}
    print(json.dumps(message, ensure_ascii=False), flush=True)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GoalControllerError(f"{path} 顶层不是 JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoalControllerError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise GoalControllerError(f"{path} 第 {line_number} 行顶层不是 JSON object")
            rows.append(payload)
    return rows


def _run(cmd: list[str], *, expect: Iterable[int] = (0,)) -> None:
    log("run_start", cmd=cmd)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    log("run_done", cmd=cmd, returncode=completed.returncode)
    if completed.returncode not in set(expect):
        raise GoalControllerError(f"命令失败 rc={completed.returncode}: {' '.join(cmd)}")


def _task_counts(task_type: str, condition: str = "clean") -> dict[str, int]:
    connection = sqlite3.connect(STATE_DB)
    try:
        rows = connection.execute(
            """
            SELECT state, COUNT(*) AS c
            FROM tasks
            WHERE task_type=? AND condition_name=?
            GROUP BY state
            ORDER BY state
            """,
            (task_type, condition),
        ).fetchall()
    finally:
        connection.close()
    counts = {str(state): int(count) for state, count in rows}
    for state in ("pending", "running", "retry_wait", "frozen", "non_applicable", "exhausted"):
        counts.setdefault(state, 0)
    return counts


def _active_plan_pids(plan_path: Path) -> list[int]:
    plan_fragment = re.escape(plan_path.name)
    pattern = f"run_claude_plan .*{plan_fragment}"
    completed = subprocess.run(
        ["pgrep", "-af", pattern],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line.split(None, 1)[0]))
        except (TypeError, ValueError):
            continue
    return pids


def _wait_for_task_terminal(task_type: str, *, plan_path: Path | None = None) -> dict[str, int]:
    while True:
        counts = _task_counts(task_type)
        active = _active_plan_pids(plan_path) if plan_path is not None else []
        log(
            "task_poll",
            task_type=task_type,
            counts=counts,
            active_pids=active,
        )
        if counts["pending"] == 0 and counts["running"] == 0 and counts["retry_wait"] == 0:
            return counts
        time.sleep(POLL_SECONDS)


def _next_batch_report(prefix: str) -> Path:
    existing = sorted(REPORTS_DIR.glob(f"{prefix}_batch*_report.json"))
    if not existing:
        batch = 1
    else:
        batch = max(
            int(path.stem.split("_batch", 1)[1].split("_report", 1)[0])
            for path in existing
            if "_batch" in path.stem
        ) + 1
    return REPORTS_DIR / f"{prefix}_batch{batch}_report.json"


def _drive_plan_batches(*, plan_path: Path, task_type: str, report_prefix: str) -> None:
    while True:
        counts = _task_counts(task_type)
        if counts["pending"] == 0 and counts["running"] == 0 and counts["retry_wait"] == 0:
            log("plan_closed", task_type=task_type, counts=counts)
            return
        active = _active_plan_pids(plan_path)
        if active:
            log("plan_wait_active_run", task_type=task_type, active_pids=active, counts=counts)
            time.sleep(POLL_SECONDS)
            continue
        report_path = _next_batch_report(report_prefix)
        cmd = [
            *CLAUDE_RESOURCE_PREFIX,
            sys.executable,
            "-m",
            RUN_CLAUDE_MODULE,
            "--plan-jsonl",
            _rel(plan_path),
            "--state-db",
            _rel(STATE_DB),
            "--attempts-dir",
            _rel(ATTEMPTS_DIR),
            "--frozen-dir",
            _rel(FROZEN_DIR),
            "--report-json",
            _rel(report_path),
            "--limit",
            str(BATCH_LIMIT),
            "--workers",
            str(WORKERS),
            "--physical-attempt-offset",
            "12",
            "--predecessor-attempt-manifest",
            _rel(PREDECESSOR_MANIFEST),
        ]
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        log(
            "plan_batch_done",
            task_type=task_type,
            report_json=str(report_path),
            returncode=completed.returncode,
        )
        if completed.returncode == 3:
            time.sleep(POLL_SECONDS)
            continue
        if completed.returncode != 0:
            raise GoalControllerError(f"{task_type} batch 失败 rc={completed.returncode}")
        time.sleep(5)


def _materialize_recovered_attempts(plan_path: Path, report_path: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            MATERIALIZE_RECOVERED_MODULE,
            "--plan-jsonl",
            _rel(plan_path),
            "--state-db",
            _rel(STATE_DB),
            "--attempts-dir",
            _rel(ATTEMPTS_DIR),
            "--report-json",
            _rel(report_path),
        ]
    )


def _audit_and_promote_pred_exhausted() -> None:
    _run(
        [
            sys.executable,
            "-m",
            AUDIT_EXHAUSTED_MODULE,
            "--plan-jsonl",
            _rel(PRED_PLAN),
            "--state-db",
            _rel(STATE_DB),
            "--attempts-dir",
            _rel(ATTEMPTS_DIR),
            "--output-jsonl",
            _rel(PRED_EXHAUSTED_AUDIT_JSONL),
            "--report-json",
            _rel(PRED_EXHAUSTED_AUDIT_REPORT),
            "--semantic-timeout-seconds",
            "300",
        ]
    )
    rows = _read_jsonl(PRED_EXHAUSTED_AUDIT_JSONL)
    promoted = 0
    for row in rows:
        if row.get("resolution") != "promotable":
            continue
        recommended_attempt_id = row.get("recommended_attempt_id")
        attempts = row.get("attempts")
        if not isinstance(recommended_attempt_id, str) or not recommended_attempt_id:
            continue
        if not isinstance(attempts, list):
            continue
        matched_path: Path | None = None
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            if attempt.get("attempt_id") == recommended_attempt_id:
                attempt_path = attempt.get("attempt_path")
                if isinstance(attempt_path, str) and attempt_path:
                    matched_path = Path(attempt_path)
                break
        if matched_path is None:
            raise GoalControllerError(f"找不到 recommended attempt 文件: {recommended_attempt_id}")
        report_path = REPORTS_DIR / f"promote_{recommended_attempt_id}.json"
        if report_path.exists():
            log("promotion_report_exists", attempt_id=recommended_attempt_id)
            continue
        _run(
            [
                sys.executable,
                "-m",
                PROMOTE_MODULE,
                "--plan-jsonl",
                _rel(PRED_PLAN),
                "--state-db",
                _rel(STATE_DB),
                "--attempt-json",
                str(matched_path.resolve()),
                "--frozen-dir",
                _rel(FROZEN_DIR),
                "--predecessor-attempt-manifest",
                _rel(PREDECESSOR_MANIFEST),
                "--audit-reason",
                "clean_pred_active_v2_exhausted_revalidation_20260902",
                "--report-json",
                _rel(report_path),
            ]
        )
        promoted += 1
    log("pred_exhausted_promotion_summary", promoted_count=promoted, audited_task_count=len(rows))


def _build_pred_index() -> None:
    _run(
        [
            sys.executable,
            "-m",
            FROZEN_INDEX_MODULE,
            "--plan-jsonl",
            _rel(PRED_PLAN),
            "--state-db",
            _rel(STATE_DB),
            "--output-jsonl",
            _rel(PRED_INDEX),
            "--summary-json",
            _rel(PRED_SUMMARY),
            "--allow-exhausted",
            "--attempts-dir",
            _rel(ATTEMPTS_DIR),
        ]
    )
    summary = _read_json(PRED_SUMMARY)
    exhausted = int(summary.get("state_counts", {}).get("exhausted", 0))
    if exhausted:
        raise GoalControllerError(f"pred frozen index 仍有 exhausted={exhausted}，不能进入 clean 正式聚合")


def _audit_pred_frozen() -> None:
    _run(
        [
            *CLAUDE_RESOURCE_PREFIX,
            sys.executable,
            "-m",
            AUDIT_FROZEN_MODULE,
            "--plan-jsonl",
            _rel(PRED_PLAN),
            "--state-db",
            _rel(STATE_DB),
            "--attempts-dir",
            _rel(ATTEMPTS_DIR),
            "--frozen-dir",
            _rel(FROZEN_DIR),
            "--output-jsonl",
            _rel(PRED_FROZEN_AUDIT_JSONL),
            "--report-json",
            _rel(PRED_FROZEN_AUDIT_REPORT),
            "--expected-plan-count",
            "2250",
            "--semantic-timeout-seconds",
            "90",
            "--workers",
            str(WORKERS),
        ]
    )


def _build_symbolic_plans() -> None:
    _run(
        [
            sys.executable,
            "-m",
            SYMBOLIC_BUILDER_MODULE,
            "--gt-frozen-index-jsonl",
            _rel(GT_INDEX),
            "--pred-frozen-index-jsonl",
            _rel(PRED_INDEX),
            "--gt-frozen-summary-json",
            _rel(GT_SUMMARY),
            "--pred-frozen-summary-json",
            _rel(PRED_SUMMARY),
            "--clean-run-metrics-csv",
            _rel(RESULTS_DIR / "clean_numeric_run_metrics.csv"),
            "--gt-plan-jsonl",
            _rel(GT_PLAN),
            "--pred-plan-jsonl",
            _rel(PRED_PLAN),
            "--phase",
            "all",
            "--equivalence-output-jsonl",
            _rel(EQ_PLAN),
            "--equivalence-non-applicable-index-jsonl",
            _rel(EQ_NON_APPLICABLE),
            "--equivalence-full-plan-jsonl",
            _rel(EQ_FULL_PLAN),
            "--structure-output-jsonl",
            _rel(STRUCT_PLAN),
            "--structure-non-applicable-index-jsonl",
            _rel(STRUCT_NON_APPLICABLE),
            "--structure-full-plan-jsonl",
            _rel(STRUCT_FULL_PLAN),
            "--report-json",
            _rel(SYMBOLIC_PLAN_REPORT),
        ]
    )


def _register_symbolic(plan_path: Path, non_applicable_path: Path, report_path: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            REGISTER_SYMBOLIC_MODULE,
            "--plan-jsonl",
            _rel(plan_path),
            "--non-applicable-index-jsonl",
            _rel(non_applicable_path),
            "--state-db",
            _rel(STATE_DB),
            "--predecessor-attempt-manifest",
            _rel(PREDECESSOR_MANIFEST),
            "--report-json",
            _rel(report_path),
        ]
    )


def _build_final_index(plan_path: Path, output_jsonl: Path, summary_json: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            FROZEN_INDEX_MODULE,
            "--plan-jsonl",
            _rel(plan_path),
            "--state-db",
            _rel(STATE_DB),
            "--output-jsonl",
            _rel(output_jsonl),
            "--summary-json",
            _rel(summary_json),
        ]
    )


def _materialize_evidence() -> None:
    _run(
        [
            sys.executable,
            "-m",
            EVIDENCE_MODULE,
            "--equivalence-plan-jsonl",
            _rel(EQ_FULL_PLAN),
            "--output-jsonl",
            _rel(EVIDENCE_JSONL),
            "--report-json",
            _rel(EVIDENCE_REPORT),
            "--expected-row-count",
            "2250",
        ]
    )


def _aggregate_clean() -> None:
    _run(
        [
            sys.executable,
            "-m",
            AGGREGATE_MODULE,
            "--numeric-csv",
            _rel(RESULTS_DIR / "clean_numeric_run_metrics.csv"),
            "--clean-numeric-preparation-report-json",
            _rel(REPORTS_DIR / "clean_numeric_preparation.json"),
            "--eff-csv",
            _rel(RESULTS_DIR / "clean_eff_run_metrics.csv"),
            "--gt-plan-jsonl",
            _rel(GT_PLAN),
            "--gt-summary-json",
            _rel(GT_SUMMARY),
            "--pred-plan-jsonl",
            _rel(PRED_PLAN),
            "--eff-preparation-report-json",
            _rel(REPORTS_DIR / "eff_preparation.json"),
            "--pred-summary-json",
            _rel(PRED_SUMMARY),
            "--equivalence-plan-jsonl",
            _rel(EQ_FULL_PLAN),
            "--equivalence-summary-json",
            _rel(EQ_SUMMARY),
            "--structure-plan-jsonl",
            _rel(STRUCT_FULL_PLAN),
            "--structure-summary-json",
            _rel(STRUCT_SUMMARY),
            "--gt-index-jsonl",
            _rel(GT_INDEX),
            "--pred-index-jsonl",
            _rel(PRED_INDEX),
            "--equivalence-index-jsonl",
            _rel(EQ_INDEX),
            "--structure-index-jsonl",
            _rel(STRUCT_INDEX),
            "--evidence-jsonl",
            _rel(EVIDENCE_JSONL),
            "--clean-run-csv",
            _rel(RESULTS_DIR / "clean_run_metrics.csv"),
            "--task-stability-csv",
            _rel(RESULTS_DIR / "task_stability.csv"),
            "--algorithm-csv",
            _rel(RESULTS_DIR / "algorithm_six_axis.csv"),
            "--report-json",
            _rel(AGGREGATE_REPORT),
            "--print-summary",
        ]
    )


def main() -> int:
    log("controller_started")
    if AGGREGATE_REPORT.exists():
        report = _read_json(AGGREGATE_REPORT)
        if report.get("status") == "ok":
            log("clean_already_completed", report_json=str(AGGREGATE_REPORT))
            return 0

    _wait_for_task_terminal("pred_simplify", plan_path=PRED_PLAN)
    _build_pred_index()
    _audit_pred_frozen()

    _build_symbolic_plans()

    _register_symbolic(EQ_PLAN, EQ_NON_APPLICABLE, EQ_REGISTER_REPORT)
    _drive_plan_batches(plan_path=EQ_PLAN, task_type="equivalence", report_prefix="clean_equivalence_w8")
    _materialize_recovered_attempts(EQ_PLAN, EQ_RECOVERED_REPORT)
    _build_final_index(EQ_FULL_PLAN, EQ_INDEX, EQ_SUMMARY)

    _register_symbolic(STRUCT_PLAN, STRUCT_NON_APPLICABLE, STRUCT_REGISTER_REPORT)
    _drive_plan_batches(plan_path=STRUCT_PLAN, task_type="stab_structure", report_prefix="clean_structure_w8")
    _materialize_recovered_attempts(STRUCT_PLAN, STRUCT_RECOVERED_REPORT)
    _build_final_index(STRUCT_FULL_PLAN, STRUCT_INDEX, STRUCT_SUMMARY)

    _materialize_evidence()
    _aggregate_clean()
    log("clean_pipeline_completed", aggregate_report=str(AGGREGATE_REPORT))
    log("noise_pending", detail="按用户要求，带噪声阶段将在 clean 完成后再继续")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GoalControllerError as exc:
        log("controller_failed", error=str(exc))
        raise SystemExit(2)
