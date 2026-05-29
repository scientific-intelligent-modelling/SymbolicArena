# Benchmark Compliance Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the stage-one benchmark compliance loop for `15 algorithms × SSR50 × seed520 × 1h`, without launching the full experiment during implementation.

**Architecture:** Add a small control-plane package under `benchmark-control/compliance/` that generates manifests, audits run artifacts, writes heartbeat state, and produces rerun queues. The real experiment outputs stay under ignored `benchmark-runs/`, while tests use temporary directories and synthetic artifacts.

**Tech Stack:** Python standard library, `pytest`, existing `scientific_intelligent_modelling/config/toolbox_config.json`, existing benchmark result conventions, existing remote launcher scripts under `check/`.

---

## Scope Boundaries

This plan implements the local control-plane and testable command-line tools. It does not start the 750-task run. Full remote execution happens only after the plan is implemented, tested, reviewed, and explicitly requested.

## File Structure

- Create `benchmark-control/compliance/lib/__init__.py`
  - Marks the compliance control code as a local importable package for script-style tools.
- Create `benchmark-control/compliance/lib/models.py`
  - Defines dataclasses and constants shared by manifest, audit, heartbeat, and rerun code.
- Create `benchmark-control/compliance/lib/manifest.py`
  - Reads algorithm config and SSR50 dataset directories, then writes `manifest/*.json` and `manifest/tasks.csv`.
- Create `benchmark-control/compliance/lib/audit.py`
  - Reads synthetic or real task output directories and writes audit CSVs.
- Create `benchmark-control/compliance/lib/heartbeat.py`
  - Writes `heartbeat.json` from manifest and audit state.
- Create `benchmark-control/compliance/lib/rerun.py`
  - Converts `audit/failure_cases.csv` into `repair/round_N/rerun_tasks.csv`.
- Create `benchmark-control/compliance/launchers/prepare_batch.py`
  - CLI entrypoint for manifest generation.
- Create `benchmark-control/compliance/launchers/audit_batch.py`
  - CLI entrypoint for audit, heartbeat, and rerun queue generation.
- Create `benchmark-control/compliance/audit-schemas/task_audit.schema.json`
  - Human-readable schema for `audit/task_audit.csv`.
- Create `benchmark-control/compliance/templates/budget_1h.json`
  - Canonical stage-one budget profile.
- Modify `benchmark-control/compliance/README.md`
  - Add exact commands for prepare and audit.
- Modify `benchmark-control/compliance/goals/1h_compliance_goal.md`
  - Add exact `/goal` operating commands after scripts exist.
- Create `tests/test_benchmark_compliance_manifest.py`
  - Tests manifest generation with temporary SSR50-style datasets.
- Create `tests/test_benchmark_compliance_audit.py`
  - Tests audit classification with synthetic task artifacts.
- Create `tests/test_benchmark_compliance_heartbeat_rerun.py`
  - Tests heartbeat and rerun queue generation.

---

### Task 1: Shared Models And Budget Template

**Files:**
- Create: `benchmark-control/compliance/lib/__init__.py`
- Create: `benchmark-control/compliance/lib/models.py`
- Create: `benchmark-control/compliance/templates/budget_1h.json`
- Test: `python -m py_compile benchmark-control/compliance/lib/models.py`

- [ ] **Step 1: Create package marker**

Create `benchmark-control/compliance/lib/__init__.py` with:

```python
"""Shared helpers for benchmark compliance control-plane scripts."""
```

- [ ] **Step 2: Create shared model definitions**

Create `benchmark-control/compliance/lib/models.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE1_SEED = 520
STAGE1_TIMEOUT_SECONDS = 3600
STAGE1_MIN_RUNTIME_SECONDS = 3300
STAGE1_PROGRESS_INTERVAL_SECONDS = 60

FAILURE_CLASSES = (
    "early_stop",
    "missing_result",
    "missing_progress",
    "metric_invalid",
    "artifact_invalid",
    "timeout_unrecovered",
    "runtime_crash",
    "dispatch_failure",
    "unknown",
)

HEARTBEAT_PHASES = (
    "preparing",
    "deploying",
    "running",
    "audit",
    "repair",
    "rerun",
    "done",
    "blocked",
)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    env: str
    regressor: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset_dir: Path


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    algorithm: str
    dataset_id: str
    dataset_dir: Path
    seed: int
    timeout_in_seconds: int
    progress_snapshot_interval_seconds: int


@dataclass(frozen=True)
class AuditRow:
    task_id: str
    algorithm: str
    dataset_id: str
    seed: int
    status: str
    runtime_seconds: float
    has_result: bool
    has_progress: bool
    metrics_valid: bool
    artifact_valid: bool
    failure_class: str
    reason: str
```

- [ ] **Step 3: Create canonical 1h budget profile**

Create `benchmark-control/compliance/templates/budget_1h.json` with:

```json
{
  "name": "stage1_1h_compliance",
  "seed": 520,
  "timeout_in_seconds": 3600,
  "min_runtime_seconds": 3300,
  "progress_snapshot_interval_seconds": 60,
  "total_algorithms": 15,
  "total_datasets": 50,
  "total_tasks": 750
}
```

- [ ] **Step 4: Compile the shared model file**

Run:

```bash
python -m py_compile benchmark-control/compliance/lib/models.py
```

Expected: command exits with status `0`.

- [ ] **Step 5: Commit**

Run:

```bash
git add benchmark-control/compliance/lib benchmark-control/compliance/templates/budget_1h.json
git commit -m "[update] 增加合规控制共享模型" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

### Task 2: Manifest Generator

**Files:**
- Create: `benchmark-control/compliance/lib/manifest.py`
- Create: `benchmark-control/compliance/launchers/prepare_batch.py`
- Test: `tests/test_benchmark_compliance_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/test_benchmark_compliance_manifest.py` with:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_dataset(root: Path, name: str) -> None:
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.yaml").write_text(
        "target:\n  name: y\nfeatures:\n  - name: x0\n",
        encoding="utf-8",
    )


def test_manifest_generation_writes_750_stage1_tasks(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("manifest")
    ssr50_root = tmp_path / "ssr50"
    for idx in range(50):
        _write_dataset(ssr50_root, f"dataset_{idx:04d}")

    config_path = tmp_path / "toolbox_config.json"
    config_path.write_text(
        json.dumps(
            {
                "tool_mapping": {
                    f"alg{idx:02d}": {"env": f"env{idx:02d}", "regressor": f"Reg{idx:02d}"}
                    for idx in range(15)
                }
            }
        ),
        encoding="utf-8",
    )
    batch_dir = tmp_path / "benchmark-runs" / "compliance" / "batch"

    summary = module.generate_manifest(
        toolbox_config_path=config_path,
        ssr50_root=ssr50_root,
        batch_dir=batch_dir,
        git_revision="abc123",
    )

    assert summary["total_algorithms"] == 15
    assert summary["total_datasets"] == 50
    assert summary["total_tasks"] == 750
    tasks_path = batch_dir / "manifest" / "tasks.csv"
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 750
    assert rows[0]["seed"] == "520"
    assert rows[0]["timeout_in_seconds"] == "3600"
    assert rows[0]["progress_snapshot_interval_seconds"] == "60"
    assert "__seed520__" in rows[0]["task_id"]
```

Create `tests/benchmark_control_compliance_manifest_import.py` with:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_for_test(name: str) -> ModuleType:
    lib_dir = Path("benchmark-control/compliance/lib")
    sys.path.insert(0, str(lib_dir))
    path = lib_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"benchmark_compliance_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_benchmark_compliance_manifest.py -q
```

Expected: FAIL because `manifest.py` does not exist or `generate_manifest` is missing.

- [ ] **Step 3: Implement manifest generation**

Create `benchmark-control/compliance/lib/manifest.py` with:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from models import (
    STAGE1_PROGRESS_INTERVAL_SECONDS,
    STAGE1_SEED,
    STAGE1_TIMEOUT_SECONDS,
)


def _read_tool_mapping(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("tool_mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"toolbox config missing tool_mapping: {path}")
    return {
        str(name): {
            "env": str(config.get("env", "")),
            "regressor": str(config.get("regressor", "")),
        }
        for name, config in sorted(mapping.items())
    }


def _read_datasets(ssr50_root: Path) -> list[dict[str, str]]:
    datasets: list[dict[str, str]] = []
    for dataset_dir in sorted(path for path in ssr50_root.iterdir() if path.is_dir()):
        metadata = dataset_dir / "metadata.yaml"
        if metadata.exists():
            datasets.append(
                {
                    "dataset_id": dataset_dir.name,
                    "dataset_dir": str(dataset_dir.resolve()),
                }
            )
    if len(datasets) != 50:
        raise ValueError(f"expected 50 SSR50 datasets, found {len(datasets)} under {ssr50_root}")
    return datasets


def generate_manifest(
    *,
    toolbox_config_path: Path,
    ssr50_root: Path,
    batch_dir: Path,
    git_revision: str,
) -> dict[str, int]:
    algorithms = _read_tool_mapping(toolbox_config_path)
    if len(algorithms) != 15:
        raise ValueError(f"expected 15 algorithms, found {len(algorithms)}")
    datasets = _read_datasets(ssr50_root)
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    (manifest_dir / "algorithms.json").write_text(
        json.dumps(algorithms, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "datasets.csv").write_text(
        _to_csv(["dataset_id", "dataset_dir"], datasets),
        encoding="utf-8",
    )
    budget = {
        "seed": STAGE1_SEED,
        "timeout_in_seconds": STAGE1_TIMEOUT_SECONDS,
        "progress_snapshot_interval_seconds": STAGE1_PROGRESS_INTERVAL_SECONDS,
    }
    (manifest_dir / "budget.json").write_text(
        json.dumps(budget, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "git_revision.txt").write_text(git_revision.strip() + "\n", encoding="utf-8")

    tasks: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for dataset in datasets:
            dataset_id = dataset["dataset_id"]
            tasks.append(
                {
                    "task_id": f"{algorithm}__seed{STAGE1_SEED}__{dataset_id}",
                    "algorithm": algorithm,
                    "dataset_id": dataset_id,
                    "dataset_dir": dataset["dataset_dir"],
                    "seed": STAGE1_SEED,
                    "timeout_in_seconds": STAGE1_TIMEOUT_SECONDS,
                    "progress_snapshot_interval_seconds": STAGE1_PROGRESS_INTERVAL_SECONDS,
                }
            )
    (manifest_dir / "tasks.csv").write_text(
        _to_csv(
            [
                "task_id",
                "algorithm",
                "dataset_id",
                "dataset_dir",
                "seed",
                "timeout_in_seconds",
                "progress_snapshot_interval_seconds",
            ],
            tasks,
        ),
        encoding="utf-8",
    )
    return {
        "total_algorithms": len(algorithms),
        "total_datasets": len(datasets),
        "total_tasks": len(tasks),
    }


def _to_csv(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
```

- [ ] **Step 4: Implement prepare CLI**

Create `benchmark-control/compliance/launchers/prepare_batch.py` with:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from manifest import generate_manifest


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolbox-config", default="scientific_intelligent_modelling/config/toolbox_config.json")
    parser.add_argument("--ssr50-root", default="sim-datasets-data/ssr50")
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    summary = generate_manifest(
        toolbox_config_path=Path(args.toolbox_config),
        ssr50_root=Path(args.ssr50_root),
        batch_dir=Path(args.batch_dir),
        git_revision=_git_revision(),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run manifest tests**

Run:

```bash
pytest tests/test_benchmark_compliance_manifest.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmark-control/compliance/lib/manifest.py benchmark-control/compliance/launchers/prepare_batch.py tests/test_benchmark_compliance_manifest.py tests/benchmark_control_compliance_manifest_import.py
git commit -m "[update] 增加合规批次 manifest 生成器" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

### Task 3: Audit Classifier

**Files:**
- Create: `benchmark-control/compliance/lib/audit.py`
- Create: `benchmark-control/compliance/launchers/audit_batch.py`
- Create: `benchmark-control/compliance/audit-schemas/task_audit.schema.json`
- Test: `tests/test_benchmark_compliance_audit.py`

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_benchmark_compliance_audit.py` with:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark_control_compliance_manifest_import import load_for_test


def _write_tasks(batch_dir: Path) -> None:
    manifest = batch_dir / "manifest"
    manifest.mkdir(parents=True)
    (manifest / "tasks.csv").write_text(
        "task_id,algorithm,dataset_id,dataset_dir,seed,timeout_in_seconds,progress_snapshot_interval_seconds\n"
        "alg__seed520__d1,alg,d1,/data/d1,520,3600,60\n"
        "alg__seed520__d2,alg,d2,/data/d2,520,3600,60\n"
        "alg__seed520__d3,alg,d3,/data/d3,520,3600,60\n",
        encoding="utf-8",
    )


def _write_result(task_dir: Path, seconds: float, metric: float | None, artifact: str | None) -> None:
    task_dir.mkdir(parents=True)
    payload = {
        "status": "ok",
        "runtime_seconds": seconds,
        "valid": {"nmse": metric},
        "id_test": {"nmse": metric},
        "ood_test": {"nmse": metric},
        "canonical_artifact": {"expression": artifact} if artifact else None,
    }
    (task_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    progress = task_dir / "progress"
    progress.mkdir()
    (progress / "minute_0001.json").write_text("{}", encoding="utf-8")
    (progress / "minute_0055.json").write_text("{}", encoding="utf-8")


def test_audit_classifies_success_early_stop_and_missing_result(tmp_path: Path) -> None:
    module = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    _write_tasks(batch_dir)
    _write_result(batch_dir / "runs" / "alg" / "seed520" / "d1", 3501.0, 0.1, "x0")
    _write_result(batch_dir / "runs" / "alg" / "seed520" / "d2", 120.0, 0.1, "x0")

    summary = module.audit_batch(batch_dir=batch_dir)

    assert summary["total_tasks"] == 3
    assert summary["failed"] == 2
    with (batch_dir / "audit" / "failure_cases.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    assert [row["failure_class"] for row in failures] == ["early_stop", "missing_result"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_benchmark_compliance_audit.py -q
```

Expected: FAIL because `audit.py` does not exist or `audit_batch` is missing.

- [ ] **Step 3: Implement audit classifier**

Create `benchmark-control/compliance/lib/audit.py` with:

```python
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from models import STAGE1_MIN_RUNTIME_SECONDS


TASK_AUDIT_FIELDS = [
    "task_id",
    "algorithm",
    "dataset_id",
    "seed",
    "status",
    "runtime_seconds",
    "has_result",
    "has_progress",
    "metrics_valid",
    "artifact_valid",
    "failure_class",
    "reason",
]


def audit_batch(*, batch_dir: Path) -> dict[str, int]:
    tasks = _read_tasks(batch_dir / "manifest" / "tasks.csv")
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = [_audit_task(batch_dir, task) for task in tasks]
    failures = [row for row in rows if row["failure_class"]]
    _write_csv(audit_dir / "task_audit.csv", TASK_AUDIT_FIELDS, rows)
    _write_csv(audit_dir / "failure_cases.csv", TASK_AUDIT_FIELDS, failures)
    _write_csv(audit_dir / "early_stop_cases.csv", TASK_AUDIT_FIELDS, [row for row in failures if row["failure_class"] == "early_stop"])
    _write_summary(audit_dir / "budget_compliance_summary.csv", rows)
    return {"total_tasks": len(rows), "failed": len(failures), "passed": len(rows) - len(failures)}


def _audit_task(batch_dir: Path, task: dict[str, str]) -> dict[str, Any]:
    task_dir = batch_dir / "runs" / task["algorithm"] / f"seed{task['seed']}" / task["dataset_id"]
    result_path = task_dir / "result.json"
    progress_dir = task_dir / "progress"
    has_progress = progress_dir.exists() and any(progress_dir.glob("minute_*.json"))
    if not result_path.exists():
        return _row(task, "failed", 0.0, False, has_progress, False, False, "missing_result", "result.json not found")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _row(task, "failed", 0.0, True, has_progress, False, False, "artifact_invalid", f"invalid result json: {exc}")
    runtime = _runtime_seconds(result)
    metrics_valid = _metrics_valid(result)
    artifact_valid = _artifact_valid(result)
    if runtime < STAGE1_MIN_RUNTIME_SECONDS:
        failure = "early_stop"
        reason = f"runtime_seconds {runtime:.3f} < {STAGE1_MIN_RUNTIME_SECONDS}"
    elif not has_progress:
        failure = "missing_progress"
        reason = "progress/minute_*.json not found"
    elif not metrics_valid:
        failure = "metric_invalid"
        reason = "valid/id_test/ood_test metrics missing or non-finite"
    elif not artifact_valid:
        failure = "artifact_invalid"
        reason = "canonical artifact or equation missing"
    else:
        failure = ""
        reason = ""
    return _row(task, "ok" if not failure else "failed", runtime, True, has_progress, metrics_valid, artifact_valid, failure, reason)


def _runtime_seconds(result: dict[str, Any]) -> float:
    value = result.get("runtime_seconds", result.get("seconds", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metrics_valid(result: dict[str, Any]) -> bool:
    for split in ("valid", "id_test", "ood_test"):
        metrics = result.get(split)
        if not isinstance(metrics, dict) or not metrics:
            return False
        numeric_values = [value for value in metrics.values() if isinstance(value, (int, float))]
        if not numeric_values or not all(math.isfinite(float(value)) for value in numeric_values):
            return False
    return True


def _artifact_valid(result: dict[str, Any]) -> bool:
    artifact = result.get("canonical_artifact")
    equation = result.get("equation")
    if isinstance(artifact, dict) and any(artifact.values()):
        return True
    return isinstance(equation, str) and bool(equation.strip())


def _row(task: dict[str, str], status: str, runtime: float, has_result: bool, has_progress: bool, metrics_valid: bool, artifact_valid: bool, failure_class: str, reason: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "algorithm": task["algorithm"],
        "dataset_id": task["dataset_id"],
        "seed": task["seed"],
        "status": status,
        "runtime_seconds": f"{runtime:.3f}",
        "has_result": str(has_result).lower(),
        "has_progress": str(has_progress).lower(),
        "metrics_valid": str(metrics_valid).lower(),
        "artifact_valid": str(artifact_valid).lower(),
        "failure_class": failure_class,
        "reason": reason,
    }


def _read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_algorithm: dict[str, dict[str, int]] = {}
    for row in rows:
        item = by_algorithm.setdefault(row["algorithm"], {"total": 0, "passed": 0, "failed": 0})
        item["total"] += 1
        if row["failure_class"]:
            item["failed"] += 1
        else:
            item["passed"] += 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "total", "passed", "failed", "compliant"])
        writer.writeheader()
        for algorithm in sorted(by_algorithm):
            item = by_algorithm[algorithm]
            writer.writerow({**{"algorithm": algorithm}, **item, "compliant": str(item["failed"] == 0).lower()})
```

- [ ] **Step 4: Implement audit CLI**

Create `benchmark-control/compliance/launchers/audit_batch.py` with:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from audit import audit_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    print(audit_batch(batch_dir=Path(args.batch_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add task audit schema**

Create `benchmark-control/compliance/audit-schemas/task_audit.schema.json` with:

```json
{
  "format": "csv",
  "primary_key": "task_id",
  "columns": [
    "task_id",
    "algorithm",
    "dataset_id",
    "seed",
    "status",
    "runtime_seconds",
    "has_result",
    "has_progress",
    "metrics_valid",
    "artifact_valid",
    "failure_class",
    "reason"
  ],
  "failure_classes": [
    "early_stop",
    "missing_result",
    "missing_progress",
    "metric_invalid",
    "artifact_invalid",
    "timeout_unrecovered",
    "runtime_crash",
    "dispatch_failure",
    "unknown"
  ]
}
```

- [ ] **Step 6: Run audit tests**

Run:

```bash
pytest tests/test_benchmark_compliance_audit.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add benchmark-control/compliance/lib/audit.py benchmark-control/compliance/launchers/audit_batch.py benchmark-control/compliance/audit-schemas/task_audit.schema.json tests/test_benchmark_compliance_audit.py
git commit -m "[update] 增加合规结果审计器" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

### Task 4: Heartbeat Writer And Rerun Queue

**Files:**
- Create: `benchmark-control/compliance/lib/heartbeat.py`
- Create: `benchmark-control/compliance/lib/rerun.py`
- Test: `tests/test_benchmark_compliance_heartbeat_rerun.py`

- [ ] **Step 1: Write failing heartbeat and rerun tests**

Create `tests/test_benchmark_compliance_heartbeat_rerun.py` with:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark_control_compliance_manifest_import import load_for_test


def test_heartbeat_marks_codex_needed_when_failures_exist(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "batch"
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n"
        "alg__seed520__d1,alg,d1,520,failed,12.0,true,true,true,true,early_stop,short\n",
        encoding="utf-8",
    )

    payload = heartbeat.write_heartbeat(batch_dir=batch_dir, phase="repair")

    assert payload["needs_codex"] is True
    assert payload["failed"] == 1
    saved = json.loads((batch_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert saved["phase"] == "repair"


def test_rerun_queue_contains_only_failed_tasks(tmp_path: Path) -> None:
    rerun = load_for_test("rerun")
    batch_dir = tmp_path / "batch"
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n"
        "alg__seed520__d1,alg,d1,520,failed,12.0,true,true,true,true,early_stop,short\n",
        encoding="utf-8",
    )

    output = rerun.write_rerun_queue(batch_dir=batch_dir, round_id=1)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"task_id": "alg__seed520__d1", "algorithm": "alg", "dataset_id": "d1", "seed": "520", "failure_class": "early_stop"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_benchmark_compliance_heartbeat_rerun.py -q
```

Expected: FAIL because `heartbeat.py` and `rerun.py` do not exist.

- [ ] **Step 3: Implement heartbeat writer**

Create `benchmark-control/compliance/lib/heartbeat.py` with:

```python
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import HEARTBEAT_PHASES


def write_heartbeat(*, batch_dir: Path, phase: str) -> dict[str, Any]:
    if phase not in HEARTBEAT_PHASES:
        raise ValueError(f"invalid heartbeat phase: {phase}")
    failures = _read_csv(batch_dir / "audit" / "failure_cases.csv")
    task_rows = _read_csv(batch_dir / "manifest" / "tasks.csv") if (batch_dir / "manifest" / "tasks.csv").exists() else []
    payload = {
        "batch_id": batch_dir.name,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(task_rows),
        "pending": 0,
        "running": 0,
        "finished": max(0, len(task_rows) - len(failures)) if task_rows else 0,
        "failed": len(failures),
        "stale": 0,
        "needs_codex": bool(failures),
        "codex_reason": _codex_reason(failures),
        "latest_audit": "audit/failure_cases.csv",
        "latest_rerun_queue": "",
    }
    (batch_dir / "heartbeat.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _codex_reason(failures: list[dict[str, str]]) -> str:
    if not failures:
        return ""
    counts: dict[str, int] = {}
    for row in failures:
        key = row.get("failure_class", "unknown") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
```

- [ ] **Step 4: Implement rerun queue writer**

Create `benchmark-control/compliance/lib/rerun.py` with:

```python
from __future__ import annotations

import csv
from pathlib import Path


RERUN_FIELDS = ["task_id", "algorithm", "dataset_id", "seed", "failure_class"]


def write_rerun_queue(*, batch_dir: Path, round_id: int) -> Path:
    failures_path = batch_dir / "audit" / "failure_cases.csv"
    repair_dir = batch_dir / "repair" / f"round_{round_id:03d}"
    repair_dir.mkdir(parents=True, exist_ok=True)
    output = repair_dir / "rerun_tasks.csv"
    rows = _read_failures(failures_path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RERUN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RERUN_FIELDS})
    return output


def _read_failures(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
```

- [ ] **Step 5: Run heartbeat and rerun tests**

Run:

```bash
pytest tests/test_benchmark_compliance_heartbeat_rerun.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmark-control/compliance/lib/heartbeat.py benchmark-control/compliance/lib/rerun.py tests/test_benchmark_compliance_heartbeat_rerun.py
git commit -m "[update] 增加合规心跳和重跑队列" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

### Task 5: Wire Audit CLI To Heartbeat And Rerun

**Files:**
- Modify: `benchmark-control/compliance/launchers/audit_batch.py`
- Test: `tests/test_benchmark_compliance_heartbeat_rerun.py`

- [ ] **Step 1: Extend test to cover audit CLI side effects**

Append to `tests/test_benchmark_compliance_heartbeat_rerun.py`:

```python
def test_audit_cli_writes_heartbeat_and_rerun_queue(tmp_path: Path) -> None:
    import subprocess
    import sys

    batch_dir = tmp_path / "batch"
    manifest = batch_dir / "manifest"
    manifest.mkdir(parents=True)
    (manifest / "tasks.csv").write_text(
        "task_id,algorithm,dataset_id,dataset_dir,seed,timeout_in_seconds,progress_snapshot_interval_seconds\n"
        "alg__seed520__d1,alg,d1,/data/d1,520,3600,60\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "benchmark-control/compliance/launchers/audit_batch.py",
            "--batch-dir",
            str(batch_dir),
            "--write-heartbeat",
            "--write-rerun",
            "--round-id",
            "1",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (batch_dir / "heartbeat.json").exists()
    assert (batch_dir / "repair" / "round_001" / "rerun_tasks.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_benchmark_compliance_heartbeat_rerun.py::test_audit_cli_writes_heartbeat_and_rerun_queue -q
```

Expected: FAIL because `audit_batch.py` does not yet accept `--write-heartbeat` and `--write-rerun`.

- [ ] **Step 3: Update audit CLI**

Replace `benchmark-control/compliance/launchers/audit_batch.py` with:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from audit import audit_batch
from heartbeat import write_heartbeat
from rerun import write_rerun_queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--write-heartbeat", action="store_true")
    parser.add_argument("--write-rerun", action="store_true")
    parser.add_argument("--round-id", type=int, default=1)
    args = parser.parse_args()
    batch_dir = Path(args.batch_dir)
    summary = audit_batch(batch_dir=batch_dir)
    if args.write_heartbeat:
        write_heartbeat(batch_dir=batch_dir, phase="repair" if summary["failed"] else "done")
    if args.write_rerun and summary["failed"]:
        write_rerun_queue(batch_dir=batch_dir, round_id=args.round_id)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI side-effect test**

Run:

```bash
pytest tests/test_benchmark_compliance_heartbeat_rerun.py::test_audit_cli_writes_heartbeat_and_rerun_queue -q
```

Expected: PASS.

- [ ] **Step 5: Run compliance control tests**

Run:

```bash
pytest tests/test_benchmark_compliance_manifest.py tests/test_benchmark_compliance_audit.py tests/test_benchmark_compliance_heartbeat_rerun.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmark-control/compliance/launchers/audit_batch.py tests/test_benchmark_compliance_heartbeat_rerun.py
git commit -m "[update] 串联合规审计心跳和重跑队列" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

### Task 6: Documentation And Goal Commands

**Files:**
- Modify: `benchmark-control/compliance/README.md`
- Modify: `benchmark-control/compliance/goals/1h_compliance_goal.md`
- Test: `rg -n "benchmark-runs|prepare_batch|audit_batch" benchmark-control/compliance`

- [ ] **Step 1: Update README with exact local commands**

Append to `benchmark-control/compliance/README.md`:

```markdown

## 本地准备命令

创建一个阶段一批次：

```bash
BATCH_ID="compliance_15alg_ssr50_seed520_1h_$(date +%Y%m%d-%H%M%S)"
python benchmark-control/compliance/launchers/prepare_batch.py \
  --toolbox-config scientific_intelligent_modelling/config/toolbox_config.json \
  --ssr50-root sim-datasets-data/ssr50 \
  --batch-dir "benchmark-runs/compliance/${BATCH_ID}"
ln -sfn "${BATCH_ID}" benchmark-runs/compliance/latest
```

审计一个批次并生成心跳与重跑队列：

```bash
python benchmark-control/compliance/launchers/audit_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --write-heartbeat \
  --write-rerun \
  --round-id 1
```

这些命令只生成控制文件和审计文件，不会启动远端全量实验。
```

- [ ] **Step 2: Update goal with exact commands**

Append to `benchmark-control/compliance/goals/1h_compliance_goal.md`:

```markdown

## 标准命令

准备批次：

```bash
BATCH_ID="compliance_15alg_ssr50_seed520_1h_$(date +%Y%m%d-%H%M%S)"
python benchmark-control/compliance/launchers/prepare_batch.py \
  --toolbox-config scientific_intelligent_modelling/config/toolbox_config.json \
  --ssr50-root sim-datasets-data/ssr50 \
  --batch-dir "benchmark-runs/compliance/${BATCH_ID}"
mkdir -p benchmark-runs/compliance
ln -sfn "${BATCH_ID}" benchmark-runs/compliance/latest
```

审计批次：

```bash
python benchmark-control/compliance/launchers/audit_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --write-heartbeat \
  --write-rerun \
  --round-id 1
```

这两个命令不启动全量实验。远端 dispatch 必须在 smoke 通过并获得明确确认后执行。
```

- [ ] **Step 3: Verify docs mention required commands**

Run:

```bash
rg -n "prepare_batch|audit_batch|benchmark-runs/compliance/latest" benchmark-control/compliance
```

Expected: output includes `README.md` and `goals/1h_compliance_goal.md`.

- [ ] **Step 4: Commit**

Run:

```bash
git add benchmark-control/compliance/README.md benchmark-control/compliance/goals/1h_compliance_goal.md
git commit -m "[update] 补充合规闭环操作说明" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

### Task 7: Final Local Verification

**Files:**
- Validate: `benchmark-control/compliance/lib/*.py`
- Validate: `benchmark-control/compliance/launchers/*.py`
- Validate: `tests/test_benchmark_compliance_*.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_benchmark_compliance_manifest.py tests/test_benchmark_compliance_audit.py tests/test_benchmark_compliance_heartbeat_rerun.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Compile scripts**

Run:

```bash
python -m py_compile \
  benchmark-control/compliance/lib/models.py \
  benchmark-control/compliance/lib/manifest.py \
  benchmark-control/compliance/lib/audit.py \
  benchmark-control/compliance/lib/heartbeat.py \
  benchmark-control/compliance/lib/rerun.py \
  benchmark-control/compliance/launchers/prepare_batch.py \
  benchmark-control/compliance/launchers/audit_batch.py
```

Expected: command exits with status `0`.

- [ ] **Step 3: Confirm ignored run directory**

Run:

```bash
git check-ignore benchmark-runs/compliance/example/heartbeat.json
```

Expected: output is `benchmark-runs/compliance/example/heartbeat.json`.

- [ ] **Step 4: Check working tree**

Run:

```bash
git status --short --branch
```

Expected: branch is clean except being ahead of origin by implementation commits.

- [ ] **Step 5: Stop before remote experiment**

Do not run remote dispatch. Report that the control-plane implementation is ready for smoke planning.

---

## Self-Review

- Spec coverage: the plan covers control directories, manifest generation, audit outputs, failure classes, heartbeat, rerun queue, goal commands, Git ignore behavior, and no-full-experiment constraint.
- Scope check: remote dispatch adaptation is intentionally not implemented in this plan; it should be a separate plan after local control-plane tests pass.
- Placeholder scan: the plan contains no unresolved placeholders or unspecified implementation steps.
- Type consistency: manifest rows, audit rows, heartbeat fields, and rerun queue fields use the same names across tasks.
