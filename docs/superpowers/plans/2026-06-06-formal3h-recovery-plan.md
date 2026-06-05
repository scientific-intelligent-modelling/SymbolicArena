# Formal 3h Recovery-First Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an independent `formal3h` control plane that recovers usable 3h snapshots from the existing 24h batch and runs only missing tasks.

**Architecture:** Add a new experiment profile, recovery launcher, missing-only scheduler filter, formal3h deploy script writer, and audit compatibility for `elapsed_seconds`. Keep `formal24h` intact and store all new runtime products under `benchmark-runs/formal3h/`.

**Tech Stack:** Python 3, existing `benchmark-control/compliance` modules, existing `check/run_e1_candidate200_12alg_load_queue.py` scheduler, pytest.

---

## File Structure

- Modify: `benchmark-control/compliance/lib/models.py`
  - Add `FORMAL3H_SPEC` and profile lookup helpers while preserving `FULL24H_SPEC`.
- Modify: `benchmark-control/compliance/launchers/prepare_batch.py`
  - Accept `formal3h_13alg_3seed_3noise` and generate `params/10800s` plus `params_smoke/600s`.
- Modify: `benchmark-control/compliance/launchers/check_stage1_readiness.py`
  - Accept the new profile and pass it to readiness checks.
- Modify: `benchmark-control/compliance/lib/readiness.py`
  - Replace formal24h-only checks with profile-driven expectations.
- Create: `benchmark-control/compliance/lib/snapshot_recovery.py`
  - Recover valid `progress/minute_0180.json` snapshots into the formal3h `runs/` layout.
- Create: `benchmark-control/compliance/launchers/recover_formal3h_from_24h.py`
  - CLI wrapper around `snapshot_recovery.py`.
- Modify: `check/run_e1_candidate200_12alg_load_queue.py`
  - Add task-id allowlist filtering after task construction.
- Create: `benchmark-control/compliance/launchers/write_full3h_queue_commands.py`
  - Generate sync, preflight, smoke, recover, and missing-only full scripts for `formal3h`.
- Modify: `benchmark-control/compliance/lib/audit.py`
  - Treat `elapsed_seconds` as a runtime fallback.
- Modify: `benchmark-control/compliance/lib/harvest.py`
  - Avoid overwriting recovered results unless a fresh full result exists.
- Test: `tests/test_benchmark_compliance_formal3h.py`
  - Cover profile, params, readiness, recovery, scheduler filtering, and audit behavior.
- Test: `tests/test_benchmark_compliance_full3h_commands.py`
  - Cover generated scripts and formal3h path isolation.

## Shared Test Helpers

Add these helpers to `tests/test_benchmark_compliance_formal3h.py` before the first test:

```python
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))


def load_compliance_module(name: str):
    path = LIB / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"formal3h_test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_formal3h_one_task_batch(batch_dir: Path, *, min_runtime_seconds: int = 10500) -> None:
    (batch_dir / "manifest").mkdir(parents=True)
    (batch_dir / "queues").mkdir()
    (batch_dir / "manifest" / "tasks.csv").write_text(
        "task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds\n"
        f"dso__seed521__clean__g0032,dso,g0032,sim-datasets-data/ssr50/g0032,521,clean,0.0,10800,{min_runtime_seconds},60\n",
        encoding="utf-8",
    )
    (batch_dir / "queues" / "ssr50_source.csv").write_text(
        "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel\n"
        "32,g0032,g0032,sim-datasets-data/ssr50/g0032,sim-datasets-data/ssr50/g0032\n",
        encoding="utf-8",
    )
```

### Task 1: Add Formal3h Profile

**Files:**
- Modify: `benchmark-control/compliance/lib/models.py`
- Test: `tests/test_benchmark_compliance_formal3h.py`

- [ ] **Step 1: Write failing profile test**

```python
from models import FORMAL3H_SPEC, get_experiment_spec


def test_formal3h_spec_matches_goal_budget() -> None:
    assert FORMAL3H_SPEC.name == "formal3h_13alg_3seed_3noise"
    assert FORMAL3H_SPEC.algorithms == (
        "gplearn",
        "pyoperon",
        "pysr",
        "dso",
        "tpsr",
        "e2esr",
        "fepysr",
        "jaxsr",
        "QLattice",
        "iMCTS",
        "udsr",
        "ragsr",
        "symbolfit",
    )
    assert FORMAL3H_SPEC.seeds == (520, 521, 522)
    assert [level.tag for level in FORMAL3H_SPEC.noise_levels] == ["clean", "noise001", "noise005"]
    assert FORMAL3H_SPEC.budget.timeout_in_seconds == 10800
    assert FORMAL3H_SPEC.budget.min_runtime_seconds == 10500
    assert FORMAL3H_SPEC.budget.progress_snapshot_interval_seconds == 60
    assert get_experiment_spec("formal3h_13alg_3seed_3noise") is FORMAL3H_SPEC
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_formal3h_spec_matches_goal_budget -q
```

Expected: FAIL with `ImportError` or missing `FORMAL3H_SPEC`.

- [ ] **Step 3: Implement minimal profile**

Add to `models.py`:

```python
FULL3H_TIMEOUT_SECONDS = 10800
FULL3H_MIN_RUNTIME_SECONDS = 10500
FULL3H_PROGRESS_INTERVAL_SECONDS = 60

FORMAL3H_SPEC = ExperimentSpec(
    name="formal3h_13alg_3seed_3noise",
    algorithms=FULL24H_ALGORITHMS,
    seeds=FULL24H_SEEDS,
    noise_levels=tuple(NoiseLevelSpec(tag=noise_tag_for_sigma(sigma), sigma=sigma) for sigma in FULL24H_NOISE_LEVELS),
    budget=BudgetSpec(
        name="formal3h",
        timeout_in_seconds=FULL3H_TIMEOUT_SECONDS,
        min_runtime_seconds=FULL3H_MIN_RUNTIME_SECONDS,
        progress_snapshot_interval_seconds=FULL3H_PROGRESS_INTERVAL_SECONDS,
    ),
)

EXPERIMENT_SPECS = {
    "stage1_1h": STAGE1_SPEC,
    STAGE1_SPEC.name: STAGE1_SPEC,
    FULL24H_SPEC.name: FULL24H_SPEC,
    FORMAL3H_SPEC.name: FORMAL3H_SPEC,
}


def get_experiment_spec(profile: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown experiment profile: {profile}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_formal3h_spec_matches_goal_budget -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark-control/compliance/lib/models.py tests/test_benchmark_compliance_formal3h.py
git commit -m "[update] 增加3小时正式实验profile" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 2: Generalize Batch Preparation

**Files:**
- Modify: `benchmark-control/compliance/launchers/prepare_batch.py`
- Test: `tests/test_benchmark_compliance_formal3h.py`

- [ ] **Step 1: Write failing prepare test**

```python
import json
import subprocess
from pathlib import Path


def test_prepare_formal3h_generates_manifest_and_params(tmp_path: Path) -> None:
    batch_dir = tmp_path / "benchmark-runs" / "formal3h" / "batch"
    result = subprocess.run(
        [
            "python",
            "benchmark-control/compliance/launchers/prepare_batch.py",
            "--profile",
            "formal3h_13alg_3seed_3noise",
            "--batch-dir",
            str(batch_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "5850" in result.stdout
    tasks = (batch_dir / "manifest" / "tasks.csv").read_text(encoding="utf-8")
    assert "10800,10500,60" in tasks
    assert (batch_dir / "params" / "gplearn_clean.json").exists()
    assert json.loads((batch_dir / "params" / "gplearn_clean.json").read_text())["timeout_in_seconds"] == 10800
    assert json.loads((batch_dir / "params_smoke" / "gplearn_clean.json").read_text())["timeout_in_seconds"] == 600
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_prepare_formal3h_generates_manifest_and_params -q
```

Expected: FAIL because `formal3h_13alg_3seed_3noise` is not an accepted profile.

- [ ] **Step 3: Replace hard-coded profile branches**

Change `prepare_batch.py` to import `get_experiment_spec` and use:

```python
FORMAL_PROFILES = {"formal24h_13alg_3seed_3noise", "formal3h_13alg_3seed_3noise"}


def _spec_for_profile(profile: str):
    return get_experiment_spec(profile)
```

Use `_spec_for_profile(args.profile)` for default tools, seeds, noise sigmas, and budget values. Generate noise params whenever `args.profile in FORMAL_PROFILES`.

- [ ] **Step 4: Run focused prepare tests**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_prepare_formal3h_generates_manifest_and_params tests/test_benchmark_compliance_manifest.py::test_prepare_formal24h_manifest_has_three_seeds_three_noise_levels -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark-control/compliance/launchers/prepare_batch.py tests/test_benchmark_compliance_formal3h.py
git commit -m "[update] 支持3小时正式批次准备" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 3: Implement Snapshot Recovery

**Files:**
- Create: `benchmark-control/compliance/lib/snapshot_recovery.py`
- Create: `benchmark-control/compliance/launchers/recover_formal3h_from_24h.py`
- Test: `tests/test_benchmark_compliance_formal3h.py`

- [ ] **Step 1: Write failing recovery test**

```python
import csv
import json
from pathlib import Path

from snapshot_recovery import recover_snapshots


def test_recover_formal3h_snapshot_into_runs_layout(tmp_path: Path) -> None:
    batch_dir = tmp_path / "formal3h"
    source_root = tmp_path / "experiments" / "formal24h_batch"
    run_dir = source_root / "dso" / "seed521" / "tasks" / "dso_s521_clean_g0032" / "iaaccn22" / "dso" / "g0032_case"
    progress_dir = run_dir / "progress"
    progress_dir.mkdir(parents=True)
    snapshot = {
        "status": "ok",
        "elapsed_seconds": 10809.949,
        "equation": "x0 + 1",
        "canonical_artifact": {"sympy": "x0 + 1"},
        "valid": {"nmse": 0.1},
        "id_test": {"nmse": 0.2},
        "ood_test": {"nmse": 0.3},
    }
    (progress_dir / "minute_0180.json").write_text(json.dumps(snapshot), encoding="utf-8")
    write_formal3h_one_task_batch(batch_dir)

    summary = recover_snapshots(
        batch_dir=batch_dir,
        source_roots=[source_root],
        source_batch="formal24h_batch",
        snapshot_name="minute_0180.json",
    )

    assert summary["recovered"] == 1
    target = batch_dir / "runs" / "dso" / "seed521" / "clean" / "g0032"
    assert json.loads((target / "result.json").read_text())["recovered_from_24h"] is True
    assert (target / "progress" / "minute_0180.json").exists()
    assert (target / "recovery_source.json").exists()
    rows = list(csv.DictReader((batch_dir / "recovery" / "recovered_tasks.csv").open()))
    assert rows[0]["task_id"] == "dso__seed521__clean__g0032"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_recover_formal3h_snapshot_into_runs_layout -q
```

Expected: FAIL because `snapshot_recovery` does not exist.

- [ ] **Step 3: Implement recovery module**

Implement `recover_snapshots(...)` with these rules:

```python
def recover_snapshots(*, batch_dir: Path, source_roots: list[Path], source_batch: str, snapshot_name: str = "minute_0180.json") -> dict[str, int]:
    ...
```

For each manifest task, reconstruct scheduler task id using queue global index, validate snapshot payload, copy it to `result.json`, copy the snapshot to `progress/`, and write recovery CSVs.

- [ ] **Step 4: Add CLI wrapper**

`recover_formal3h_from_24h.py` accepts:

```text
--batch-dir
--source-batch
--snapshot-name
--source-root
```

It prints `recovery_summary.json`.

- [ ] **Step 5: Run recovery tests**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_recover_formal3h_snapshot_into_runs_layout -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmark-control/compliance/lib/snapshot_recovery.py benchmark-control/compliance/launchers/recover_formal3h_from_24h.py tests/test_benchmark_compliance_formal3h.py
git commit -m "[update] 增加3小时快照恢复器" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 4: Add Missing-Only Scheduler Filter

**Files:**
- Modify: `check/run_e1_candidate200_12alg_load_queue.py`
- Test: `tests/test_load_queue_scheduler.py`

- [ ] **Step 1: Write failing scheduler test**

```python
def test_scheduler_filters_tasks_by_allowlist(tmp_path: Path) -> None:
    allowlist = tmp_path / "missing_tasks.csv"
    allowlist.write_text("task_id\npysr_s520_clean_g0001\n", encoding="utf-8")
    tasks = scheduler._build_tasks(
        [{"global_index": "1", "dataset_dir": "sim-datasets-data/ssr50/g0001", "dataset_name": "g0001"},
         {"global_index": "2", "dataset_dir": "sim-datasets-data/ssr50/g0002", "dataset_name": "g0002"}],
        tools=["pysr"],
        seeds=[520],
        noise_sigmas=[0.0],
        queue_root=tmp_path / "queue",
        params_root=tmp_path / "params",
    )
    filtered = scheduler._filter_tasks_by_allowlist(tasks, allowlist)
    assert [task.task_id for task in filtered] == ["pysr_s520_clean_g0001"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_load_queue_scheduler.py::test_scheduler_filters_tasks_by_allowlist -q
```

Expected: FAIL because allowlist filtering is not implemented.

- [ ] **Step 3: Implement CLI and filter**

Add CLI argument:

```python
parser.add_argument("--task-id-allowlist-csv", default=None)
```

After task construction:

```python
allowed_task_ids = _read_task_id_allowlist(Path(args.task_id_allowlist_csv)) if args.task_id_allowlist_csv else None
if allowed_task_ids is not None:
    tasks = [task for task in tasks if task["task_id"] in allowed_task_ids]
```

`_read_task_id_allowlist` reads a CSV with a `task_id` column and rejects an empty allowlist.

- [ ] **Step 4: Run scheduler tests**

```bash
pytest tests/test_load_queue_scheduler.py::test_scheduler_filters_tasks_by_allowlist -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add check/run_e1_candidate200_12alg_load_queue.py tests/test_load_queue_scheduler.py
git commit -m "[update] 支持缺口任务队列过滤" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 5: Add Formal3h Deploy Scripts

**Files:**
- Create: `benchmark-control/compliance/launchers/write_full3h_queue_commands.py`
- Test: `tests/test_benchmark_compliance_full3h_commands.py`

- [ ] **Step 1: Write failing deploy-script test**

```python
import importlib.util
from pathlib import Path


def load_full3h_launcher():
    path = Path("benchmark-control/compliance/launchers/write_full3h_queue_commands.py")
    spec = importlib.util.spec_from_file_location("full3h_commands", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_full3h_commands_use_formal3h_paths(tmp_path: Path) -> None:
    module = load_full3h_launcher()
    batch_dir = tmp_path / "benchmark-runs" / "formal3h" / "batch"
    paths = module.write_full3h_queue_commands(batch_dir=batch_dir)
    scripts = {path.name: path.read_text(encoding="utf-8") for path in paths}
    assert "benchmark-runs/formal3h/latest" in scripts["00_sync_code_and_batch_to_iaaccn22.sh"]
    assert "recover_formal3h_from_24h.py" in scripts["02_recover_from_formal24h_from_iaaccn22.sh"]
    assert "--task-id-allowlist-csv benchmark-runs/formal3h/latest/recovery/missing_tasks.csv" in scripts["04_full_dispatch_from_iaaccn22.sh"]
    assert "--params-root benchmark-runs/formal3h/latest/params" in scripts["04_full_dispatch_from_iaaccn22.sh"]
    assert "--session-prefix formal3h_full_" in scripts["04_full_dispatch_from_iaaccn22.sh"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_compliance_full3h_commands.py::test_write_full3h_commands_use_formal3h_paths -q
```

Expected: FAIL because `write_full3h_queue_commands.py` does not exist.

- [ ] **Step 3: Implement script writer**

Create scripts:

```text
00_sync_code_and_batch_to_iaaccn22.sh
01_preflight_from_iaaccn22.sh
02_recover_from_formal24h_from_iaaccn22.sh
03_smoke_dispatch_from_iaaccn22.sh
04_full_dispatch_from_iaaccn22.sh
```

Use `benchmark-runs/formal3h/latest` everywhere. The full dispatch command must include:

```text
--task-id-allowlist-csv benchmark-runs/formal3h/latest/recovery/missing_tasks.csv
--params-root benchmark-runs/formal3h/latest/params
--session-prefix formal3h_full_
--host-session-count-prefix formal3h_full_
```

- [ ] **Step 4: Run deploy-script tests**

```bash
pytest tests/test_benchmark_compliance_full3h_commands.py::test_write_full3h_commands_use_formal3h_paths -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark-control/compliance/launchers/write_full3h_queue_commands.py tests/test_benchmark_compliance_full3h_commands.py
git commit -m "[update] 增加3小时正式实验部署脚本" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 6: Make Audit and Harvest Recovery-Aware

**Files:**
- Modify: `benchmark-control/compliance/lib/audit.py`
- Modify: `benchmark-control/compliance/lib/harvest.py`
- Test: `tests/test_benchmark_compliance_formal3h.py`

- [ ] **Step 1: Write failing audit test**

```python
def test_audit_accepts_elapsed_seconds_for_recovered_snapshot(tmp_path: Path) -> None:
    batch_dir = tmp_path / "formal3h"
    write_formal3h_one_task_batch(batch_dir, min_runtime_seconds=10500)
    run_dir = batch_dir / "runs" / "dso" / "seed521" / "clean" / "g0032"
    (run_dir / "progress").mkdir(parents=True)
    (run_dir / "progress" / "minute_0180.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "elapsed_seconds": 10809.949,
                "equation": "x0 + 1",
                "canonical_artifact": {"sympy": "x0 + 1"},
                "valid": {"nmse": 0.1},
                "id_test": {"nmse": 0.2},
                "ood_test": {"nmse": 0.3},
                "recovered_from_24h": True,
            }
        ),
        encoding="utf-8",
    )
    summary = audit_batch(batch_dir=batch_dir)
    assert summary == {"total_tasks": 1, "failed": 0, "passed": 1}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_audit_accepts_elapsed_seconds_for_recovered_snapshot -q
```

Expected: FAIL with `early_stop` because `_runtime_seconds` ignores `elapsed_seconds`.

- [ ] **Step 3: Implement runtime fallback**

Change `_runtime_seconds`:

```python
value = result.get("runtime_seconds", result.get("seconds", result.get("elapsed_seconds", 0.0)))
```

- [ ] **Step 4: Preserve recovered results during harvest**

In `_copy_candidate`, if `target_dir/result.json` exists and contains `recovered_from_24h=true`, allow overwrite only when copying a fresh full `result.json`; never overwrite a recovered result with a missing or invalid source.

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/test_benchmark_compliance_formal3h.py::test_audit_accepts_elapsed_seconds_for_recovered_snapshot tests/test_benchmark_compliance_harvest.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmark-control/compliance/lib/audit.py benchmark-control/compliance/lib/harvest.py tests/test_benchmark_compliance_formal3h.py
git commit -m "[fix] 兼容3小时恢复结果审计" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 7: Wire Readiness and Local Dry Run

**Files:**
- Modify: `benchmark-control/compliance/launchers/check_stage1_readiness.py`
- Modify: `benchmark-control/compliance/lib/readiness.py`
- Test: `tests/test_benchmark_compliance_readiness.py`

- [ ] **Step 1: Write failing readiness test**

```python
def test_readiness_passes_for_formal3h_batch(tmp_path: Path) -> None:
    batch_dir = tmp_path / "formal3h"
    subprocess.run(
        [
            "python",
            "benchmark-control/compliance/launchers/prepare_batch.py",
            "--profile",
            "formal3h_13alg_3seed_3noise",
            "--batch-dir",
            str(batch_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    readiness = load_compliance_module("readiness")
    result = readiness.check_readiness(batch_dir=batch_dir, profile="formal3h_13alg_3seed_3noise")
    assert result["ready"] is True
    assert result["total_tasks"] == 5850
    assert result["total_algorithms"] == 13
    assert result["total_noise_levels"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_compliance_readiness.py::test_readiness_passes_for_formal3h_batch -q
```

Expected: FAIL because readiness only accepts stage1 and formal24h.

- [ ] **Step 3: Generalize profile expectations**

Use `get_experiment_spec(profile)` in readiness and compute expected algorithms, seeds, noise tags, timeout, min runtime, progress interval, and expected task count from the selected spec.

- [ ] **Step 4: Generate local formal3h batch**

```bash
BATCH_ID="formal3h_13alg_ssr50_seed520-522_noise0-001-005_localcheck"
rm -rf "benchmark-runs/formal3h/${BATCH_ID}"
mkdir -p "benchmark-runs/formal3h/${BATCH_ID}"
ln -sfn "${BATCH_ID}" benchmark-runs/formal3h/latest
python benchmark-control/compliance/launchers/prepare_batch.py \
  --profile formal3h_13alg_3seed_3noise \
  --batch-dir "benchmark-runs/formal3h/${BATCH_ID}"
python benchmark-control/compliance/launchers/write_full3h_queue_commands.py \
  --batch-dir benchmark-runs/formal3h/latest
python benchmark-control/compliance/launchers/check_stage1_readiness.py \
  --profile formal3h_13alg_3seed_3noise \
  --batch-dir benchmark-runs/formal3h/latest
```

Expected: readiness reports `ready=true`.

- [ ] **Step 5: Run focused readiness tests**

```bash
pytest tests/test_benchmark_compliance_readiness.py::test_readiness_passes_for_formal3h_batch -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmark-control/compliance/launchers/check_stage1_readiness.py benchmark-control/compliance/lib/readiness.py tests/test_benchmark_compliance_readiness.py
git commit -m "[update] 接入3小时正式实验就绪检查" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

### Task 8: Final Verification and Push

**Files:**
- Verify: `benchmark-control/compliance`
- Verify: `check/run_e1_candidate200_12alg_load_queue.py`
- Verify: `tests`

- [ ] **Step 1: Run focused compliance tests**

```bash
pytest \
  tests/test_benchmark_compliance_formal3h.py \
  tests/test_benchmark_compliance_full3h_commands.py \
  tests/test_benchmark_compliance_manifest.py \
  tests/test_benchmark_compliance_params.py \
  tests/test_benchmark_compliance_readiness.py \
  tests/test_benchmark_compliance_harvest.py \
  tests/test_benchmark_compliance_audit.py \
  tests/test_load_queue_scheduler.py -q
```

Expected: PASS.

- [ ] **Step 2: Check worktree**

```bash
git status --short
```

Expected: no uncommitted code changes except ignored `benchmark-runs/` local dry-run artifacts.

- [ ] **Step 3: Push commits**

```bash
git push
```

Expected: branch `main` updates on `origin`.

## Execution Notes

- Do not stop existing 24h remote jobs without explicit user confirmation.
- Do not delete historical batches.
- Do not clean remote data disks.
- Do not commit `benchmark-runs/`.
- Use `iaaccn22` as the hub and internal IPs `10.10.100.23~29` for fan-out.
- For complex remote logic, upload scripts to `/tmp` and execute script files instead of inline shell/Python.
