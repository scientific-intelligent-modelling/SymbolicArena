# Full 24h 13 算法三 Seed 三噪声实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a spec-driven control plane for `13 algorithms × SSR50 × seeds(520,521,522) × noise(0,0.01,0.05) × 24h`, without launching the full experiment during implementation.

**Architecture:** Generalize the existing 1h compliance control plane from hard-coded Stage1 constants into explicit experiment profiles. Add noise-aware manifest, params generation, scheduler task IDs, harvest paths, audit schema, readiness gates, and deploy scripts so the 5850-task batch can be prepared, preflighted, smoked, audited, repaired, and rerun safely.

**Tech Stack:** Python standard library, `pytest`, existing `benchmark-control/compliance/` control plane, existing `check/run_e1_candidate200_12alg_load_queue.py` scheduler, existing `scientific_intelligent_modelling/benchmarks/runner.py` train-label-noise support.

---

## Scope Boundaries

This plan implements local control-plane support and generates safe launch scripts. It does not start smoke or full runs. Remote execution happens only after implementation, tests, local readiness, remote sync, and explicit user instruction or active `/goal resume`.

## File Structure

- Modify `benchmark-control/compliance/lib/models.py`
  - Replace single Stage1 constants with reusable budget/profile dataclasses while preserving 1h defaults.
- Modify `benchmark-control/compliance/lib/manifest.py`
  - Generate manifest rows for arbitrary algorithms, seeds, noise levels, timeout, and progress interval.
- Modify `benchmark-control/compliance/launchers/prepare_batch.py`
  - Add CLI args for experiment profile, selected tools, seeds, noise levels, timeout, min runtime, and output family.
- Create `benchmark-control/compliance/lib/params.py`
  - Generate per-noise params JSON files from the existing E1 params source.
- Modify `check/run_e1_candidate200_12alg_load_queue.py`
  - Add first-class `--noise-sigmas` support and include `noise_tag` in task IDs and selected params names.
- Modify `benchmark-control/compliance/lib/harvest.py`
  - Harvest noise-aware scheduler task IDs into `runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/`.
- Modify `benchmark-control/compliance/lib/audit.py`
  - Read noise-aware run paths and include `noise_tag` / `noise_sigma` in audit CSVs.
- Modify `benchmark-control/compliance/audit-schemas/task_audit.schema.json`
  - Add noise columns.
- Modify `benchmark-control/compliance/lib/readiness.py`
  - Validate 5850 tasks, 13 algorithms, 3 seeds, 3 noise levels, 24h timeout, and generated params.
- Create `benchmark-control/compliance/launchers/write_full24h_queue_commands.py`
  - Generate preflight, smoke, and full scripts for the formal24h batch.
- Create `benchmark-control/compliance/templates/budget_24h_13alg_3seed_3noise.json`
  - Canonical budget profile for this experiment.
- Modify `benchmark-control/compliance/README.md`
  - Document the 24h profile and exact safe prepare/preflight/smoke/full workflow.
- Test files:
  - Modify `tests/test_benchmark_compliance_manifest.py`
  - Create `tests/test_benchmark_compliance_params.py`
  - Modify `tests/test_load_queue_scheduler.py`
  - Modify `tests/test_benchmark_compliance_harvest.py`
  - Modify `tests/test_benchmark_compliance_audit.py`
  - Modify `tests/test_benchmark_compliance_readiness.py`
  - Create `tests/test_benchmark_compliance_full24h_commands.py`

---

### Task 1: Experiment Profile Models

**Files:**
- Modify: `benchmark-control/compliance/lib/models.py`
- Test: `python -m py_compile benchmark-control/compliance/lib/models.py`

- [ ] **Step 1: Add reusable profile dataclasses**

Replace the top-level constant block with this compatible structure:

```python
STAGE1_SEED = 520
STAGE1_TIMEOUT_SECONDS = 3600
STAGE1_MIN_RUNTIME_SECONDS = 3300
STAGE1_PROGRESS_INTERVAL_SECONDS = 60

FULL24H_SEEDS = (520, 521, 522)
FULL24H_NOISE_LEVELS = (0.0, 0.01, 0.05)
FULL24H_TIMEOUT_SECONDS = 86400
FULL24H_MIN_RUNTIME_SECONDS = 82800
FULL24H_PROGRESS_INTERVAL_SECONDS = 60
FULL24H_ALGORITHMS = (
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


@dataclass(frozen=True)
class NoiseLevelSpec:
    tag: str
    sigma: float


@dataclass(frozen=True)
class BudgetSpec:
    name: str
    timeout_in_seconds: int
    min_runtime_seconds: int
    progress_snapshot_interval_seconds: int


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    algorithms: tuple[str, ...]
    seeds: tuple[int, ...]
    noise_levels: tuple[NoiseLevelSpec, ...]
    budget: BudgetSpec
    expected_datasets: int = 50


def noise_tag_for_sigma(sigma: float) -> str:
    if float(sigma) == 0.0:
        return "clean"
    scaled = int(round(float(sigma) * 100))
    return f"noise{scaled:03d}"


STAGE1_SPEC = ExperimentSpec(
    name="stage1_1h_compliance",
    algorithms=(
        "QLattice",
        "drsr",
        "dso",
        "e2esr",
        "fepysr",
        "gplearn",
        "iMCTS",
        "jaxsr",
        "llmsr",
        "pyoperon",
        "pysr",
        "ragsr",
        "symbolfit",
        "tpsr",
        "udsr",
    ),
    seeds=(STAGE1_SEED,),
    noise_levels=(NoiseLevelSpec(tag="clean", sigma=0.0),),
    budget=BudgetSpec(
        name="stage1_1h",
        timeout_in_seconds=STAGE1_TIMEOUT_SECONDS,
        min_runtime_seconds=STAGE1_MIN_RUNTIME_SECONDS,
        progress_snapshot_interval_seconds=STAGE1_PROGRESS_INTERVAL_SECONDS,
    ),
)


FULL24H_SPEC = ExperimentSpec(
    name="formal24h_13alg_3seed_3noise",
    algorithms=FULL24H_ALGORITHMS,
    seeds=FULL24H_SEEDS,
    noise_levels=tuple(NoiseLevelSpec(tag=noise_tag_for_sigma(sigma), sigma=sigma) for sigma in FULL24H_NOISE_LEVELS),
    budget=BudgetSpec(
        name="formal24h",
        timeout_in_seconds=FULL24H_TIMEOUT_SECONDS,
        min_runtime_seconds=FULL24H_MIN_RUNTIME_SECONDS,
        progress_snapshot_interval_seconds=FULL24H_PROGRESS_INTERVAL_SECONDS,
    ),
)
```

- [ ] **Step 2: Keep existing dataclasses compatible**

Extend `TaskSpec` with noise fields without removing existing fields:

```python
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    algorithm: str
    dataset_id: str
    dataset_dir: Path
    seed: int
    timeout_in_seconds: int
    progress_snapshot_interval_seconds: int
    noise_tag: str = "clean"
    noise_sigma: float = 0.0
```

- [ ] **Step 3: Compile models**

Run:

```bash
python -m py_compile benchmark-control/compliance/lib/models.py
```

Expected: exit code `0`.

---

### Task 2: Noise-Aware Manifest Generation

**Files:**
- Modify: `benchmark-control/compliance/lib/manifest.py`
- Modify: `benchmark-control/compliance/launchers/prepare_batch.py`
- Modify: `tests/test_benchmark_compliance_manifest.py`

- [ ] **Step 1: Write failing full24h manifest test**

Append this test to `tests/test_benchmark_compliance_manifest.py`:

```python
def test_full24h_manifest_generation_writes_5850_noise_aware_tasks(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("manifest")
    ssr50_root = tmp_path / "ssr50"
    for index in range(50):
        _write_dataset(ssr50_root, f"d{index:02d}")

    toolbox_config = tmp_path / "toolbox_config.json"
    toolbox_config.write_text(
        json.dumps(
            {
                "tool_mapping": {
                    name: {"env": "sim_base", "regressor": name}
                    for name in (
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
                        "llmsr",
                        "drsr",
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    batch_dir = tmp_path / "benchmark-runs" / "formal24h" / "batch"
    summary = module.generate_manifest(
        toolbox_config_path=toolbox_config,
        ssr50_root=ssr50_root,
        batch_dir=batch_dir,
        git_revision="abc123",
        dataset_dir_base=tmp_path,
        algorithms=("gplearn", "pyoperon", "pysr", "dso", "tpsr", "e2esr", "fepysr", "jaxsr", "QLattice", "iMCTS", "udsr", "ragsr", "symbolfit"),
        seeds=(520, 521, 522),
        noise_sigmas=(0.0, 0.01, 0.05),
        timeout_in_seconds=86400,
        progress_snapshot_interval_seconds=60,
        min_runtime_seconds=82800,
    )

    assert summary == {"total_algorithms": 13, "total_datasets": 50, "total_tasks": 5850}
    with (batch_dir / "manifest" / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5850
    assert {row["seed"] for row in rows} == {"520", "521", "522"}
    assert {row["noise_tag"] for row in rows} == {"clean", "noise001", "noise005"}
    assert {row["noise_sigma"] for row in rows} == {"0.0", "0.01", "0.05"}
    assert all("__seed" in row["task_id"] and "__noise" in row["task_id"] or "__clean__" in row["task_id"] for row in rows)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
pytest tests/test_benchmark_compliance_manifest.py::test_full24h_manifest_generation_writes_5850_noise_aware_tasks -q
```

Expected: fail because `generate_manifest()` does not accept `algorithms`, `seeds`, or `noise_sigmas`.

- [ ] **Step 3: Extend manifest signature**

Modify `generate_manifest()` to accept:

```python
algorithms: tuple[str, ...] | None = None,
seeds: tuple[int, ...] | None = None,
noise_sigmas: tuple[float, ...] | None = None,
timeout_in_seconds: int = STAGE1_TIMEOUT_SECONDS,
progress_snapshot_interval_seconds: int = STAGE1_PROGRESS_INTERVAL_SECONDS,
min_runtime_seconds: int = STAGE1_MIN_RUNTIME_SECONDS,
```

Implementation rules:

```python
selected_algorithms = tuple(algorithms or tuple(_read_tool_mapping(toolbox_config_path)))
noise_levels = tuple(
    {"noise_tag": noise_tag_for_sigma(sigma), "noise_sigma": float(sigma)}
    for sigma in (noise_sigmas or (0.0,))
)
```

Filter `algorithms.json` to selected algorithms and raise `ValueError` if any requested algorithm is absent from toolbox config.

- [ ] **Step 4: Extend manifest outputs**

Write `manifest/noise_levels.csv` with columns:

```text
noise_tag,noise_sigma
```

Write `manifest/budget.json` with:

```json
{
  "timeout_in_seconds": 86400,
  "min_runtime_seconds": 82800,
  "progress_snapshot_interval_seconds": 60,
  "total_algorithms": 13,
  "total_datasets": 50,
  "total_seeds": 3,
  "total_noise_levels": 3,
  "total_tasks": 5850
}
```

- [ ] **Step 5: Extend task CSV fields**

Use these fields in order:

```python
[
    "task_id",
    "algorithm",
    "dataset_id",
    "dataset_dir",
    "seed",
    "noise_tag",
    "noise_sigma",
    "timeout_in_seconds",
    "min_runtime_seconds",
    "progress_snapshot_interval_seconds",
]
```

Task ID format:

```python
f"{algorithm}__seed{seed}__{noise_tag}__{dataset_id}"
```

- [ ] **Step 6: Add prepare CLI args**

Add to `prepare_batch.py`:

```python
parser.add_argument("--profile", choices=["stage1_1h", "formal24h_13alg_3seed_3noise"], default="stage1_1h")
parser.add_argument("--tools", nargs="*", default=None)
parser.add_argument("--seeds", nargs="*", type=int, default=None)
parser.add_argument("--noise-sigmas", nargs="*", type=float, default=None)
parser.add_argument("--timeout-in-seconds", type=int, default=None)
parser.add_argument("--min-runtime-seconds", type=int, default=None)
parser.add_argument("--progress-snapshot-interval-seconds", type=int, default=None)
```

When `--profile formal24h_13alg_3seed_3noise`, pass the 13 algorithms, seeds `520 521 522`, noise sigmas `0 0.01 0.05`, timeout `86400`, min runtime `82800`, interval `60`.

- [ ] **Step 7: Run manifest tests**

Run:

```bash
pytest tests/test_benchmark_compliance_manifest.py -q
```

Expected: all manifest tests pass.

---

### Task 3: 24h Noise Params Generator

**Files:**
- Create: `benchmark-control/compliance/lib/params.py`
- Create: `tests/test_benchmark_compliance_params.py`
- Create: `benchmark-control/compliance/templates/budget_24h_13alg_3seed_3noise.json`

- [ ] **Step 1: Write params generation tests**

Create `tests/test_benchmark_compliance_params.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def test_generate_noise_params_writes_13_by_3_files(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("params")
    source = tmp_path / "source"
    source.mkdir()
    for tool in ("gplearn", "pyoperon", "pysr", "dso", "tpsr", "e2esr", "fepysr", "jaxsr", "qlattice", "imcts", "udsr", "ragsr", "symbolfit"):
        (source / f"{tool}.json").write_text(json.dumps({"timeout_in_seconds": 3600, "niterations": 1000000}), encoding="utf-8")

    output = tmp_path / "params"
    summary = module.generate_noise_params(
        source_params_root=source,
        output_params_root=output,
        tools=("gplearn", "pyoperon", "pysr", "dso", "tpsr", "e2esr", "fepysr", "jaxsr", "qlattice", "imcts", "udsr", "ragsr", "symbolfit"),
        noise_sigmas=(0.0, 0.01, 0.05),
        timeout_in_seconds=86400,
        progress_snapshot_interval_seconds=60,
    )

    assert summary == {"tools": 13, "noise_levels": 3, "params_files": 39}
    clean = json.loads((output / "pysr__clean.json").read_text(encoding="utf-8"))
    noisy = json.loads((output / "pysr__noise001.json").read_text(encoding="utf-8"))
    assert clean["timeout_in_seconds"] == 86400
    assert clean["train_label_noise_enabled"] is False
    assert clean["train_label_noise_sigma"] == 0.0
    assert noisy["train_label_noise_enabled"] is True
    assert noisy["train_label_noise_sigma"] == 0.01
```

- [ ] **Step 2: Run failing test**

Run:

```bash
pytest tests/test_benchmark_compliance_params.py -q
```

Expected: fail because `params.py` does not exist.

- [ ] **Step 3: Implement params generator**

Create `benchmark-control/compliance/lib/params.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import noise_tag_for_sigma


def generate_noise_params(
    *,
    source_params_root: Path,
    output_params_root: Path,
    tools: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    timeout_in_seconds: int,
    progress_snapshot_interval_seconds: int,
) -> dict[str, int]:
    output_params_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for tool in tools:
        source_path = source_params_root / f"{tool}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"参数文件不存在: {source_path}")
        base = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError(f"参数文件必须是 JSON object: {source_path}")
        for sigma in noise_sigmas:
            tag = noise_tag_for_sigma(sigma)
            payload: dict[str, Any] = dict(base)
            payload["timeout_in_seconds"] = int(timeout_in_seconds)
            payload["progress_snapshot_interval_seconds"] = int(progress_snapshot_interval_seconds)
            payload["train_label_noise_sigma"] = float(sigma)
            payload["train_label_noise_enabled"] = bool(float(sigma) > 0.0)
            (output_params_root / f"{tool}__{tag}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            count += 1
    return {"tools": len(tools), "noise_levels": len(noise_sigmas), "params_files": count}
```

- [ ] **Step 4: Create budget template**

Create `benchmark-control/compliance/templates/budget_24h_13alg_3seed_3noise.json`:

```json
{
  "name": "formal24h_13alg_3seed_3noise",
  "seeds": [520, 521, 522],
  "noise_sigmas": [0.0, 0.01, 0.05],
  "timeout_in_seconds": 86400,
  "min_runtime_seconds": 82800,
  "progress_snapshot_interval_seconds": 60,
  "total_algorithms": 13,
  "total_datasets": 50,
  "total_tasks": 5850
}
```

- [ ] **Step 5: Run params tests**

Run:

```bash
pytest tests/test_benchmark_compliance_params.py -q
```

Expected: pass.

---

### Task 4: Scheduler Noise Dimension

**Files:**
- Modify: `check/run_e1_candidate200_12alg_load_queue.py`
- Modify: `tests/test_load_queue_scheduler.py`

- [ ] **Step 1: Add failing scheduler test**

Append to `tests/test_load_queue_scheduler.py`:

```python
def test_build_tasks_includes_noise_dimension_in_task_ids(tmp_path: Path) -> None:
    scheduler = _load_scheduler()
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
```

- [ ] **Step 2: Run failing scheduler test**

Run:

```bash
pytest tests/test_load_queue_scheduler.py::test_build_tasks_includes_noise_dimension_in_task_ids -q
```

Expected: fail because `_build_tasks()` has no `noise_sigmas` parameter.

- [ ] **Step 3: Extend `QueueTask`**

Add fields:

```python
noise_tag: str = "clean"
noise_sigma: float = 0.0
```

- [ ] **Step 4: Extend `_build_tasks()`**

Add parameter:

```python
noise_sigmas: list[float] | None = None
```

Loop order:

```python
for seed in seeds:
    for noise_sigma in noise_sigmas or [0.0]:
        noise_tag = _noise_tag_for_sigma(noise_sigma)
        for tool in tools:
            ...
```

Task ID:

```python
task_id = f"{tool}_s{seed}_{noise_tag}_g{int(global_index):04d}"
```

Params name for non-LLM tools:

```python
params_name = f"{base_params_name}__{noise_tag}"
```

This formal24h batch excludes LLM tools, but preserve existing LLM logic for Stage1 by only appending noise suffix when `noise_tag != "clean"` or when `--params-root` contains suffixed files.

- [ ] **Step 5: Add CLI args**

Add:

```python
parser.add_argument("--noise-sigmas", nargs="*", type=float, default=[0.0])
```

Pass `args.noise_sigmas` into `_build_tasks()`.

- [ ] **Step 6: Extend state fields**

In `_initial_state()`, add:

```python
"noise_tag": task.noise_tag,
"noise_sigma": task.noise_sigma,
```

In `_load_or_init_state()`, validate those fields exactly like `params_name`.

- [ ] **Step 7: Run scheduler tests**

Run:

```bash
pytest tests/test_load_queue_scheduler.py -q
```

Expected: pass.

---

### Task 5: Noise-Aware Harvest And Audit

**Files:**
- Modify: `benchmark-control/compliance/lib/harvest.py`
- Modify: `benchmark-control/compliance/lib/audit.py`
- Modify: `benchmark-control/compliance/audit-schemas/task_audit.schema.json`
- Modify: `tests/test_benchmark_compliance_harvest.py`
- Modify: `tests/test_benchmark_compliance_audit.py`

- [ ] **Step 1: Add harvest test for noise path**

Append a test that writes a manifest task with:

```csv
task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds
pysr__seed520__noise001__d0,pysr,d0,sim-datasets-data/ssr50/d0,520,noise001,0.01,86400,82800,60
```

Expected harvest target:

```text
runs/pysr/seed520/noise001/d0/result.json
```

- [ ] **Step 2: Add audit test for noise path**

Append a test that creates:

```text
runs/pysr/seed520/noise001/d0/result.json
runs/pysr/seed520/noise001/d0/progress/minute_0001.json
```

Expected audit row includes:

```text
noise_tag=noise001
noise_sigma=0.01
```

- [ ] **Step 3: Run failing tests**

Run:

```bash
pytest tests/test_benchmark_compliance_harvest.py tests/test_benchmark_compliance_audit.py -q
```

Expected: fail because current paths omit noise.

- [ ] **Step 4: Modify harvest path**

When manifest row has `noise_tag`, harvest to:

```python
target_dir = batch_dir / "runs" / task["algorithm"] / f"seed{seed}" / task.get("noise_tag", "clean") / dataset_id
```

Derive scheduler task ID:

```python
noise_tag = task.get("noise_tag", "clean")
scheduler_task_id = f"{tool_key}_s{seed}_{noise_tag}_g{global_index:04d}"
```

Keep backward compatibility: if `noise_tag` column is absent, use old `f"{tool_key}_s{seed}_g{global_index:04d}"`.

- [ ] **Step 5: Modify audit path and fields**

Add to `TASK_AUDIT_FIELDS` after `seed`:

```python
"noise_tag",
"noise_sigma",
```

Audit path:

```python
noise_tag = task.get("noise_tag", "clean")
task_dir = batch_dir / "runs" / task["algorithm"] / f"seed{task['seed']}" / noise_tag / task["dataset_id"]
```

Backward compatibility: if no `noise_tag` in manifest, use old path.

- [ ] **Step 6: Update audit schema**

Add columns:

```json
"noise_tag",
"noise_sigma"
```

- [ ] **Step 7: Run harvest and audit tests**

Run:

```bash
pytest tests/test_benchmark_compliance_harvest.py tests/test_benchmark_compliance_audit.py -q
```

Expected: pass.

---

### Task 6: Formal24h Readiness Gate

**Files:**
- Modify: `benchmark-control/compliance/lib/readiness.py`
- Modify: `benchmark-control/compliance/launchers/check_stage1_readiness.py`
- Modify: `tests/test_benchmark_compliance_readiness.py`

- [ ] **Step 1: Add failing readiness test**

Append a test that creates a formal24h batch with:

- 5850 tasks.
- 13 algorithms.
- 50 datasets.
- 3 seeds.
- 3 noise levels.
- timeout `86400`.
- min runtime `82800`.
- progress interval `60`.
- 39 generated params files.

Expected:

```python
assert summary["ready"] is True
assert summary["total_tasks"] == 5850
assert summary["total_algorithms"] == 13
assert summary["total_noise_levels"] == 3
```

- [ ] **Step 2: Run failing readiness test**

Run:

```bash
pytest tests/test_benchmark_compliance_readiness.py::test_readiness_passes_for_formal24h_batch -q
```

Expected: fail because readiness is hard-coded to 750/15/seed520/3600.

- [ ] **Step 3: Add profile-aware readiness**

Add CLI argument:

```python
parser.add_argument("--profile", choices=["stage1_1h", "formal24h_13alg_3seed_3noise"], default="stage1_1h")
```

For `formal24h_13alg_3seed_3noise`, expected values:

```python
expected_tasks = 5850
expected_algorithms = 13
expected_seeds = {"520", "521", "522"}
expected_noise_tags = {"clean", "noise001", "noise005"}
expected_timeout = {"86400"}
expected_min_runtime = {"82800"}
expected_interval = {"60"}
```

- [ ] **Step 4: Validate generated params**

For each selected algorithm and noise tag, require:

```text
params/<algorithm>__<noise_tag>.json
```

Each file must contain:

```json
"timeout_in_seconds": 86400
"progress_snapshot_interval_seconds": 60
"train_label_noise_sigma": <sigma>
```

- [ ] **Step 5: Run readiness tests**

Run:

```bash
pytest tests/test_benchmark_compliance_readiness.py -q
```

Expected: pass.

---

### Task 7: Formal24h Queue Command Generator

**Files:**
- Create: `benchmark-control/compliance/launchers/write_full24h_queue_commands.py`
- Create: `tests/test_benchmark_compliance_full24h_commands.py`
- Modify: `benchmark-control/compliance/README.md`

- [ ] **Step 1: Write command generator test**

Create `tests/test_benchmark_compliance_full24h_commands.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_launcher():
    path = Path("benchmark-control/compliance/launchers/write_full24h_queue_commands.py")
    spec = importlib.util.spec_from_file_location("full24h_commands", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_full24h_queue_commands_use_13_tools_3_seeds_3_noise(tmp_path: Path) -> None:
    launcher = _load_launcher()
    batch_dir = tmp_path / "benchmark-runs" / "formal24h" / "batch"
    paths = launcher.write_full24h_queue_commands(batch_dir=batch_dir)
    assert {path.name for path in paths} == {
        "01_preflight_from_iaaccn22.sh",
        "02_smoke_dispatch_from_iaaccn22.sh",
        "03_full_dispatch_from_iaaccn22.sh",
    }

    full = (batch_dir / "deploy" / "03_full_dispatch_from_iaaccn22.sh").read_text(encoding="utf-8")
    assert "--tools gplearn pyoperon pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit" in full
    assert "--seeds 520 521 522" in full
    assert "--noise-sigmas 0 0.01 0.05" in full
    assert "--expected-total-tasks 5850" in full
    assert "compliance_1h_" not in full
```

- [ ] **Step 2: Run failing test**

Run:

```bash
pytest tests/test_benchmark_compliance_full24h_commands.py -q
```

Expected: fail because launcher does not exist.

- [ ] **Step 3: Implement command generator**

Create the launcher by adapting `write_stage1_queue_commands.py` with these constants:

```python
TOOLS = "gplearn pyoperon pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit"
HOSTS = "iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29"
SEEDS = "520 521 522"
NOISE_SIGMAS = "0 0.01 0.05"
EXPECTED_SMOKE_TASKS = 234
EXPECTED_FULL_TASKS = 5850
```

Use:

```text
benchmark-runs/formal24h/latest
```

Use params root:

```text
benchmark-runs/formal24h/latest/params
```

For smoke dispatch, use the separate short-budget params root:

```text
benchmark-runs/formal24h/latest/params_smoke
```

Use session prefixes:

```text
formal24h_smoke_
formal24h_full_
```

- [ ] **Step 4: Add README section**

Document safe commands:

```bash
BATCH_ID="formal24h_13alg_ssr50_seed520-522_noise0-001-005_$(date +%Y%m%d-%H%M%S)"
mkdir -p "benchmark-runs/formal24h/${BATCH_ID}"
ln -sfn "${BATCH_ID}" benchmark-runs/formal24h/latest

python benchmark-control/compliance/launchers/prepare_batch.py \
  --profile formal24h_13alg_3seed_3noise \
  --toolbox-config scientific_intelligent_modelling/config/toolbox_config.json \
  --ssr50-root sim-datasets-data/ssr50 \
  --batch-dir "benchmark-runs/formal24h/${BATCH_ID}"

python benchmark-control/compliance/launchers/write_full24h_queue_commands.py \
  --batch-dir benchmark-runs/formal24h/latest

python benchmark-control/compliance/launchers/check_stage1_readiness.py \
  --profile formal24h_13alg_3seed_3noise \
  --batch-dir benchmark-runs/formal24h/latest
```

- [ ] **Step 5: Run command tests**

Run:

```bash
pytest tests/test_benchmark_compliance_full24h_commands.py -q
```

Expected: pass.

---

### Task 8: Local Verification Without Starting Experiments

**Files:**
- No new files.
- Validation only.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest \
  tests/test_benchmark_compliance_manifest.py \
  tests/test_benchmark_compliance_params.py \
  tests/test_load_queue_scheduler.py \
  tests/test_benchmark_compliance_harvest.py \
  tests/test_benchmark_compliance_audit.py \
  tests/test_benchmark_compliance_readiness.py \
  tests/test_benchmark_compliance_full24h_commands.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Generate a local formal24h batch**

Run:

```bash
BATCH_ID="formal24h_13alg_ssr50_seed520-522_noise0-001-005_localcheck"
rm -rf "benchmark-runs/formal24h/${BATCH_ID}"
mkdir -p "benchmark-runs/formal24h/${BATCH_ID}"
ln -sfn "${BATCH_ID}" benchmark-runs/formal24h/latest

python benchmark-control/compliance/launchers/prepare_batch.py \
  --profile formal24h_13alg_3seed_3noise \
  --toolbox-config scientific_intelligent_modelling/config/toolbox_config.json \
  --ssr50-root sim-datasets-data/ssr50 \
  --batch-dir "benchmark-runs/formal24h/${BATCH_ID}"

python benchmark-control/compliance/launchers/write_full24h_queue_commands.py \
  --batch-dir benchmark-runs/formal24h/latest

python benchmark-control/compliance/launchers/check_stage1_readiness.py \
  --profile formal24h_13alg_3seed_3noise \
  --batch-dir benchmark-runs/formal24h/latest
```

Expected:

```text
ready=true
total_tasks=5850
total_algorithms=13
total_datasets=50
```

- [ ] **Step 3: Confirm no experiment started**

Run:

```bash
find benchmark-runs/formal24h/latest -maxdepth 3 -type d | sort | sed -n '1,120p'
```

Expected: only control-plane directories such as `manifest`, `params`, `queues`, `deploy`, and `readiness`; no `runs` with real result artifacts and no remote `experiments` collection.

- [ ] **Step 4: Check Git ignores run artifacts**

Run:

```bash
git status --short benchmark-runs/formal24h
```

Expected: no tracked output.

---

### Task 9: Commit Implementation

**Files:**
- All modified control-plane files and tests from previous tasks.

- [ ] **Step 1: Review diff**

Run:

```bash
git diff -- benchmark-control check tests docs
```

Expected: only control-plane, scheduler, tests, and docs changes; no `benchmark-runs/` artifacts.

- [ ] **Step 2: Commit**

Run:

```bash
git add benchmark-control check tests docs
git commit -m "[update] 支持24小时三噪声正式实验控制面" -m "Co-Authored-By: lilmortyj <781113402@qq.com>"
```

Expected: commit succeeds.

---

## Self-Review

- Spec coverage: covers 13 algorithms, SSR50, seeds `520/521/522`, noise `0/0.01/0.05`, 24h timeout, 60s progress, manifest, params, scheduler, harvest, audit, readiness, smoke/full gates, and no full launch during implementation.
- Placeholder scan: no task depends on an unspecified future file or manual hand edit.
- Type consistency: `noise_tag` uses `clean/noise001/noise005`; `noise_sigma` remains float in JSON and string in CSV.
- Backward compatibility: Stage1 1h defaults remain available and existing tests must continue passing.
