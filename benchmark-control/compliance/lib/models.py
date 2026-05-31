from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


STAGE1_SPEC = ExperimentSpec(
    name="stage1_1h_compliance",
    algorithms=(
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
