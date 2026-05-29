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
