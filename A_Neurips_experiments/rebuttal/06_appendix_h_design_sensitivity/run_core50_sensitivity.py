#!/usr/bin/env python3
"""Reproduce Appendix H Core50 joint-weight sensitivity when the baseline is replayable.

The historical Core50 selector is not available in this repository.  This runner therefore
has a deliberately strict first gate: the default (0.45, 0.35, 0.20) selector replay must
produce exactly the frozen 50-task manifest.  A failed gate writes only ``audit.json`` and
never fabricates a 5000-configuration result.

When the gate passes, the runner samples each of the three weights independently from its
prespecified +/- 0.05 interval, renormalizes every row, and replays the same selector for
5000 configurations.  The append-only result files are transactionally ordered as
membership rows first, then a configuration row, then a checkpoint.  On restart, incomplete
configurations are discarded and rerun, so a killed process cannot silently inflate a count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check import analyze_core50_membership_sensitivity as selector  # noqa: E402


PROBE4 = selector.PROBE4
DEFAULT_WEIGHTS = (0.45, 0.35, 0.20)
WEIGHT_DELTA = 0.05
DEFAULT_N_CONFIGS = 5000
DEFAULT_SUBSET_SIZE = 50
DEFAULT_RESERVOIR_SIZE = 664
DEFAULT_SEED = 20260910
DEFAULT_RESTARTS = 6
DEFAULT_GRID_RESTARTS = 2

RAW_WEIGHT_COLUMNS = (
    "raw_w_coverage",
    "raw_w_mean_info",
    "raw_w_balance",
)
NORMALIZED_WEIGHT_COLUMNS = (
    "w_coverage",
    "w_mean_info",
    "w_balance",
)

AUDIT_FILE = "audit.json"
CHECKPOINT_FILE = "checkpoint.json"
WEIGHTS_FILE = "joint_weights.csv"
CONFIG_RESULTS_FILE = "joint_config_results.csv"
MEMBERSHIPS_FILE = "selected_memberships.csv"
FREQUENCY_FILE = "membership_frequency.csv"
MANIFEST_FILE = "core50_manifest.csv"


class GateBlocked(RuntimeError):
    """Raised internally when a required reproducibility gate fails."""


class CheckpointMismatch(RuntimeError):
    """Raised when an existing partial result does not match this invocation."""


@dataclass(frozen=True)
class PreparedInputs:
    dataset_level: pd.DataFrame
    dataset_algorithm: pd.DataFrame
    core_manifest: pd.DataFrame
    problem: Any
    frozen_ids: set[str]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, header=not path.exists() or path.stat().st_size == 0)
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_joint_weights(
    n_configs: int = DEFAULT_N_CONFIGS,
    seed: int = DEFAULT_SEED,
    defaults: Iterable[float] = DEFAULT_WEIGHTS,
    delta: float = WEIGHT_DELTA,
) -> pd.DataFrame:
    """Sample independent local weights and normalize each configuration row.

    ``raw_w_*`` are retained as evidence of the actual independent draws.  The ``w_*``
    columns are the only weights passed to the selector.
    """
    if n_configs <= 0:
        raise ValueError("n_configs must be positive")
    defaults_array = np.asarray(tuple(defaults), dtype=float)
    if defaults_array.shape != (3,) or np.any(defaults_array <= 0):
        raise ValueError("defaults must contain three positive weights")
    if delta < 0 or np.any(defaults_array - delta <= 0):
        raise ValueError("delta must keep all raw weights positive")

    rng = np.random.default_rng(seed)
    raw = rng.uniform(
        defaults_array - delta,
        defaults_array + delta,
        size=(n_configs, 3),
    )
    normalized = raw / raw.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(raw, columns=RAW_WEIGHT_COLUMNS)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "config_index": np.arange(n_configs, dtype=int),
                    "config_id": [f"joint_{i:04d}" for i in range(n_configs)],
                    "sampling_seed": int(seed),
                    "sampling_delta": float(delta),
                    "raw_weight_sum": raw.sum(axis=1),
                }
            ),
            frame,
            pd.DataFrame(normalized, columns=NORMALIZED_WEIGHT_COLUMNS),
        ],
        axis=1,
    )
    return frame


def _args_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def load_prepared_inputs(
    dataset_level_path: Path,
    dataset_algorithm_path: Path,
    core_manifest_path: Path,
) -> PreparedInputs:
    """Load and normalize the exact inputs consumed by the shared selector."""
    for path in (dataset_level_path, dataset_algorithm_path, core_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_level_raw = pd.read_csv(dataset_level_path)
    dataset_algorithm = pd.read_csv(dataset_algorithm_path)
    core_manifest = pd.read_csv(core_manifest_path)
    dataset_level = selector.prepare_dataset_table(dataset_level_raw, dataset_algorithm)
    frozen_ids = selector.core_ids_from_manifest(dataset_level, core_manifest)
    problem = selector.SelectionProblem(dataset_level)
    return PreparedInputs(
        dataset_level=dataset_level,
        dataset_algorithm=dataset_algorithm,
        core_manifest=core_manifest,
        problem=problem,
        frozen_ids=frozen_ids,
    )


def _build_starts(
    problem: Any,
    frozen_ids: set[str],
    constraints: Any,
    seed: int,
    restarts: int,
) -> list[np.ndarray]:
    if restarts < 1:
        raise ValueError("restarts must be at least one")
    frozen = problem.indices(frozen_ids)
    if not problem.is_feasible(frozen, constraints):
        raise GateBlocked("frozen Core50 is infeasible under nominal constraints")
    starts: list[np.ndarray] = [frozen]
    rng = np.random.default_rng(seed)
    for _ in range(restarts - 1):
        starts.append(problem.construct_feasible(constraints, rng))
    return selector.deduplicate_starts(starts)


def select_for_objective(
    problem: Any,
    starts: Iterable[np.ndarray],
    objective: Any,
    constraints: Any,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Small indirection to make the selector transaction easy to test."""
    return selector.choose_best(problem, starts, objective, constraints)


def _selection_ids(problem: Any, selected: Iterable[int]) -> set[str]:
    return problem.ids(selected)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _baseline_gate(
    prepared: PreparedInputs,
    seed: int,
    restarts: int,
    subset_size: int,
) -> tuple[dict[str, Any], list[np.ndarray], Any, Any]:
    objective = selector.ObjectiveSpec(*DEFAULT_WEIGHTS).normalized()
    constraints = selector.ConstraintSpec()
    if len(prepared.frozen_ids) != subset_size:
        raise GateBlocked(
            f"frozen manifest has {len(prepared.frozen_ids)} tasks, expected {subset_size}"
        )
    starts = _build_starts(prepared.problem, prepared.frozen_ids, constraints, seed, restarts)
    selected, history, best_start = select_for_objective(
        prepared.problem,
        starts,
        objective,
        constraints,
    )
    selected_ids = _selection_ids(prepared.problem, selected)
    if len(selected_ids) != subset_size:
        raise GateBlocked(f"default selector returned {len(selected_ids)} tasks")
    if not prepared.problem.is_feasible(selected, constraints):
        raise GateBlocked("default selector returned an infeasible subset")
    terms = prepared.problem.objective_terms(selected, objective)
    overlap = len(selected_ids & prepared.frozen_ids)
    baseline = {
        "baseline_gate": (
            "pass_exact_reproduction"
            if selected_ids == prepared.frozen_ids
            else "fail_exact_reproduction"
        ),
        "baseline_frozen_overlap": overlap,
        "baseline_frozen_jaccard": _jaccard(selected_ids, prepared.frozen_ids),
        "baseline_selected_count": len(selected_ids),
        "baseline_objective": terms["objective"],
        "baseline_coverage": terms["coverage"],
        "baseline_mean_information": terms["mean_information"],
        "baseline_balance": terms["balance"],
        "baseline_best_start_index": best_start,
        "baseline_unique_starts": len(starts),
        "baseline_accepted_swaps": len(history),
        "baseline_selected_ids": sorted(selected_ids),
    }
    return baseline, starts, objective, constraints


def _rank_metrics(
    dataset_algorithm: pd.DataFrame,
    all_ids: set[str],
    selected_ids: set[str],
) -> dict[str, Any]:
    required = {"median_log_id_nmse", "median_log_ood_nmse"}
    if not required.issubset(dataset_algorithm.columns):
        return {}
    full_scores = selector.method_scores(dataset_algorithm, all_ids)
    subset_scores = selector.method_scores(dataset_algorithm, selected_ids)
    return selector.ranking_metrics(full_scores, subset_scores)


def _config_result(
    prepared: PreparedInputs,
    weight_row: pd.Series,
    starts: list[np.ndarray],
    constraints: Any,
    subset_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    objective = selector.ObjectiveSpec(
        float(weight_row["w_coverage"]),
        float(weight_row["w_mean_info"]),
        float(weight_row["w_balance"]),
    ).normalized()
    selected, history, best_start = select_for_objective(
        prepared.problem,
        starts,
        objective,
        constraints,
    )
    selected_ids = _selection_ids(prepared.problem, selected)
    if len(selected_ids) != subset_size or not prepared.problem.is_feasible(selected, constraints):
        raise GateBlocked("sampled configuration returned an invalid Core50 subset")
    terms = prepared.problem.objective_terms(selected, objective)
    ids_sorted = sorted(selected_ids)
    selected_frame = prepared.dataset_level.set_index("dataset_id").loc[ids_sorted].reset_index()
    membership = selected_frame[
        [
            "dataset_id",
            "dataset_name",
            "dataset_rel",
            "family",
            "subgroup",
            "basename",
            "difficulty_bin",
            "failure_mode",
            "info_score",
        ]
    ].copy()
    membership.insert(0, "selection_rank", np.arange(1, len(membership) + 1))
    membership.insert(0, "config_id", str(weight_row["config_id"]))
    membership.insert(1, "config_index", int(weight_row["config_index"]))
    membership["is_frozen_core50"] = membership["dataset_id"].isin(prepared.frozen_ids)
    result: dict[str, Any] = {
        "config_id": str(weight_row["config_id"]),
        "config_index": int(weight_row["config_index"]),
        "status": "ok",
        "selected_count": len(selected_ids),
        "feasible": True,
        "overlap_with_frozen": len(selected_ids & prepared.frozen_ids),
        "jaccard_with_frozen": _jaccard(selected_ids, prepared.frozen_ids),
        "objective": terms["objective"],
        "coverage": terms["coverage"],
        "structural_coverage": terms["structural_coverage"],
        "response_coverage": terms["response_coverage"],
        "mean_information": terms["mean_information"],
        "balance": terms["balance"],
        "structural_balance": terms["structural_balance"],
        "response_balance": terms["response_balance"],
        "best_start_index": int(best_start),
        "accepted_swaps": len(history),
    }
    result.update(
        {
            column: float(weight_row[column])
            for column in RAW_WEIGHT_COLUMNS + NORMALIZED_WEIGHT_COLUMNS
        }
    )
    result.update(_rank_metrics(prepared.dataset_algorithm, set(prepared.dataset_level["dataset_id"]), selected_ids))
    membership["w_coverage"] = float(weight_row["w_coverage"])
    membership["w_mean_info"] = float(weight_row["w_mean_info"])
    membership["w_balance"] = float(weight_row["w_balance"])
    return result, membership


def repair_checkpoint(output: Path, subset_size: int = DEFAULT_SUBSET_SIZE) -> set[str]:
    """Remove rows left by a killed transaction and return complete config IDs."""
    result_path = output / CONFIG_RESULTS_FILE
    membership_path = output / MEMBERSHIPS_FILE
    if not result_path.exists() and not membership_path.exists():
        return set()

    if result_path.exists():
        results = pd.read_csv(result_path)
    else:
        results = pd.DataFrame(columns=["config_id", "status"])
    if membership_path.exists():
        memberships = pd.read_csv(membership_path)
    else:
        memberships = pd.DataFrame(columns=["config_id", "dataset_id"])
    if "config_id" not in results or "status" not in results:
        raise CheckpointMismatch("config result checkpoint lacks config_id/status")
    if "config_id" not in memberships or "dataset_id" not in memberships:
        raise CheckpointMismatch("membership checkpoint lacks config_id/dataset_id")

    results = results.drop_duplicates("config_id", keep="last")
    candidate_ids = set(results.loc[results["status"].eq("ok"), "config_id"].astype(str))
    memberships = memberships.copy()
    memberships["config_id"] = memberships["config_id"].astype(str)
    memberships["dataset_id"] = memberships["dataset_id"].astype(str)
    counts = memberships.groupby("config_id").agg(
        rows=("dataset_id", "size"), unique_tasks=("dataset_id", "nunique")
    )
    complete = {
        config_id
        for config_id in candidate_ids
        if config_id in counts.index
        and int(counts.loc[config_id, "rows"]) == subset_size
        and int(counts.loc[config_id, "unique_tasks"]) == subset_size
    }
    if not complete:
        clean_results = results.iloc[0:0].copy()
        clean_memberships = memberships.iloc[0:0].copy()
    else:
        clean_results = results[results["config_id"].astype(str).isin(complete)].copy()
        clean_memberships = memberships[memberships["config_id"].isin(complete)].copy()
    _write_csv(result_path, clean_results)
    _write_csv(membership_path, clean_memberships)
    return complete


def _validate_weights_file(
    path: Path,
    expected: pd.DataFrame,
    delta: float,
) -> pd.DataFrame:
    if not path.exists():
        _write_csv(path, expected)
        return expected
    current = pd.read_csv(path)
    if list(current.columns) != list(expected.columns) or len(current) != len(expected):
        raise CheckpointMismatch("joint weight checkpoint shape/columns mismatch")
    if current["config_id"].astype(str).tolist() != expected["config_id"].tolist():
        raise CheckpointMismatch("joint weight checkpoint config IDs mismatch")
    for column in RAW_WEIGHT_COLUMNS:
        defaults = DEFAULT_WEIGHTS[RAW_WEIGHT_COLUMNS.index(column)]
        values = pd.to_numeric(current[column], errors="coerce").to_numpy()
        if np.any(~np.isfinite(values)) or np.any(values < defaults - delta - 1e-12) or np.any(values > defaults + delta + 1e-12):
            raise CheckpointMismatch(f"joint weight checkpoint violates {column} bounds")
    normalized = current[list(NORMALIZED_WEIGHT_COLUMNS)].to_numpy(dtype=float)
    if np.any(~np.isfinite(normalized)) or not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-12):
        raise CheckpointMismatch("joint weight checkpoint is not row-normalized")
    if not np.allclose(
        current[list(NORMALIZED_WEIGHT_COLUMNS)].to_numpy(dtype=float),
        current[list(RAW_WEIGHT_COLUMNS)].to_numpy(dtype=float)
        / current["raw_weight_sum"].to_numpy(dtype=float)[:, None],
        atol=1e-12,
    ):
        raise CheckpointMismatch("joint weight checkpoint normalization is inconsistent")
    return current


def build_membership_frequency(
    dataset_level: pd.DataFrame,
    memberships: pd.DataFrame,
    n_configs: int,
) -> pd.DataFrame:
    if len(memberships) != n_configs * DEFAULT_SUBSET_SIZE:
        raise ValueError("membership row count does not match the 5000 x 50 contract")
    counts = memberships.groupby("dataset_id")["config_id"].nunique()
    output = dataset_level[
        [
            "dataset_id",
            "dataset_name",
            "dataset_rel",
            "family",
            "subgroup",
            "difficulty_bin",
            "failure_mode",
        ]
    ].copy()
    output["selection_count"] = output["dataset_id"].map(counts).fillna(0).astype(int)
    output["selection_frequency"] = output["selection_count"] / float(n_configs)
    output["selected_every_config"] = output["selection_count"].eq(n_configs)
    output["selected_at_least_75pct"] = output["selection_frequency"].ge(0.75)
    return output.sort_values(
        ["selection_frequency", "dataset_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _base_audit(
    args: argparse.Namespace,
    prepared: PreparedInputs | None = None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "runner": "appendix_h_core50_joint_weight_sensitivity_v1",
        "status": "starting",
        "baseline_gate": "not_run",
        "dataset_level": str(_args_value(args, "dataset_level", "")),
        "dataset_algorithm": str(_args_value(args, "dataset_algorithm", "")),
        "core_manifest": str(_args_value(args, "core_manifest", "")),
        "output": str(_args_value(args, "output", "")),
        "seed": int(_args_value(args, "seed", DEFAULT_SEED)),
        "delta": float(_args_value(args, "delta", WEIGHT_DELTA)),
        "default_weights": list(DEFAULT_WEIGHTS),
        "n_configurations_requested": int(_args_value(args, "n_configs", DEFAULT_N_CONFIGS)),
        "subset_size": int(_args_value(args, "subset_size", DEFAULT_SUBSET_SIZE)),
        "expected_reservoir_size": int(
            _args_value(args, "expected_reservoir_size", DEFAULT_RESERVOIR_SIZE)
        ),
        "selector_module": str(Path(selector.__file__).resolve()),
        "outputs_written": [],
    }
    if prepared is not None:
        audit.update(
            {
                "dataset_count": len(prepared.dataset_level),
                "dataset_algorithm_rows": len(prepared.dataset_algorithm),
                "frozen_core_count": len(prepared.frozen_ids),
            }
        )
    return audit


def _source_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    return {
        name: sha256(Path(getattr(args, name)))
        for name in ("dataset_level", "dataset_algorithm", "core_manifest")
    }


def _checkpoint_metadata(args: argparse.Namespace, fingerprints: dict[str, str]) -> dict[str, Any]:
    return {
        "runner": "appendix_h_core50_joint_weight_sensitivity_v1",
        "seed": int(args.seed),
        "delta": float(args.delta),
        "default_weights": list(DEFAULT_WEIGHTS),
        "n_configs": int(args.n_configs),
        "subset_size": int(args.subset_size),
        "fingerprints": fingerprints,
    }


def _validate_existing_checkpoint(
    output: Path,
    args: argparse.Namespace,
    fingerprints: dict[str, str],
) -> None:
    checkpoint = output / CHECKPOINT_FILE
    if not checkpoint.exists():
        return
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    expected = _checkpoint_metadata(args, fingerprints)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CheckpointMismatch(f"checkpoint metadata mismatch at {key}")


def _write_frequency_and_finalize(
    output: Path,
    prepared: PreparedInputs,
    results: pd.DataFrame,
    n_configs: int,
    audit: dict[str, Any],
) -> dict[str, Any]:
    memberships = pd.read_csv(output / MEMBERSHIPS_FILE)
    if len(results) != n_configs or results["config_id"].nunique() != n_configs:
        raise CheckpointMismatch("final config result count is not complete")
    if not results["status"].astype(str).eq("ok").all():
        raise CheckpointMismatch("final config results contain a non-ok status")
    if len(memberships) != n_configs * DEFAULT_SUBSET_SIZE:
        raise CheckpointMismatch("final membership count is not 250000")
    counts = memberships.groupby("config_id").size()
    if len(counts) != n_configs or not counts.eq(DEFAULT_SUBSET_SIZE).all():
        raise CheckpointMismatch("some final configurations do not have 50 members")
    frequency = build_membership_frequency(prepared.dataset_level, memberships, n_configs)
    _write_csv(output / FREQUENCY_FILE, frequency)
    _write_csv(output / MANIFEST_FILE, prepared.core_manifest)

    overlaps = pd.to_numeric(results["overlap_with_frozen"], errors="coerce")
    jaccards = pd.to_numeric(results["jaccard_with_frozen"], errors="coerce")
    audit.update(
        {
            "status": "complete",
            "baseline_gate": "pass_exact_reproduction",
            "configurations": int(len(results)),
            "membership_rows": int(len(memberships)),
            "frequency_rows": int(len(frequency)),
            "manifest_rows": int(len(prepared.core_manifest)),
            "jaccard": {
                "mean": float(jaccards.mean()),
                "min": float(jaccards.min()),
                "max": float(jaccards.max()),
                "overlap_min": int(overlaps.min()),
                "overlap_max": int(overlaps.max()),
            },
            "selection_frequency": {
                "tasks_selected_every_config": int(frequency["selected_every_config"].sum()),
                "tasks_selected_at_least_75pct": int(frequency["selected_at_least_75pct"].sum()),
            },
            "outputs_written": [
                WEIGHTS_FILE,
                CONFIG_RESULTS_FILE,
                MEMBERSHIPS_FILE,
                FREQUENCY_FILE,
                MANIFEST_FILE,
                CHECKPOINT_FILE,
                AUDIT_FILE,
            ],
        }
    )
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit = _base_audit(args)
    _write_json(output / AUDIT_FILE, audit)

    try:
        if int(args.subset_size) != DEFAULT_SUBSET_SIZE:
            raise GateBlocked("Appendix H requires subset_size=50")
        if float(args.delta) != WEIGHT_DELTA:
            raise GateBlocked("Appendix H requires an independent +/-0.05 perturbation")
        prepared = load_prepared_inputs(args.dataset_level, args.dataset_algorithm, args.core_manifest)
        audit.update(
            {
                "dataset_count": len(prepared.dataset_level),
                "dataset_algorithm_rows": len(prepared.dataset_algorithm),
                "frozen_core_count": len(prepared.frozen_ids),
            }
        )
        if len(prepared.dataset_level) != int(args.expected_reservoir_size):
            raise GateBlocked(
                f"reservoir has {len(prepared.dataset_level)} tasks, expected {args.expected_reservoir_size}"
            )
        if len(prepared.dataset_algorithm) != len(prepared.dataset_level) * len(selector.PROBE4):
            raise GateBlocked("dataset-algorithm input does not contain the complete Probe-4 panel")
        fingerprints = _source_fingerprints(args)
        audit["source_sha256"] = fingerprints
        _validate_existing_checkpoint(output, args, fingerprints)
        baseline, starts, _, constraints = _baseline_gate(
            prepared,
            int(args.seed),
            int(args.restarts),
            int(args.subset_size),
        )
        audit.update(baseline)
        _write_json(output / AUDIT_FILE, audit)
        if baseline["baseline_gate"] != "pass_exact_reproduction":
            raise GateBlocked("default selector does not reproduce frozen Core50 exactly")

        expected_weights = sample_joint_weights(
            int(args.n_configs), int(args.seed), DEFAULT_WEIGHTS, float(args.delta)
        )
        weights = _validate_weights_file(output / WEIGHTS_FILE, expected_weights, float(args.delta))
        completed = (
            repair_checkpoint(output, int(args.subset_size))
            if bool(_args_value(args, "resume", True))
            else set()
        )
        unknown = completed - set(weights["config_id"].astype(str))
        if unknown:
            raise CheckpointMismatch(f"checkpoint contains unknown configuration IDs: {sorted(unknown)[:3]}")

        for _, weight_row in weights.iterrows():
            config_id = str(weight_row["config_id"])
            if config_id in completed:
                continue
            result, membership = _config_result(
                prepared,
                weight_row,
                starts,
                constraints,
                int(args.subset_size),
            )
            # The config row is the commit marker: repair_checkpoint only trusts it when
            # the preceding 50 membership rows are present and unique.
            _append_csv(output / MEMBERSHIPS_FILE, membership)
            _append_csv(output / CONFIG_RESULTS_FILE, pd.DataFrame([result]))
            completed.add(config_id)
            _write_json(
                output / CHECKPOINT_FILE,
                {
                    **_checkpoint_metadata(args, fingerprints),
                    "status": "running",
                    "completed_configurations": len(completed),
                    "membership_rows": len(completed) * int(args.subset_size),
                },
            )

        results = pd.read_csv(output / CONFIG_RESULTS_FILE)
        final = _write_frequency_and_finalize(
            output,
            prepared,
            results,
            int(args.n_configs),
            audit,
        )
        _write_json(
            output / CHECKPOINT_FILE,
            {
                **_checkpoint_metadata(args, fingerprints),
                "status": "complete",
                "completed_configurations": int(args.n_configs),
                "membership_rows": int(args.n_configs) * int(args.subset_size),
            },
        )
        _write_json(output / AUDIT_FILE, final)
        return final
    except (GateBlocked, CheckpointMismatch, FileNotFoundError, ValueError, KeyError) as exc:
        audit.update(
            {
                "status": "blocked",
                "baseline_gate": audit.get("baseline_gate", "not_run"),
                "failure": str(exc),
                "failure_type": type(exc).__name__,
            }
        )
        if "baseline_frozen_overlap" not in audit:
            audit["baseline_frozen_overlap"] = None
        _write_json(output / AUDIT_FILE, audit)
        return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Appendix H Core50 joint weight sensitivity")
    parser.add_argument(
        "--dataset-level",
        type=Path,
        default=REPO_ROOT
        / "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508/probe4_postprocess_dataset_level.csv",
    )
    parser.add_argument(
        "--dataset-algorithm",
        type=Path,
        default=REPO_ROOT
        / "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508/probe4_postprocess_dataset_algorithm.csv",
    )
    parser.add_argument(
        "--core-manifest",
        type=Path,
        default=REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--delta", type=float, default=WEIGHT_DELTA)
    parser.add_argument("--n-configs", type=int, default=DEFAULT_N_CONFIGS)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--expected-reservoir-size", type=int, default=DEFAULT_RESERVOIR_SIZE)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--grid-restarts", type=int, default=DEFAULT_GRID_RESTARTS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = run(args)
    print(json.dumps(_json_safe(audit), ensure_ascii=False, indent=2))
    return 0 if audit.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
