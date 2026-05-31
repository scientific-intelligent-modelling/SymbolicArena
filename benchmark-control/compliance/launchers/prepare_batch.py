from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from manifest import generate_manifest
from models import FULL24H_SPEC, STAGE1_SPEC
from params import generate_noise_params


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _parse_tools(raw: list[str] | None, *, profile: str) -> tuple[str, ...]:
    if raw:
        return tuple(raw)
    if profile == "formal24h_13alg_3seed_3noise":
        return FULL24H_SPEC.algorithms
    return STAGE1_SPEC.algorithms


def _parse_seeds(raw: list[int] | None, *, profile: str) -> tuple[int, ...]:
    if raw:
        return tuple(raw)
    if profile == "formal24h_13alg_3seed_3noise":
        return FULL24H_SPEC.seeds
    return STAGE1_SPEC.seeds


def _parse_noise_sigmas(raw: list[float] | None, *, profile: str) -> tuple[float, ...]:
    if raw:
        return tuple(float(value) for value in raw)
    if profile == "formal24h_13alg_3seed_3noise":
        return tuple(level.sigma for level in FULL24H_SPEC.noise_levels)
    return tuple(level.sigma for level in STAGE1_SPEC.noise_levels)


def _params_tool_name(tool: str) -> str:
    return {
        "QLattice": "qlattice",
        "iMCTS": "imcts",
    }.get(tool, tool)


def _budget_value(value: int | None, *, profile: str, field: str) -> int:
    if value is not None:
        return value
    spec = FULL24H_SPEC if profile == "formal24h_13alg_3seed_3noise" else STAGE1_SPEC
    return int(getattr(spec.budget, field))


def _write_queue_sources(*, batch_dir: Path, dataset_rows: list[dict[str, str]]) -> None:
    queue_dir = batch_dir / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "global_index": str(index),
            "dataset_id": row["dataset_id"],
            "dataset_name": row["dataset_id"],
            "dataset_dir": row["dataset_dir"],
            "dataset_rel": row["dataset_dir"],
        }
        for index, row in enumerate(dataset_rows, start=1)
    ]
    _write_csv(queue_dir / "ssr50_source.csv", rows)
    _write_csv(queue_dir / "smoke_2datasets_source.csv", rows[:2])


def _read_manifest_datasets(batch_dir: Path) -> list[dict[str, str]]:
    import csv

    with (batch_dir / "manifest" / "datasets.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    if not rows:
        raise ValueError(f"拒绝写空 CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["stage1_1h", "formal24h_13alg_3seed_3noise"], default="stage1_1h")
    parser.add_argument("--toolbox-config", default="scientific_intelligent_modelling/config/toolbox_config.json")
    parser.add_argument("--ssr50-root", default="sim-datasets-data/ssr50")
    parser.add_argument("--source-params-root", default="exp-planning/02.E1选择验证/generated/params")
    parser.add_argument("--tools", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--noise-sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--timeout-in-seconds", type=int, default=None)
    parser.add_argument("--min-runtime-seconds", type=int, default=None)
    parser.add_argument("--progress-snapshot-interval-seconds", type=int, default=None)
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    batch_dir = _resolve_repo_path(args.batch_dir)
    tools = _parse_tools(args.tools, profile=args.profile)
    seeds = _parse_seeds(args.seeds, profile=args.profile)
    noise_sigmas = _parse_noise_sigmas(args.noise_sigmas, profile=args.profile)
    timeout_in_seconds = _budget_value(args.timeout_in_seconds, profile=args.profile, field="timeout_in_seconds")
    min_runtime_seconds = _budget_value(args.min_runtime_seconds, profile=args.profile, field="min_runtime_seconds")
    progress_interval = _budget_value(
        args.progress_snapshot_interval_seconds,
        profile=args.profile,
        field="progress_snapshot_interval_seconds",
    )
    summary = generate_manifest(
        toolbox_config_path=_resolve_repo_path(args.toolbox_config),
        ssr50_root=_resolve_repo_path(args.ssr50_root),
        batch_dir=batch_dir,
        git_revision=_git_revision(),
        dataset_dir_base=ROOT,
        algorithms=tools if args.profile == "formal24h_13alg_3seed_3noise" else None,
        seeds=seeds,
        noise_sigmas=noise_sigmas,
        timeout_in_seconds=timeout_in_seconds,
        min_runtime_seconds=min_runtime_seconds,
        progress_snapshot_interval_seconds=progress_interval,
    )
    _write_queue_sources(batch_dir=batch_dir, dataset_rows=_read_manifest_datasets(batch_dir))
    if args.profile == "formal24h_13alg_3seed_3noise":
        params_tools = tuple(_params_tool_name(tool) for tool in tools)
        params_summary = generate_noise_params(
            source_params_root=_resolve_repo_path(args.source_params_root),
            output_params_root=batch_dir / "params",
            tools=params_tools,
            noise_sigmas=noise_sigmas,
            timeout_in_seconds=timeout_in_seconds,
            progress_snapshot_interval_seconds=progress_interval,
        )
        smoke_params_summary = generate_noise_params(
            source_params_root=_resolve_repo_path(args.source_params_root),
            output_params_root=batch_dir / "params_smoke",
            tools=params_tools,
            noise_sigmas=noise_sigmas,
            timeout_in_seconds=600,
            progress_snapshot_interval_seconds=progress_interval,
        )
        summary = {
            **summary,
            **params_summary,
            "smoke_params_files": smoke_params_summary["params_files"],
            "smoke_timeout_in_seconds": 600,
        }
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
