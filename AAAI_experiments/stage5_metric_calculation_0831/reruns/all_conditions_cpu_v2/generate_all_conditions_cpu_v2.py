#!/usr/bin/env python3
"""Generate one CPU-aware queue for the selected native-fidelity noise reruns.

The generator deliberately keeps the five native_fidelity_v1 allowlists as
frozen inputs and emits one merged allowlist.  It never starts a remote
controller; ``run_dry_run=True`` only invokes the queue scheduler's local
``--dry-run`` path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASSET_ROOT = Path(__file__).resolve().parent

NATIVE_ROOT = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/reruns/native_fidelity_v1"
)
FORMAL_ROOT = (
    REPO_ROOT
    / "benchmark-runs/formal3h/"
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658"
)
DRSR_FORMAL_ROOT = (
    REPO_ROOT
    / "benchmark-runs/drsr3h/"
    "drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355"
)
QUEUE_SCRIPT = REPO_ROOT / "check/run_e1_candidate200_12alg_load_queue.py"

SOURCE_CSV = FORMAL_ROOT / "queues/ssr50_source.csv"
DRSR_SOURCE_CSV = DRSR_FORMAL_ROOT / "queues/ssr50_source.csv"

HOSTS = tuple(f"iaaccn{number}" for number in range(22, 30))
TOOLS = ("jaxsr", "imcts", "drsr")
SEEDS = (520, 521, 522)
NOISE_SIGMAS = (0.01, 0.05)
TIMEOUT_SECONDS = 10800
SNAPSHOT_SECONDS = 60
MAX_CPU_USED_RATIO = 0.95
MAX_LOAD_RATIO = 0.95
MAX_MEMORY_USED_RATIO = 0.95
MAX_JOBS_PER_HOST = 18
MAX_NEW_JOBS_PER_HOST_PER_POLL = 18
LOAD_TIERS = "0.50:18,0.70:18,0.85:18,0.95:18"
EXPECTED_TOTAL = 143
EXPECTED_ALGORITHM_COUNTS = {"jaxsr": 63, "imcts": 79, "drsr": 1}
EXPECTED_CONDITION_COUNTS = {"noise001": 79, "noise005": 64}
BATCH_NAME = "all_conditions_cpu_v2_noise_formal"
SESSION_PREFIX = "all_conditions_cpu_v2_"

ALLOWLIST_SPECS = (
    ("jaxsr", "noise001", "jaxsr_noise001_task_allowlist.csv", 35),
    ("jaxsr", "noise005", "jaxsr_noise005_task_allowlist.csv", 28),
    ("imcts", "noise001", "imcts_noise001_task_allowlist.csv", 44),
    ("imcts", "noise005", "imcts_noise005_task_allowlist.csv", 35),
    ("drsr", "noise005", "drsr_noise005_task_allowlist.csv", 1),
)

PARAM_SPECS = (
    (FORMAL_ROOT / "params/jaxsr__noise001.json", "jaxsr__noise001.json"),
    (FORMAL_ROOT / "params/jaxsr__noise005.json", "jaxsr__noise005.json"),
    (FORMAL_ROOT / "params/imcts__noise001.json", "imcts__noise001.json"),
    (FORMAL_ROOT / "params/imcts__noise005.json", "imcts__noise005.json"),
    (DRSR_FORMAL_ROOT / "params/drsr__noise005.json", "drsr__noise005.json"),
    # The queue resolves DRSR's model bucket through its base formal file.
    (DRSR_FORMAL_ROOT / "params/drsr.json", "drsr.json"),
)

PRIMARY_ALLOWLIST_FIELDS = (
    "task_id",
    "algorithm",
    "dataset_id",
    "seed",
    "condition",
    "noise_sigma",
    "params_name",
    "input_csv_sha256",
    "selection_category",
    "selection_reason",
    "selection_threshold",
    "id_quality_delta_from_native",
    "ood_quality_delta_from_native",
    "valid_output",
    "invalid_reason",
    "replay_error",
    "source_allowlist",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames or not rows:
        raise ValueError(f"CSV 为空或没有表头: {path}")
    return rows, fieldnames


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"拒绝写空 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _copy_file(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"冻结输入不存在: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _sha256(destination)


def _noise_sigma(condition: str) -> float:
    return {"noise001": 0.01, "noise005": 0.05}[condition]


def _source_index(rows: list[dict[str, str]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "").strip()
        global_index = str(row.get("global_index") or "").strip()
        if not dataset_id or not global_index:
            raise ValueError("source CSV 缺少 dataset_id/global_index")
        if dataset_id in index or int(global_index) in index.values():
            raise ValueError("source CSV 的 dataset_id/global_index 不是一一对应")
        index[dataset_id] = int(global_index)
    if len(index) != 50:
        raise ValueError(f"formal source 应为 50 个任务，实际 {len(index)}")
    return index


def _validate_sources() -> tuple[list[dict[str, str]], dict[str, Any]]:
    source_rows, source_fields = _read_csv(SOURCE_CSV)
    drsr_rows, drsr_fields = _read_csv(DRSR_SOURCE_CSV)
    if source_rows != drsr_rows or source_fields != drsr_fields:
        raise ValueError("iMCTS/JAXSR 与 DRSR formal source 不一致，无法进入单一 queue")
    if len(source_rows) != 50:
        raise ValueError(f"formal source 应为 50 行，实际 {len(source_rows)}")
    return source_rows, {
        "path": str(SOURCE_CSV.resolve()),
        "sha256": _sha256(SOURCE_CSV),
        "rows": len(source_rows),
        "fields": source_fields,
        "drsr_source_sha256": _sha256(DRSR_SOURCE_CSV),
    }


def _validate_allowlist_row(
    row: Mapping[str, str],
    *,
    algorithm: str,
    condition: str,
    source_index: Mapping[str, int],
) -> None:
    if str(row.get("algorithm") or "").strip().lower() != algorithm:
        raise ValueError(f"allowlist algorithm 漂移: {row}")
    if str(row.get("condition") or "").strip() != condition:
        raise ValueError(f"allowlist condition 漂移: {row}")
    try:
        seed = int(str(row.get("seed") or ""))
        sigma = float(str(row.get("noise_sigma") or ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"allowlist seed/noise_sigma 非法: {row}") from exc
    if seed not in SEEDS or not math.isclose(sigma, _noise_sigma(condition), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"allowlist seed/noise_sigma 不符合 formal 网格: {row}")
    dataset_id = str(row.get("dataset_id") or "").strip()
    global_index = source_index.get(dataset_id)
    if global_index is None:
        raise ValueError(f"allowlist 数据集不在 50-task source 中: {row}")
    expected_task_id = f"{algorithm}_s{seed}_{condition}_g{global_index:04d}"
    if str(row.get("task_id") or "").strip() != expected_task_id:
        raise ValueError(
            f"allowlist task_id 与 source 不一致: {row.get('task_id')!r} != {expected_task_id!r}"
        )
    expected_params = f"{algorithm}__{condition}"
    if str(row.get("params_name") or "").strip() != expected_params:
        raise ValueError(f"allowlist params_name 漂移: {row}")


def _read_and_merge_allowlists(
    asset_root: Path, source_index: Mapping[str, int]
) -> tuple[list[dict[str, str]], dict[str, Any], list[Path]]:
    merged_by_task_id: dict[str, dict[str, str]] = {}
    selection_reports: list[dict[str, Any]] = []
    frozen_paths: list[Path] = []
    for algorithm, condition, filename, expected_count in ALLOWLIST_SPECS:
        source_path = NATIVE_ROOT / "manifests" / filename
        source_rows, _ = _read_csv(source_path)
        if len(source_rows) != expected_count:
            raise ValueError(
                f"{filename} 期望 {expected_count} 行，实际 {len(source_rows)}"
            )
        frozen_path = asset_root / "frozen_inputs/native_fidelity_v1_allowlists" / filename
        _copy_file(source_path, frozen_path)
        frozen_paths.append(frozen_path)
        for row in source_rows:
            _validate_allowlist_row(
                row,
                algorithm=algorithm,
                condition=condition,
                source_index=source_index,
            )
            task_id = str(row["task_id"]).strip()
            normalized = dict(row)
            normalized["source_allowlist"] = filename
            previous = merged_by_task_id.get(task_id)
            if previous is not None:
                comparable_previous = {
                    key: value for key, value in previous.items() if key != "source_allowlist"
                }
                comparable_current = {
                    key: value for key, value in normalized.items() if key != "source_allowlist"
                }
                if comparable_previous != comparable_current:
                    raise ValueError(f"重复 task_id 的 allowlist 记录冲突: {task_id}")
                continue
            merged_by_task_id[task_id] = normalized
        selection_reports.append(
            {
                "algorithm": algorithm,
                "condition": condition,
                "source": str(source_path.resolve()),
                "source_sha256": _sha256(source_path),
                "expected_rows": expected_count,
                "read_rows": len(source_rows),
            }
        )

    rows = sorted(
        merged_by_task_id.values(),
        key=lambda row: (
            str(row.get("algorithm")),
            str(row.get("condition")),
            int(str(row.get("seed") or "0")),
            str(row.get("dataset_id")),
        ),
    )
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"合并去重后应为 {EXPECTED_TOTAL} 条，实际 {len(rows)}")
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("合并 allowlist 存在重复 task_id")
    algorithm_counts = dict(sorted(Counter(row["algorithm"] for row in rows).items()))
    condition_counts = dict(sorted(Counter(row["condition"] for row in rows).items()))
    if algorithm_counts != dict(sorted(EXPECTED_ALGORITHM_COUNTS.items())):
        raise ValueError(f"算法计数漂移: {algorithm_counts}")
    if condition_counts != dict(sorted(EXPECTED_CONDITION_COUNTS.items())):
        raise ValueError(f"condition 计数漂移: {condition_counts}")
    report = {
        "source_allowlists": selection_reports,
        "merged_count": len(rows),
        "deduplicated_count": len(rows),
        "task_ids_unique": True,
        "algorithm_counts": algorithm_counts,
        "condition_counts": condition_counts,
    }
    return rows, report, frozen_paths


def _validate_params(asset_root: Path) -> tuple[dict[str, Any], list[Path]]:
    params_dir = asset_root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    report: dict[str, Any] = {}
    for source, filename in PARAM_SPECS:
        destination = params_dir / filename
        _copy_file(source, destination)
        copied.append(destination)
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"formal 参数不是合法 JSON: {source}") from exc
        if int(payload.get("timeout_in_seconds") or 0) != TIMEOUT_SECONDS:
            raise ValueError(f"formal 参数 timeout 不是 {TIMEOUT_SECONDS}: {source}")
        if int(payload.get("progress_snapshot_interval_seconds") or 0) != SNAPSHOT_SECONDS:
            raise ValueError(f"formal 参数 snapshot 不是 {SNAPSHOT_SECONDS}: {source}")
        report[filename] = {
            "source": str(source.resolve()),
            "sha256": _sha256(destination),
            "timeout_in_seconds": payload.get("timeout_in_seconds"),
            "progress_snapshot_interval_seconds": payload.get(
                "progress_snapshot_interval_seconds"
            ),
            "train_label_noise_sigma": payload.get("train_label_noise_sigma"),
        }
    return report, copied


def _render_command(
    asset_root: Path,
    *,
    mode: str,
    include_future_cpu_flag: bool = True,
) -> str:
    script_dir = '"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"'
    asset_expr = f'"{script_dir}/.."'
    repo_expr = f'"{asset_expr}/../../../.."'
    queue_root = f'"$ASSET_ROOT/queues/{mode}"'
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"SCRIPT_DIR={script_dir}",
        'ASSET_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"',
        'REPO_ROOT="$(cd -- "$ASSET_ROOT/../../../.." && pwd)"',
        'LOG_DIR="$ASSET_ROOT/logs"',
        'mkdir -p "$LOG_DIR"',
        f'exec > >(tee -a "$LOG_DIR/controller_{mode}.log") 2>&1',
        'cd "$REPO_ROOT"',
        'export PYTHONPATH=.',
        # iaaccn22 的系统 hostname 不是集群别名；显式声明 controller 为本机，
        # 避免调度器通过外层 SSH 配置反连自身。
        'export SIM_QUEUE_CONTROLLER_IS_LOCAL=1',
        'QUEUE_SCRIPT="$REPO_ROOT/check/run_e1_candidate200_12alg_load_queue.py"',
        'SOURCE_CSV="$ASSET_ROOT/frozen_inputs/ssr50_source.csv"',
        'PARAMS_ROOT="$ASSET_ROOT/params"',
        'ALLOWLIST="$ASSET_ROOT/manifests/all_conditions_noise_task_allowlist.csv"',
        f'QUEUE_ROOT={queue_root}',
        'mkdir -p "$QUEUE_ROOT"',
        "",
        'python "$QUEUE_SCRIPT" \\',
        f'  --batch-name {BATCH_NAME}_{mode} \\',
        '  --source-csv "$SOURCE_CSV" \\',
        '  --expected-rows 50 \\',
        '  --queue-root "$QUEUE_ROOT" \\',
        '  --params-root "$PARAMS_ROOT" \\',
        '  --task-id-allowlist-csv "$ALLOWLIST" \\',
        "  --hosts " + " ".join(HOSTS) + " \\",
        "  --tools " + " ".join(TOOLS) + " \\",
        "  --seeds " + " ".join(str(seed) for seed in SEEDS) + " \\",
        "  --noise-sigmas 0.01 0.05 \\",
        "  --controller-host iaaccn22 \\",
        "  --use-internal-ips \\",
        "  --remote-root /home/zhangziwen/workplace/scientific-intelligent-modelling \\",
        "  --remote-data-root /home/zhangziwen/sim-datasets-data \\",
        f"  --max-jobs-per-host {MAX_JOBS_PER_HOST} \\",
        f"  --max-new-jobs-per-host-per-poll {MAX_NEW_JOBS_PER_HOST_PER_POLL} \\",
        f'  --load-tier-new-jobs "{LOAD_TIERS}" \\',
        f"  --max-load-ratio {MAX_LOAD_RATIO} \\",
        f"  --max-memory-used-ratio {MAX_MEMORY_USED_RATIO} \\",
        "  --min-free-mem-gb 0 \\",
        f"  --session-prefix {SESSION_PREFIX} \\",
        f"  --host-session-count-prefix {SESSION_PREFIX} \\",
        "  --poll-seconds 60 \\",
        "  --retry-limit 1 \\",
        "  --llm-model-assignment from-params \\",
        "  --llm-default-bucket turbo \\",
        "  --llm-model-bucket-limits base:0,turbo:1",
    ]
    if include_future_cpu_flag:
        lines[-1] += f" \\\n  --max-cpu-used-ratio {MAX_CPU_USED_RATIO}"
    if mode == "preflight":
        lines[-1] += ' \\\n  --preflight-only \\\n  --preflight-report "$ASSET_ROOT/reports/preflight_report.json"'
    elif mode == "dry_run":
        lines[-1] += " \\\n  --dry-run \\\n  --once"
    else:
        lines[-1] += " \\\n  --force-rerun-existing"
    return "\n".join(lines) + "\n"


def _write_commands(asset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for mode, filename in (
        ("preflight", "run_all_conditions_preflight.sh"),
        ("dry_run", "run_all_conditions_dry_run.sh"),
        ("formal", "run_all_conditions_formal.sh"),
    ):
        path = asset_root / "commands" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_command(asset_root, mode=mode), encoding="utf-8")
        path.chmod(0o755)
        paths.append(path)
    return paths


def _queue_supports_cpu_flag() -> bool:
    """Detect whether the checked-out scheduler already exposes the new guard."""
    try:
        return "--max-cpu-used-ratio" in QUEUE_SCRIPT.read_text(encoding="utf-8")
    except OSError:
        return False


def _compatibility_dry_run_args(
    asset_root: Path, *, include_future_cpu_flag: bool
) -> list[str]:
    args = [
        "python",
        str(QUEUE_SCRIPT),
        "--batch-name",
        f"{BATCH_NAME}_dry_run_validation",
        "--source-csv",
        str(asset_root / "frozen_inputs/ssr50_source.csv"),
        "--expected-rows",
        "50",
        "--queue-root",
        str(asset_root / "queues/dry_run_validation"),
        "--params-root",
        str(asset_root / "params"),
        "--task-id-allowlist-csv",
        str(asset_root / "manifests/all_conditions_noise_task_allowlist.csv"),
        "--hosts",
        *HOSTS,
        "--tools",
        *TOOLS,
        "--seeds",
        *(str(seed) for seed in SEEDS),
        "--noise-sigmas",
        *(str(sigma) for sigma in NOISE_SIGMAS),
        "--controller-host",
        "iaaccn22",
        "--use-internal-ips",
        "--max-jobs-per-host",
        str(MAX_JOBS_PER_HOST),
        "--max-new-jobs-per-host-per-poll",
        str(MAX_NEW_JOBS_PER_HOST_PER_POLL),
        "--load-tier-new-jobs",
        LOAD_TIERS,
        "--max-load-ratio",
        str(MAX_LOAD_RATIO),
        "--max-memory-used-ratio",
        str(MAX_MEMORY_USED_RATIO),
        "--session-prefix",
        SESSION_PREFIX,
        "--host-session-count-prefix",
        SESSION_PREFIX,
        "--poll-seconds",
        "60",
        "--retry-limit",
        "1",
        "--llm-model-assignment",
        "from-params",
        "--llm-default-bucket",
        "turbo",
        "--llm-model-bucket-limits",
        "base:0,turbo:1",
    ]
    if include_future_cpu_flag:
        args.extend(["--max-cpu-used-ratio", str(MAX_CPU_USED_RATIO)])
    args.extend(["--dry-run", "--once"])
    return args


def _parse_queue_start(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "queue_start":
            return payload
    raise ValueError(f"dry-run 输出没有 queue_start: {stdout[-4000:]}")


def _run_dry_run(asset_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    include_future_cpu_flag = _queue_supports_cpu_flag()
    command = _compatibility_dry_run_args(
        asset_root, include_future_cpu_flag=include_future_cpu_flag
    )
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("本地 dry-run 超过 60 秒") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "本地 dry-run 失败: "
            f"returncode={result.returncode}\nstdout={result.stdout[-4000:]}\n"
            f"stderr={result.stderr[-4000:]}"
        )
    queue_start = _parse_queue_start(result.stdout)
    if int(queue_start.get("tasks") or 0) != EXPECTED_TOTAL:
        raise ValueError(f"scheduler dry-run 任务数漂移: {queue_start}")
    state_path = (
        asset_root
        / "queues/dry_run_validation/state/"
        f"{BATCH_NAME}_dry_run_validation.state.json"
    )
    if not state_path.is_file():
        raise RuntimeError(f"dry-run 未生成 state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_tasks = state.get("tasks") or {}
    if len(state_tasks) != EXPECTED_TOTAL:
        raise ValueError(f"dry-run state 任务数漂移: {len(state_tasks)}")
    state_algorithm_counts = dict(
        sorted(Counter(str(task.get("tool")) for task in state_tasks.values()).items())
    )
    state_condition_counts = dict(
        sorted(Counter(str(task.get("noise_tag")) for task in state_tasks.values()).items())
    )
    expected_algorithm_counts = dict(sorted(EXPECTED_ALGORITHM_COUNTS.items()))
    expected_condition_counts = dict(sorted(EXPECTED_CONDITION_COUNTS.items()))
    if state_algorithm_counts != expected_algorithm_counts:
        raise ValueError(f"dry-run 算法计数漂移: {state_algorithm_counts}")
    if state_condition_counts != expected_condition_counts:
        raise ValueError(f"dry-run condition 计数漂移: {state_condition_counts}")
    task_ids_unique = len(state_tasks) == len(set(state_tasks))
    if not task_ids_unique:
        raise ValueError("dry-run state task IDs 不唯一")
    report = {
        "status": "ok",
        "validation_mode": (
            "local_scheduler_dry_run"
            if include_future_cpu_flag
            else "local_scheduler_dry_run_without_future_cpu_cli_flag"
        ),
        "future_flag_excluded": None if include_future_cpu_flag else "--max-cpu-used-ratio",
        "command": command,
        "queue_start": queue_start,
        "state_path": str(state_path),
        "task_count": len(state_tasks),
        "algorithm_counts": state_algorithm_counts,
        "condition_counts": state_condition_counts,
        "task_ids_unique": task_ids_unique,
        "expected": dict(expected),
    }
    _write_json(asset_root / "reports/dry_run_validation.json", report)
    return report


def _write_allowlist(asset_root: Path, rows: list[dict[str, str]]) -> Path:
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in PRIMARY_ALLOWLIST_FIELDS
        }
    )
    fields = list(PRIMARY_ALLOWLIST_FIELDS) + extra_fields
    path = asset_root / "manifests/all_conditions_noise_task_allowlist.csv"
    _write_csv(path, rows, fields)
    return path


def _manifest_files(asset_root: Path) -> list[Path]:
    roots = (
        "commands",
        "frozen_inputs",
        "manifests",
        "params",
        "queues",
        "reports",
        "README.md",
    )
    files: list[Path] = []
    for root in roots:
        path = asset_root / root
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            continue
        files.extend(
            item
            for item in path.rglob("*")
            if (
                item.is_file()
                and "__pycache__" not in item.parts
                and not item.name.endswith(".pyc")
                and item.relative_to(asset_root).as_posix() != "reports/generation_report.json"
            )
        )
    return sorted(files)


def _write_manifest(asset_root: Path, metadata: Mapping[str, Any]) -> tuple[Path, Path, str]:
    files = {}
    for path in _manifest_files(asset_root):
        files[path.relative_to(asset_root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest_path = asset_root / "asset_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "all_conditions_cpu_v2.asset_manifest.v1",
            "metadata": metadata,
            "files": files,
        },
    )
    manifest_sha = _sha256(manifest_path)
    hash_path = asset_root / "asset_manifest.sha256"
    hash_path.write_text(f"{manifest_sha}  asset_manifest.json\n", encoding="utf-8")
    return manifest_path, hash_path, manifest_sha


def _write_readme(asset_root: Path, report: Mapping[str, Any]) -> Path:
    text = f"""# all_conditions_cpu_v2

这是 Stage5 `native_fidelity_v1` 噪声精确重跑的单一 CPU 感知队列资产，不启动远端任务。

## 任务范围

- 合并去重后的 allowlist：**{report['task_count']}** 条，所有 `task_id` 唯一。
- 算法计数：JAXSR **{report['algorithm_counts']['jaxsr']}**、iMCTS **{report['algorithm_counts']['imcts']}**、DRSR **{report['algorithm_counts']['drsr']}**。
- 条件计数：`noise001` **{report['condition_counts']['noise001']}**、`noise005` **{report['condition_counts']['noise005']}**。
- 一个 controller、一个 allowlist、一个 queue；两个噪声条件通过同一条 `--noise-sigmas 0.01 0.05` 入队。
- 主机固定为 `iaaccn22` 到 `iaaccn29` 共 8 台，controller 为 `iaaccn22`。
- 每台最多 18 个任务，8 台首轮容量为 `8 x 18 = 144`，因此 143 条任务可以一次性铺开；这里的“打满”指把剩余任务并发提交，不代表物理占满 2048 核。每个任务仍是单 worker，并受 CPU 权重、load 和内存熔断约束。

五个输入 allowlist 原样冻结在 `frozen_inputs/native_fidelity_v1_allowlists/`，合并结果为 `manifests/all_conditions_noise_task_allowlist.csv`。formal 参数复制到 `params/`，每个参数均检查 `timeout_in_seconds=10800` 与 `progress_snapshot_interval_seconds=60`。`drsr.json` 是 DRSR 从参数推断模型桶所需的 base formal 参数；实际噪声任务使用 `drsr__noise005.json`。

## 调度约束

生成的三个入口都使用即将加入 scheduler 的 `--max-cpu-used-ratio {MAX_CPU_USED_RATIO}`。`--max-jobs-per-host {MAX_JOBS_PER_HOST}` 和每轮最多新增 `{MAX_NEW_JOBS_PER_HOST_PER_POLL}` 个任务是高位保险与首轮铺开配置；load 和内存阈值均为 `{MAX_LOAD_RATIO}`，仍保留熔断。controller 日志持久化到 `logs/controller_*.log`，队列状态与事件日志在各自 `queues/*/state/` 下；本目录新增入口不使用 `/tmp`。

生成器会根据当前 checkout 的 scheduler CLI 自动选择 dry-run 验收方式：若已声明 `--max-cpu-used-ratio`，本地验收会带该参数；若尚未落地，则只在本地兼容验收时暂时省略它。正式生成的 `run_all_conditions_dry_run.sh`、`run_all_conditions_preflight.sh` 和 `run_all_conditions_formal.sh` 始终保留该参数。当前 scheduler 的旧 preflight/SSH 提交实现若仍创建瞬时 `/tmp` 文件，需要由 scheduler 本身另行收口；本资产不把 controller 日志放在 `/tmp`。

## 旧轨迹与重跑边界

既有 **4500 条旧 noise trajectory** 在回收后走后处理路径；它们不需要因为本次内部目标证据而全部重跑。本目录只生成冻结审计明确要求的 143 条补充 noise rerun，**无需全重跑** 4500 条旧 noise 轨迹，也无需重跑完整 6750 网格。

## 执行入口

依次使用：

1. `commands/run_all_conditions_preflight.sh`
2. `commands/run_all_conditions_dry_run.sh`
3. `commands/run_all_conditions_formal.sh`

dry-run 只构建本地 state，不连接远端；本次生成过程没有启动 preflight 或 formal。输入和输出文件的 SHA256 记录在 `asset_manifest.json`，其自身的哈希记录在 `asset_manifest.sha256`。
"""
    path = asset_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def generate_assets(
    *, asset_root: Path | str | None = None, run_dry_run: bool = True
) -> dict[str, Any]:
    """Create the complete asset set and optionally validate the local queue."""
    root = Path(asset_root or DEFAULT_ASSET_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs/.gitkeep").write_text("", encoding="utf-8")

    source_rows, source_report = _validate_sources()
    source_index = _source_index(source_rows)
    merged_rows, allowlist_report, frozen_allowlists = _read_and_merge_allowlists(
        root, source_index
    )
    allowlist_path = _write_allowlist(root, merged_rows)
    frozen_source_path = root / "frozen_inputs/ssr50_source.csv"
    _copy_file(SOURCE_CSV, frozen_source_path)
    params_report, copied_params = _validate_params(root)
    command_paths = _write_commands(root)

    report: dict[str, Any] = {
        "status": "ok",
        "asset_root": str(root),
        "task_count": len(merged_rows),
        "algorithm_counts": dict(sorted(Counter(row["algorithm"] for row in merged_rows).items())),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in merged_rows).items())),
        "task_ids_unique": len({row["task_id"] for row in merged_rows}) == len(merged_rows),
        "hosts": list(HOSTS),
        "tools": list(TOOLS),
        "seeds": list(SEEDS),
        "noise_sigmas": list(NOISE_SIGMAS),
        "timeout_in_seconds": TIMEOUT_SECONDS,
        "progress_snapshot_interval_seconds": SNAPSHOT_SECONDS,
        "max_cpu_used_ratio": MAX_CPU_USED_RATIO,
        "max_load_ratio": MAX_LOAD_RATIO,
        "max_memory_used_ratio": MAX_MEMORY_USED_RATIO,
        "max_jobs_per_host": MAX_JOBS_PER_HOST,
        "max_new_jobs_per_host_per_poll": MAX_NEW_JOBS_PER_HOST_PER_POLL,
        "first_round_capacity": len(HOSTS) * MAX_JOBS_PER_HOST,
        "first_round_tasks_fit": len(merged_rows) <= len(HOSTS) * MAX_JOBS_PER_HOST,
        "physical_core_fill_note": "143 single-worker tasks do not physically fill 2048 CPU cores; first-round fill means concurrent submission.",
        "source": source_report,
        "allowlists": allowlist_report,
        "params": params_report,
        "frozen_files": {
            "source_csv": str(frozen_source_path),
            "allowlists": [str(path) for path in frozen_allowlists],
            "params": [str(path) for path in copied_params],
        },
        "commands": [str(path) for path in command_paths],
        "dry_run": None,
        "old_noise_trajectory_policy": {
            "old_noise_trajectories": 4500,
            "action": "recycle_for_postprocessing",
            "full_rerun_required": False,
        },
    }
    if run_dry_run:
        report["dry_run"] = _run_dry_run(root, report)
    _write_json(root / "reports/generation_report.json", report)
    _write_readme(root, report)
    manifest_path, hash_path, manifest_sha = _write_manifest(root, report)
    report["manifest"] = {
        "path": str(manifest_path),
        "sha256_path": str(hash_path),
        "sha256": manifest_sha,
        "excluded_from_file_hashes": ["reports/generation_report.json"],
    }
    # generation_report contains the manifest digest, so it is intentionally
    # excluded from the manifest file list to avoid a circular hash.
    _write_json(root / "reports/generation_report.json", report)
    return report


def main() -> None:
    result = generate_assets()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
