#!/usr/bin/env python3
"""生成最新 SymbolFit noise300 final numeric 与 180 分钟 EFF 交付指标。"""

from __future__ import annotations

import os

# 必须在 NumPy 及 canonical replay 导入前限制本地数值库线程。
for _name in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import efficiency_from_qualities, phi_nmse
from .performance_replay import (
    EVALUATION_PATH,
    PerformanceReplayCache,
    replay_payload_performance,
)


CONDITIONS = ("noise001", "noise005")
SEEDS = (520, 521, 522)
HORIZON = 180
EXPECTED_RUNS = 300
EXPECTED_POINTS = EXPECTED_RUNS * HORIZON
EXECUTION_RUNNER_COMMIT = "2835ec2ab987edd2b40f6e25e6e5e59eb138dfc0"
EXECUTION_RUNNER_PATH = "scientific_intelligent_modelling/benchmarks/runner.py"
EXECUTION_SOURCE_MARKERS = (
    'return min(candidates, key=lambda item: item["internal_loss"])',
    "_predict_from_canonical_artifact(canonical_artifact, dataset.id_test.X)",
    "_predict_from_canonical_artifact(canonical_artifact, dataset.ood_test.X)",
    'payload["source_internal_loss"] = candidate.get("internal_loss")',
    'payload["record_type"] = "budget_end_internal_best"',
)


class SymbolFitReleaseMetricError(ValueError):
    """输入冻结包或指标结果不满足发布契约。"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SymbolFitReleaseMetricError(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SymbolFitReleaseMetricError(f"JSON 顶层不是 object: {path}")
    return payload


def _finite_nonnegative(value: object, *, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise SymbolFitReleaseMetricError(f"{context} 缺失或不是数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SymbolFitReleaseMetricError(f"{context} 不是数值: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise SymbolFitReleaseMetricError(f"{context} 必须是非负有限数值: {value!r}")
    return number


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=list(fields), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_frozen_path(freeze_root: Path, row: Mapping[str, str], field: str) -> Path:
    relative = str(row.get(field) or "").strip()
    if not relative:
        raise SymbolFitReleaseMetricError(f"{row.get('task_id')}: {field} 为空")
    return freeze_root / "hosts" / row["host"] / relative


def read_freeze_manifest(freeze_root: Path) -> list[dict[str, str]]:
    path = freeze_root / "manifests" / "symbolfit_noise_latest_runs.csv"
    if not path.is_file():
        raise SymbolFitReleaseMetricError(f"冻结 manifest 不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != EXPECTED_RUNS:
        raise SymbolFitReleaseMetricError(f"冻结 manifest 应有 {EXPECTED_RUNS} 行，实际 {len(rows)}")
    task_ids = [row["task_id"] for row in rows]
    logical_keys = [row["logical_key"] for row in rows]
    if len(set(task_ids)) != EXPECTED_RUNS or len(set(logical_keys)) != EXPECTED_RUNS:
        raise SymbolFitReleaseMetricError("冻结 manifest 的 task_id/logical_key 不唯一")
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        condition = row["condition"]
        seed = int(row["seed"])
        if condition not in CONDITIONS or seed not in SEEDS:
            raise SymbolFitReleaseMetricError(f"冻结行 condition/seed 非法: {row['logical_key']}")
        if row.get("algorithm") != "SymbolFit" or row.get("validation_status") != "passed":
            raise SymbolFitReleaseMetricError(f"冻结行算法/验收状态非法: {row['logical_key']}")
        if row.get("replacement_authorization") not in {
            "not_authorized_collected_candidate_only",
            "final_and_eff",
        }:
            raise SymbolFitReleaseMetricError(f"冻结行原始授权字段非法: {row['logical_key']}")
        counts[(condition, seed)] += 1
    expected_counts = {(condition, seed): 50 for condition in CONDITIONS for seed in SEEDS}
    if counts != expected_counts:
        raise SymbolFitReleaseMetricError(f"冻结网格不完整: {counts}")
    return sorted(
        rows,
        key=lambda row: (row["condition"], int(row["seed"]), int(row["dataset_global_index"])),
    )


def execution_source_evidence(
    *, repo_root: Path, preflight_report: Path, execution_commit: str
) -> dict[str, Any]:
    report = _read_json(preflight_report)
    runner_hashes: set[str] = set()
    verified_hosts: list[str] = []
    for host in report.get("hosts", []):
        if not isinstance(host, Mapping):
            continue
        runner = host.get("files", {}).get("runner") if isinstance(host.get("files"), Mapping) else None
        if not isinstance(runner, Mapping) or runner.get("matches_expected") is not True:
            continue
        sha = str(runner.get("sha256") or "")
        if len(sha) == 64:
            runner_hashes.add(sha)
            verified_hosts.append(str(host.get("host") or ""))
    if len(runner_hashes) != 1:
        raise SymbolFitReleaseMetricError(
            f"preflight 没有给出唯一已验证 runner SHA: {sorted(runner_hashes)}"
        )
    execution_sha = next(iter(runner_hashes))
    process = subprocess.run(
        ["git", "show", f"{execution_commit}:{EXECUTION_RUNNER_PATH}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise SymbolFitReleaseMetricError(
            f"无法读取执行期 runner Git blob: {process.stderr.decode('utf-8', errors='replace')}"
        )
    source_bytes = process.stdout
    blob_sha = hashlib.sha256(source_bytes).hexdigest()
    if blob_sha != execution_sha:
        raise SymbolFitReleaseMetricError(
            f"执行期 runner blob SHA 不匹配: {blob_sha} != {execution_sha}"
        )
    source = source_bytes.decode("utf-8")
    missing_markers = [marker for marker in EXECUTION_SOURCE_MARKERS if marker not in source]
    if missing_markers:
        raise SymbolFitReleaseMetricError(f"执行期 runner 缺少指标来源标记: {missing_markers}")
    return {
        "preflight_report": str(preflight_report.resolve()),
        "preflight_report_sha256": _sha256_file(preflight_report),
        "execution_runner_commit": execution_commit,
        "execution_runner_path": EXECUTION_RUNNER_PATH,
        "execution_runner_sha256": execution_sha,
        "preflight_verified_hosts": sorted(verified_hosts),
        "verified_source_markers": list(EXECUTION_SOURCE_MARKERS),
        "candidate_selection": "minimum_source_internal_loss",
        "candidate_evaluation": "canonical artifact on frozen ID/OOD splits",
        "test_metric_used_for_candidate_selection": False,
        "worker_code_binding": (
            "preflight 直接绑定已通过主机；其余 worker 由启动前 fanout 脚本按 SHA 同步，"
            "快照本身未嵌入逐 run 代码 SHA"
        ),
    }


def _verify_result_source(
    freeze_root: Path, row: Mapping[str, str]
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_frozen_path(freeze_root, row, "frozen_result_path")
    if not path.is_file() or _sha256_file(path) != row["frozen_result_sha256"]:
        raise SymbolFitReleaseMetricError(f"{row['logical_key']}: frozen result SHA 不匹配")
    payload = _read_json(path)
    if (
        payload.get("status") != "ok"
        or payload.get("tool") != "symbolfit"
        or payload.get("dataset") != row["dataset_id"]
        or int(payload.get("seed", -1)) != int(row["seed"])
        or payload.get("condition") != row["condition"]
    ):
        raise SymbolFitReleaseMetricError(f"{row['logical_key']}: frozen result 身份不匹配")
    equation = payload.get("equation")
    artifact = payload.get("canonical_artifact")
    if not isinstance(equation, str) or not equation.strip() or not isinstance(artifact, Mapping):
        raise SymbolFitReleaseMetricError(f"{row['logical_key']}: final equation/artifact 缺失")
    if _sha256_text(equation) != row["final_equation_sha256"]:
        raise SymbolFitReleaseMetricError(f"{row['logical_key']}: final equation SHA 不匹配")
    artifact_sha = _sha256_text(_canonical_json(artifact))
    if artifact_sha != row["canonical_artifact_sha256"]:
        raise SymbolFitReleaseMetricError(f"{row['logical_key']}: canonical artifact SHA 不匹配")
    return path, payload


def _replay_dataset_group(
    items: Sequence[tuple[dict[str, str], Path, dict[str, Any]]],
    *, repo_root: Path,
) -> list[tuple[dict[str, str], Path, dict[str, Any], dict[str, Any]]]:
    cache = PerformanceReplayCache()
    output = []
    for row, path, payload in items:
        replay = replay_payload_performance(
            payload,
            algorithm="symbolfit",
            repo_root=repo_root,
            cache=cache,
            task_id=row["task_id"],
            condition=row["condition"],
            result_sha256=row["frozen_result_sha256"],
        )
        output.append((row, path, payload, replay))
    return output


FINAL_FIELDS = (
    "logical_key",
    "algorithm",
    "dataset_id",
    "dataset_global_index",
    "seed",
    "condition",
    "task_id",
    "host",
    "result_path",
    "result_sha256",
    "equation",
    "equation_sha256",
    "source_canonical_artifact_sha256",
    "replay_canonical_artifact_sha256",
    "artifact_rebuilt",
    "native_id_nmse",
    "native_ood_nmse",
    "evaluation_status",
    "id_nmse",
    "ood_nmse",
    "id_quality",
    "ood_quality",
    "valid_output",
    "invalid_reason",
    "evaluation_path",
    "id_nmse_delta_from_native",
    "ood_nmse_delta_from_native",
    "replacement_scope",
)


def prepare_final_numeric(
    rows: Sequence[dict[str, str]],
    *,
    freeze_root: Path,
    repo_root: Path,
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[str, list[tuple[dict[str, str], Path, dict[str, Any]]]] = defaultdict(list)
    payload_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        path, payload = _verify_result_source(freeze_root, row)
        groups[row["dataset_id"]].append((row, path, payload))
        payload_by_key[row["logical_key"]] = payload

    replayed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_replay_dataset_group, items, repo_root=repo_root)
            for _, items in sorted(groups.items())
        ]
        for future in as_completed(futures):
            replayed.extend(future.result())

    output: list[dict[str, Any]] = []
    for row, path, payload, replay in replayed:
        native_id = _finite_nonnegative(payload.get("id_test", {}).get("nmse"), context="native.id")
        native_ood = _finite_nonnegative(payload.get("ood_test", {}).get("nmse"), context="native.ood")
        valid_output = bool(replay.get("valid_output"))
        if valid_output:
            id_metrics = replay.get("id_test")
            ood_metrics = replay.get("ood_test")
            if not isinstance(id_metrics, Mapping) or not isinstance(ood_metrics, Mapping):
                raise SymbolFitReleaseMetricError(f"{row['logical_key']}: replay 缺少 ID/OOD")
            id_nmse: float | str = _finite_nonnegative(
                id_metrics.get("nmse"), context=f"{row['logical_key']}.id"
            )
            ood_nmse: float | str = _finite_nonnegative(
                ood_metrics.get("nmse"), context=f"{row['logical_key']}.ood"
            )
            id_quality = phi_nmse(id_nmse)
            ood_quality = phi_nmse(ood_nmse)
            id_delta: float | str = float(id_nmse) - native_id
            ood_delta: float | str = float(ood_nmse) - native_ood
            evaluation_status = "valid"
        else:
            id_nmse = ""
            ood_nmse = ""
            id_quality = 0.0
            ood_quality = 0.0
            id_delta = ""
            ood_delta = ""
            evaluation_status = "invalid_output"
        equation = str(payload["equation"])
        output.append(
            {
                "logical_key": row["logical_key"],
                "algorithm": "SymbolFit",
                "dataset_id": row["dataset_id"],
                "dataset_global_index": int(row["dataset_global_index"]),
                "seed": int(row["seed"]),
                "condition": row["condition"],
                "task_id": row["task_id"],
                "host": row["host"],
                "result_path": _relative_or_absolute(path, repo_root),
                "result_sha256": row["frozen_result_sha256"],
                "equation": equation,
                "equation_sha256": _sha256_text(equation),
                "source_canonical_artifact_sha256": row["canonical_artifact_sha256"],
                "replay_canonical_artifact_sha256": replay["canonical_artifact_sha256"],
                "artifact_rebuilt": str(bool(replay.get("artifact_rebuilt"))).lower(),
                "native_id_nmse": native_id,
                "native_ood_nmse": native_ood,
                "evaluation_status": evaluation_status,
                "id_nmse": id_nmse,
                "ood_nmse": ood_nmse,
                "id_quality": id_quality,
                "ood_quality": ood_quality,
                "valid_output": str(valid_output).lower(),
                "invalid_reason": str(replay.get("invalid_reason") or ""),
                "evaluation_path": EVALUATION_PATH,
                "id_nmse_delta_from_native": id_delta,
                "ood_nmse_delta_from_native": ood_delta,
                "replacement_scope": "final_and_eff",
            }
        )
    output.sort(
        key=lambda row: (row["condition"], int(row["seed"]), int(row["dataset_global_index"]))
    )
    return output, payload_by_key


MINUTE_FIELDS = (
    "logical_key",
    "algorithm",
    "dataset_id",
    "dataset_global_index",
    "seed",
    "condition",
    "task_id",
    "host",
    "minute",
    "record_type",
    "candidate_available",
    "candidate_expression",
    "candidate_expression_sha256",
    "canonical_artifact_sha256",
    "source_internal_loss",
    "source_complexity",
    "candidate_attempt",
    "candidate_first_discovered_minute",
    "candidate_source",
    "carry_forward",
    "carry_forward_from_minute",
    "effective_source",
    "id_nmse",
    "ood_nmse",
    "id_quality",
    "ood_quality",
    "combined_quality",
    "q_star",
    "relative_progress",
    "cumulative_eff",
    "metric_source",
    "test_metric_used_for_candidate_selection",
    "snapshot_payload_sha256",
    "source_trajectory_path",
    "source_trajectory_sha256",
)


def build_run_trajectory(
    row: Mapping[str, str],
    payloads: Sequence[Mapping[str, Any]],
    *,
    source_trajectory_path: str,
    source_trajectory_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(payloads) != HORIZON:
        raise SymbolFitReleaseMetricError(
            f"{row['logical_key']}: 分钟记录应为 {HORIZON}，实际 {len(payloads)}"
        )
    minute_rows: list[dict[str, Any]] = []
    previous_artifact_sha: str | None = None
    current_candidate_start: int | None = None
    combined: list[float] = []
    id_qualities: list[float] = []
    ood_qualities: list[float] = []

    for minute, payload in enumerate(payloads, start=1):
        if int(payload.get("checkpoint_index", -1)) != minute:
            raise SymbolFitReleaseMetricError(
                f"{row['logical_key']}: minute_{minute:04d} checkpoint 不匹配"
            )
        if (
            str(payload.get("tool", "")).lower() != "symbolfit"
            or payload.get("dataset") != row["dataset_id"]
            or int(payload.get("seed", -1)) != int(row["seed"])
            or payload.get("condition") != row["condition"]
        ):
            raise SymbolFitReleaseMetricError(
                f"{row['logical_key']}: minute_{minute:04d} 身份不匹配"
            )
        record_type = str(payload.get("record_type") or "")
        if minute == HORIZON:
            if record_type != "budget_end_internal_best":
                raise SymbolFitReleaseMetricError(f"{row['logical_key']}: minute180 类型非法")
        elif record_type not in {"periodic_heartbeat", "periodic_best"}:
            raise SymbolFitReleaseMetricError(
                f"{row['logical_key']}: minute_{minute:04d} 类型非法: {record_type}"
            )

        candidate_available = payload.get("candidate_available") is True
        equation = ""
        equation_sha = ""
        artifact_sha = ""
        internal_loss: float | str = ""
        source_complexity: Any = ""
        attempt: Any = ""
        first_minute: Any = ""
        candidate_source = ""
        carry_forward = False
        carry_from: int | str = ""
        effective_source = "no_candidate"
        id_nmse: float | str = ""
        ood_nmse: float | str = ""
        q_id = 0.0
        q_ood = 0.0

        if candidate_available:
            if payload.get("status") != "ok":
                raise SymbolFitReleaseMetricError(
                    f"{row['logical_key']}: minute_{minute:04d} candidate status 非 ok"
                )
            equation_raw = payload.get("equation")
            artifact = payload.get("canonical_artifact")
            if not isinstance(equation_raw, str) or not equation_raw.strip():
                raise SymbolFitReleaseMetricError(
                    f"{row['logical_key']}: minute_{minute:04d} equation 缺失"
                )
            if not isinstance(artifact, Mapping) or artifact.get("artifact_valid") is not True:
                raise SymbolFitReleaseMetricError(
                    f"{row['logical_key']}: minute_{minute:04d} canonical artifact 无效"
                )
            candidate_source = str(payload.get("candidate_source") or "")
            if candidate_source != "symbolfit_active_pysr_hall_of_fame":
                raise SymbolFitReleaseMetricError(
                    f"{row['logical_key']}: minute_{minute:04d} 非内部搜索候选"
                )
            internal_loss = _finite_nonnegative(
                payload.get("source_internal_loss"),
                context=f"{row['logical_key']}.minute{minute}.internal_loss",
            )
            id_nmse = _finite_nonnegative(
                payload.get("id_test", {}).get("nmse"),
                context=f"{row['logical_key']}.minute{minute}.id_nmse",
            )
            ood_nmse = _finite_nonnegative(
                payload.get("ood_test", {}).get("nmse"),
                context=f"{row['logical_key']}.minute{minute}.ood_nmse",
            )
            equation = equation_raw.strip()
            equation_sha = _sha256_text(equation)
            artifact_sha = _sha256_text(_canonical_json(artifact))
            q_id = phi_nmse(id_nmse)
            q_ood = phi_nmse(ood_nmse)
            source_complexity = payload.get("source_complexity")
            attempt = payload.get("candidate_attempt")
            first_minute = payload.get("candidate_first_discovered_minute")
            if artifact_sha == previous_artifact_sha:
                carry_forward = True
                carry_from = current_candidate_start if current_candidate_start is not None else minute - 1
                effective_source = f"carry_forward_from_minute_{int(carry_from):04d}"
            else:
                current_candidate_start = minute
                effective_source = candidate_source
            previous_artifact_sha = artifact_sha
        elif payload.get("status") != "running" or record_type != "periodic_heartbeat":
            raise SymbolFitReleaseMetricError(
                f"{row['logical_key']}: minute_{minute:04d} 无候选记录不合法"
            )

        q = (q_id + q_ood) / 2.0
        id_qualities.append(q_id)
        ood_qualities.append(q_ood)
        combined.append(q)
        minute_rows.append(
            {
                "logical_key": row["logical_key"],
                "algorithm": "SymbolFit",
                "dataset_id": row["dataset_id"],
                "dataset_global_index": int(row["dataset_global_index"]),
                "seed": int(row["seed"]),
                "condition": row["condition"],
                "task_id": row["task_id"],
                "host": row["host"],
                "minute": minute,
                "record_type": record_type,
                "candidate_available": str(candidate_available).lower(),
                "candidate_expression": equation,
                "candidate_expression_sha256": equation_sha,
                "canonical_artifact_sha256": artifact_sha,
                "source_internal_loss": internal_loss,
                "source_complexity": source_complexity,
                "candidate_attempt": attempt,
                "candidate_first_discovered_minute": first_minute,
                "candidate_source": candidate_source,
                "carry_forward": str(carry_forward).lower(),
                "carry_forward_from_minute": carry_from,
                "effective_source": effective_source,
                "id_nmse": id_nmse,
                "ood_nmse": ood_nmse,
                "id_quality": q_id,
                "ood_quality": q_ood,
                "combined_quality": q,
                "metric_source": "execution_runner_canonical_capture",
                "test_metric_used_for_candidate_selection": "false",
                "snapshot_payload_sha256": _sha256_text(_canonical_json(payload)),
                "source_trajectory_path": source_trajectory_path,
                "source_trajectory_sha256": source_trajectory_sha256,
            }
        )

    q_star = max(combined)
    q_star_minute = combined.index(q_star) + 1 if q_star > 0.0 else ""
    progress = [value / q_star if q_star > 0.0 else 0.0 for value in combined]
    running_sum = 0.0
    for index, item in enumerate(minute_rows, start=1):
        running_sum += progress[index - 1]
        item["q_star"] = q_star
        item["relative_progress"] = progress[index - 1]
        item["cumulative_eff"] = running_sum / index
    m_eff = efficiency_from_qualities(combined, horizon=HORIZON)
    summary = {
        "logical_key": row["logical_key"],
        "algorithm": "SymbolFit",
        "dataset_id": row["dataset_id"],
        "dataset_global_index": int(row["dataset_global_index"]),
        "seed": int(row["seed"]),
        "condition": row["condition"],
        "task_id": row["task_id"],
        "host": row["host"],
        "q_star": q_star,
        "q_star_first_minute": q_star_minute,
        "m_eff": m_eff,
        "eff_score": 100.0 * m_eff,
        "candidate_snapshot_count": sum(item["candidate_available"] == "true" for item in minute_rows),
        "heartbeat_snapshot_count": sum(item["candidate_available"] == "false" for item in minute_rows),
        "source_trajectory_path": source_trajectory_path,
        "source_trajectory_sha256": source_trajectory_sha256,
        "evaluation_path": "internal_loss_selected_execution_runner_canonical_capture.v1",
        "replacement_scope": "final_and_eff",
        "id_qualities": id_qualities,
        "ood_qualities": ood_qualities,
        "combined_qualities": combined,
    }
    return minute_rows, summary


def prepare_trajectories(
    rows: Sequence[dict[str, str]],
    *,
    freeze_root: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    minute_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        trajectory_path = _resolve_frozen_path(freeze_root, row, "trajectory_path")
        trajectory_sha = _sha256_file(trajectory_path)
        if trajectory_sha != row["trajectory_sha256"]:
            raise SymbolFitReleaseMetricError(f"{row['logical_key']}: trajectory SHA 不匹配")
        payloads: list[dict[str, Any]] = []
        with gzip.open(trajectory_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise SymbolFitReleaseMetricError(
                            f"{row['logical_key']}: trajectory 含非 object"
                        )
                    payloads.append(value)
        relative_path = _relative_or_absolute(trajectory_path, repo_root)
        run_minutes, summary = build_run_trajectory(
            row,
            payloads,
            source_trajectory_path=relative_path,
            source_trajectory_sha256=trajectory_sha,
        )
        minute_rows.extend(run_minutes)
        summaries.append(summary)
    if len(minute_rows) != EXPECTED_POINTS or len(summaries) != EXPECTED_RUNS:
        raise SymbolFitReleaseMetricError(
            f"轨迹规模错误: runs={len(summaries)} points={len(minute_rows)}"
        )
    return minute_rows, summaries


RUN_BASE_FIELDS = (
    "logical_key",
    "algorithm",
    "dataset_id",
    "dataset_global_index",
    "seed",
    "condition",
    "task_id",
    "host",
    "q_star",
    "q_star_first_minute",
    "m_eff",
    "eff_score",
    "candidate_snapshot_count",
    "heartbeat_snapshot_count",
    "source_trajectory_path",
    "source_trajectory_sha256",
    "evaluation_path",
    "replacement_scope",
)


def wide_run_rows(summaries: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    quality_fields = [
        field
        for prefix in ("q_id", "q_ood", "q")
        for field in (f"{prefix}_{minute:04d}" for minute in range(1, HORIZON + 1))
    ]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = {field: summary[field] for field in RUN_BASE_FIELDS}
        for minute, value in enumerate(summary["id_qualities"], start=1):
            row[f"q_id_{minute:04d}"] = value
        for minute, value in enumerate(summary["ood_qualities"], start=1):
            row[f"q_ood_{minute:04d}"] = value
        for minute, value in enumerate(summary["combined_qualities"], start=1):
            row[f"q_{minute:04d}"] = value
        rows.append(row)
    return rows, [*RUN_BASE_FIELDS, *quality_fields]


CURVE_FIELDS = (
    "condition",
    "algorithm",
    "minute",
    "run_count",
    "valid_candidate_count",
    "mean_id_quality",
    "mean_ood_quality",
    "mean_combined_quality",
    "mean_relative_progress",
    "mean_cumulative_eff",
    "formal_eff_at_minute180",
    "evaluation_path",
)


def condition_curves(minute_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in minute_rows:
        grouped[(str(row["condition"]), int(row["minute"]))].append(row)
    curves: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for minute in range(1, HORIZON + 1):
            points = grouped[(condition, minute)]
            if len(points) != 150:
                raise SymbolFitReleaseMetricError(
                    f"{condition} minute{minute}: 应有150条，实际{len(points)}"
                )
            cumulative = sum(float(row["cumulative_eff"]) for row in points) / len(points)
            curves.append(
                {
                    "condition": condition,
                    "algorithm": "SymbolFit",
                    "minute": minute,
                    "run_count": len(points),
                    "valid_candidate_count": sum(row["candidate_available"] == "true" for row in points),
                    "mean_id_quality": sum(float(row["id_quality"]) for row in points) / len(points),
                    "mean_ood_quality": sum(float(row["ood_quality"]) for row in points) / len(points),
                    "mean_combined_quality": sum(float(row["combined_quality"]) for row in points) / len(points),
                    "mean_relative_progress": sum(float(row["relative_progress"]) for row in points) / len(points),
                    "mean_cumulative_eff": 100.0 * cumulative,
                    "formal_eff_at_minute180": 100.0 * cumulative if minute == HORIZON else "",
                    "evaluation_path": "internal_loss_selected_execution_runner_canonical_capture.v1",
                }
            )
    return curves


def _file_record(path: Path, repo_root: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative_or_absolute(path, repo_root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    freeze_root = args.freeze_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.max_workers <= 4:
        raise SymbolFitReleaseMetricError("max_workers 必须在 1..4")

    freeze_rows = read_freeze_manifest(freeze_root)
    source_evidence = execution_source_evidence(
        repo_root=repo_root,
        preflight_report=args.preflight_report.resolve(),
        execution_commit=args.execution_runner_commit,
    )
    final_rows, _ = prepare_final_numeric(
        freeze_rows,
        freeze_root=freeze_root,
        repo_root=repo_root,
        max_workers=args.max_workers,
    )
    minute_rows, run_summaries = prepare_trajectories(
        freeze_rows,
        freeze_root=freeze_root,
        repo_root=repo_root,
    )
    wide_rows, wide_fields = wide_run_rows(run_summaries)
    curves = condition_curves(minute_rows)

    final_path = output_root / "final_300_numeric.csv"
    minute_path = output_root / "minute_54000_long.csv.gz"
    run_path = output_root / "run_300_eff_wide.csv"
    curve_path = output_root / "condition_curves_360.csv"
    _write_csv(final_path, final_rows, FINAL_FIELDS)
    _write_gzip_csv(minute_path, minute_rows, MINUTE_FIELDS)
    _write_csv(run_path, wide_rows, wide_fields)
    _write_csv(curve_path, curves, CURVE_FIELDS)

    final_by_key = {row["logical_key"]: row for row in final_rows}
    endpoint_differences = sorted(
        row["logical_key"] for row in freeze_rows if row["endpoint_matches_final"] == "false"
    )
    metric_contract = {
        "schema_version": "symbolfit_release_metric_contract_v1",
        "scope": "final_and_eff",
        "algorithm": "SymbolFit",
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "datasets": 50,
        "runs": EXPECTED_RUNS,
        "horizon_minutes": HORIZON,
        "final_numeric": {
            "evaluation_path": EVALUATION_PATH,
            "source": "latest final result canonical artifact",
            "invalid_canonical_output": "ID/OOD quality 置 0，并保留 native 指标仅供审计",
            "worker_limit": args.max_workers,
            "blas_threads_per_worker": 1,
        },
        "minute_numeric": {
            "evaluation_path": "execution_runner_canonical_capture",
            "candidate_selector": "minimum source_internal_loss among observed internal candidates",
            "test_metric_used_for_candidate_selection": False,
            "heartbeat_quality": 0.0,
            "reuse_reason": "快照已由指纹冻结的执行期 runner 用同一 canonical artifact 计算 ID/OOD",
            "source_binding_limit": "worker 快照未逐条内嵌执行代码 SHA；依赖 preflight 与 SHA fanout 证据",
        },
        "quality_mapping": "phi(x)=1-(clip(log10(max(x,1e-12)),-12,2)+12)/14",
        "combined_quality": "q(t)=(q_id(t)+q_ood(t))/2",
        "eff": "m_eff=(1/180)*sum_t(q(t)/max_t q(t)); q*=0 时 m_eff=0",
        "q_star_role": "仅用于运行内相对进度归一化，不用于候选选择",
        "endpoint_policy": "EFF 始终使用 minute180 内部搜索候选，不用 final result 回填",
        "endpoint_differs_from_final_keys": endpoint_differences,
        "llm_calls": 0,
        "training_runs": 0,
    }
    contract_path = output_root / "metric_contract.json"
    contract_path.write_text(
        json.dumps(metric_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    valid_final_rows = [row for row in final_rows if row["valid_output"] == "true"]
    invalid_final_rows = [row for row in final_rows if row["valid_output"] == "false"]
    max_final_id_delta = max(
        (abs(float(row["id_nmse_delta_from_native"])) for row in valid_final_rows),
        default=0.0,
    )
    max_final_ood_delta = max(
        (abs(float(row["ood_nmse_delta_from_native"])) for row in valid_final_rows),
        default=0.0,
    )
    curve_endpoints = {
        row["condition"]: float(row["formal_eff_at_minute180"])
        for row in curves
        if int(row["minute"]) == HORIZON
    }
    validation_report = {
        "schema_version": "symbolfit_release_metrics_validation_v1",
        "status": "passed",
        "final_run_count": len(final_rows),
        "minute_point_count": len(minute_rows),
        "run_eff_count": len(wide_rows),
        "curve_point_count": len(curves),
        "unique_final_logical_keys": len(final_by_key),
        "final_valid_count": len(valid_final_rows),
        "final_invalid_output_count": len(invalid_final_rows),
        "final_invalid_output_keys": [row["logical_key"] for row in invalid_final_rows],
        "unique_minute_logical_keys": len({row["logical_key"] for row in minute_rows}),
        "candidate_point_count": sum(row["candidate_available"] == "true" for row in minute_rows),
        "heartbeat_point_count": sum(row["candidate_available"] == "false" for row in minute_rows),
        "carry_forward_point_count": sum(row["carry_forward"] == "true" for row in minute_rows),
        "unique_minute_canonical_artifacts": len(
            {row["canonical_artifact_sha256"] for row in minute_rows if row["canonical_artifact_sha256"]}
        ),
        "max_final_id_nmse_delta_from_native": max_final_id_delta,
        "max_final_ood_nmse_delta_from_native": max_final_ood_delta,
        "endpoint_matches_final_count": EXPECTED_RUNS - len(endpoint_differences),
        "endpoint_differs_from_final_count": len(endpoint_differences),
        "endpoint_differs_from_final_keys": endpoint_differences,
        "condition_eff_scores": curve_endpoints,
        "unresolved_count": 0,
        "api_calls": 0,
        "llm_calls": 0,
        "training_runs": 0,
        "max_workers": args.max_workers,
        "blas_threads_per_worker": 1,
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_manifest = {
        "schema_version": "symbolfit_release_metrics_sources_v1",
        "status": "passed",
        "source_evidence": source_evidence,
        "inputs": {
            "freeze_manifest": _file_record(
                freeze_root / "manifests" / "symbolfit_noise_latest_runs.csv", repo_root, rows=300
            ),
            "freeze_validation": _file_record(
                freeze_root / "reports" / "symbolfit_noise_latest_validation.json", repo_root
            ),
            "preflight_report": _file_record(args.preflight_report.resolve(), repo_root),
            "runner_fanout_script": _file_record(
                args.preflight_report.resolve().parent
                / "commands"
                / "fanout_runner_from_iaaccn22.sh",
                repo_root,
            ),
            "state": _file_record(
                freeze_root
                / "control"
                / "symbolfit_noise_internal_progress_v1_20260910_full.state.json",
                repo_root,
            ),
            "source_csv": _file_record(freeze_root / "control" / "ssr50_source.csv", repo_root, rows=50),
        },
        "implementation": {
            "this_pipeline": _file_record(Path(__file__).resolve(), repo_root),
            "performance_replay": _file_record(
                Path(__file__).resolve().with_name("performance_replay.py"), repo_root
            ),
            "metrics": _file_record(Path(__file__).resolve().with_name("metrics.py"), repo_root),
        },
        "outputs": {
            "final_numeric": _file_record(final_path, repo_root, rows=300),
            "minute_long": _file_record(minute_path, repo_root, rows=EXPECTED_POINTS),
            "run_eff_wide": _file_record(run_path, repo_root, rows=300),
            "condition_curves": _file_record(curve_path, repo_root, rows=360),
            "metric_contract": _file_record(contract_path, repo_root),
            "validation_report": _file_record(report_path, repo_root),
        },
    }
    source_manifest_path = output_root / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    override_manifest = {
        "schema_version": "symbolfit_noise_final_and_eff_override_v1",
        "status": "passed",
        "scope": "final_and_eff",
        "authorization": "user_explicit_2026-09-13",
        "algorithm": "SymbolFit",
        "conditions": list(CONDITIONS),
        "replacement_count": EXPECTED_RUNS,
        "final_replacement_count": EXPECTED_RUNS,
        "eff_replacement_count": EXPECTED_RUNS,
        "replacement_keys": [row["logical_key"] for row in freeze_rows],
        "final_source_policy": "latest frozen final result canonical replay",
        "eff_source_policy": "latest minute1..180 internal-loss-selected snapshots",
        "endpoint_policy": "do not replace minute180 with final result",
        "endpoint_differs_from_final_keys": endpoint_differences,
        "source_freeze_manifest": _file_record(
            freeze_root / "manifests" / "symbolfit_noise_latest_runs.csv", repo_root, rows=300
        ),
        "source_manifest": _file_record(source_manifest_path, repo_root),
        "outputs": source_manifest["outputs"],
        "api_calls": 0,
        "llm_calls": 0,
        "training_runs": 0,
    }
    override_path = output_root / "final_and_eff_override_manifest.json"
    override_path.write_text(
        json.dumps(override_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output_root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{_sha256_file(path)}  {path.relative_to(output_root)}\n")
    print(json.dumps(validation_report, ensure_ascii=False, indent=2, sort_keys=True))
    return validation_report


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    stage_root = repo_root / "AAAI_experiments" / "stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=stage_root / "work" / "final_release_20260913" / "symbolfit_noise_latest",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=stage_root
        / "reruns"
        / "symbolfit_noise_internal_progress_v1_20260910"
        / "preflight_report.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=stage_root
        / "work"
        / "final_release_20260913"
        / "release_v2"
        / "symbolfit_metrics",
    )
    parser.add_argument("--execution-runner-commit", default=EXECUTION_RUNNER_COMMIT)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main() -> int:
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
