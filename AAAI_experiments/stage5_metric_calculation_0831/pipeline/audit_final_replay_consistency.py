"""审计冻结 final 指标与统一 canonical replay 的一致性及来源。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class FinalReplayAuditError(ValueError):
    """输入覆盖、来源绑定或数值字段不满足审计契约。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _logical_key(source: Mapping[str, Any]) -> str:
    algorithm = str(source.get("algorithm") or "")
    dataset = str(source.get("dataset_id") or "")
    condition = str(source.get("noise_tag") or "")
    raw_seed = source.get("seed")
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise FinalReplayAuditError(f"source.seed 无效: {raw_seed!r}") from exc
    if isinstance(raw_seed, bool) or str(raw_seed) != str(seed):
        raise FinalReplayAuditError(f"source.seed 不是 canonical 整数: {raw_seed!r}")
    if not algorithm or not dataset or not condition or "::" in algorithm or "::" in dataset:
        raise FinalReplayAuditError(f"source logical identity 无效: {source!r}")
    return f"{algorithm}::{dataset}::s{seed}::{condition}"


def _reference_source(payload: Mapping[str, Any]) -> str:
    markers: dict[str, bool] = {}
    for field in ("recovered_from_timeout", "recovered_from_error"):
        value = payload.get(field, False)
        if not isinstance(value, bool):
            raise FinalReplayAuditError(f"{field} 必须是 JSON bool")
        markers[field] = value
    if all(markers.values()):
        raise FinalReplayAuditError("recovery source 标记不能同时为 true")
    if markers["recovered_from_timeout"]:
        return "canonical_recovery_timeout"
    if markers["recovered_from_error"]:
        return "canonical_recovery_error"
    return "native_predict"


def _finite_float(value: object, *, context: str, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalReplayAuditError(f"{context} 不是数值: {value!r}") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise FinalReplayAuditError(f"{context} 不是有限值: {value!r}")
    return number


def _run_identity(
    row: Mapping[str, Any], *, expected_condition: str
) -> tuple[str, str, int, str]:
    algorithm = str(row.get("algorithm") or "")
    dataset = str(row.get("dataset_id") or "")
    condition = str(row.get("noise_tag") or "")
    if condition != expected_condition:
        raise FinalReplayAuditError(
            f"run CSV condition={condition!r}，预期 {expected_condition!r}"
        )
    raw_seed = row.get("seed")
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise FinalReplayAuditError(f"run CSV seed 无效: {raw_seed!r}") from exc
    if isinstance(raw_seed, bool) or str(raw_seed) != str(seed):
        raise FinalReplayAuditError(f"run CSV seed 不是 canonical 整数: {raw_seed!r}")
    if not algorithm or not dataset or "::" in algorithm or "::" in dataset:
        raise FinalReplayAuditError("run CSV algorithm/dataset 无效")
    canonical_key = f"{algorithm}::{dataset}::s{seed}::{condition}"
    if row.get("logical_key") != canonical_key:
        raise FinalReplayAuditError(
            f"run CSV logical_key 与 algorithm/dataset/seed/condition 不一致: "
            f"{row.get('logical_key')!r} != {canonical_key!r}"
        )
    return algorithm, dataset, seed, condition


def _validate_grid(
    identities: Sequence[tuple[str, str, int, str]],
    *,
    expected_runs: int,
    expected_algorithm_count: int,
    expected_dataset_count: int,
    expected_seeds: Sequence[int],
) -> None:
    seeds = tuple(int(seed) for seed in expected_seeds)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise FinalReplayAuditError("期望 seed 网格为空或重复")
    expected_grid_size = expected_algorithm_count * expected_dataset_count * len(seeds)
    if expected_grid_size != expected_runs:
        raise FinalReplayAuditError(
            f"期望网格大小 {expected_grid_size} 与 expected_runs={expected_runs} 不一致"
        )
    algorithms = {identity[0] for identity in identities}
    datasets = {identity[1] for identity in identities}
    actual = {(algorithm, dataset, seed) for algorithm, dataset, seed, _ in identities}
    if len(actual) != len(identities):
        raise FinalReplayAuditError("run CSV 网格存在重复 algorithm/dataset/seed")
    if len(algorithms) != expected_algorithm_count:
        raise FinalReplayAuditError(
            f"算法网格应为 {expected_algorithm_count}，实际为 {len(algorithms)}"
        )
    if len(datasets) != expected_dataset_count:
        raise FinalReplayAuditError(
            f"数据集网格应为 {expected_dataset_count}，实际为 {len(datasets)}"
        )
    expected = set(product(algorithms, datasets, seeds))
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise FinalReplayAuditError(f"run CSV 不是完整笛卡尔网格: missing={missing}, extra={extra}")


def _validate_evaluation_fields(row: Mapping[str, Any], *, key: str) -> str:
    status = str(row.get("evaluation_status") or "")
    if status not in {"valid", "invalid_output", "replay_unavailable"}:
        raise FinalReplayAuditError(f"{key}.evaluation_status 枚举无效: {status!r}")
    valid_output = str(row.get("valid_output") or "")
    invalid_reason = str(row.get("invalid_reason") or "")
    replay_error = str(row.get("replay_error") or "")
    id_nmse = row.get("id_nmse")
    ood_nmse = row.get("ood_nmse")
    id_delta = row.get("id_quality_delta_from_native")
    ood_delta = row.get("ood_quality_delta_from_native")
    pair_values = (id_nmse, ood_nmse)
    delta_values = (id_delta, ood_delta)
    if status == "valid":
        valid = (
            valid_output == "true"
            and not invalid_reason
            and not replay_error
            and all(value not in (None, "") for value in pair_values + delta_values)
        )
    elif status == "invalid_output":
        valid = (
            valid_output == "false"
            and bool(invalid_reason)
            and not replay_error
            and all(value in (None, "") for value in pair_values)
            and all(value not in (None, "") for value in delta_values)
        )
    else:
        valid = (
            valid_output == ""
            and not invalid_reason
            and bool(replay_error)
            and all(value in (None, "") for value in pair_values + delta_values)
        )
    if not valid:
        raise FinalReplayAuditError(f"{key}.{status} 字段组合不一致")
    return status


def _relative_error(reference: float, replay: float) -> float:
    return abs(replay - reference) / max(abs(reference), 1.0e-300)


def load_freeze_provenance(
    freeze_paths: Sequence[Path],
    *,
    expected_condition: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """加载冻结 result 元数据，并验证 raw_text SHA 与 logical key 唯一性。"""

    provenance: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for raw_path in freeze_paths:
        path = raw_path.resolve()
        inputs.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        try:
            handle = gzip.open(path, "rt", encoding="utf-8")
        except OSError as exc:
            raise FinalReplayAuditError(f"无法打开冻结包 {path}: {exc}") from exc
        with handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    source = record["source"]
                    result = record["result"]
                    raw_text = result["raw_text"]
                    payload = json.loads(raw_text)
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise FinalReplayAuditError(
                        f"{path}:{line_number} 冻结记录无法解析: {exc}"
                    ) from exc
                if not isinstance(source, Mapping) or not isinstance(payload, Mapping):
                    raise FinalReplayAuditError(f"{path}:{line_number} source/result 必须是 object")
                if str(source.get("noise_tag")) != expected_condition:
                    raise FinalReplayAuditError(
                        f"{path}:{line_number} condition={source.get('noise_tag')!r}，"
                        f"预期 {expected_condition!r}"
                    )
                expected_sha = result.get("sha256")
                actual_sha = _sha256_bytes(str(raw_text).encode("utf-8"))
                if expected_sha != actual_sha:
                    raise FinalReplayAuditError(f"{path}:{line_number} result raw_text SHA 漂移")
                key = _logical_key(source)
                if key in provenance:
                    raise FinalReplayAuditError(f"冻结包存在重复 logical_key: {key}")
                provenance[key] = {
                    "reference_source": _reference_source(payload),
                    "termination_reason": str(payload.get("termination_reason") or ""),
                    "timeout_type": str(payload.get("timeout_type") or ""),
                    "result_sha256": actual_sha,
                    "bundle_path": str(path),
                }
    return provenance, inputs


def audit_final_replay_consistency(
    *,
    run_csv: Path,
    freeze_paths: Sequence[Path],
    condition: str,
    output_csv: Path,
    output_report: Path,
    expected_runs: int = 2250,
    expected_algorithm_count: int = 15,
    expected_dataset_count: int = 50,
    expected_seeds: Sequence[int] = (520, 521, 522),
    relative_nmse_tolerance: float = 1.0e-6,
    quality_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """生成逐 run 与逐算法审计，严格区分 native 和恢复回放。"""

    if relative_nmse_tolerance < 0 or quality_tolerance < 0:
        raise FinalReplayAuditError("审计阈值必须非负")
    provenance, freeze_inputs = load_freeze_provenance(
        freeze_paths,
        expected_condition=condition,
    )
    with run_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != expected_runs:
        raise FinalReplayAuditError(
            f"run CSV 行数应为 {expected_runs}，实际为 {len(source_rows)}"
        )
    identities = [
        _run_identity(row, expected_condition=condition) for row in source_rows
    ]
    _validate_grid(
        identities,
        expected_runs=expected_runs,
        expected_algorithm_count=expected_algorithm_count,
        expected_dataset_count=expected_dataset_count,
        expected_seeds=expected_seeds,
    )
    if len(provenance) != expected_runs:
        raise FinalReplayAuditError(
            f"冻结 provenance 行数应为 {expected_runs}，实际为 {len(provenance)}"
        )

    seen: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = str(row.get("logical_key") or "")
        if not key or key in seen:
            raise FinalReplayAuditError(f"run CSV logical_key 缺失或重复: {key!r}")
        seen.add(key)
        meta = provenance.get(key)
        if meta is None:
            raise FinalReplayAuditError(f"run CSV 在冻结 provenance 中缺失: {key}")
        if row.get("result_sha256") != meta["result_sha256"]:
            raise FinalReplayAuditError(f"{key}: run CSV result SHA 与冻结包不一致")

        evaluation_status = _validate_evaluation_fields(row, key=key)
        reference_id = _finite_float(
            row.get("native_id_nmse"),
            context=f"{key}.reference_id_nmse",
            nonnegative=True,
        )
        reference_ood = _finite_float(
            row.get("native_ood_nmse"),
            context=f"{key}.reference_ood_nmse",
            nonnegative=True,
        )
        id_quality_delta = row.get("id_quality_delta_from_native")
        ood_quality_delta = row.get("ood_quality_delta_from_native")
        quality_comparable = id_quality_delta not in (None, "") and ood_quality_delta not in (None, "")
        if quality_comparable:
            id_quality_delta_value = _finite_float(
                id_quality_delta, context=f"{key}.id_quality_delta"
            )
            ood_quality_delta_value = _finite_float(
                ood_quality_delta, context=f"{key}.ood_quality_delta"
            )
            quality_consistent: bool | None = (
                abs(id_quality_delta_value) <= quality_tolerance
                and abs(ood_quality_delta_value) <= quality_tolerance
            )
        else:
            id_quality_delta_value = None
            ood_quality_delta_value = None
            quality_consistent = None

        nmse_comparable = (
            evaluation_status == "valid"
            and row.get("id_nmse") not in (None, "")
            and row.get("ood_nmse") not in (None, "")
        )
        if nmse_comparable:
            replay_id = _finite_float(
                row.get("id_nmse"), context=f"{key}.replay_id_nmse", nonnegative=True
            )
            replay_ood = _finite_float(
                row.get("ood_nmse"), context=f"{key}.replay_ood_nmse", nonnegative=True
            )
            id_relative_error = _relative_error(reference_id, replay_id)
            ood_relative_error = _relative_error(reference_ood, replay_ood)
            nmse_consistent: bool | None = (
                id_relative_error <= relative_nmse_tolerance
                and ood_relative_error <= relative_nmse_tolerance
            )
        else:
            replay_id = None
            replay_ood = None
            id_relative_error = None
            ood_relative_error = None
            nmse_consistent = None

        reference_source = str(meta["reference_source"])
        audit_rows.append(
            {
                "logical_key": key,
                "algorithm": str(row.get("algorithm") or ""),
                "dataset_id": str(row.get("dataset_id") or ""),
                "seed": str(row.get("seed") or ""),
                "condition": condition,
                "reference_source": reference_source,
                "native_fidelity_applicable": str(reference_source == "native_predict").lower(),
                "termination_reason": meta["termination_reason"],
                "timeout_type": meta["timeout_type"],
                "evaluation_status": evaluation_status,
                "reference_id_nmse": f"{reference_id:.17g}",
                "reference_ood_nmse": f"{reference_ood:.17g}",
                "replay_id_nmse": "" if replay_id is None else f"{replay_id:.17g}",
                "replay_ood_nmse": "" if replay_ood is None else f"{replay_ood:.17g}",
                "id_relative_nmse_error": ""
                if id_relative_error is None
                else f"{id_relative_error:.17g}",
                "ood_relative_nmse_error": ""
                if ood_relative_error is None
                else f"{ood_relative_error:.17g}",
                "strict_nmse_consistent": ""
                if nmse_consistent is None
                else str(nmse_consistent).lower(),
                "id_quality_delta": ""
                if id_quality_delta_value is None
                else f"{id_quality_delta_value:.17g}",
                "ood_quality_delta": ""
                if ood_quality_delta_value is None
                else f"{ood_quality_delta_value:.17g}",
                "formal_quality_consistent": ""
                if quality_consistent is None
                else str(quality_consistent).lower(),
                "replay_error": str(row.get("replay_error") or ""),
                "invalid_reason": str(row.get("invalid_reason") or ""),
                "result_sha256": meta["result_sha256"],
                "bundle_path": meta["bundle_path"],
            }
        )

    extra = sorted(set(provenance) - seen)
    if extra:
        raise FinalReplayAuditError(f"冻结 provenance 存在未消费记录: {extra[:5]}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit_rows:
        counts = by_algorithm[row["algorithm"]]
        counts["runs"] += 1
        counts[f"reference_source:{row['reference_source']}"] += 1
        counts[f"evaluation_status:{row['evaluation_status']}"] += 1
        scope = "native" if row["native_fidelity_applicable"] == "true" else "recovery"
        strict = row["strict_nmse_consistent"]
        formal = row["formal_quality_consistent"]
        counts[f"{scope}_strict_{'unavailable' if strict == '' else strict}"] += 1
        counts[f"{scope}_formal_{'unavailable' if formal == '' else formal}"] += 1

    algorithm_summaries: list[dict[str, Any]] = []
    for algorithm in sorted(by_algorithm):
        counts = by_algorithm[algorithm]
        algorithm_summaries.append(
            {
                "algorithm": algorithm,
                "runs": counts["runs"],
                "native_predict_runs": counts["reference_source:native_predict"],
                "canonical_recovery_timeout_runs": counts[
                    "reference_source:canonical_recovery_timeout"
                ],
                "canonical_recovery_error_runs": counts[
                    "reference_source:canonical_recovery_error"
                ],
                "replay_valid": counts["evaluation_status:valid"],
                "replay_invalid_output": counts["evaluation_status:invalid_output"],
                "replay_unavailable": counts["evaluation_status:replay_unavailable"],
                "native_strict_pass": counts["native_strict_true"],
                "native_strict_fail": counts["native_strict_false"],
                "native_strict_unavailable": counts["native_strict_unavailable"],
                "native_formal_pass": counts["native_formal_true"],
                "native_formal_fail": counts["native_formal_false"],
                "native_formal_unavailable": counts["native_formal_unavailable"],
                "recovery_strict_pass": counts["recovery_strict_true"],
                "recovery_strict_fail": counts["recovery_strict_false"],
                "recovery_strict_unavailable": counts["recovery_strict_unavailable"],
                "recovery_formal_pass": counts["recovery_formal_true"],
                "recovery_formal_fail": counts["recovery_formal_false"],
                "recovery_formal_unavailable": counts["recovery_formal_unavailable"],
            }
        )

    native_formal_failures = sum(item["native_formal_fail"] for item in algorithm_summaries)
    native_formal_unavailable = sum(
        item["native_formal_unavailable"] for item in algorithm_summaries
    )
    replay_unavailable = sum(
        item["replay_unavailable"] for item in algorithm_summaries
    )
    contract_ok = replay_unavailable == 0
    report = {
        "status": "ok" if contract_ok else "error",
        "contract_ok": contract_ok,
        "condition": condition,
        "thresholds": {
            "relative_nmse": relative_nmse_tolerance,
            "absolute_quality": quality_tolerance,
        },
        "grid": {
            "expected_runs": expected_runs,
            "algorithm_count": expected_algorithm_count,
            "dataset_count": expected_dataset_count,
            "seeds": [int(seed) for seed in expected_seeds],
            "complete_cartesian_product": True,
        },
        "summary": {
            "runs": len(audit_rows),
            "algorithms": len(algorithm_summaries),
            "native_predict_runs": sum(
                item["native_predict_runs"] for item in algorithm_summaries
            ),
            "canonical_recovery_runs": sum(
                item["canonical_recovery_timeout_runs"]
                + item["canonical_recovery_error_runs"]
                for item in algorithm_summaries
            ),
            "native_formal_failures": native_formal_failures,
            "native_formal_unavailable": native_formal_unavailable,
            "replay_unavailable": replay_unavailable,
            "native_formal_ready": native_formal_failures == 0
            and native_formal_unavailable == 0
            and replay_unavailable == 0,
        },
        "algorithms": algorithm_summaries,
        "inputs": {
            "run_csv": {
                "path": str(run_csv.resolve()),
                "sha256": _sha256_file(run_csv),
            },
            "freeze_bundles": freeze_inputs,
        },
        "output": {
            "run_audit_csv": str(output_csv.resolve()),
            "sha256": _sha256_file(output_csv),
            "row_count": len(audit_rows),
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计 final/reference 与 canonical replay 一致性")
    parser.add_argument("--run-csv", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, action="append", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=2250)
    parser.add_argument("--expected-algorithm-count", type=int, default=15)
    parser.add_argument("--expected-dataset-count", type=int, default=50)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[520, 521, 522])
    parser.add_argument("--relative-nmse-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--quality-tolerance", type=float, default=1.0e-6)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = audit_final_replay_consistency(
        run_csv=args.run_csv.resolve(),
        freeze_paths=[path.resolve() for path in args.freeze],
        condition=args.condition,
        output_csv=args.output_csv.resolve(),
        output_report=args.output_report.resolve(),
        expected_runs=args.expected_runs,
        expected_algorithm_count=args.expected_algorithm_count,
        expected_dataset_count=args.expected_dataset_count,
        expected_seeds=args.expected_seeds,
        relative_nmse_tolerance=args.relative_nmse_tolerance,
        quality_tolerance=args.quality_tolerance,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if report["contract_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
