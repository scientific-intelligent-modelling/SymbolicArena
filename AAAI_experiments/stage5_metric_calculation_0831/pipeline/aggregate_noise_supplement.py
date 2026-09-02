"""聚合 Stage5 噪声条件补充指标，不改动 clean 正式六轴。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SEEDS = (520, 521, 522)
NOISE_CONDITIONS = ("noise001", "noise005")
DEFAULT_EXPECTED_RUNS_PER_CONDITION = 2250
DEFAULT_EXPECTED_ALGORITHMS = 15
DEFAULT_EXPECTED_DATASETS = 50
EQUIVALENCE_STATES = {"frozen", "non_applicable"}
EQUIVALENCE_DECISIONS = {"equivalent", "not_equivalent", "undetermined"}
EQUIVALENCE_EVIDENCE_BASES = {
    "symbolic_proof",
    "numerical_support",
    "structural_analysis",
    "mixed",
    "insufficient",
}
RUN_FIELDS = (
    "logical_key",
    "algorithm",
    "dataset_id",
    "seed",
    "noise_tag",
    "task_id",
    "host",
    "valid_output",
    "pred_state",
    "equivalence_state",
    "id_quality",
    "ood_quality",
    "m_eff",
    "equivalence_decision",
    "equivalent",
    "tree_similarity",
    "variable_f1",
    "operator_f1",
    "m_sym",
    "reference_complexity",
    "predicted_complexity",
    "m_min",
    "gt_logical_id",
    "pred_logical_id",
    "equivalence_logical_id",
)
FORMAL_SIX_AXIS_FIELDS = (
    "m_eff",
    "tree_similarity",
    "variable_f1",
    "operator_f1",
    "m_sym",
    "reference_complexity",
    "predicted_complexity",
    "m_min",
    "gt_logical_id",
)
SUPPLEMENT_FIELDS = (
    "algorithm",
    "noise_tag",
    "run_count",
    "valid_output_count",
    "valid_output_rate",
    "mean_id_quality",
    "mean_ood_quality",
    "clean_mean_id_quality",
    "clean_mean_ood_quality",
    "id_quality_retention",
    "id_quality_drop",
    "ood_quality_retention",
    "ood_quality_drop",
    "equivalence_applicable_count",
    "equivalence_non_applicable_count",
    "equivalence_covered_count",
    "equivalence_decided_count",
    "equivalent_count",
    "equivalence_coverage",
    "equivalence_rate",
)
LOGICAL_KEY_RE = re.compile(r"^([^:]+)::([^:]+)::s(520|521|522)::(clean|noise001|noise005)$")
EQUIVALENCE_LOGICAL_ID_RE = re.compile(
    r"^equivalence::([a-z0-9_-]+)::(g\d{4})::s(520|521|522)::"
    r"(noise001|noise005)(?:::v([1-9]\d*))?$"
)


class AggregateNoiseSupplementError(ValueError):
    """噪声补充指标输入不满足闭环契约。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage5_root() -> Path:
    return _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"


def _raise(message: str) -> None:
    raise AggregateNoiseSupplementError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    _raise(f"{context} 必须为 true/false，实际为 {value!r}")
    raise AssertionError("unreachable")


def _nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(f"{context} 必须是非空字符串")
    return value.strip()


def _parse_seed(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        _raise(f"{context} 不是合法 seed")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AggregateNoiseSupplementError(f"{context} 不是合法 seed: {value!r}") from exc
    if seed not in SEEDS:
        _raise(f"{context} 必须在 {SEEDS} 中，实际为 {seed}")
    return seed


def _unit_float(value: object, *, context: str) -> float:
    if value is None or isinstance(value, bool):
        _raise(f"{context} 缺失或不是数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AggregateNoiseSupplementError(f"{context} 不是合法数值: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        _raise(f"{context} 必须在 [0, 1] 内，实际为 {value!r}")
    return number


def _artifact_info(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
    if row_count is not None:
        info["row_count"] = row_count
    return info


def _read_csv(path: Path, *, label: str, required_fields: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        _raise(f"{label} 不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(required_fields - set(fields))
        if missing:
            _raise(f"{label} 缺少字段: {missing}")
        rows = [dict(row) for row in reader]
    return rows, fields


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _raise(f"{label} 不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                _raise(f"{label}:{line_number} 不得是空行")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise AggregateNoiseSupplementError(f"{label}:{line_number} 不是合法 JSON") from exc
            if not isinstance(row, dict):
                _raise(f"{label}:{line_number} 顶层必须是 object")
            rows.append(row)
    return rows


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        _raise(f"{label} 不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AggregateNoiseSupplementError(f"{label} 不是合法 JSON") from exc
    if not isinstance(payload, dict):
        _raise(f"{label} 顶层必须是 object")
    return payload


def _validate_logical_key(row: Mapping[str, object], *, condition: str, context: str) -> tuple[str, str, int]:
    logical_key = _nonempty_string(row.get("logical_key"), context=f"{context}.logical_key")
    match = LOGICAL_KEY_RE.fullmatch(logical_key)
    if match is None:
        _raise(f"{context}.logical_key 格式非法: {logical_key!r}")
    algorithm = _nonempty_string(row.get("algorithm"), context=f"{context}.algorithm")
    dataset_id = _nonempty_string(row.get("dataset_id"), context=f"{context}.dataset_id")
    seed = _parse_seed(row.get("seed"), context=f"{context}.seed")
    noise_tag = _nonempty_string(row.get("noise_tag"), context=f"{context}.noise_tag")
    if (match.group(1), match.group(2), int(match.group(3)), match.group(4)) != (
        algorithm,
        dataset_id,
        seed,
        condition,
    ) or noise_tag != condition:
        _raise(f"{context} logical_key 与 algorithm/dataset_id/seed/noise_tag 不一致")
    return algorithm, dataset_id, seed


def _validate_grid(
    keys: set[tuple[str, str, int]],
    *,
    algorithms: set[str],
    datasets: set[str],
    expected_runs: int,
    expected_algorithms: int,
    expected_datasets: int,
    context: str,
) -> None:
    if len(keys) != expected_runs:
        _raise(f"{context} 行数应为 {expected_runs}，实际为 {len(keys)}")
    if len(algorithms) != expected_algorithms:
        _raise(f"{context} 算法数应为 {expected_algorithms}，实际为 {len(algorithms)}")
    if len(datasets) != expected_datasets:
        _raise(f"{context} 数据集数应为 {expected_datasets}，实际为 {len(datasets)}")
    expected = {(algorithm, dataset, seed) for algorithm in algorithms for dataset in datasets for seed in SEEDS}
    missing = expected - keys
    unexpected = keys - expected
    if missing or unexpected:
        _raise(
            f"{context} algorithm/dataset/seed 网格不完整: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def _load_clean_rows(
    path: Path,
    *,
    expected_runs: int,
    expected_algorithms: int,
    expected_datasets: int,
) -> tuple[list[dict[str, str]], dict[tuple[str, str, int], dict[str, Any]], set[str], set[str], dict[str, Any]]:
    rows, fields = _read_csv(path, label="clean_run_metrics.csv", required_fields=set(RUN_FIELDS))
    if tuple(fields) != RUN_FIELDS:
        _raise("clean_run_metrics.csv 字段或字段顺序不符合正式 run 指标契约")
    clean: dict[tuple[str, str, int], dict[str, Any]] = {}
    algorithms: set[str] = set()
    datasets: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        context = f"clean_run_metrics.csv:{row_number}"
        key = _validate_logical_key(row, condition="clean", context=context)
        if key in clean:
            _raise(f"clean_run_metrics.csv 出现重复键: {key}")
        valid = _parse_bool(row["valid_output"], context=f"{context}.valid_output")
        pred_state = _nonempty_string(row["pred_state"], context=f"{context}.pred_state")
        equivalence_state = _nonempty_string(
            row["equivalence_state"], context=f"{context}.equivalence_state"
        )
        if pred_state not in {"frozen", "non_applicable", "exhausted"}:
            _raise(f"{context}.pred_state 枚举非法: {pred_state!r}")
        if equivalence_state not in EQUIVALENCE_STATES:
            _raise(f"{context}.equivalence_state 枚举非法: {equivalence_state!r}")
        decision = _nonempty_string(
            row["equivalence_decision"], context=f"{context}.equivalence_decision"
        )
        if decision not in EQUIVALENCE_DECISIONS | {"non_applicable"}:
            _raise(f"{context}.equivalence_decision 枚举非法: {decision!r}")
        equivalent = _parse_bool(row["equivalent"], context=f"{context}.equivalent")
        if equivalent != (decision == "equivalent"):
            _raise(f"{context}.equivalent 与 equivalence_decision 不一致")
        id_quality = _unit_float(row["id_quality"], context=f"{context}.id_quality")
        ood_quality = _unit_float(row["ood_quality"], context=f"{context}.ood_quality")
        if not valid and (id_quality != 0.0 or ood_quality != 0.0):
            _raise(f"{context} valid_output=false 时质量必须为 0")
        clean[key] = {"id_quality": id_quality, "ood_quality": ood_quality, "valid": valid}
        algorithms.add(key[0])
        datasets.add(key[1])
    _validate_grid(
        set(clean),
        algorithms=algorithms,
        datasets=datasets,
        expected_runs=expected_runs,
        expected_algorithms=expected_algorithms,
        expected_datasets=expected_datasets,
        context="clean_run_metrics.csv",
    )
    return rows, clean, algorithms, datasets, _artifact_info(path, row_count=len(rows))


def _load_numeric_rows(
    path: Path,
    *,
    condition: str,
    expected_runs: int,
    algorithms: set[str],
    datasets: set[str],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    required = {
        "logical_key",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "valid_output",
        "id_quality",
        "ood_quality",
    }
    rows, _ = _read_csv(path, label=f"{condition}_numeric_run_metrics.csv", required_fields=required)
    numeric: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        context = f"{condition}_numeric_run_metrics.csv:{row_number}"
        key = _validate_logical_key(row, condition=condition, context=context)
        if key in numeric:
            _raise(f"{condition}_numeric_run_metrics.csv 出现重复键: {key}")
        valid = _parse_bool(row["valid_output"], context=f"{context}.valid_output")
        id_quality = _unit_float(row["id_quality"], context=f"{context}.id_quality")
        ood_quality = _unit_float(row["ood_quality"], context=f"{context}.ood_quality")
        if not valid and (id_quality != 0.0 or ood_quality != 0.0):
            _raise(f"{context} valid_output=false 时质量必须为 0")
        numeric[key] = {
            "row": row,
            "valid": valid,
            "id_quality": id_quality,
            "ood_quality": ood_quality,
        }
    _validate_grid(
        set(numeric),
        algorithms=algorithms,
        datasets=datasets,
        expected_runs=expected_runs,
        expected_algorithms=len(algorithms),
        expected_datasets=len(datasets),
        context=f"{condition}_numeric_run_metrics.csv",
    )
    return numeric, _artifact_info(path, row_count=len(rows))


def _resolve_declared_path(value: object, *, summary_path: Path, context: str) -> Path:
    declared = Path(_nonempty_string(value, context=context))
    if declared.is_absolute():
        return declared.resolve()
    repo_candidate = (_repo_root() / declared).resolve()
    summary_candidate = (summary_path.parent / declared).resolve()
    return repo_candidate if repo_candidate.exists() else summary_candidate


def _load_equivalence_bundle(
    *,
    condition: str,
    plan_path: Path,
    index_path: Path,
    summary_path: Path,
    numeric: Mapping[tuple[str, str, int], Mapping[str, Any]],
    expected_runs: int,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    plan_rows = _read_jsonl(plan_path, label=f"{condition} equivalence plan")
    if len(plan_rows) != expected_runs:
        _raise(f"{condition} equivalence plan 行数应为 {expected_runs}，实际为 {len(plan_rows)}")
    plan_sha = _sha256_file(plan_path)
    plan_by_logical_id: dict[str, dict[str, Any]] = {}
    plan_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    evaluation_keys: set[str] = set()
    for row_number, row in enumerate(plan_rows, start=1):
        context = f"{condition} equivalence plan:{row_number}"
        if row.get("condition") != condition or row.get("task_type") != "equivalence":
            _raise(f"{context} condition/task_type 不一致")
        logical_id = _nonempty_string(row.get("logical_id"), context=f"{context}.logical_id")
        match = EQUIVALENCE_LOGICAL_ID_RE.fullmatch(logical_id)
        if (
            match is None
            or match.group(4) != condition
            or (match.group(5) is not None and int(match.group(5)) < 2)
        ):
            _raise(f"{context}.logical_id 格式或 condition 非法: {logical_id!r}")
        evaluation_key = _nonempty_string(
            row.get("evaluation_key"), context=f"{context}.evaluation_key"
        )
        if len(evaluation_key) != 64 or any(ch not in "0123456789abcdef" for ch in evaluation_key.lower()):
            _raise(f"{context}.evaluation_key 必须为 SHA256")
        if logical_id in plan_by_logical_id or evaluation_key in evaluation_keys:
            _raise(f"{condition} equivalence plan 出现重复 logical_id/evaluation_key")
        request = row.get("request")
        if not isinstance(request, Mapping):
            _raise(f"{context}.request 必须为 object")
        algorithm = _nonempty_string(request.get("algorithm"), context=f"{context}.request.algorithm")
        dataset_id = _nonempty_string(request.get("dataset_id"), context=f"{context}.request.dataset_id")
        dataset_index = _nonempty_string(
            request.get("dataset_index"), context=f"{context}.request.dataset_index"
        )
        seed = _parse_seed(request.get("seed"), context=f"{context}.request.seed")
        if request.get("noise_tag") != condition:
            _raise(f"{context}.request.noise_tag 必须为 {condition}")
        if algorithm.lower() != match.group(1).lower() or dataset_index != match.group(2) or seed != int(match.group(3)):
            _raise(f"{context} logical_id 与 request 不一致")
        key = (algorithm, dataset_id, seed)
        numeric_row = numeric.get(key)
        if numeric_row is None:
            _raise(f"{context} 无对应 numeric run: {key}")
        if key in plan_by_key:
            _raise(f"{condition} equivalence plan 出现重复 run 键: {key}")
        prediction_valid = _parse_bool(
            request.get("prediction_valid_output"),
            context=f"{context}.request.prediction_valid_output",
        )
        if prediction_valid != bool(numeric_row["valid"]):
            _raise(f"{context}.request.prediction_valid_output 与 numeric 不一致")
        prediction_task_id = request.get("prediction_task_id")
        if prediction_task_id is not None and str(prediction_task_id) != str(numeric_row["row"]["task_id"]):
            _raise(f"{context}.request.prediction_task_id 与 numeric 不一致")
        plan_by_logical_id[logical_id] = row
        plan_by_key[key] = row
        evaluation_keys.add(evaluation_key)
    if set(plan_by_key) != set(numeric):
        _raise(f"{condition} equivalence plan 与 numeric 网格不一致")

    index_rows = _read_jsonl(index_path, label=f"{condition} equivalence index")
    if len(index_rows) != expected_runs:
        _raise(f"{condition} equivalence index 行数应为 {expected_runs}，实际为 {len(index_rows)}")
    adjudication_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    logical_ids_seen: set[str] = set()
    state_counts: Counter[str] = Counter()
    for row_number, row in enumerate(index_rows, start=1):
        context = f"{condition} equivalence index:{row_number}"
        if row.get("condition") != condition or row.get("task_type") != "equivalence":
            _raise(f"{context} condition/task_type 不一致")
        if row.get("plan_sha256") != plan_sha:
            _raise(f"{context}.plan_sha256 与输入 plan 不一致")
        logical_id = _nonempty_string(row.get("logical_id"), context=f"{context}.logical_id")
        if logical_id in logical_ids_seen:
            _raise(f"{condition} equivalence index 出现重复 logical_id: {logical_id}")
        plan_row = plan_by_logical_id.get(logical_id)
        if plan_row is None or row.get("evaluation_key") != plan_row.get("evaluation_key"):
            _raise(f"{context} 未绑定对应 plan 行")
        request = plan_row["request"]
        assert isinstance(request, Mapping)
        key = (str(request["algorithm"]), str(request["dataset_id"]), int(request["seed"]))
        state = _nonempty_string(row.get("state"), context=f"{context}.state")
        if state not in EQUIVALENCE_STATES:
            _raise(f"{context}.state 枚举非法: {state!r}")
        structured_output = row.get("structured_output")
        if state == "frozen":
            if not isinstance(structured_output, Mapping):
                _raise(f"{context}.structured_output 缺失")
            decision = _nonempty_string(
                structured_output.get("decision"), context=f"{context}.decision"
            )
            if decision not in EQUIVALENCE_DECISIONS:
                _raise(f"{context}.decision 枚举非法: {decision!r}")
            evidence_basis = _nonempty_string(
                structured_output.get("evidence_basis"), context=f"{context}.evidence_basis"
            )
            if evidence_basis not in EQUIVALENCE_EVIDENCE_BASES:
                _raise(f"{context}.evidence_basis 枚举非法: {evidence_basis!r}")
            pred_state = "frozen" if request.get("prediction_logical_id") else "non_applicable"
        else:
            if structured_output is not None:
                _raise(f"{context} non_applicable 时 structured_output 必须为 null")
            non_applicable = row.get("non_applicable")
            if not isinstance(non_applicable, Mapping):
                _raise(f"{context}.non_applicable 缺失")
            _nonempty_string(non_applicable.get("reason"), context=f"{context}.non_applicable.reason")
            decision = "non_applicable"
            pred_state = "frozen" if request.get("prediction_logical_id") else "non_applicable"
        adjudication_by_key[key] = {
            "state": state,
            "decision": decision,
            "equivalent": decision == "equivalent",
            "pred_state": pred_state,
            "logical_id": logical_id,
            "pred_logical_id": str(request.get("prediction_logical_id") or ""),
        }
        logical_ids_seen.add(logical_id)
        state_counts[state] += 1
    if set(adjudication_by_key) != set(numeric):
        _raise(f"{condition} equivalence index 与 numeric 网格不一致")

    summary = _read_json(summary_path, label=f"{condition} equivalence summary")
    if summary.get("status") != "ok":
        _raise(f"{condition} equivalence summary.status 必须为 ok")
    if summary.get("plan_sha256") != plan_sha:
        _raise(f"{condition} equivalence summary.plan_sha256 与 plan 不一致")
    if summary.get("output_sha256") != _sha256_file(index_path):
        _raise(f"{condition} equivalence summary.output_sha256 与 index 不一致")
    if summary.get("row_count") != len(index_rows):
        _raise(f"{condition} equivalence summary.row_count 与 index 不一致")
    declared_counts = summary.get("state_counts")
    if not isinstance(declared_counts, Mapping):
        _raise(f"{condition} equivalence summary.state_counts 缺失")
    for state in EQUIVALENCE_STATES:
        try:
            declared_count = int(declared_counts.get(state, 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise AggregateNoiseSupplementError(
                f"{condition} equivalence summary.state_counts.{state} 非法"
            ) from exc
        if declared_count != state_counts[state]:
            _raise(f"{condition} equivalence summary.state_counts.{state} 与 index 不一致")
    if set(declared_counts) - EQUIVALENCE_STATES:
        _raise(f"{condition} equivalence summary.state_counts 含非法枚举")
    if _resolve_declared_path(
        summary.get("plan_jsonl"), summary_path=summary_path, context="summary.plan_jsonl"
    ) != plan_path.resolve():
        _raise(f"{condition} equivalence summary.plan_jsonl 与输入 plan 路径不一致")
    if _resolve_declared_path(
        summary.get("output_jsonl"), summary_path=summary_path, context="summary.output_jsonl"
    ) != index_path.resolve():
        _raise(f"{condition} equivalence summary.output_jsonl 与输入 index 路径不一致")

    info = {
        "plan_jsonl": _artifact_info(plan_path, row_count=len(plan_rows)),
        "index_jsonl": _artifact_info(index_path, row_count=len(index_rows)),
        "summary_json": _artifact_info(summary_path),
    }
    return adjudication_by_key, info


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.17g}"


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def aggregate_noise_supplement(
    *,
    clean_run_csv: Path,
    noise001_numeric_csv: Path,
    noise005_numeric_csv: Path,
    noise001_equivalence_plan_jsonl: Path,
    noise001_equivalence_index_jsonl: Path,
    noise001_equivalence_summary_json: Path,
    noise005_equivalence_plan_jsonl: Path,
    noise005_equivalence_index_jsonl: Path,
    noise005_equivalence_summary_json: Path,
    all_conditions_csv: Path,
    noise_supplement_csv: Path,
    report_json: Path,
    expected_runs_per_condition: int = DEFAULT_EXPECTED_RUNS_PER_CONDITION,
    expected_algorithms: int = DEFAULT_EXPECTED_ALGORITHMS,
    expected_datasets: int = DEFAULT_EXPECTED_DATASETS,
) -> dict[str, Any]:
    if expected_runs_per_condition != expected_algorithms * expected_datasets * len(SEEDS):
        _raise("expected_runs_per_condition 必须等于 expected_algorithms * expected_datasets * 3")
    clean_rows, clean, algorithms, datasets, clean_info = _load_clean_rows(
        clean_run_csv,
        expected_runs=expected_runs_per_condition,
        expected_algorithms=expected_algorithms,
        expected_datasets=expected_datasets,
    )
    numeric_paths = {
        "noise001": noise001_numeric_csv,
        "noise005": noise005_numeric_csv,
    }
    bundle_paths = {
        "noise001": (
            noise001_equivalence_plan_jsonl,
            noise001_equivalence_index_jsonl,
            noise001_equivalence_summary_json,
        ),
        "noise005": (
            noise005_equivalence_plan_jsonl,
            noise005_equivalence_index_jsonl,
            noise005_equivalence_summary_json,
        ),
    }
    numeric_by_condition: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    adjudication_by_condition: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    input_info: dict[str, dict[str, Any]] = {"clean_run_csv": clean_info}
    for condition in NOISE_CONDITIONS:
        numeric, numeric_info = _load_numeric_rows(
            numeric_paths[condition],
            condition=condition,
            expected_runs=expected_runs_per_condition,
            algorithms=algorithms,
            datasets=datasets,
        )
        plan_path, index_path, summary_path = bundle_paths[condition]
        adjudication, bundle_info = _load_equivalence_bundle(
            condition=condition,
            plan_path=plan_path,
            index_path=index_path,
            summary_path=summary_path,
            numeric=numeric,
            expected_runs=expected_runs_per_condition,
        )
        numeric_by_condition[condition] = numeric
        adjudication_by_condition[condition] = adjudication
        input_info[f"{condition}_numeric_csv"] = numeric_info
        input_info[f"{condition}_equivalence_plan_jsonl"] = bundle_info["plan_jsonl"]
        input_info[f"{condition}_equivalence_index_jsonl"] = bundle_info["index_jsonl"]
        input_info[f"{condition}_equivalence_summary_json"] = bundle_info["summary_json"]

    noise_run_rows: list[dict[str, object]] = []
    grouped_noise: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for condition in NOISE_CONDITIONS:
        numeric = numeric_by_condition[condition]
        adjudication = adjudication_by_condition[condition]
        for key in sorted(numeric, key=lambda item: (item[0], item[1], item[2])):
            numeric_item = numeric[key]
            numeric_row = numeric_item["row"]
            equivalent_item = adjudication[key]
            row: dict[str, object] = {field: "" for field in RUN_FIELDS}
            row.update(
                {
                    "logical_key": numeric_row["logical_key"],
                    "algorithm": key[0],
                    "dataset_id": key[1],
                    "seed": key[2],
                    "noise_tag": condition,
                    "task_id": numeric_row["task_id"],
                    "host": numeric_row["host"],
                    "valid_output": str(numeric_item["valid"]).lower(),
                    "pred_state": equivalent_item["pred_state"],
                    "equivalence_state": equivalent_item["state"],
                    "id_quality": _format_float(numeric_item["id_quality"]),
                    "ood_quality": _format_float(numeric_item["ood_quality"]),
                    "equivalence_decision": equivalent_item["decision"],
                    "equivalent": str(equivalent_item["equivalent"]).lower(),
                    "pred_logical_id": equivalent_item["pred_logical_id"],
                    "equivalence_logical_id": equivalent_item["logical_id"],
                }
            )
            if any(row[field] != "" for field in FORMAL_SIX_AXIS_FIELDS):
                _raise(f"{condition} 噪声行不得写入正式六轴字段")
            noise_run_rows.append(row)
            grouped_noise[(key[0], condition)].append(row)

    all_rows: list[dict[str, object]] = [dict(row) for row in clean_rows] + noise_run_rows
    expected_all_rows = expected_runs_per_condition * 3
    if len(all_rows) != expected_all_rows:
        _raise(f"run_metrics_all_conditions.csv 行数应为 {expected_all_rows}，实际为 {len(all_rows)}")
    all_logical_keys = [str(row["logical_key"]) for row in all_rows]
    if len(set(all_logical_keys)) != len(all_logical_keys):
        _raise("run_metrics_all_conditions.csv 出现重复 logical_key")

    clean_means: dict[str, tuple[float, float]] = {}
    for algorithm in sorted(algorithms):
        algorithm_clean = [value for key, value in clean.items() if key[0] == algorithm]
        clean_means[algorithm] = (
            sum(value["id_quality"] for value in algorithm_clean) / len(algorithm_clean),
            sum(value["ood_quality"] for value in algorithm_clean) / len(algorithm_clean),
        )

    supplement_rows: list[dict[str, object]] = []
    expected_runs_per_algorithm = expected_datasets * len(SEEDS)
    for algorithm in sorted(algorithms):
        clean_id_mean, clean_ood_mean = clean_means[algorithm]
        for condition in NOISE_CONDITIONS:
            rows = grouped_noise[(algorithm, condition)]
            if len(rows) != expected_runs_per_algorithm:
                _raise(f"{algorithm}/{condition} run 数不完整")
            mean_id = sum(float(row["id_quality"]) for row in rows) / len(rows)
            mean_ood = sum(float(row["ood_quality"]) for row in rows) / len(rows)
            valid_count = sum(_parse_bool(row["valid_output"], context="valid_output") for row in rows)
            non_applicable_count = sum(row["equivalence_state"] == "non_applicable" for row in rows)
            applicable_count = len(rows) - non_applicable_count
            covered_count = sum(row["equivalence_state"] == "frozen" for row in rows)
            decided_count = sum(
                row["equivalence_decision"] in {"equivalent", "not_equivalent"} for row in rows
            )
            equivalent_count = sum(row["equivalence_decision"] == "equivalent" for row in rows)
            id_retention = _ratio(mean_id, clean_id_mean)
            ood_retention = _ratio(mean_ood, clean_ood_mean)
            supplement_rows.append(
                {
                    "algorithm": algorithm,
                    "noise_tag": condition,
                    "run_count": len(rows),
                    "valid_output_count": valid_count,
                    "valid_output_rate": _format_float(valid_count / len(rows)),
                    "mean_id_quality": _format_float(mean_id),
                    "mean_ood_quality": _format_float(mean_ood),
                    "clean_mean_id_quality": _format_float(clean_id_mean),
                    "clean_mean_ood_quality": _format_float(clean_ood_mean),
                    "id_quality_retention": _format_float(id_retention),
                    "id_quality_drop": _format_float(None if id_retention is None else 1.0 - id_retention),
                    "ood_quality_retention": _format_float(ood_retention),
                    "ood_quality_drop": _format_float(None if ood_retention is None else 1.0 - ood_retention),
                    "equivalence_applicable_count": applicable_count,
                    "equivalence_non_applicable_count": non_applicable_count,
                    "equivalence_covered_count": covered_count,
                    "equivalence_decided_count": decided_count,
                    "equivalent_count": equivalent_count,
                    "equivalence_coverage": _format_float(
                        None if applicable_count == 0 else covered_count / applicable_count
                    ),
                    "equivalence_rate": _format_float(
                        None if decided_count == 0 else equivalent_count / decided_count
                    ),
                }
            )
    expected_supplement_rows = expected_algorithms * len(NOISE_CONDITIONS)
    if len(supplement_rows) != expected_supplement_rows:
        _raise(f"noise_supplement.csv 行数应为 {expected_supplement_rows}，实际为 {len(supplement_rows)}")

    _write_csv_atomic(all_conditions_csv, RUN_FIELDS, all_rows)
    _write_csv_atomic(noise_supplement_csv, SUPPLEMENT_FIELDS, supplement_rows)
    output_info = {
        "all_conditions_csv": _artifact_info(all_conditions_csv, row_count=len(all_rows)),
        "noise_supplement_csv": _artifact_info(noise_supplement_csv, row_count=len(supplement_rows)),
    }
    summary = {
        "all_conditions_row_count": len(all_rows),
        "clean_row_count": len(clean_rows),
        "noise001_row_count": len(numeric_by_condition["noise001"]),
        "noise005_row_count": len(numeric_by_condition["noise005"]),
        "noise_supplement_row_count": len(supplement_rows),
        "algorithm_count": len(algorithms),
        "dataset_count": len(datasets),
        "conditions": ["clean", *NOISE_CONDITIONS],
        "formal_six_axis_conditions": ["clean"],
    }
    report = {
        "inputs": input_info,
        "outputs": output_info,
        "summary": summary,
        "summary_sha256": _sha256_text(
            _canonical_json({"inputs": input_info, "outputs": output_info, "summary": summary})
        ),
    }
    _write_json_atomic(report_json, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    stage5_root = _stage5_root()
    parser = argparse.ArgumentParser(description="聚合 Stage5 clean/noise001/noise005 run 与噪声补充指标")
    parser.add_argument("--clean-run-csv", type=Path, default=stage5_root / "results/clean_run_metrics.csv")
    parser.add_argument(
        "--noise001-numeric-csv",
        type=Path,
        default=stage5_root / "results/noise001_numeric_run_metrics.csv",
    )
    parser.add_argument(
        "--noise005-numeric-csv",
        type=Path,
        default=stage5_root / "results/noise005_numeric_run_metrics.csv",
    )
    for condition in NOISE_CONDITIONS:
        parser.add_argument(f"--{condition}-equivalence-plan-jsonl", type=Path, required=True)
        parser.add_argument(f"--{condition}-equivalence-index-jsonl", type=Path, required=True)
        parser.add_argument(f"--{condition}-equivalence-summary-json", type=Path, required=True)
    parser.add_argument(
        "--all-conditions-csv",
        type=Path,
        default=stage5_root / "results/run_metrics_all_conditions.csv",
    )
    parser.add_argument(
        "--noise-supplement-csv",
        type=Path,
        default=stage5_root / "results/noise_supplement.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=stage5_root / "reports/aggregate_noise_supplement.json",
    )
    parser.add_argument(
        "--expected-runs-per-condition", type=int, default=DEFAULT_EXPECTED_RUNS_PER_CONDITION
    )
    parser.add_argument("--expected-algorithms", type=int, default=DEFAULT_EXPECTED_ALGORITHMS)
    parser.add_argument("--expected-datasets", type=int, default=DEFAULT_EXPECTED_DATASETS)
    parser.add_argument("--print-summary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = aggregate_noise_supplement(
            clean_run_csv=args.clean_run_csv.resolve(),
            noise001_numeric_csv=args.noise001_numeric_csv.resolve(),
            noise005_numeric_csv=args.noise005_numeric_csv.resolve(),
            noise001_equivalence_plan_jsonl=args.noise001_equivalence_plan_jsonl.resolve(),
            noise001_equivalence_index_jsonl=args.noise001_equivalence_index_jsonl.resolve(),
            noise001_equivalence_summary_json=args.noise001_equivalence_summary_json.resolve(),
            noise005_equivalence_plan_jsonl=args.noise005_equivalence_plan_jsonl.resolve(),
            noise005_equivalence_index_jsonl=args.noise005_equivalence_index_jsonl.resolve(),
            noise005_equivalence_summary_json=args.noise005_equivalence_summary_json.resolve(),
            all_conditions_csv=args.all_conditions_csv.resolve(),
            noise_supplement_csv=args.noise_supplement_csv.resolve(),
            report_json=args.report_json.resolve(),
            expected_runs_per_condition=args.expected_runs_per_condition,
            expected_algorithms=args.expected_algorithms,
            expected_datasets=args.expected_datasets,
        )
    except AggregateNoiseSupplementError as exc:
        print(json.dumps({"fatal_error": str(exc)}, ensure_ascii=False, sort_keys=True), file=__import__("sys").stderr)
        return 2
    if args.print_summary:
        print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
