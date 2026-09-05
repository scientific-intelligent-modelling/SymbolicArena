"""将 result.json 本地快照转换为 simplify builder 可审计冻结包。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from .source_provenance import SourceProvenanceError, reject_aborted_fullcpu_source


SUPPORTED_CONDITIONS = ("clean", "noise001", "noise005")
CONDITION_SIGMA = {"clean": 0.0, "noise001": 0.01, "noise005": 0.05}
TASK_ID_RE = re.compile(
    r"^(?P<slug>[a-z0-9]+)_s(?P<seed>520|521|522)_"
    r"(?P<condition>clean|noise001|noise005)_g(?P<dataset_index>\d{4})$"
)
PARAMETER_RE = re.compile(r"\bparams\s*\[")


class ResultFreezeAdapterError(ValueError):
    """本地结果索引与冻结源之间无法建立可信绑定。"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, *, context: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ResultFreezeAdapterError(f"{context} 缺少 CSV 表头: {path}")
        return [dict(row) for row in reader]


def _parse_bool(value: object, *, context: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ResultFreezeAdapterError(f"{context} 必须为 true/false")


def _parse_nmse(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ResultFreezeAdapterError(f"{context} 不是有限非负数")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultFreezeAdapterError(f"{context} 不是有限非负数") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ResultFreezeAdapterError(f"{context} 不是有限非负数")
    return parsed


def _payload_nmse(payload: Mapping[str, Any], split: str, *, context: str) -> float:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        raise ResultFreezeAdapterError(f"{context}.{split} 缺失")
    return _parse_nmse(block.get("nmse"), context=f"{context}.{split}.nmse")


def _same_nmse(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _validate_noise_contract(payload: Mapping[str, Any], *, condition: str, task_id: str) -> None:
    block = payload.get("train_label_noise")
    if not isinstance(block, Mapping):
        if condition == "clean":
            return
        raise ResultFreezeAdapterError(f"{task_id}: 缺少 train_label_noise")
    enabled = block.get("enabled")
    requested = block.get("requested")
    if not isinstance(enabled, bool) or not isinstance(requested, bool):
        raise ResultFreezeAdapterError(f"{task_id}: train_label_noise 标志必须是布尔值")
    expected_enabled = condition != "clean"
    if enabled != expected_enabled or requested != expected_enabled:
        raise ResultFreezeAdapterError(f"{task_id}: train_label_noise 标志与 {condition} 不一致")
    sigma = _parse_nmse(
        0.0 if block.get("sigma") is None and condition == "clean" else block.get("sigma"),
        context=f"{task_id}.train_label_noise.sigma",
    )
    if not math.isclose(sigma, CONDITION_SIGMA[condition], rel_tol=0.0, abs_tol=1e-12):
        raise ResultFreezeAdapterError(f"{task_id}: train_label_noise.sigma 与 {condition} 不一致")


def _formula_source(payload: Mapping[str, Any]) -> tuple[str, str]:
    artifact = payload.get("canonical_artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    candidates = (
        ("canonical_artifact.instantiated_expression", artifact.get("instantiated_expression")),
        ("canonical_artifact.normalized_expression", artifact.get("normalized_expression")),
        ("canonical_artifact.return_expression_source", artifact.get("return_expression_source")),
        ("equation", payload.get("equation")),
    )
    for source, value in candidates:
        if isinstance(value, str) and value.strip():
            return source, value.strip()
    return "missing", ""


def _atomic_write_gzip_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as raw:
        tmp_path = Path(raw.name)
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text_handle:
                for row in rows:
                    text_handle.write(_canonical_json(row) + "\n")
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _quality(nmse: float) -> float:
    clipped = min(2.0, max(-12.0, math.log10(max(nmse, 1e-12))))
    return 1.0 - ((clipped + 12.0) / 14.0)


def _atomic_write_numeric_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "logical_key", "algorithm", "dataset_id", "seed", "noise_tag", "task_id",
        "host", "result_sha256", "valid_output", "formula_source", "id_nmse",
        "ood_nmse", "id_quality", "ood_quality",
    ]
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False, mode="w",
        encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def build_result_freeze_bundle(
    *,
    source_runs_csv: Path,
    result_index_csv: Path,
    condition: str,
    output_jsonl_gz: Path,
    report_json: Path,
    expected_count: int | None = 2250,
    numeric_metrics_csv: Path | None = None,
) -> dict[str, Any]:
    """交叉验证 source_runs、index 与本地 result.json 后原子冻结。"""
    if condition not in SUPPORTED_CONDITIONS:
        raise ResultFreezeAdapterError(f"未知 condition: {condition!r}")
    source_rows_all = _read_csv(source_runs_csv.resolve(), context="source_runs")
    for row in source_rows_all:
        try:
            reject_aborted_fullcpu_source(
                row,
                context=f"source_runs[{row.get('logical_key', '<unknown>')}]",
            )
        except SourceProvenanceError as exc:
            raise ResultFreezeAdapterError(str(exc)) from exc
    source_rows = [row for row in source_rows_all if row.get("noise_tag") == condition]
    index_rows_all = _read_csv(result_index_csv.resolve(), context="result_index")
    index_rows = [row for row in index_rows_all if row.get("noise_tag") == condition]
    if expected_count is not None and len(source_rows) != expected_count:
        raise ResultFreezeAdapterError(
            f"{condition} source_runs 行数不符: 期望 {expected_count}，实际 {len(source_rows)}"
        )
    if len(index_rows) != len(source_rows):
        raise ResultFreezeAdapterError(
            f"{condition} result_index 行数不符: 期望 {len(source_rows)}，实际 {len(index_rows)}"
        )
    source_by_key = {row.get("logical_key", ""): row for row in source_rows}
    index_by_key = {row.get("logical_key", ""): row for row in index_rows}
    if "" in source_by_key or len(source_by_key) != len(source_rows):
        raise ResultFreezeAdapterError("source_runs logical_key 缺失或重复")
    if "" in index_by_key or len(index_by_key) != len(index_rows):
        raise ResultFreezeAdapterError("result_index logical_key 缺失或重复")
    if set(source_by_key) != set(index_by_key):
        raise ResultFreezeAdapterError("source_runs 与 result_index logical_key 集合不一致")

    bundle_rows: list[dict[str, Any]] = []
    formula_counts: dict[str, int] = {}
    canonical_missing = 0
    unresolved_params = 0
    numeric_rows: list[dict[str, Any]] = []
    for logical_key in sorted(source_by_key):
        source = source_by_key[logical_key]
        index = index_by_key[logical_key]
        task_id = source.get("task_id", "")
        match = TASK_ID_RE.fullmatch(task_id)
        if match is None or match.group("condition") != condition:
            raise ResultFreezeAdapterError(f"task_id 与 condition 不一致: {task_id!r}")
        seed = int(source.get("seed", "-1"))
        if seed != int(match.group("seed")):
            raise ResultFreezeAdapterError(f"{task_id}: seed 与 task_id 不一致")
        expected_logical_key = f"{source.get('algorithm')}::{source.get('dataset_id')}::s{seed}::{condition}"
        if logical_key != expected_logical_key:
            raise ResultFreezeAdapterError(f"{task_id}: logical_key 非 canonical")
        exact_fields = (
            "batch", "algorithm", "dataset_id", "noise_tag", "task_id", "host"
        )
        for field in exact_fields:
            if index.get(field) != source.get(field):
                raise ResultFreezeAdapterError(f"{task_id}: index.{field} 与 source_runs 不一致")
        if int(index.get("seed", "-1")) != seed:
            raise ResultFreezeAdapterError(f"{task_id}: index.seed 与 source_runs 不一致")
        if index.get("remote_result_path") != source.get("path"):
            raise ResultFreezeAdapterError(f"{task_id}: remote_result_path 与 source_runs 不一致")
        if not _parse_bool(index.get("metrics_match"), context=f"{task_id}.metrics_match"):
            raise ResultFreezeAdapterError(f"{task_id}: index 标记 NMSE 不匹配")

        local_path = Path(index.get("local_result_path", "")).resolve()
        if not local_path.is_file():
            raise ResultFreezeAdapterError(f"{task_id}: 本地 result.json 不存在: {local_path}")
        actual_sha = _sha256_file(local_path)
        if index.get("local_sha256") != actual_sha:
            raise ResultFreezeAdapterError(f"{task_id}: local_sha256 漂移")
        raw_bytes = local_path.read_bytes()
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResultFreezeAdapterError(f"{task_id}: result.json 不是 UTF-8") from exc
        try:
            payload_raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ResultFreezeAdapterError(f"{task_id}: result.json 解析失败") from exc
        if not isinstance(payload_raw, Mapping):
            raise ResultFreezeAdapterError(f"{task_id}: result.json 根必须是对象")
        payload = dict(payload_raw)
        if payload.get("status") != source.get("status") or index.get("status") != source.get("status"):
            raise ResultFreezeAdapterError(f"{task_id}: status 在 source/index/result 间不一致")
        _validate_noise_contract(payload, condition=condition, task_id=task_id)

        source_id = _parse_nmse(source.get("id_nmse"), context=f"{task_id}.source.id_nmse")
        source_ood = _parse_nmse(source.get("ood_nmse"), context=f"{task_id}.source.ood_nmse")
        result_id = _payload_nmse(payload, "id_test", context=task_id)
        result_ood = _payload_nmse(payload, "ood_test", context=task_id)
        index_values = (
            _parse_nmse(index.get("id_nmse_source"), context=f"{task_id}.index.id_nmse_source"),
            _parse_nmse(index.get("id_nmse_result"), context=f"{task_id}.index.id_nmse_result"),
            _parse_nmse(index.get("ood_nmse_source"), context=f"{task_id}.index.ood_nmse_source"),
            _parse_nmse(index.get("ood_nmse_result"), context=f"{task_id}.index.ood_nmse_result"),
        )
        comparisons = (
            (source_id, result_id), (source_ood, result_ood),
            (source_id, index_values[0]), (result_id, index_values[1]),
            (source_ood, index_values[2]), (result_ood, index_values[3]),
        )
        if not all(_same_nmse(left, right) for left, right in comparisons):
            raise ResultFreezeAdapterError(f"{task_id}: NMSE 在 source/index/result 间漂移")

        formula_source, expression = _formula_source(payload)
        formula_counts[formula_source] = formula_counts.get(formula_source, 0) + 1
        if not isinstance(payload.get("canonical_artifact"), Mapping):
            canonical_missing += 1
        if PARAMETER_RE.search(expression):
            unresolved_params += 1
        source_public = {
            "algorithm": source["algorithm"],
            "batch": source["batch"],
            "dataset_id": source["dataset_id"],
            "host": source["host"],
            "noise_tag": condition,
            "path": source["path"],
            "seed": str(seed),
            "source_row_sha256": _sha256_bytes(_canonical_json(source).encode("utf-8")),
            "task_id": task_id,
        }
        bundle_rows.append(
            {
                "source": source_public,
                "result": {"raw_text": raw_text, "sha256": actual_sha},
            }
        )
        numeric_rows.append(
            {
                "logical_key": logical_key,
                "algorithm": source["algorithm"],
                "dataset_id": source["dataset_id"],
                "seed": seed,
                "noise_tag": condition,
                "task_id": task_id,
                "host": source["host"],
                "result_sha256": actual_sha,
                "valid_output": str(bool(expression) and payload.get("status") == "ok").lower(),
                "formula_source": (
                    "canonical_artifact" if formula_source.startswith("canonical_artifact.") else formula_source
                ),
                "id_nmse": format(result_id, ".17g"),
                "ood_nmse": format(result_ood, ".17g"),
                "id_quality": format(_quality(result_id), ".17g"),
                "ood_quality": format(_quality(result_ood), ".17g"),
            }
        )

    _atomic_write_gzip_jsonl(output_jsonl_gz.resolve(), bundle_rows)
    if numeric_metrics_csv is not None:
        _atomic_write_numeric_csv(numeric_metrics_csv.resolve(), numeric_rows)
    report: dict[str, Any] = {
        "schema_version": "result_freeze_adapter.v1",
        "condition": condition,
        "source_runs_csv": str(source_runs_csv.resolve()),
        "source_runs_sha256": _sha256_file(source_runs_csv.resolve()),
        "result_index_csv": str(result_index_csv.resolve()),
        "result_index_sha256": _sha256_file(result_index_csv.resolve()),
        "output_jsonl_gz": str(output_jsonl_gz.resolve()),
        "output_sha256": _sha256_file(output_jsonl_gz.resolve()),
        "row_count": len(bundle_rows),
        "formula_source_counts": dict(sorted(formula_counts.items())),
        "canonical_artifact_missing_count": canonical_missing,
        "unresolved_parameter_count": unresolved_params,
        "status": "ok",
    }
    if numeric_metrics_csv is not None:
        report["numeric_metrics_csv"] = str(numeric_metrics_csv.resolve())
        report["numeric_metrics_sha256"] = _sha256_file(numeric_metrics_csv.resolve())
    _atomic_write_json(report_json.resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Stage5 result simplify 可信冻结包")
    parser.add_argument("--source-runs-csv", type=Path, required=True)
    parser.add_argument("--result-index-csv", type=Path, required=True)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
    parser.add_argument("--output-jsonl-gz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2250)
    parser.add_argument("--numeric-metrics-csv", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_result_freeze_bundle(
        source_runs_csv=args.source_runs_csv,
        result_index_csv=args.result_index_csv,
        condition=args.condition,
        output_jsonl_gz=args.output_jsonl_gz,
        report_json=args.report_json,
        expected_count=args.expected_count,
        numeric_metrics_csv=args.numeric_metrics_csv,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
