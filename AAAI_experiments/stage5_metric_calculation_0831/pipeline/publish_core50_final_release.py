#!/usr/bin/env python3
"""从显式冻结配置发布 Core-50 当前最终结果，不调用模型或重跑训练。"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import (
    RunQuality,
    efficiency_from_qualities,
    minimality_score,
    numerical_consistency,
    phi_nmse,
    stability_score,
    symbolic_fidelity_score,
)
from .prepare_core50_final_release import _artifact_expression
from .symbolic_evidence import (
    build_symbolic_artifact,
    operator_f1,
    tree_similarity,
    variable_f1,
)


CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = (520, 521, 522)
PAIRS = ((520, 521), (520, 522), (521, 522))
HORIZON = 180
RUNS_PER_CONDITION = 2250
TASKS_PER_CONDITION = 750
CURVES_PER_CONDITION = 2700
TOTAL_RUNS = 6750
FORBIDDEN_SOURCE = "all_15alg_fullcpu_v1"
SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b|\bBearer\s+[A-Za-z0-9._-]{16,}")
Key = tuple[str, str, int, str]
TaskKey = tuple[str, str, str]


class PublicationError(ValueError):
    """冻结输入、表达式绑定或发布网格不满足合同。"""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def algorithm_slug(value: object) -> str:
    return re.sub(r"[-_\s]", "", str(value).strip().casefold())


def run_key(
    algorithm: object,
    dataset_id: object,
    seed: object,
    condition: object,
) -> Key:
    key = (
        algorithm_slug(algorithm),
        str(dataset_id).strip(),
        int(seed),
        str(condition).strip(),
    )
    if not key[0] or not key[1] or key[2] not in SEEDS or key[3] not in CONDITIONS:
        raise PublicationError(f"非法运行身份: {key}")
    return key


def logical_key(key: Key, display_algorithm: str | None = None) -> str:
    return f"{display_algorithm or key[0]}::{key[1]}::s{key[2]}::{key[3]}"


def expression_fingerprint(expression: str) -> str:
    """只忽略格式与 np/math 限定，绝不把变量置换视为相同。"""

    class DropModulePrefix(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            node = self.generic_visit(node)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in {"np", "numpy", "math"}:
                    return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            return node

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        tree = ast.fix_missing_locations(DropModulePrefix().visit(tree))
        rendered = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        rendered = "".join(expression.split())
    return sha256_text(rendered)


def require_same_expression(left: object, right: object, *, context: str) -> None:
    if not isinstance(left, str) or not left.strip():
        raise PublicationError(f"{context}: 左表达式为空")
    if not isinstance(right, str) or not right.strip():
        raise PublicationError(f"{context}: 右表达式为空")
    if expression_fingerprint(left) != expression_fingerprint(right):
        raise PublicationError(f"{context}: 表达式绑定不一致")


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    sha256: str
    rows: int
    format: str
    generation: str = "preexisting"
    defaults: Mapping[str, Any] | None = None

    @classmethod
    def from_json(cls, raw: object, *, repo_root: Path, context: str) -> "ArtifactSpec":
        if not isinstance(raw, Mapping):
            raise PublicationError(f"{context} 必须是 artifact object")
        required = {"path", "sha256", "rows", "format"}
        missing = sorted(required - set(raw))
        if missing:
            raise PublicationError(f"{context} 缺少字段: {missing}")
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        expected_sha = str(raw["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise PublicationError(f"{context}.sha256 非法")
        try:
            rows = int(raw["rows"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise PublicationError(f"{context}.rows 非法") from exc
        if rows < 0:
            raise PublicationError(f"{context}.rows 必须非负")
        format_name = str(raw["format"])
        if format_name not in {"csv", "csv.gz", "jsonl", "jsonl.gz", "json", "text"}:
            raise PublicationError(f"{context}.format 不支持: {format_name}")
        generation = str(raw.get("generation", "preexisting"))
        if generation not in {"new", "preexisting", "mixed"}:
            raise PublicationError(f"{context}.generation 非法")
        defaults = raw.get("defaults")
        if defaults is not None and not isinstance(defaults, Mapping):
            raise PublicationError(f"{context}.defaults 必须是 object")
        return cls(path, expected_sha, rows, format_name, generation, defaults)

    def verify(self) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise PublicationError(f"冻结输入必须是已物化文件: {self.path}")
        actual = sha256_file(self.path)
        if actual != self.sha256:
            raise PublicationError(f"冻结输入 SHA 漂移: {self.path}: {actual} != {self.sha256}")

    def read_rows(self) -> list[dict[str, Any]]:
        self.verify()
        if self.format.startswith("csv"):
            opener = gzip.open if self.format.endswith(".gz") else open
            with opener(self.path, "rt", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        elif self.format.startswith("jsonl"):
            opener = gzip.open if self.format.endswith(".gz") else open
            rows = []
            with opener(self.path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise PublicationError(f"{self.path}:{line_number} 不是 object")
                    rows.append(value)
        elif self.format == "json":
            rows = [_read_json_object(self.path)]
        else:
            rows = [{"line": line} for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != self.rows:
            raise PublicationError(f"{self.path}: 声明 {self.rows} 行，实际 {len(rows)}")
        return rows

    def verify_rows(self) -> None:
        self.verify()
        count = 0
        opener = gzip.open if self.format.endswith(".gz") else open
        if self.format.startswith("csv"):
            with opener(self.path, "rt", encoding="utf-8", newline="") as handle:
                for _ in csv.DictReader(handle):
                    count += 1
        elif self.format.startswith("jsonl"):
            with opener(self.path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise PublicationError(f"{self.path}:{line_number} 不是 object")
                    count += 1
        elif self.format == "json":
            _read_json_object(self.path)
            count = 1
        else:
            count = sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line)
        if count != self.rows:
            raise PublicationError(f"{self.path}: 声明 {self.rows} 行，实际 {count}")

    def evidence(self, repo_root: Path) -> dict[str, Any]:
        try:
            path = str(self.path.relative_to(repo_root.resolve()))
        except ValueError:
            path = str(self.path)
        return {
            "path": path,
            "sha256": self.sha256,
            "rows": self.rows,
            "format": self.format,
            "generation": self.generation,
            "defaults": dict(self.defaults or {}),
        }


@dataclass(frozen=True)
class ConditionInputs:
    raw_results: ArtifactSpec
    numeric: ArtifactSpec
    pred_plan: ArtifactSpec
    pred_index: ArtifactSpec
    equivalence_plan: ArtifactSpec
    equivalence_index: ArtifactSpec
    equivalence_overrides: tuple[ArtifactSpec, ...]
    structure_plan: ArtifactSpec
    structure_index: ArtifactSpec
    structure_overlays: tuple[tuple[ArtifactSpec, ArtifactSpec], ...]
    trajectory_base: ArtifactSpec | None
    trajectory_overrides: tuple[ArtifactSpec, ...]


@dataclass(frozen=True)
class EffConditionInputs:
    run_status: ArtifactSpec
    native_trajectories: ArtifactSpec
    minute_metrics: ArtifactSpec
    algorithm_curves: ArtifactSpec
    algorithm_summary: ArtifactSpec
    unavailable: ArtifactSpec
    manifest: ArtifactSpec


@dataclass(frozen=True)
class EffRevisionInputs:
    conditions: Mapping[str, EffConditionInputs]
    manifest: ArtifactSpec
    validation_report: ArtifactSpec
    checksums: ArtifactSpec


@dataclass(frozen=True)
class PublicationConfig:
    repo_root: Path
    output_root: Path
    dataset_manifest: ArtifactSpec
    semantic_runs: ArtifactSpec
    downstream_composition_manifest: ArtifactSpec
    gt_plan: ArtifactSpec
    gt_index: ArtifactSpec
    conditions: Mapping[str, ConditionInputs]
    eff_revision: EffRevisionInputs
    max_workers: int
    accepted_preexisting_transports: tuple[str, ...]
    new_evidence_transport: str


def load_config(path: Path) -> PublicationConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "core50_final_release_publish_config_v1":
        raise PublicationError("config schema_version 不匹配")
    repo_root = Path(str(raw.get("repo_root") or _repo_root())).resolve()
    raw_output_root = raw.get("output_root")
    if not isinstance(raw_output_root, str) or not raw_output_root.strip():
        raise PublicationError("config.output_root 必须是非空路径")
    output_root = Path(raw_output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    formal_root = (repo_root / "AAAI_experiments" / "Core50_final_20260913").resolve()
    if output_root == formal_root or formal_root in output_root.parents:
        raise PublicationError("发布器拒绝直接覆盖正式目录；先输出 publication_candidate")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PublicationError("config.artifacts 缺失")
    ground_truth_raw = artifacts.get("ground_truth")
    if not isinstance(ground_truth_raw, Mapping):
        raise PublicationError("config.artifacts.ground_truth 缺失")
    conditions_raw = artifacts.get("conditions")
    if not isinstance(conditions_raw, Mapping) or set(conditions_raw) != set(CONDITIONS):
        raise PublicationError("config.artifacts.conditions 必须恰含 clean/noise001/noise005")

    def spec(value: object, context: str) -> ArtifactSpec:
        return ArtifactSpec.from_json(value, repo_root=repo_root, context=context)

    conditions: dict[str, ConditionInputs] = {}
    for condition in CONDITIONS:
        section = conditions_raw[condition]
        if not isinstance(section, Mapping):
            raise PublicationError(f"conditions.{condition} 必须是 object")
        pred = section.get("prediction")
        eq = section.get("equivalence")
        structure = section.get("structure")
        trajectory = section.get("trajectory")
        if not all(isinstance(item, Mapping) for item in (pred, eq, structure, trajectory)):
            raise PublicationError(f"conditions.{condition} 缺少 prediction/equivalence/structure/trajectory")
        overrides_raw = trajectory.get("overrides", [])
        if not isinstance(overrides_raw, list):
            raise PublicationError(f"conditions.{condition}.trajectory.overrides 必须是数组")
        eq_overrides_raw = eq.get("audit_overrides", [])
        structure_overlays_raw = structure.get("audit_overlays", [])
        if not isinstance(eq_overrides_raw, list) or not isinstance(structure_overlays_raw, list):
            raise PublicationError(f"conditions.{condition} audit_overrides 必须是数组")
        structure_overlays = []
        for index, value in enumerate(structure_overlays_raw):
            if not isinstance(value, Mapping):
                raise PublicationError(f"{condition}.structure.audit_overlays[{index}] 必须是object")
            structure_overlays.append(
                (
                    spec(value.get("plan"), f"{condition}.structure.audit_overlays[{index}].plan"),
                    spec(value.get("index"), f"{condition}.structure.audit_overlays[{index}].index"),
                )
            )
        if trajectory.get("mode") != "eff_revision_v3" or overrides_raw:
            raise PublicationError(
                f"{condition} trajectory必须使用mode=eff_revision_v3且不得另加override"
            )
        trajectory_base = None
        conditions[condition] = ConditionInputs(
            raw_results=spec(section.get("raw_results"), f"{condition}.raw_results"),
            numeric=spec(section.get("numeric"), f"{condition}.numeric"),
            pred_plan=spec(pred.get("plan"), f"{condition}.prediction.plan"),
            pred_index=spec(pred.get("index"), f"{condition}.prediction.index"),
            equivalence_plan=spec(eq.get("plan"), f"{condition}.equivalence.plan"),
            equivalence_index=spec(eq.get("index"), f"{condition}.equivalence.index"),
            equivalence_overrides=tuple(
                spec(value, f"{condition}.equivalence.audit_overrides[{index}]")
                for index, value in enumerate(eq_overrides_raw)
            ),
            structure_plan=spec(structure.get("plan"), f"{condition}.structure.plan"),
            structure_index=spec(structure.get("index"), f"{condition}.structure.index"),
            structure_overlays=tuple(structure_overlays),
            trajectory_base=trajectory_base,
            trajectory_overrides=tuple(
                spec(value, f"{condition}.trajectory.overrides[{index}]")
                for index, value in enumerate(overrides_raw)
            ),
        )
    policy = raw.get("llm_transport_policy")
    if not isinstance(policy, Mapping):
        raise PublicationError("config.llm_transport_policy 缺失")
    accepted = policy.get("accepted_preexisting", ["routify", "claude_cli", "yapi"])
    if not isinstance(accepted, list) or not all(isinstance(item, str) for item in accepted):
        raise PublicationError("accepted_preexisting 必须是字符串数组")
    new_transport = str(policy.get("new_evidence_required") or "")
    if new_transport != "routify":
        raise PublicationError("新证据 transport 必须固定为 routify")
    max_workers = int(raw.get("max_workers", 4))
    if not 1 <= max_workers <= 4:
        raise PublicationError("max_workers 必须在 1..4")
    eff_raw = artifacts.get("eff_revision")
    if not isinstance(eff_raw, Mapping):
        raise PublicationError("config.artifacts.eff_revision 缺失")
    eff_conditions_raw = eff_raw.get("conditions")
    if not isinstance(eff_conditions_raw, Mapping) or set(eff_conditions_raw) != set(CONDITIONS):
        raise PublicationError("eff_revision.conditions必须恰含三个条件")
    eff_conditions: dict[str, EffConditionInputs] = {}
    for condition in CONDITIONS:
        section = eff_conditions_raw[condition]
        if not isinstance(section, Mapping):
            raise PublicationError(f"eff_revision.conditions.{condition}必须是object")
        eff_conditions[condition] = EffConditionInputs(
            run_status=spec(section.get("run_status"), f"eff_revision.{condition}.run_status"),
            native_trajectories=spec(
                section.get("native_trajectories"),
                f"eff_revision.{condition}.native_trajectories",
            ),
            minute_metrics=spec(
                section.get("minute_metrics"), f"eff_revision.{condition}.minute_metrics"
            ),
            algorithm_curves=spec(
                section.get("algorithm_curves"), f"eff_revision.{condition}.algorithm_curves"
            ),
            algorithm_summary=spec(
                section.get("algorithm_summary"), f"eff_revision.{condition}.algorithm_summary"
            ),
            unavailable=spec(
                section.get("unavailable"), f"eff_revision.{condition}.unavailable"
            ),
            manifest=spec(section.get("manifest"), f"eff_revision.{condition}.manifest"),
        )
    eff_revision = EffRevisionInputs(
        conditions=eff_conditions,
        manifest=spec(eff_raw.get("manifest"), "eff_revision.manifest"),
        validation_report=spec(
            eff_raw.get("validation_report"), "eff_revision.validation_report"
        ),
        checksums=spec(eff_raw.get("checksums"), "eff_revision.checksums"),
    )
    return PublicationConfig(
        repo_root=repo_root,
        output_root=output_root,
        dataset_manifest=spec(artifacts.get("dataset_manifest"), "dataset_manifest"),
        semantic_runs=spec(artifacts.get("semantic_runs"), "semantic_runs"),
        downstream_composition_manifest=spec(
            artifacts.get("downstream_composition_manifest"),
            "downstream_composition_manifest",
        ),
        gt_plan=spec(ground_truth_raw.get("plan"), "ground_truth.plan"),
        gt_index=spec(ground_truth_raw.get("index"), "ground_truth.index"),
        conditions=conditions,
        eff_revision=eff_revision,
        max_workers=max_workers,
        accepted_preexisting_transports=tuple(accepted),
        new_evidence_transport=new_transport,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def plan_request(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    request = plan.get("request")
    if isinstance(request, Mapping):
        return request
    normalized = plan.get("normalized_input")
    request = normalized.get("request") if isinstance(normalized, Mapping) else None
    if not isinstance(request, Mapping):
        raise PublicationError(f"plan 缺少 request: {plan.get('logical_id')}")
    return request


def plan_index_pairs(
    plan_spec: ArtifactSpec,
    index_spec: ArtifactSpec,
    *,
    label: str,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    plans = plan_spec.read_rows()
    indexes = index_spec.read_rows()
    by_id: dict[str, dict[str, Any]] = {}
    for index in indexes:
        logical_id = str(index.get("logical_id") or "")
        if not logical_id or logical_id in by_id:
            raise PublicationError(f"{label} index logical_id 缺失或重复: {logical_id}")
        by_id[logical_id] = index
    output: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for plan in plans:
        logical_id = str(plan.get("logical_id") or "")
        if not logical_id or logical_id in output or logical_id not in by_id:
            raise PublicationError(f"{label} plan logical_id 缺失、重复或无 index: {logical_id}")
        index = by_id[logical_id]
        if index.get("evaluation_key") != plan.get("evaluation_key"):
            raise PublicationError(f"{label} evaluation_key 漂移: {logical_id}")
        if index.get("state") not in {"frozen", "non_applicable", "unresolved"}:
            raise PublicationError(f"{label} index 未冻结或非正式non_applicable: {logical_id}")
        output[logical_id] = (plan, index)
    extra = sorted(set(by_id) - set(output))
    if extra:
        raise PublicationError(f"{label} index 含无 plan 项: {extra[:3]}")
    return output


def _unique_by_run(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
    identity_fn,
) -> dict[Key, dict[str, Any]]:
    output: dict[Key, dict[str, Any]] = {}
    for row in rows:
        key = identity_fn(row)
        if key in output:
            raise PublicationError(f"{label} 重复运行: {key}")
        output[key] = row
    return output


def _source_identity(source: Mapping[str, Any]) -> Key:
    return run_key(
        source.get("algorithm"),
        source.get("dataset_id"),
        source.get("seed"),
        source.get("condition") or source.get("noise_tag"),
    )


def validate_raw_record(record: Mapping[str, Any]) -> tuple[Key, dict[str, Any]]:
    source = record.get("source")
    result = record.get("result")
    if not isinstance(source, Mapping) or not isinstance(result, Mapping):
        raise PublicationError("raw record 缺少 source/result")
    text = result.get("raw_text")
    expected_sha = result.get("sha256")
    if not isinstance(text, str) or not isinstance(expected_sha, str):
        raise PublicationError("raw record 缺少 raw_text/sha256")
    if sha256_text(text) != expected_sha:
        raise PublicationError("raw result SHA 漂移")
    serialized_source = canonical_json(source)
    if FORBIDDEN_SOURCE in serialized_source or FORBIDDEN_SOURCE in text:
        raise PublicationError("最终来源命中已废弃 fullcpu 批次")
    if SECRET_RE.search(text):
        raise PublicationError("raw result 含疑似凭据")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise PublicationError("raw result 顶层不是 object")
    key = _source_identity(source)
    payload_key = run_key(
        payload.get("tool"),
        payload.get("dataset"),
        payload.get("seed"),
        payload.get("condition") or source.get("noise_tag"),
    )
    if payload_key != key:
        raise PublicationError(f"raw payload 与 source 身份不一致: {key} != {payload_key}")
    if not isinstance(payload.get("equation"), str) or not payload["equation"].strip():
        raise PublicationError(f"raw result 缺少原始公式: {key}")
    return key, payload


def load_raw_results(spec: ArtifactSpec, *, condition: str) -> dict[Key, dict[str, Any]]:
    records = spec.read_rows()
    output: dict[Key, dict[str, Any]] = {}
    for record in records:
        key, payload = validate_raw_record(record)
        if key[3] != condition or key in output:
            raise PublicationError(f"{condition} raw condition错误或重复: {key}")
        output[key] = {"record": record, "payload": payload}
    if len(output) != RUNS_PER_CONDITION:
        raise PublicationError(f"{condition} raw 应有2250条，实际{len(output)}")
    return output


def semantic_identity(row: Mapping[str, Any]) -> Key:
    return run_key(
        row.get("algorithm"), row.get("dataset_id"), row.get("seed"), row.get("condition")
    )


def load_semantic_runs(spec: ArtifactSpec) -> dict[Key, dict[str, Any]]:
    output = _unique_by_run(
        spec.read_rows(), label="semantic_runs", identity_fn=semantic_identity
    )
    if len(output) != TOTAL_RUNS:
        raise PublicationError(f"semantic_runs 应有6750条，实际{len(output)}")
    for key, row in output.items():
        required = (
            "source_result_sha256",
            "raw_equation",
            "raw_equation_sha256",
            "canonical_artifact",
            "canonical_artifact_sha256",
            "effective_raw_semantic_expression",
            "effective_raw_semantic_expression_sha256",
        )
        if any(row.get(field) is None or row.get(field) == "" for field in required):
            raise PublicationError(f"semantic_runs 缺少当前语义字段: {key}")
        if sha256_text(str(row["raw_equation"])) != row["raw_equation_sha256"]:
            raise PublicationError(f"semantic raw equation SHA 漂移: {key}")
        if sha256_text(canonical_json(row["canonical_artifact"])) != row["canonical_artifact_sha256"]:
            raise PublicationError(f"semantic canonical artifact SHA 漂移: {key}")
        if sha256_text(str(row["effective_raw_semantic_expression"])) != row[
            "effective_raw_semantic_expression_sha256"
        ]:
            raise PublicationError(f"semantic effective expression SHA 漂移: {key}")
        feature_names = row.get("feature_names")
        if not isinstance(feature_names, list) or not all(
            isinstance(name, str) for name in feature_names
        ):
            raise PublicationError(f"semantic feature_names 非法: {key}")
        derived = _artifact_expression(row["canonical_artifact"], feature_names)
        require_same_expression(
            derived,
            row["effective_raw_semantic_expression"],
            context=f"semantic canonical named expression {key}",
        )
    return output


def bind_raw_and_semantic(
    raw: Mapping[Key, Mapping[str, Any]],
    semantic: Mapping[Key, Mapping[str, Any]],
) -> None:
    if set(raw) != {key for key in semantic if key[3] == next(iter(raw))[3]}:
        raise PublicationError("raw 与 semantic condition 网格不同")
    for key, source in raw.items():
        row = semantic[key]
        record = source["record"]
        payload = source["payload"]
        if row["source_result_sha256"] != record["result"]["sha256"]:
            raise PublicationError(f"semantic 仍绑定旧 result: {key}")
        if row["raw_equation"] != payload["equation"]:
            raise PublicationError(f"semantic raw equation 与当前 result 不一致: {key}")


def _index_response_container(
    index: Mapping[str, Any],
    *,
    repo_root: Path,
    generation: str,
    new_transport: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = index.get("result_path")
    expected_sha = index.get("result_sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_sha, str):
        raise PublicationError(f"冻结 index 缺少 result_path/result_sha256: {index.get('logical_id')}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise PublicationError(f"冻结 Opus 响应缺失或 SHA 漂移: {path}")
    container = _read_json_object(path)
    if container.get("evaluation_key") != index.get("evaluation_key"):
        raise PublicationError(f"冻结响应 evaluation_key 与 index 不一致: {index.get('logical_id')}")
    if container.get("logical_id") != index.get("logical_id"):
        raise PublicationError(f"冻结响应 logical_id 与 index 不一致: {index.get('logical_id')}")
    metadata = container.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if generation == "new":
        channel = str(metadata.get("api_channel") or "").casefold()
        base_url = str(metadata.get("api_base_url") or "").casefold()
        if new_transport not in channel and new_transport not in base_url:
            raise PublicationError(f"新 Opus 证据不是 Routify: {index.get('logical_id')}")
    return container, metadata


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicationError(f"JSON 顶层不是 object: {path}")
    return value


def compact_llm_evidence(
    plan: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    repo_root: Path,
    generation: str,
    new_transport: str,
) -> dict[str, Any]:
    if index.get("state") == "unresolved":
        return {
            "logical_id": index["logical_id"],
            "evaluation_key": index["evaluation_key"],
            "task_type": index.get("task_type") or index.get("task_kind"),
            "condition": index.get("condition") or plan.get("condition"),
            "state": "unresolved",
            "dependencies": list(plan.get("dependencies") or []),
            "input_expression": None,
            "effective_expression": None,
            "expression_resolution": None,
            "structured_output": None,
            "unresolved": index.get("unresolved"),
            "response_path": None,
            "response_sha256": None,
            "evidence_generation": generation,
            "network_request": True,
        }
    if index.get("state") == "non_applicable":
        request = plan_request(plan)
        return {
            "logical_id": index["logical_id"],
            "evaluation_key": index["evaluation_key"],
            "task_type": index.get("task_type") or index.get("task_kind"),
            "condition": index.get("condition") or plan.get("condition"),
            "state": "non_applicable",
            "dependencies": list(plan.get("dependencies") or []),
            "input_expression": request.get("expression"),
            "effective_expression": None,
            "expression_resolution": None,
            "structured_output": None,
            "non_applicable": index.get("non_applicable"),
            "evidence_path": index.get("evidence_path") or plan.get("evidence_path"),
            "evidence_sha256": index.get("evidence_sha256") or plan.get("evidence_sha256"),
            "response_path": None,
            "response_sha256": None,
            "requested_model": None,
            "response_model": None,
            "requested_effort": None,
            "api_channel": None,
            "api_base_url": None,
            "transport_version": None,
            "evidence_generation": generation,
            "network_request": False,
        }
    container, metadata = _index_response_container(
        index,
        repo_root=repo_root,
        generation=generation,
        new_transport=new_transport,
    )
    requested_model = str(metadata.get("requested_model") or "")
    response_model = str(
        metadata.get("response_model")
        or (container.get("envelope") or {}).get("model")
        or ""
    )
    if "opus" not in (requested_model + response_model).casefold():
        raise PublicationError(f"冻结响应缺少 Opus 模型证据: {index.get('logical_id')}")
    request = plan_request(plan)
    return {
        "logical_id": index["logical_id"],
        "evaluation_key": index["evaluation_key"],
        "task_type": index.get("task_type") or index.get("task_kind"),
        "condition": index.get("condition") or plan.get("condition"),
        "state": index["state"],
        "dependencies": list(plan.get("dependencies") or []),
        "input_expression": request.get("expression"),
        "effective_expression": index.get("effective_expression"),
        "expression_resolution": index.get("expression_resolution"),
        "structured_output": index.get("structured_output"),
        "response_path": index.get("result_path"),
        "response_sha256": index["result_sha256"],
        "requested_model": requested_model,
        "response_model": response_model,
        "requested_effort": metadata.get("requested_effort"),
        "api_channel": metadata.get("api_channel"),
        "api_base_url": metadata.get("api_base_url"),
        "transport_version": metadata.get("transport_version"),
        "evidence_generation": generation,
    }


def row_generation(index: Mapping[str, Any], spec: ArtifactSpec) -> str:
    if spec.generation != "mixed":
        return spec.generation
    value = index.get("evidence_generation") or index.get("generation")
    if value not in {"new", "preexisting"}:
        raise PublicationError(
            f"mixed index 行必须声明 evidence_generation: {index.get('logical_id')}"
        )
    return str(value)


def _request_result_sha(request: Mapping[str, Any]) -> str | None:
    evidence = request.get("ast_source_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    for value in (
        request.get("result_raw_sha256"),
        request.get("result_sha256"),
        request.get("prediction_result_sha256"),
        evidence.get("result_raw_sha256"),
        evidence.get("result_sha256"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _plan_run_identity(plan: Mapping[str, Any], condition: str) -> Key:
    request = plan_request(plan)
    seed = request.get("seed")
    if seed is None:
        match = re.search(r"_s(520|521|522)_", str(request.get("task_id") or plan.get("logical_id")))
        if match is None:
            raise PublicationError(f"plan 缺少 seed: {plan.get('logical_id')}")
        seed = int(match.group(1))
    return run_key(
        request.get("algorithm_slug") or request.get("algorithm"),
        request.get("dataset_id"),
        seed,
        request.get("noise_tag") or plan.get("condition") or condition,
    )


def bind_predictions(
    pairs: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    condition: str,
    raw: Mapping[Key, Mapping[str, Any]],
    semantic: Mapping[Key, Mapping[str, Any]],
    index_spec: ArtifactSpec,
    config: PublicationConfig,
) -> tuple[dict[Key, dict[str, Any]], list[dict[str, Any]]]:
    bound: dict[Key, dict[str, Any]] = {}
    compact: list[dict[str, Any]] = []
    for plan, index in pairs.values():
        key = _plan_run_identity(plan, condition)
        if key in bound or key not in raw:
            raise PublicationError(f"prediction plan 网格重复或越界: {key}")
        request = plan_request(plan)
        current = semantic[key]
        source_sha = _request_result_sha(request)
        if source_sha != raw[key]["record"]["result"]["sha256"]:
            raise PublicationError(f"prediction plan 仍绑定旧 result: {key}")
        require_same_expression(
            request.get("expression"),
            current["effective_raw_semantic_expression"],
            context=f"prediction plan/current semantic {key}",
        )
        effective = index.get("effective_expression")
        if not isinstance(effective, str) or not effective.strip():
            raise PublicationError(f"prediction frozen index 缺少有效表达式: {key}")
        bound[key] = {
            "plan": plan,
            "index": index,
            "input_expression": str(request["expression"]),
            "effective_expression": effective.strip(),
            "evaluation_key": str(index["evaluation_key"]),
            "logical_id": str(index["logical_id"]),
        }
        compact.append(
            compact_llm_evidence(
                plan,
                index,
                repo_root=config.repo_root,
                generation=row_generation(index, index_spec),
                new_transport=config.new_evidence_transport,
            )
        )
    if set(bound) != set(raw):
        raise PublicationError(f"{condition} prediction plan/index 不是2250完整网格")
    return bound, compact


def bind_ground_truth(
    pairs: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    dataset_manifest: ArtifactSpec,
    index_spec: ArtifactSpec,
    config: PublicationConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    datasets = dataset_manifest.read_rows()
    dataset_names = {
        str(row.get("dataset_name") or row.get("dataset_id") or ""): row for row in datasets
    }
    if len(dataset_names) != 50 or "" in dataset_names:
        raise PublicationError("dataset manifest 不是50个唯一任务")
    output: dict[str, dict[str, Any]] = {}
    compact: list[dict[str, Any]] = []
    for plan, index in pairs.values():
        request = plan_request(plan)
        dataset_id = str(request.get("dataset_id") or "")
        if dataset_id not in dataset_names or dataset_id in output:
            raise PublicationError(f"GT plan 数据集越界或重复: {dataset_id}")
        effective = index.get("effective_expression")
        if not isinstance(effective, str) or not effective.strip():
            raise PublicationError(f"GT frozen index 缺少有效表达式: {dataset_id}")
        original = request.get("expression") or request.get("original_expression")
        if not isinstance(original, str) or not original.strip():
            raise PublicationError(f"GT plan 缺少原始表达式: {dataset_id}")
        source_checksums = (request.get("ast_source_evidence") or {}).get("source_checksums")
        if isinstance(source_checksums, Mapping):
            manifest_row = dataset_names[dataset_id]
            mapping = {
                "formula_py_sha256": "formula_sha256",
                "metadata_yaml_sha256": "metadata_sha256",
                "train_csv_sha256": "train_sha256",
                "valid_csv_sha256": "valid_sha256",
                "id_test_csv_sha256": "id_test_sha256",
                "ood_test_csv_sha256": "ood_test_sha256",
            }
            for source_name, manifest_name in mapping.items():
                if source_name in source_checksums and manifest_row.get(manifest_name):
                    if source_checksums[source_name] != manifest_row[manifest_name]:
                        raise PublicationError(f"GT 数据文件 SHA 绑定漂移: {dataset_id}/{source_name}")
        artifact = build_symbolic_artifact(effective.strip())
        output[dataset_id] = {
            "dataset_id": dataset_id,
            "original_expression": original.strip(),
            "effective_expression": effective.strip(),
            "evaluation_key": str(index["evaluation_key"]),
            "logical_id": str(index["logical_id"]),
            "artifact": artifact,
            "artifact_sha256": str(artifact["artifact_sha256"]),
        }
        compact.append(
            compact_llm_evidence(
                plan,
                index,
                repo_root=config.repo_root,
                generation=row_generation(index, index_spec),
                new_transport=config.new_evidence_transport,
            )
        )
    if set(output) != set(dataset_names):
        raise PublicationError("GT plan/index 未覆盖全部50任务")
    return output, compact


def _binding_entries(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if "frozen_evaluation_key" in value or "plan_evaluation_key" in value:
                entries.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(request)
    for prefix in ("prediction_a", "prediction_b"):
        evaluation_key = request.get(f"{prefix}_frozen_evaluation_key")
        expression = request.get(f"effective_{prefix}_expression") or request.get(
            f"simplified_{prefix}_expression"
        )
        if isinstance(evaluation_key, str) and isinstance(expression, str):
            entries.append(
                {
                    "frozen_evaluation_key": evaluation_key,
                    "frozen_effective_expression": expression,
                }
            )
    unique: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for entry in entries:
        if id(entry) not in seen:
            seen.add(id(entry))
            unique.append(entry)
    return unique


def _binding_key(entry: Mapping[str, Any]) -> str | None:
    for field in ("frozen_evaluation_key", "plan_evaluation_key", "evaluation_key"):
        value = entry.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _binding_expression(entry: Mapping[str, Any]) -> str | None:
    for field in (
        "frozen_effective_expression",
        "frozen_simplified_expression",
        "effective_expression",
        "expression",
    ):
        value = entry.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def require_dependencies(
    plan: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    context: str,
) -> None:
    dependencies = plan.get("dependencies")
    if not isinstance(dependencies, list):
        raise PublicationError(f"{context}: dependencies 未精确绑定当前上游")
    request = plan_request(plan)
    declared_dependencies = set(map(str, dependencies))
    if declared_dependencies != set(expected):
        standalone_bindings = request.get("frozen_dependency_evaluation_keys")
        if (
            declared_dependencies
            or not isinstance(standalone_bindings, list)
            or set(map(str, standalone_bindings)) != set(expected)
        ):
            raise PublicationError(f"{context}: dependencies 未精确绑定当前上游")
    entries = _binding_entries(request)
    by_key = {_binding_key(entry): entry for entry in entries if _binding_key(entry)}
    if not set(expected) <= set(by_key):
        raise PublicationError(f"{context}: request binding 缺少当前 evaluation key")
    for evaluation_key, expression in expected.items():
        require_same_expression(
            _binding_expression(by_key[evaluation_key]),
            expression,
            context=f"{context}/{evaluation_key}",
        )


def bind_equivalence(
    pairs: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    condition: str,
    predictions: Mapping[Key, Mapping[str, Any]],
    ground_truth: Mapping[str, Mapping[str, Any]],
    index_spec: ArtifactSpec,
    override_specs: Sequence[ArtifactSpec],
    config: PublicationConfig,
) -> tuple[dict[Key, dict[str, Any]], list[dict[str, Any]]]:
    output: dict[Key, dict[str, Any]] = {}
    compact: list[dict[str, Any]] = []
    compact_by_logical_id: dict[str, dict[str, Any]] = {}
    for plan, index in pairs.values():
        key = _plan_run_identity(plan, condition)
        if key in output or key not in predictions:
            raise PublicationError(f"equivalence plan 网格重复或越界: {key}")
        prediction = predictions[key]
        gt = ground_truth[key[1]]
        require_dependencies(
            plan,
            {
                prediction["evaluation_key"]: prediction["effective_expression"],
                gt["evaluation_key"]: gt["effective_expression"],
            },
            context=f"equivalence {key}",
        )
        decision = None
        if index.get("state") == "non_applicable":
            decision = "non_applicable"
        elif index.get("state") != "unresolved":
            decision = (index.get("structured_output") or {}).get("decision")
        if decision not in {"equivalent", "not_equivalent", "undetermined", "non_applicable"}:
            raise PublicationError(f"equivalence decision 非法: {key}/{decision}")
        output[key] = {
            "plan": plan,
            "index": index,
            "decision": decision,
            "underlying_decision": decision,
            "evaluation_key": str(index["evaluation_key"]),
            "audit_overlay_applied": False,
            "audit_source": None,
        }
        evidence = compact_llm_evidence(
            plan,
            index,
            repo_root=config.repo_root,
            generation=row_generation(index, index_spec),
            new_transport=config.new_evidence_transport,
        )
        evidence["underlying_structured_output"] = evidence["structured_output"]
        evidence["effective_decision"] = decision
        evidence["audit_overlay_applied"] = False
        compact.append(evidence)
        compact_by_logical_id[str(index["logical_id"])] = evidence
    if set(output) != set(predictions):
        raise PublicationError(f"{condition} equivalence plan/index 不是2250完整网格")
    by_logical_id = {str(item["index"]["logical_id"]): key for key, item in output.items()}
    applied: set[str] = set()
    for spec in override_specs:
        for override in spec.read_rows():
            target = str(override.get("target_logical_id") or "")
            if not target:
                raise PublicationError("equivalence audit override缺少target")
            if target in applied:
                raise PublicationError(f"equivalence audit override重复: {target}")
            if target not in by_logical_id:
                continue
            before = override.get("before_decision")
            after = override.get("after_decision")
            if after not in {"equivalent", "not_equivalent", "undetermined"}:
                raise PublicationError(f"equivalence audit override decision非法: {target}/{after}")
            key = by_logical_id[target]
            item = output[key]
            if item["underlying_decision"] != before:
                continue
            audit_sha = str(override.get("source_final_record_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", audit_sha):
                raise PublicationError(f"equivalence audit override缺少审计记录SHA: {target}")
            item["decision"] = after
            item["audit_overlay_applied"] = True
            item["audit_source"] = {
                "source_audit_logical_id": override.get("source_audit_logical_id"),
                "source_final_record_sha256": audit_sha,
                "override_manifest_sha256": spec.sha256,
            }
            evidence = compact_by_logical_id[target]
            evidence["effective_decision"] = after
            evidence["audit_overlay_applied"] = True
            evidence["audit_source"] = item["audit_source"]
            applied.add(target)
    return output, compact


def _structure_identity(plan: Mapping[str, Any], condition: str) -> tuple[str, str, int, int, str]:
    request = plan_request(plan)
    seed_a = request.get("seed_a")
    seed_b = request.get("seed_b")
    if seed_a is None or seed_b is None:
        match = re.search(r"s(520|521|522)-s?(520|521|522)", str(plan.get("logical_id")))
        if match is None:
            raise PublicationError(f"structure plan 缺少 seed pair: {plan.get('logical_id')}")
        seed_a, seed_b = int(match.group(1)), int(match.group(2))
    pair = tuple(sorted((int(seed_a), int(seed_b))))
    if pair not in PAIRS:
        raise PublicationError(f"structure seed pair 非法: {pair}")
    return (
        algorithm_slug(request.get("algorithm_slug") or request.get("algorithm")),
        str(request.get("dataset_id") or ""),
        pair[0],
        pair[1],
        str(request.get("noise_tag") or plan.get("condition") or condition),
    )


def validate_structure_overlay_coverage(
    output: Mapping[tuple[str, str, int, int, str], Mapping[str, Any]],
    *,
    current_matches: set[tuple[str, str, int, int, str]],
    applied: set[tuple[str, str, int, int, str]],
    condition: str,
    has_overlays: bool,
) -> None:
    """验证当前公式匹配的 overlay 均已应用，或仅因显式 unresolved 被隔离。"""

    if condition != "clean" or not has_overlays:
        return
    if len(current_matches) != 10:
        raise PublicationError(
            "clean current structure overlay应匹配10条: "
            f"matched={len(current_matches)}"
        )
    unresolved_matches = {
        identity
        for identity in current_matches
        if output[identity].get("state") == "unresolved"
    }
    expected_applied = current_matches - unresolved_matches
    if applied != expected_applied:
        raise PublicationError(
            "clean current structure overlay应用集合不符: "
            f"matched={len(current_matches)} unresolved={len(unresolved_matches)} "
            f"applied={len(applied)} expected={len(expected_applied)}"
        )


def build_publication_limitations(
    condition_reports: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    limitations = [
        "三个条件的EFF均由2250条可审计原生轨迹完整计算，无legacy fallback",
        "本发布只含最终六轴值，不宣称逐分钟 SYM/MIN/STAB 已补齐",
    ]
    clean_unresolved = int(
        condition_reports.get("clean", {}).get("structure_unresolved_task_count", 0)
    )
    if clean_unresolved:
        limitations.insert(
            1,
            "clean的gplearn g0029 s520-s521结构裁决显式unresolved，因此该task的m_stab及gplearn STAB为空",
        )
    return limitations


def bind_structure(
    pairs: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    condition: str,
    predictions: Mapping[Key, Mapping[str, Any]],
    index_spec: ArtifactSpec,
    overlay_specs: Sequence[tuple[ArtifactSpec, ArtifactSpec]],
    config: PublicationConfig,
) -> tuple[dict[tuple[str, str, int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    output: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    compact: list[dict[str, Any]] = []
    compact_by_logical_id: dict[str, dict[str, Any]] = {}
    for plan, index in pairs.values():
        identity = _structure_identity(plan, condition)
        algorithm, dataset_id, seed_a, seed_b, plan_condition = identity
        if identity in output or plan_condition != condition:
            raise PublicationError(f"structure plan 重复或condition错误: {identity}")
        left_key = (algorithm, dataset_id, seed_a, condition)
        right_key = (algorithm, dataset_id, seed_b, condition)
        if left_key not in predictions or right_key not in predictions:
            raise PublicationError(f"structure plan 越出 prediction 网格: {identity}")
        left = predictions[left_key]
        right = predictions[right_key]
        require_dependencies(
            plan,
            {
                left["evaluation_key"]: left["effective_expression"],
                right["evaluation_key"]: right["effective_expression"],
            },
            context=f"structure {identity}",
        )
        decision = (
            "non_applicable"
            if index.get("state") == "non_applicable"
            else (index.get("structured_output") or {}).get("decision")
        )
        if index.get("state") == "unresolved":
            if decision is not None or not isinstance(index.get("unresolved"), Mapping):
                raise PublicationError(f"structure unresolved含伪造decision或缺证据: {identity}")
        elif decision not in {
            "mathematically_equivalent",
            "same_canonical_structure",
            "different_structure",
            "undetermined",
            "non_applicable",
        }:
            raise PublicationError(f"structure decision 非法: {identity}/{decision}")
        output[identity] = {
            "decision": decision,
            "underlying_decision": decision,
            "evaluation_key": str(index["evaluation_key"]),
            "plan": plan,
            "index": index,
            "audit_overlay_applied": False,
            "audit_source": None,
            "state": index.get("state"),
        }
        evidence = compact_llm_evidence(
            plan,
            index,
            repo_root=config.repo_root,
            generation=row_generation(index, index_spec),
            new_transport=config.new_evidence_transport,
        )
        evidence["underlying_structured_output"] = evidence["structured_output"]
        evidence["effective_decision"] = decision
        evidence["audit_overlay_applied"] = False
        compact.append(evidence)
        compact_by_logical_id[str(index["logical_id"])] = evidence
    expected = {
        (key[0], key[1], seed_a, seed_b, condition)
        for key in predictions
        if key[2] == 520
        for seed_a, seed_b in PAIRS
    }
    if set(output) != expected:
        raise PublicationError(f"{condition} structure plan/index 不是2250完整pair网格")
    applied: set[tuple[str, str, int, int, str]] = set()
    current_matches: set[tuple[str, str, int, int, str]] = set()
    for plan_spec, overlay_index_spec in overlay_specs:
        overlay_pairs = plan_index_pairs(
            plan_spec,
            overlay_index_spec,
            label=f"{condition}.structure.audit_overlay",
        )
        for overlay_plan, overlay_index in overlay_pairs.values():
            identity = _structure_identity(overlay_plan, condition)
            if identity in applied:
                raise PublicationError(f"structure audit overlay重复: {identity}")
            if identity not in output:
                continue
            algorithm, dataset_id, seed_a, seed_b, _ = identity
            left = predictions[(algorithm, dataset_id, seed_a, condition)]
            right = predictions[(algorithm, dataset_id, seed_b, condition)]
            expected_bindings = {
                left["evaluation_key"]: left["effective_expression"],
                right["evaluation_key"]: right["effective_expression"],
            }
            entries = _binding_entries(plan_request(overlay_plan))
            observed_expressions = [
                expression_fingerprint(expression)
                for entry in entries
                if isinstance((expression := _binding_expression(entry)), str)
            ]
            expected_expressions = [
                expression_fingerprint(expression) for expression in expected_bindings.values()
            ]
            if sorted(set(observed_expressions)) != sorted(set(expected_expressions)):
                continue
            current_matches.add(identity)
            if output[identity].get("state") == "unresolved":
                quarantined_overlay = {
                    "reason": "current_expression_match_but_user_required_opus_unresolved",
                    "overlay_logical_id": overlay_index.get("logical_id"),
                    "overlay_result_sha256": overlay_index.get("result_sha256"),
                    "overlay_plan_artifact_sha256": plan_spec.sha256,
                    "overlay_index_artifact_sha256": overlay_index_spec.sha256,
                }
                output[identity]["audit_overlay_not_applied"] = quarantined_overlay
                compact_by_logical_id[str(output[identity]["index"]["logical_id"])][
                    "audit_overlay_not_applied"
                ] = quarantined_overlay
                continue
            decision = (overlay_index.get("structured_output") or {}).get("decision")
            if decision not in {
                "mathematically_equivalent",
                "same_canonical_structure",
                "different_structure",
                "undetermined",
            }:
                raise PublicationError(f"structure audit overlay decision非法: {identity}/{decision}")
            item = output[identity]
            item["decision"] = decision
            item["audit_overlay_applied"] = True
            item["audit_source"] = {
                "overlay_logical_id": overlay_index.get("logical_id"),
                "overlay_evaluation_key": overlay_index.get("evaluation_key"),
                "overlay_result_sha256": overlay_index.get("result_sha256"),
                "overlay_revision_sha256": overlay_index.get("overlay_revision_sha256"),
                "overlay_plan_artifact_sha256": plan_spec.sha256,
                "overlay_index_artifact_sha256": overlay_index_spec.sha256,
            }
            base_logical_id = str(item["index"]["logical_id"])
            evidence = compact_by_logical_id[base_logical_id]
            evidence["effective_decision"] = decision
            evidence["audit_overlay_applied"] = True
            evidence["audit_source"] = item["audit_source"]
            evidence["audit_overlay_structured_output"] = overlay_index.get("structured_output")
            applied.add(identity)
    validate_structure_overlay_coverage(
        output,
        current_matches=current_matches,
        applied=applied,
        condition=condition,
        has_overlays=bool(overlay_specs),
    )
    return output, compact


def _numeric_identity(row: Mapping[str, Any], condition: str) -> Key:
    return run_key(
        row.get("algorithm"),
        row.get("dataset_id"),
        row.get("seed"),
        row.get("condition") or row.get("noise_tag") or condition,
    )


def _finite(value: object, *, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise PublicationError(f"{context} 缺失或非数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PublicationError(f"{context} 非数值: {value!r}") from exc
    if not math.isfinite(number):
        raise PublicationError(f"{context} 非有限: {value!r}")
    return number


def bind_numeric(
    spec: ArtifactSpec,
    *,
    condition: str,
    raw: Mapping[Key, Mapping[str, Any]],
    semantic: Mapping[Key, Mapping[str, Any]],
) -> dict[Key, dict[str, Any]]:
    output: dict[Key, dict[str, Any]] = {}
    for row in spec.read_rows():
        key = _numeric_identity(row, condition)
        if key in output or key not in raw:
            raise PublicationError(f"numeric 网格重复或越界: {key}")
        if row.get("result_sha256") != raw[key]["record"]["result"]["sha256"]:
            raise PublicationError(f"numeric 仍绑定旧 result: {key}")
        artifact_sha = (
            row.get("canonical_artifact_sha256")
            or row.get("replay_canonical_artifact_sha256")
            or row.get("source_canonical_artifact_sha256")
        )
        if artifact_sha != semantic[key]["canonical_artifact_sha256"]:
            raise PublicationError(f"numeric canonical artifact 与当前 semantic 不一致: {key}")
        valid = str(row.get("valid_output", "")).casefold() == "true"
        id_quality = _finite(row.get("id_quality"), context=f"{key}.id_quality")
        ood_quality = _finite(row.get("ood_quality"), context=f"{key}.ood_quality")
        if not (0.0 <= id_quality <= 1.0 and 0.0 <= ood_quality <= 1.0):
            raise PublicationError(f"numeric quality 越界: {key}")
        if valid:
            id_nmse = _finite(row.get("id_nmse"), context=f"{key}.id_nmse")
            ood_nmse = _finite(row.get("ood_nmse"), context=f"{key}.ood_nmse")
            if id_nmse < 0 or ood_nmse < 0:
                raise PublicationError(f"numeric NMSE 为负: {key}")
            if not math.isclose(id_quality, phi_nmse(id_nmse), rel_tol=0, abs_tol=2e-14):
                raise PublicationError(f"numeric ID quality 未按 phi 计算: {key}")
            if not math.isclose(ood_quality, phi_nmse(ood_nmse), rel_tol=0, abs_tol=2e-14):
                raise PublicationError(f"numeric OOD quality 未按 phi 计算: {key}")
        elif id_quality != 0.0 or ood_quality != 0.0:
            raise PublicationError(f"invalid numeric 必须质量置0: {key}")
        output[key] = dict(row)
    if set(output) != set(raw):
        raise PublicationError(f"{condition} numeric 不是2250完整网格")
    return output


def _trajectory_identity(row: Mapping[str, Any], condition: str) -> Key:
    return run_key(
        row.get("algorithm"),
        row.get("dataset_id"),
        row.get("seed"),
        row.get("condition") or row.get("noise_tag") or condition,
    )


def validate_eff_status_row(row: Mapping[str, Any], *, key: Key) -> str:
    availability = str(row.get("availability_status") or "")
    if availability not in {"available", "unavailable"}:
        raise PublicationError(f"EFF revision availability非法: {key}/{availability}")
    if availability == "available":
        q_star = _finite(row.get("q_star"), context=f"{key}.q_star")
        m_eff = _finite(row.get("m_eff"), context=f"{key}.m_eff")
        if q_star < 0 or not 0 <= m_eff <= 1:
            raise PublicationError(f"EFF revision available数值非法: {key}")
    else:
        if any(row.get(field) not in {None, ""} for field in ("q_star", "m_eff", "eff_score")):
            raise PublicationError(f"EFF unavailable不得填0或部分值: {key}")
        if not row.get("unavailable_reason") or not row.get("required_action"):
            raise PublicationError(f"EFF unavailable缺少原因/动作: {key}")
    return availability


def validate_eff_curve_primary(row: Mapping[str, Any], *, formal: bool, context: str) -> None:
    primary_fields = (
        "mean_id_quality",
        "mean_ood_quality",
        "mean_quality",
        "mean_relative_progress",
        "cumulative_eff_score",
    )
    if formal:
        for field in primary_fields:
            _finite(row.get(field), context=f"{context}.{field}")
    elif any(row.get(field) not in {None, ""} for field in primary_fields):
        raise PublicationError(f"EFF不完整算法curve不得发布部分主均值: {context}")


def load_eff_revision(
    inputs: EffConditionInputs,
    *,
    condition: str,
    expected_keys: set[Key],
) -> tuple[
    dict[Key, dict[str, Any]],
    dict[Key, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = inputs.manifest.read_rows()[0]
    if (
        manifest.get("condition") != condition
        or int(manifest.get("available_run_count", -1)) != RUNS_PER_CONDITION
        or int(manifest.get("unavailable_run_count", -1)) != 0
        or int(manifest.get("formal_ready_algorithm_count", -1)) != 15
        or manifest.get("candidate_selection_uses_id_ood") is not False
        or manifest.get("legacy_fallback_used") is not False
    ):
        raise PublicationError(f"EFFv3 {condition} manifest不满足完整正式合同")
    if inputs.unavailable.read_rows():
        raise PublicationError(f"EFFv3 {condition} unavailable必须为空")

    status: dict[Key, dict[str, Any]] = {}
    for row in inputs.run_status.read_rows():
        key = _trajectory_identity(row, condition)
        if key[3] != condition or key in status:
            raise PublicationError(f"EFFv3 run status重复或condition错误: {key}")
        if validate_eff_status_row(row, key=key) != "available":
            raise PublicationError(f"EFFv3 run必须全部available: {key}")
        status[key] = dict(row)
    if set(status) != expected_keys or len(status) != RUNS_PER_CONDITION:
        raise PublicationError(f"EFFv3 {condition} run status不是2250完整网格")

    trajectories: dict[Key, dict[str, Any]] = {}
    for row in inputs.native_trajectories.read_rows():
        key = _trajectory_identity(row, condition)
        if key in trajectories or key not in status:
            raise PublicationError(f"EFFv3 native trajectory重复或越界: {key}")
        arrays = {}
        for field in (
            "expression",
            "id_quality",
            "ood_quality",
            "quality",
            "valid_output",
            "objective_field",
            "objective_value",
            "incumbent_source_minute",
            "trajectory_source",
            "source_path",
            "source_sha256",
        ):
            value = row.get(field)
            if not isinstance(value, list) or len(value) != HORIZON:
                raise PublicationError(f"EFFv3 {key}.{field}不是180点")
            arrays[field] = value
        qualities = [_finite(value, context=f"{key}.quality") for value in arrays["quality"]]
        id_values = [_finite(value, context=f"{key}.id_quality") for value in arrays["id_quality"]]
        ood_values = [_finite(value, context=f"{key}.ood_quality") for value in arrays["ood_quality"]]
        if any(
            not math.isclose(q, (qid + qood) / 2.0, rel_tol=0, abs_tol=2e-14)
            for q, qid, qood in zip(qualities, id_values, ood_values)
        ):
            raise PublicationError(f"EFFv3 {key} quality不等于ID/OOD均值")
        m_eff = efficiency_from_qualities(qualities, horizon=HORIZON)
        if not math.isclose(m_eff, float(status[key]["m_eff"]), rel_tol=0, abs_tol=2e-13):
            raise PublicationError(f"EFFv3 {key} trajectory/status m_eff不一致")
        normalized = {
            "logical_key": row["logical_key"],
            "algorithm": row["algorithm"],
            "algorithm_slug": key[0],
            "dataset_id": key[1],
            "seed": key[2],
            "condition": condition,
            "task_id": row.get("task_id", ""),
            "host": row.get("host", ""),
            "source_tier": "eff_revision_v3_native",
            "trajectory_basis": "algorithm_native_internal_best_so_far.v1",
            "formal_ready": "true",
            "source_artifact_path": str(inputs.native_trajectories.path),
            "source_artifact_sha256": inputs.native_trajectories.sha256,
            "q_star": max(qualities),
            "m_eff": m_eff,
            "id_ood_quality_available": "true",
            "availability_status": "available",
            "unavailable_reason": "",
            "required_action": "",
        }
        for minute, value in enumerate(id_values, start=1):
            normalized[f"id_q_{minute:04d}"] = value
        for minute, value in enumerate(ood_values, start=1):
            normalized[f"ood_q_{minute:04d}"] = value
        for minute, value in enumerate(qualities, start=1):
            normalized[f"q_{minute:04d}"] = value
        trajectories[key] = normalized
    if set(trajectories) != expected_keys:
        raise PublicationError(f"EFFv3 {condition} native trajectory不是2250完整网格")

    minute_rows = inputs.minute_metrics.read_rows()
    minute_keys: set[tuple[Key, int]] = set()
    for row in minute_rows:
        key = _trajectory_identity(row, condition)
        minute = int(row["minute"])
        identity = (key, minute)
        if key not in expected_keys or not 1 <= minute <= HORIZON or identity in minute_keys:
            raise PublicationError(f"EFFv3 minute identity非法或重复: {identity}")
        minute_keys.add(identity)
        qid = _finite(row.get("id_quality"), context=f"{identity}.id")
        qood = _finite(row.get("ood_quality"), context=f"{identity}.ood")
        q = _finite(row.get("quality"), context=f"{identity}.q")
        if not math.isclose(q, (qid + qood) / 2.0, rel_tol=0, abs_tol=2e-14):
            raise PublicationError(f"EFFv3 minute quality不等于ID/OOD均值: {identity}")
        trajectory = trajectories[key]
        if not math.isclose(q, float(trajectory[f"q_{minute:04d}"]), rel_tol=0, abs_tol=2e-14):
            raise PublicationError(f"EFFv3 minute/native trajectory漂移: {identity}")
    if len(minute_keys) != RUNS_PER_CONDITION * HORIZON:
        raise PublicationError(f"EFFv3 {condition} minute表不是405000行")

    summary_rows = inputs.algorithm_summary.read_rows()
    summaries = {algorithm_slug(row["algorithm"]): row for row in summary_rows}
    if len(summaries) != 15:
        raise PublicationError(f"EFFv3 {condition} summary不是15算法")
    for algorithm, row in summaries.items():
        if (
            row.get("condition") != condition
            or int(row["available_run_count"]) != 150
            or int(row["unavailable_run_count"]) != 0
            or str(row["formal_ready"]).casefold() != "true"
        ):
            raise PublicationError(f"EFFv3 summary非150/150 ready: {condition}/{algorithm}")
        derived = 100.0 * sum(
            float(item["m_eff"]) for key, item in status.items() if key[0] == algorithm
        ) / 150.0
        if not math.isclose(float(row["EFF"]), derived, rel_tol=0, abs_tol=2e-12):
            raise PublicationError(f"EFFv3 summary不可由run复现: {condition}/{algorithm}")

    curve_rows = inputs.algorithm_curves.read_rows()
    curve_keys = {(algorithm_slug(row["algorithm"]), int(row["minute"])) for row in curve_rows}
    if len(curve_rows) != CURVES_PER_CONDITION or len(curve_keys) != CURVES_PER_CONDITION:
        raise PublicationError(f"EFFv3 {condition} curve不是15x180")
    for row in curve_rows:
        if row.get("condition") != condition or str(row.get("formal_ready")).casefold() != "true":
            raise PublicationError(f"EFFv3 curve非正式ready: {condition}")
        validate_eff_curve_primary(
            row,
            formal=True,
            context=f"{condition}/{row['algorithm']}/minute{row['minute']}",
        )
    return status, trajectories, minute_rows, curve_rows, summary_rows


def validate_eff_revision_v3(inputs: EffRevisionInputs) -> list[tuple[Path, str]]:
    manifest = inputs.manifest.read_rows()[0]
    validation = inputs.validation_report.read_rows()[0]
    if (
        manifest.get("schema_version") != "three_condition_native_eff_revision.v3"
        or manifest.get("status") != "complete"
        or manifest.get("candidate_selection_uses_id_ood") is not False
        or manifest.get("legacy_fallback_used") is not False
        or validation.get("status") != "passed"
        or validation.get("manifest_status") != "complete"
    ):
        raise PublicationError("EFF revision v3全局manifest/validation不满足正式合同")
    root = inputs.checksums.path.parent.resolve()
    entries: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for row in inputs.checksums.read_rows():
        line = str(row.get("line") or "")
        match = re.fullmatch(r"([0-9a-f]{64})\s{2}(.+)", line)
        if match is None:
            raise PublicationError("EFFv3 SHA256SUMS格式非法")
        relative = Path(match.group(2))
        path = (root / relative).resolve()
        if root not in path.parents or path in seen or path.is_symlink() or not path.is_file():
            raise PublicationError(f"EFFv3 SHA路径越界、重复或未物化: {relative}")
        if sha256_file(path) != match.group(1):
            raise PublicationError(f"EFFv3 SHA漂移: {relative}")
        entries.append((path, match.group(1)))
        seen.add(path)
    configured = {
        spec.path.resolve()
        for _, spec in _eff_specs(inputs)
        if spec is not inputs.checksums
    }
    if not configured <= seen:
        missing = sorted(str(path) for path in configured - seen)
        raise PublicationError(f"EFFv3 SHA未覆盖配置输入: {missing[:3]}")
    return entries


def _eff_specs(inputs: EffRevisionInputs) -> list[tuple[str, ArtifactSpec]]:
    specs = [
        ("manifest", inputs.manifest),
        ("validation_report", inputs.validation_report),
        ("checksums", inputs.checksums),
    ]
    for condition, item in inputs.conditions.items():
        specs.extend(
            [
                (f"{condition}/run_status", item.run_status),
                (f"{condition}/native_trajectories", item.native_trajectories),
                (f"{condition}/minute_metrics", item.minute_metrics),
                (f"{condition}/algorithm_curves", item.algorithm_curves),
                (f"{condition}/algorithm_summary", item.algorithm_summary),
                (f"{condition}/unavailable", item.unavailable),
                (f"{condition}/manifest", item.manifest),
            ]
        )
    return specs


def merge_trajectories(
    base_spec: ArtifactSpec,
    override_specs: Sequence[ArtifactSpec],
    *,
    condition: str,
    expected_keys: set[Key],
) -> tuple[dict[Key, dict[str, Any]], dict[Key, ArtifactSpec]]:
    output: dict[Key, dict[str, Any]] = {}
    sources: dict[Key, ArtifactSpec] = {}
    for row in base_spec.read_rows():
        key = _trajectory_identity(row, condition)
        if key in output:
            raise PublicationError(f"trajectory base 重复: {key}")
        output[key] = dict(row)
        sources[key] = base_spec
    if set(output) != expected_keys:
        raise PublicationError(f"{condition} trajectory base 不是2250完整网格")
    replaced: set[Key] = set()
    for spec in override_specs:
        for row in spec.read_rows():
            key = _trajectory_identity(row, condition)
            if key[3] != condition:
                continue
            if key not in output or key in replaced:
                raise PublicationError(f"trajectory override 越界或重复替换: {key}")
            output[key] = dict(row)
            sources[key] = spec
            replaced.add(key)
    return output, sources


def normalize_trajectory(
    row: Mapping[str, Any],
    *,
    key: Key,
    source_spec: ArtifactSpec,
) -> dict[str, Any]:
    defaults = dict(source_spec.defaults or {})
    qualities: list[float] = []
    id_qualities: list[float | str] = []
    ood_qualities: list[float | str] = []
    for minute in range(1, HORIZON + 1):
        q = _finite(row.get(f"q_{minute:04d}"), context=f"{key}.q_{minute:04d}")
        if not 0.0 <= q <= 1.0:
            raise PublicationError(f"trajectory q 越界: {key}/minute{minute}")
        qualities.append(q)
        id_value = row.get(f"id_q_{minute:04d}", row.get(f"q_id_{minute:04d}", ""))
        ood_value = row.get(f"ood_q_{minute:04d}", row.get(f"q_ood_{minute:04d}", ""))
        if id_value in {None, ""} or ood_value in {None, ""}:
            id_qualities.append("")
            ood_qualities.append("")
        else:
            q_id = _finite(id_value, context=f"{key}.id_q_{minute:04d}")
            q_ood = _finite(ood_value, context=f"{key}.ood_q_{minute:04d}")
            if not math.isclose(q, (q_id + q_ood) / 2.0, rel_tol=0, abs_tol=2e-14):
                raise PublicationError(f"trajectory q != mean(ID,OOD): {key}/minute{minute}")
            id_qualities.append(q_id)
            ood_qualities.append(q_ood)
    m_eff = efficiency_from_qualities(qualities, horizon=HORIZON)
    if row.get("m_eff") not in {None, ""} and not math.isclose(
        m_eff, float(row["m_eff"]), rel_tol=0, abs_tol=2e-13
    ):
        raise PublicationError(f"trajectory m_eff 不可复现: {key}")
    q_star = max(qualities)
    basis = str(row.get("trajectory_basis") or defaults.get("trajectory_basis") or "")
    if not basis:
        raise PublicationError(f"trajectory_basis 缺失: {key}")
    source_tier = str(row.get("source_tier") or defaults.get("source_tier") or "")
    formal_raw = row.get("formal_ready", defaults.get("formal_ready"))
    if formal_raw is None and isinstance(defaults.get("formal_ready_source_tiers"), list):
        formal_ready = source_tier in defaults["formal_ready_source_tiers"]
    else:
        formal_ready = (
            str(formal_raw).casefold() == "true"
            if not isinstance(formal_raw, bool)
            else formal_raw
        )
    output = {
        "logical_key": logical_key(key, str(row.get("algorithm") or key[0])),
        "algorithm": str(row.get("algorithm") or key[0]),
        "algorithm_slug": key[0],
        "dataset_id": key[1],
        "seed": key[2],
        "condition": key[3],
        "task_id": row.get("task_id", ""),
        "host": row.get("host", ""),
        "source_tier": source_tier,
        "trajectory_basis": basis,
        "formal_ready": str(formal_ready).lower(),
        "source_artifact_path": str(source_spec.path),
        "source_artifact_sha256": source_spec.sha256,
        "q_star": q_star,
        "m_eff": m_eff,
        "id_ood_quality_available": str(all(value != "" for value in id_qualities)).lower(),
    }
    for minute, value in enumerate(id_qualities, start=1):
        output[f"id_q_{minute:04d}"] = value
    for minute, value in enumerate(ood_qualities, start=1):
        output[f"ood_q_{minute:04d}"] = value
    for minute, value in enumerate(qualities, start=1):
        output[f"q_{minute:04d}"] = value
    return output


TRAJECTORY_BASE_FIELDS = (
    "logical_key",
    "algorithm",
    "algorithm_slug",
    "dataset_id",
    "seed",
    "condition",
    "task_id",
    "host",
    "source_tier",
    "trajectory_basis",
    "formal_ready",
    "source_artifact_path",
    "source_artifact_sha256",
    "q_star",
    "m_eff",
    "id_ood_quality_available",
)


def trajectory_fields() -> list[str]:
    return [
        *TRAJECTORY_BASE_FIELDS,
        *(f"id_q_{minute:04d}" for minute in range(1, HORIZON + 1)),
        *(f"ood_q_{minute:04d}" for minute in range(1, HORIZON + 1)),
        *(f"q_{minute:04d}" for minute in range(1, HORIZON + 1)),
    ]


def build_curves(trajectories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[str(row["algorithm_slug"])].append(row)
    curves = []
    for algorithm in sorted(grouped):
        runs = grouped[algorithm]
        if len(runs) != 150:
            raise PublicationError(f"curve 算法运行数不是150: {algorithm}")
        running_relative = [0.0] * len(runs)
        for minute in range(1, HORIZON + 1):
            q_values = [float(row[f"q_{minute:04d}"]) for row in runs]
            relative = [
                q / float(row["q_star"]) if float(row["q_star"]) > 0.0 else 0.0
                for row, q in zip(runs, q_values)
            ]
            running_relative = [left + right for left, right in zip(running_relative, relative)]
            id_values = [
                float(row[f"id_q_{minute:04d}"])
                for row in runs
                if row[f"id_q_{minute:04d}"] not in {None, ""}
            ]
            ood_values = [
                float(row[f"ood_q_{minute:04d}"])
                for row in runs
                if row[f"ood_q_{minute:04d}"] not in {None, ""}
            ]
            bases = sorted({str(row["trajectory_basis"]) for row in runs})
            curves.append(
                {
                    "algorithm": runs[0]["algorithm"],
                    "algorithm_slug": algorithm,
                    "condition": runs[0]["condition"],
                    "minute": minute,
                    "run_count": len(runs),
                    "id_ood_quality_count": min(len(id_values), len(ood_values)),
                    "mean_id_quality": sum(id_values) / len(id_values) if id_values else "",
                    "mean_ood_quality": sum(ood_values) / len(ood_values) if ood_values else "",
                    "mean_combined_quality": sum(q_values) / len(q_values),
                    "mean_relative_progress": sum(relative) / len(relative),
                    "mean_cumulative_eff": 100.0
                    * sum(value / minute for value in running_relative)
                    / len(running_relative),
                    "trajectory_basis": ";".join(bases),
                    "formal_ready": str(all(row["formal_ready"] == "true" for row in runs)).lower(),
                }
            )
    if len(curves) != CURVES_PER_CONDITION:
        raise PublicationError(f"condition curve 应有2700行，实际{len(curves)}")
    return curves


def validate_trajectory_readiness(
    trajectories: Sequence[Mapping[str, Any]], *, condition: str
) -> None:
    if condition == "clean":
        raise PublicationError("clean轨迹必须通过eff_revision fail-closed索引处理")
    symbolfit = [row for row in trajectories if row["algorithm_slug"] == "symbolfit"]
    others = [row for row in trajectories if row["algorithm_slug"] != "symbolfit"]
    if len(symbolfit) != 150 or any(row["formal_ready"] != "true" for row in symbolfit):
        raise PublicationError(f"{condition} 最新SymbolFit 150轨迹必须formal_ready=true")
    if len(others) != 2100 or any(row["formal_ready"] != "false" for row in others):
        raise PublicationError(f"{condition} 其余observed轨迹必须formal_ready=false")


def build_run_rows(
    *,
    condition: str,
    raw: Mapping[Key, Mapping[str, Any]],
    semantic: Mapping[Key, Mapping[str, Any]],
    predictions: Mapping[Key, Mapping[str, Any]],
    equivalence: Mapping[Key, Mapping[str, Any]],
    ground_truth: Mapping[str, Mapping[str, Any]],
    numeric: Mapping[Key, Mapping[str, Any]],
    trajectories: Mapping[Key, Mapping[str, Any]],
    trajectory_sources: Mapping[Key, ArtifactSpec],
    config: PublicationConfig,
    condition_inputs: ConditionInputs,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(raw):
        source = raw[key]
        semantic_row = semantic[key]
        prediction = predictions[key]
        eq = equivalence[key]
        gt = ground_truth[key[1]]
        number = numeric[key]
        trajectory = trajectories[key]
        pred_artifact = build_symbolic_artifact(prediction["effective_expression"])
        gt_artifact = gt["artifact"]
        valid = str(number.get("valid_output", "")).casefold() == "true"
        decision = str(eq["decision"])
        t = tree_similarity(gt_artifact, pred_artifact) if valid else 0.0
        v = variable_f1(gt_artifact, pred_artifact) if valid else 0.0
        o = operator_f1(gt_artifact, pred_artifact) if valid else 0.0
        c_ref = int(gt_artifact["node_count"])
        c_pred = int(pred_artifact["node_count"]) if valid else 0
        m_sym = symbolic_fidelity_score(
            equivalent=decision == "equivalent",
            tree_similarity=t,
            variable_f1=v,
            operator_f1=o,
            valid=valid,
        )
        m_min = minimality_score(c_ref, max(c_pred, 1), valid=valid)
        display_algorithm = str(number.get("algorithm") or semantic_row.get("algorithm") or key[0])
        payload = source["payload"]
        record = source["record"]
        rows.append(
            {
                "logical_key": logical_key(key, display_algorithm),
                "algorithm": display_algorithm,
                "algorithm_slug": key[0],
                "dataset_id": key[1],
                "seed": key[2],
                "condition": condition,
                "task_id": (record.get("source") or {}).get("task_id", ""),
                "status": payload.get("status"),
                "valid_output": str(valid).lower(),
                "raw_original_equation": payload.get("equation"),
                "raw_original_equation_sha256": semantic_row["raw_equation_sha256"],
                "canonical_named_expression": semantic_row["effective_raw_semantic_expression"],
                "canonical_named_expression_sha256": semantic_row[
                    "effective_raw_semantic_expression_sha256"
                ],
                "canonical_artifact_sha256": semantic_row["canonical_artifact_sha256"],
                "opus5_effective_expression": prediction["effective_expression"],
                "opus5_effective_expression_sha256": sha256_text(
                    prediction["effective_expression"]
                ),
                "ground_truth_original_expression": gt["original_expression"],
                "ground_truth_effective_expression": gt["effective_expression"],
                "ground_truth_artifact_sha256": gt["artifact_sha256"],
                "equivalence_decision": decision,
                "underlying_equivalence_decision": eq["underlying_decision"],
                "equivalence_audit_overlay_applied": str(eq["audit_overlay_applied"]).lower(),
                "equivalence_audit_source": canonical_json(eq["audit_source"])
                if eq["audit_source"]
                else "",
                "equivalent": str(decision == "equivalent").lower(),
                "tree_similarity": t,
                "variable_f1": v,
                "operator_f1": o,
                "C_ref": c_ref,
                "C_pred": c_pred,
                "m_sym": m_sym,
                "m_min": m_min,
                "id_nmse": number.get("id_nmse", ""),
                "ood_nmse": number.get("ood_nmse", ""),
                "id_quality": number["id_quality"],
                "ood_quality": number["ood_quality"],
                "m_eff": trajectory["m_eff"],
                "eff_availability_status": trajectory.get("availability_status", "available"),
                "eff_unavailable_reason": trajectory.get("unavailable_reason", ""),
                "eff_required_action": trajectory.get("required_action", ""),
                "trajectory_basis": trajectory["trajectory_basis"],
                "trajectory_formal_ready": trajectory["formal_ready"],
                "raw_result_sha256": record["result"]["sha256"],
                "raw_results_artifact_sha256": condition_inputs.raw_results.sha256,
                "semantic_runs_artifact_sha256": config.semantic_runs.sha256,
                "numeric_artifact_sha256": condition_inputs.numeric.sha256,
                "pred_plan_artifact_sha256": condition_inputs.pred_plan.sha256,
                "pred_index_artifact_sha256": condition_inputs.pred_index.sha256,
                "pred_evaluation_key": prediction["evaluation_key"],
                "equivalence_plan_artifact_sha256": condition_inputs.equivalence_plan.sha256,
                "equivalence_index_artifact_sha256": condition_inputs.equivalence_index.sha256,
                "equivalence_evaluation_key": eq["evaluation_key"],
                "trajectory_artifact_sha256": trajectory_sources[key].sha256,
                "symbolic_component_basis": "recomputed_from_current_effective_pred_and_gt",
                "assembly_complete": str(trajectory["m_eff"] not in {None, ""}).lower(),
            }
        )
    if len(rows) != RUNS_PER_CONDITION:
        raise PublicationError(f"{condition} run_final 应为2250行")
    return rows


def build_task_stability(
    run_rows: Sequence[Mapping[str, Any]],
    structure: Mapping[tuple[str, str, int, int, str], Mapping[str, Any]],
    *,
    condition: str,
    structure_plan_sha: str,
    structure_index_sha: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(str(row["algorithm_slug"]), str(row["dataset_id"]))].append(row)
    task_rows: list[dict[str, Any]] = []
    valid_labels = {"mathematically_equivalent", "same_canonical_structure"}
    for (algorithm, dataset_id), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in group) != SEEDS:
            raise PublicationError(f"STAB seed网格不完整: {algorithm}/{dataset_id}/{condition}")
        decisions: list[str | None] = []
        evaluation_keys = []
        items = []
        for seed_a, seed_b in PAIRS:
            item = structure[(algorithm, dataset_id, seed_a, seed_b, condition)]
            decisions.append(item["decision"])
            evaluation_keys.append(str(item["evaluation_key"]))
            items.append(item)
        qualities = [
            RunQuality(
                id_quality=float(row["id_quality"]),
                ood_quality=float(row["ood_quality"]),
                valid=str(row["valid_output"]) == "true",
            )
            for row in group
        ]
        unresolved = any(item.get("state") == "unresolved" for item in items)
        if unresolved:
            numerical = numerical_consistency(qualities)
            validity = sum(run.validated().valid for run in qualities) / 3.0
            structural: float | str = ""
            m_stab: float | str = ""
        else:
            stab = stability_score(
                qualities,
                structural_pair_results=[decision in valid_labels for decision in decisions],
            )
            numerical = stab.numerical_consistency
            validity = stab.validity
            structural = stab.structural_consistency
            m_stab = stab.score
        task_rows.append(
            {
                "algorithm": group[0]["algorithm"],
                "algorithm_slug": algorithm,
                "dataset_id": dataset_id,
                "condition": condition,
                "pair_520_521": decisions[0] if decisions[0] is not None else "",
                "pair_520_522": decisions[1] if decisions[1] is not None else "",
                "pair_521_522": decisions[2] if decisions[2] is not None else "",
                "pair_520_521_underlying": structure[
                    (algorithm, dataset_id, 520, 521, condition)
                ]["underlying_decision"],
                "pair_520_522_underlying": structure[
                    (algorithm, dataset_id, 520, 522, condition)
                ]["underlying_decision"],
                "pair_521_522_underlying": structure[
                    (algorithm, dataset_id, 521, 522, condition)
                ]["underlying_decision"],
                "pair_520_521_audit_source": canonical_json(
                    structure[(algorithm, dataset_id, 520, 521, condition)]["audit_source"]
                )
                if structure[(algorithm, dataset_id, 520, 521, condition)]["audit_source"]
                else "",
                "pair_520_522_audit_source": canonical_json(
                    structure[(algorithm, dataset_id, 520, 522, condition)]["audit_source"]
                )
                if structure[(algorithm, dataset_id, 520, 522, condition)]["audit_source"]
                else "",
                "pair_521_522_audit_source": canonical_json(
                    structure[(algorithm, dataset_id, 521, 522, condition)]["audit_source"]
                )
                if structure[(algorithm, dataset_id, 521, 522, condition)]["audit_source"]
                else "",
                "pair_520_521_evaluation_key": evaluation_keys[0],
                "pair_520_522_evaluation_key": evaluation_keys[1],
                "pair_521_522_evaluation_key": evaluation_keys[2],
                "numerical_consistency": numerical,
                "validity": validity,
                "structural_consistency": structural,
                "m_stab": m_stab,
                "structure_plan_artifact_sha256": structure_plan_sha,
                "structure_index_artifact_sha256": structure_index_sha,
                "assembly_complete": str(not unresolved).lower(),
                "unresolved_structure": str(unresolved).lower(),
            }
        )
    if len(task_rows) != TASKS_PER_CONDITION:
        raise PublicationError(f"{condition} task_stability 应为750行，实际{len(task_rows)}")
    return task_rows


def aggregate_six_axis(
    run_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    run_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        run_groups[str(row["algorithm_slug"])].append(row)
    for row in task_rows:
        task_groups[str(row["algorithm_slug"])].append(row)
    output = []
    for algorithm in sorted(run_groups):
        runs = run_groups[algorithm]
        tasks = task_groups[algorithm]
        if len(runs) != 150 or len(tasks) != 50:
            raise PublicationError(f"six-axis 网格不完整: {algorithm}/{condition}")
        eff_values = [float(row["m_eff"]) for row in runs if row["m_eff"] not in {None, ""}]
        stab_values = [float(row["m_stab"]) for row in tasks if row["m_stab"] not in {None, ""}]
        axes: dict[str, float | str] = {
            "ID": 100.0 * sum(float(row["id_quality"]) for row in runs) / 150.0,
            "OOD": 100.0 * sum(float(row["ood_quality"]) for row in runs) / 150.0,
            "SYM": 100.0 * sum(float(row["m_sym"]) for row in runs) / 150.0,
            "MIN": 100.0 * sum(float(row["m_min"]) for row in runs) / 150.0,
            "EFF": 100.0 * sum(eff_values) / 150.0 if len(eff_values) == 150 else "",
            "STAB": 100.0 * sum(stab_values) / 50.0 if len(stab_values) == 50 else "",
        }
        assembly_complete = all(
            value != "" and math.isfinite(float(value)) for value in axes.values()
        )
        trajectory_ready = all(row["trajectory_formal_ready"] == "true" for row in runs)
        output.append(
            {
                "algorithm": runs[0]["algorithm"],
                "algorithm_slug": algorithm,
                "condition": condition,
                "run_count": len(runs),
                "task_count": len(tasks),
                "eff_available_run_count": len(eff_values),
                "eff_unavailable_run_count": 150 - len(eff_values),
                "stab_available_task_count": len(stab_values),
                "stab_unavailable_task_count": 50 - len(stab_values),
                **axes,
                "assembly_complete": str(assembly_complete).lower(),
                "formal_ready": str(assembly_complete and trajectory_ready).lower(),
                "trajectory_formal_ready": str(trajectory_ready).lower(),
                "trajectory_bases": ";".join(
                    sorted({str(row["trajectory_basis"]) for row in runs})
                ),
            }
        )
    if len(output) != 15:
        raise PublicationError(f"{condition} six-axis 应为15行")
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                    for row in rows:
                        text.write(canonical_json(row) + "\n")
    else:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
    gzip_output: bool = False,
) -> None:
    if not rows:
        raise PublicationError(f"拒绝写出空CSV: {path}")
    columns = list(fields or dict.fromkeys(field for row in rows for field in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _copy_checked(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise PublicationError(f"复制来源不是已物化文件: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != sha256_file(source):
        raise PublicationError(f"复制后 SHA 不一致: {destination}")


def _all_specs(config: PublicationConfig) -> list[tuple[str, ArtifactSpec]]:
    specs = [
        ("dataset_manifest", config.dataset_manifest),
        ("semantic_runs", config.semantic_runs),
        ("downstream_composition_manifest", config.downstream_composition_manifest),
        ("ground_truth.plan", config.gt_plan),
        ("ground_truth.index", config.gt_index),
    ]
    for condition, inputs in config.conditions.items():
        specs.extend(
            [
                (f"{condition}.raw_results", inputs.raw_results),
                (f"{condition}.numeric", inputs.numeric),
                (f"{condition}.prediction.plan", inputs.pred_plan),
                (f"{condition}.prediction.index", inputs.pred_index),
                (f"{condition}.equivalence.plan", inputs.equivalence_plan),
                (f"{condition}.equivalence.index", inputs.equivalence_index),
                (f"{condition}.structure.plan", inputs.structure_plan),
                (f"{condition}.structure.index", inputs.structure_index),
            ]
        )
        if inputs.trajectory_base is not None:
            specs.append((f"{condition}.trajectory.base", inputs.trajectory_base))
        specs.extend(
            (f"{condition}.equivalence.audit_override[{index}]", spec)
            for index, spec in enumerate(inputs.equivalence_overrides)
        )
        for index, (plan, frozen_index) in enumerate(inputs.structure_overlays):
            specs.append((f"{condition}.structure.audit_overlay[{index}].plan", plan))
            specs.append((f"{condition}.structure.audit_overlay[{index}].index", frozen_index))
        specs.extend(
            (f"{condition}.trajectory.override[{index}]", spec)
            for index, spec in enumerate(inputs.trajectory_overrides)
        )
    eff = config.eff_revision
    specs.extend(
        [
            ("eff_revision.manifest", eff.manifest),
            ("eff_revision.validation_report", eff.validation_report),
            ("eff_revision.checksums", eff.checksums),
        ]
    )
    for condition, condition_eff in eff.conditions.items():
        specs.extend(
            [
                (f"eff_revision.{condition}.run_status", condition_eff.run_status),
                (
                    f"eff_revision.{condition}.native_trajectories",
                    condition_eff.native_trajectories,
                ),
                (f"eff_revision.{condition}.minute_metrics", condition_eff.minute_metrics),
                (f"eff_revision.{condition}.algorithm_curves", condition_eff.algorithm_curves),
                (f"eff_revision.{condition}.algorithm_summary", condition_eff.algorithm_summary),
                (f"eff_revision.{condition}.unavailable", condition_eff.unavailable),
                (f"eff_revision.{condition}.manifest", condition_eff.manifest),
            ]
        )
    return specs


def check_config_inputs(config: PublicationConfig) -> dict[str, Any]:
    artifacts = []
    for label, spec in _all_specs(config):
        spec.verify_rows()
        artifacts.append({"label": label, **spec.evidence(config.repo_root)})
    return {
        "schema_version": "core50_final_release_input_preflight_v1",
        "status": "passed",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "output_root": str(config.output_root),
        "api_calls": 0,
        "training_runs": 0,
    }


def _composition_identity(plan: Mapping[str, Any], kind: str, condition: str) -> object:
    if kind == "equivalence":
        return _plan_run_identity(plan, condition)
    return _structure_identity(plan, condition)


def compose_downstream_indexes(
    *,
    final_status_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise PublicationError(f"composer输出已存在，拒绝覆盖: {output_root}")
    status = _read_json_object(final_status_path.resolve())
    if status.get("status") != "completed_with_frozen_clean_structure_exception":
        raise PublicationError("FINAL_OPUS_STATUS状态不符合最终合成合同")
    status_sha = sha256_file(final_status_path.resolve())
    outputs: dict[str, Any] = {}
    output_root.mkdir(parents=True)
    for name, section in sorted(status.get("downstream_indexes", {}).items()):
        condition, kind = name.rsplit("_", 1)
        if condition not in CONDITIONS or kind not in {"equivalence", "structure"}:
            raise PublicationError(f"FINAL_OPUS_STATUS downstream名称非法: {name}")

        paths = {
            "full_plan": Path(section["benchmark_full_plan_jsonl"]).resolve(),
            "active_plan": Path(section["active_refresh_plan_jsonl"]).resolve(),
            "reused": Path(section["reused_index_jsonl"]).resolve(),
            "new": Path(section["index_jsonl"]).resolve(),
            "non_applicable": Path(section["non_applicable_jsonl"]).resolve(),
        }
        expected_hashes = {
            "full_plan": section["benchmark_full_plan_sha256"],
            "active_plan": section["active_refresh_plan_sha256"],
            "reused": sha256_file(paths["reused"]),
            "new": section["index_sha256"],
            "non_applicable": sha256_file(paths["non_applicable"]),
        }
        for label, path in paths.items():
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hashes[label]:
                raise PublicationError(f"composer输入缺失或SHA漂移: {name}/{label}")

        full_plans = [dict(row) for row in ArtifactSpec(paths["full_plan"], expected_hashes["full_plan"], 2250, "jsonl").read_rows()]
        active_plans = [
            dict(row)
            for row in ArtifactSpec(
                paths["active_plan"],
                expected_hashes["active_plan"],
                int(section["active_refresh_plan_count"]),
                "jsonl",
            ).read_rows()
        ]
        full_by_identity = {
            _composition_identity(plan, kind, condition): plan for plan in full_plans
        }
        active_by_logical = {str(plan["logical_id"]): plan for plan in active_plans}
        active_by_identity = {
            _composition_identity(plan, kind, condition): plan for plan in active_plans
        }
        if len(full_by_identity) != 2250 or len(active_by_identity) != len(active_plans):
            raise PublicationError(f"composer plan identity重复: {name}")

        composed_plans = dict(full_by_identity)
        composed_plans.update(active_by_identity)
        composed_indexes: dict[object, dict[str, Any]] = {}

        def add_index(index: Mapping[str, Any], source_kind: str) -> None:
            logical_id = str(index.get("logical_id") or "")
            plan = active_by_logical.get(logical_id) if source_kind == "new" else None
            if plan is None:
                plan = next((item for item in full_plans if item["logical_id"] == logical_id), None)
            if plan is None:
                raise PublicationError(f"composer index没有对应plan: {name}/{logical_id}")
            identity = _composition_identity(plan, kind, condition)
            current_plan = composed_plans[identity]
            if current_plan["evaluation_key"] != index.get("evaluation_key"):
                raise PublicationError(f"composer index与current plan evaluation_key漂移: {name}/{logical_id}")
            if identity in composed_indexes:
                raise PublicationError(f"composer index identity重复: {name}/{identity}")
            output = dict(index)
            if source_kind == "non_applicable":
                output["state"] = "non_applicable"
                output["structured_output"] = None
                output["effective_expression"] = None
                output["non_applicable"] = {
                    "reason": output.get("reason")
                    or (output.get("evidence_payload") or {}).get("reason")
                    or "upstream_expression_or_seed_invalid",
                    "evidence_path": output.get("evidence_path"),
                    "evidence_sha256": output.get("evidence_sha256"),
                }
                output["network_request"] = False
            if source_kind == "new" and output.get("state") is None:
                result_path = Path(str(output.get("result_path") or ""))
                if not result_path.is_absolute():
                    result_path = _repo_root() / result_path
                if (
                    not result_path.is_file()
                    or sha256_file(result_path) != output.get("result_sha256")
                ):
                    raise PublicationError(f"composer new frozen响应缺失或SHA漂移: {logical_id}")
                container = _read_json_object(result_path)
                validation = container.get("validation") or {}
                structured = container.get("structured_output")
                if validation.get("ok") is not True or not isinstance(structured, Mapping):
                    raise PublicationError(f"composer new frozen响应未通过结构校验: {logical_id}")
                output["state"] = "frozen"
                output["structured_output"] = dict(structured)
                output["effective_expression"] = container.get("effective_expression")
                output["expression_resolution"] = container.get("expression_resolution")
            output["evidence_generation"] = (
                "new" if source_kind == "new" else "preexisting"
            )
            output["composition_source"] = source_kind
            output["source_plan_sha256"] = output.get("plan_sha256")
            output["plan_sha256"] = sha256_text(canonical_json(current_plan))
            output["composed_plan_logical_id"] = current_plan["logical_id"]
            composed_indexes[identity] = output

        for source_kind, path, count in (
            ("preexisting", paths["reused"], int(section["reused_count"])),
            ("new", paths["new"], int(section["row_count"])),
            ("non_applicable", paths["non_applicable"], int(section["non_applicable_count"])),
        ):
            for index in ArtifactSpec(path, expected_hashes[source_kind if source_kind != "preexisting" else "reused"], count, "jsonl").read_rows():
                add_index(index, source_kind)

        unresolved_count = 0
        if name == "clean_structure":
            missing = set(composed_plans) - set(composed_indexes)
            if len(missing) != 1:
                raise PublicationError(f"clean structure孤例数量不是1: {len(missing)}")
            identity = next(iter(missing))
            if identity[:4] != ("gplearn", "feynman-ii.6.11", 520, 521):
                # 数据集身份仍以当前full plan为准，逻辑编号必须是g0029。
                plan = composed_plans[identity]
                if "::g0029::s520-s521" not in str(plan["logical_id"]):
                    raise PublicationError(f"clean structure孤例不是gplearn g0029: {identity}")
            plan = composed_plans[identity]
            exception = status.get("frozen_exception") or {}
            attempts = []
            status_root = final_status_path.resolve().parent
            for task in exception.get("task_rows", []):
                evaluation_key = str(task["evaluation_key"])
                attempt_paths = sorted(
                    (status_root / "downstream_clean" / "attempts").glob(
                        f"{evaluation_key}.a*.json"
                    )
                )
                if len(attempt_paths) != 3:
                    raise PublicationError(f"clean structure孤例每版本attempt不是3: {evaluation_key}")
                for attempt_path in attempt_paths:
                    attempt = _read_json_object(attempt_path)
                    validation = attempt.get("validation") or {}
                    if validation.get("ok") is not False:
                        raise PublicationError("clean structure孤例attempt意外通过，拒绝伪造unresolved")
                    attempts.append(
                        {
                            "path": str(attempt_path),
                            "sha256": sha256_file(attempt_path),
                            "evaluation_key": evaluation_key,
                            "attempt_id": attempt.get("attempt_id"),
                            "error_class": validation.get("error_class"),
                            "error_message": validation.get("error_message"),
                        }
                    )
            if len(attempts) != 6:
                raise PublicationError("clean structure孤例总attempt证据不是6")
            composed_indexes[identity] = {
                "logical_id": plan["logical_id"],
                "evaluation_key": plan["evaluation_key"],
                "task_type": "stab_structure",
                "condition": "clean",
                "state": "unresolved",
                "structured_output": None,
                "effective_expression": None,
                "evidence_generation": "new",
                "composition_source": "frozen_exception_unresolved",
                "plan_sha256": sha256_text(canonical_json(plan)),
                "composed_plan_logical_id": plan["logical_id"],
                "unresolved": {
                    "reason": exception.get("reason"),
                    "final_opus_status_path": str(final_status_path.resolve()),
                    "final_opus_status_sha256": status_sha,
                    "attempts": attempts,
                },
            }
            unresolved_count = 1

        if set(composed_indexes) != set(composed_plans) or len(composed_indexes) != 2250:
            raise PublicationError(f"composer最终网格不完整: {name}")
        ordered_identities = sorted(composed_plans, key=str)
        plan_path = output_root / f"{name}_current_plan.jsonl"
        index_path = output_root / f"{name}_current_mixed_index.jsonl"
        _write_jsonl(plan_path, (composed_plans[key] for key in ordered_identities), gzip_output=False)
        _write_jsonl(index_path, (composed_indexes[key] for key in ordered_identities), gzip_output=False)
        outputs[name] = {
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path), "rows": 2250},
            "index": {"path": str(index_path), "sha256": sha256_file(index_path), "rows": 2250},
            "source_counts": dict(Counter(row["composition_source"] for row in composed_indexes.values())),
            "unresolved_count": unresolved_count,
        }
    manifest = {
        "schema_version": "core50_current_downstream_composition_v1",
        "status": "complete_with_one_explicit_unresolved",
        "final_opus_status": {
            "path": str(final_status_path.resolve()),
            "sha256": status_sha,
        },
        "outputs": outputs,
        "api_calls": 0,
        "state_mutations": 0,
    }
    _write_json(output_root / "compose_manifest.json", manifest)
    return manifest


def publish(config: PublicationConfig, *, config_path: Path) -> dict[str, Any]:
    if config.output_root.exists():
        raise PublicationError(f"输出目录已存在，拒绝覆盖: {config.output_root}")
    staging_root = config.output_root.with_name(f".{config.output_root.name}.staging")
    if staging_root.exists():
        raise PublicationError(f"发布暂存目录已存在，拒绝覆盖: {staging_root}")
    preflight = check_config_inputs(config)
    eff_archive_entries = validate_eff_revision_v3(config.eff_revision)
    semantic = load_semantic_runs(config.semantic_runs)
    gt_pairs = plan_index_pairs(config.gt_plan, config.gt_index, label="ground_truth")
    ground_truth, gt_evidence = bind_ground_truth(
        gt_pairs,
        dataset_manifest=config.dataset_manifest,
        index_spec=config.gt_index,
        config=config,
    )

    raw_by_condition: dict[str, dict[Key, dict[str, Any]]] = {}
    algorithms: set[str] = set()
    datasets: set[str] = set()
    for condition, inputs in config.conditions.items():
        raw = load_raw_results(inputs.raw_results, condition=condition)
        bind_raw_and_semantic(raw, semantic)
        raw_by_condition[condition] = raw
        algorithms.update(key[0] for key in raw)
        datasets.update(key[1] for key in raw)
    expected = {
        (algorithm, dataset, seed, condition)
        for algorithm in algorithms
        for dataset in datasets
        for seed in SEEDS
        for condition in CONDITIONS
    }
    all_raw_keys = set().union(*(set(rows) for rows in raw_by_condition.values()))
    if len(algorithms) != 15 or len(datasets) != 50 or all_raw_keys != expected:
        raise PublicationError(
            f"最终 raw 不是15x50x3x3网格: alg={len(algorithms)} data={len(datasets)} runs={len(all_raw_keys)}"
        )
    if set(ground_truth) != datasets or set(semantic) != expected:
        raise PublicationError("GT/semantic 与最终raw网格不一致")

    staging_root.mkdir(parents=True)
    _copy_checked(config_path.resolve(), staging_root / "provenance" / "publish_config.json")
    _copy_checked(
        config.downstream_composition_manifest.path,
        staging_root / "provenance" / "downstream_composition_manifest.json",
    )
    _write_json(staging_root / "provenance" / "input_preflight.json", preflight)
    eff_audit_root = staging_root / "provenance" / "eff_revision_v3"
    eff_source_root = config.eff_revision.checksums.path.parent.resolve()
    for source, _ in eff_archive_entries:
        _copy_checked(source, eff_audit_root / source.relative_to(eff_source_root))
    _copy_checked(config.eff_revision.checksums.path, eff_audit_root / "SHA256SUMS")
    _write_jsonl(
        staging_root / "ground_truth" / "opus5_ground_truth.jsonl",
        gt_evidence,
        gzip_output=False,
    )
    gt_rows = [
        {
            "dataset_id": dataset_id,
            "original_expression": row["original_expression"],
            "opus5_effective_expression": row["effective_expression"],
            "artifact_sha256": row["artifact_sha256"],
            "logical_id": row["logical_id"],
            "evaluation_key": row["evaluation_key"],
        }
        for dataset_id, row in sorted(ground_truth.items())
    ]
    _write_csv(staging_root / "ground_truth" / "current_references.csv", gt_rows)

    condition_reports: dict[str, Any] = {}
    all_complete = True
    total_formal_algorithms = 0
    for condition in CONDITIONS:
        inputs = config.conditions[condition]
        raw = raw_by_condition[condition]
        pred_pairs = plan_index_pairs(inputs.pred_plan, inputs.pred_index, label=f"{condition}.prediction")
        predictions, pred_evidence = bind_predictions(
            pred_pairs,
            condition=condition,
            raw=raw,
            semantic=semantic,
            index_spec=inputs.pred_index,
            config=config,
        )
        eq_pairs = plan_index_pairs(
            inputs.equivalence_plan,
            inputs.equivalence_index,
            label=f"{condition}.equivalence",
        )
        equivalence, eq_evidence = bind_equivalence(
            eq_pairs,
            condition=condition,
            predictions=predictions,
            ground_truth=ground_truth,
            index_spec=inputs.equivalence_index,
            override_specs=inputs.equivalence_overrides,
            config=config,
        )
        structure_pairs = plan_index_pairs(
            inputs.structure_plan,
            inputs.structure_index,
            label=f"{condition}.structure",
        )
        structure, structure_evidence = bind_structure(
            structure_pairs,
            condition=condition,
            predictions=predictions,
            index_spec=inputs.structure_index,
            overlay_specs=inputs.structure_overlays,
            config=config,
        )
        numeric = bind_numeric(
            inputs.numeric,
            condition=condition,
            raw=raw,
            semantic=semantic,
        )
        eff_status, trajectories, minute_metrics, curves, _ = load_eff_revision(
            config.eff_revision.conditions[condition],
            condition=condition,
            expected_keys=set(raw),
        )
        trajectory_sources = {
            key: config.eff_revision.conditions[condition].native_trajectories
            for key in trajectories
        }
        run_rows = build_run_rows(
            condition=condition,
            raw=raw,
            semantic=semantic,
            predictions=predictions,
            equivalence=equivalence,
            ground_truth=ground_truth,
            numeric=numeric,
            trajectories=trajectories,
            trajectory_sources=trajectory_sources,
            config=config,
            condition_inputs=inputs,
        )
        task_rows = build_task_stability(
            run_rows,
            structure,
            condition=condition,
            structure_plan_sha=inputs.structure_plan.sha256,
            structure_index_sha=inputs.structure_index.sha256,
        )
        six_axis = aggregate_six_axis(run_rows, task_rows, condition=condition)
        trajectory_rows = [trajectories[key] for key in sorted(trajectories)]
        target = staging_root / "results" / condition
        _write_jsonl(
            target / "raw_results.jsonl.gz",
            (raw[key]["record"] for key in sorted(raw)),
            gzip_output=True,
        )
        _write_csv(target / "run_final.csv", run_rows)
        _write_jsonl(target / "opus5_prediction.jsonl", pred_evidence, gzip_output=False)
        _write_jsonl(target / "opus5_equivalence.jsonl", eq_evidence, gzip_output=False)
        _write_jsonl(target / "opus5_structure.jsonl", structure_evidence, gzip_output=False)
        _write_csv(target / "six_axis.csv", six_axis)
        _write_csv(target / "task_stability.csv", task_rows)
        _write_csv(
            target / "run_eff_status.csv",
            [eff_status[key] for key in sorted(eff_status)],
        )
        _write_csv(
            target / "run_trajectories_180min.csv.gz",
            trajectory_rows,
            fields=trajectory_fields(),
            gzip_output=True,
        )
        _write_csv(target / "curves_2700.csv", curves)
        _write_csv(
            target / "id_ood_eff_minute.csv.gz",
            minute_metrics,
            gzip_output=True,
        )
        complete = all(row["assembly_complete"] == "true" for row in six_axis)
        all_complete = all_complete and complete
        ready_count = sum(row["formal_ready"] == "true" for row in six_axis)
        total_formal_algorithms += ready_count
        condition_reports[condition] = {
            "raw_results": len(raw),
            "run_final": len(run_rows),
            "prediction_evidence": len(pred_evidence),
            "equivalence_evidence": len(eq_evidence),
            "structure_evidence": len(structure_evidence),
            "task_stability": len(task_rows),
            "six_axis": len(six_axis),
            "trajectory_runs": len(trajectory_rows),
            "eff_status_runs": len(eff_status),
            "id_ood_eff_minute_rows": len(minute_metrics),
            "curves": len(curves),
            "assembly_complete": complete,
            "formal_ready_algorithm_count": ready_count,
            "trajectory_basis_counts": dict(
                Counter(row["trajectory_basis"] for row in trajectories.values())
            ),
            "trajectory_formal_ready_run_count": sum(
                row["formal_ready"] == "true" for row in trajectories.values()
            ),
            "eff_unavailable_run_count": sum(
                row.get("availability_status") == "unavailable"
                for row in trajectories.values()
            ),
            "structure_unresolved_task_count": sum(
                row["unresolved_structure"] == "true" for row in task_rows
            ),
        }

    source_manifest = {
        "schema_version": "core50_final_release_sources_v1",
        "status": "passed",
        "inputs": {label: spec.evidence(config.repo_root) for label, spec in _all_specs(config)},
        "publish_config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path.resolve()),
        },
        "forbidden_sources": [FORBIDDEN_SOURCE],
        "llm_transport_policy": {
            "new_evidence_required": config.new_evidence_transport,
            "accepted_preexisting": list(config.accepted_preexisting_transports),
            "rejudge_preexisting_by_transport": False,
        },
        "max_workers": config.max_workers,
        "api_calls": 0,
        "llm_calls": 0,
        "training_runs": 0,
    }
    _write_json(staging_root / "provenance" / "source_manifest.json", source_manifest)
    report = {
        "schema_version": "core50_final_release_publication_report_v1",
        "status": "passed" if all_complete else "passed_fail_closed",
        "current_only": True,
        "run_count": TOTAL_RUNS,
        "algorithm_count": len(algorithms),
        "dataset_count": len(datasets),
        "conditions": condition_reports,
        "assembly_complete": all_complete,
        "formal_ready_algorithm_condition_count": total_formal_algorithms,
        "formal_ready": total_formal_algorithms == 45,
        "limitations": build_publication_limitations(condition_reports),
        "api_calls": 0,
        "llm_calls": 0,
        "training_runs": 0,
    }
    _write_json(staging_root / "PUBLICATION_REPORT.json", report)
    files = sorted(
        path for path in staging_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (staging_root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(staging_root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    staging_root.replace(config.output_root)
    return report


def artifact_template(path: str, rows: int, format_name: str, **extra: Any) -> dict[str, Any]:
    return {"path": path, "sha256": "<64-lowercase-hex>", "rows": rows, "format": format_name, **extra}


def config_template() -> dict[str, Any]:
    conditions = {}
    for condition in CONDITIONS:
        clean = condition == "clean"
        conditions[condition] = {
            "raw_results": artifact_template(f"<merged-{condition}-raw.jsonl.gz>", 2250, "jsonl.gz"),
            "numeric": artifact_template(f"<merged-{condition}-numeric.csv>", 2250, "csv"),
            "prediction": {
                "plan": artifact_template(f"<all-{condition}-pred-plan.jsonl>", 2250, "jsonl"),
                "index": artifact_template(
                    f"<all-{condition}-pred-index.jsonl>", 2250, "jsonl", generation="mixed"
                ),
            },
            "equivalence": {
                "plan": artifact_template(f"<all-{condition}-eq-plan.jsonl>", 2250, "jsonl"),
                "index": artifact_template(
                    f"<all-{condition}-eq-index.jsonl>", 2250, "jsonl", generation="mixed"
                ),
                "audit_overrides": [
                    artifact_template("<clean-equivalence-audit-overrides.jsonl>", 28, "jsonl")
                ]
                if clean
                else [],
            },
            "structure": {
                "plan": artifact_template(f"<all-{condition}-structure-plan.jsonl>", 2250, "jsonl"),
                "index": artifact_template(
                    f"<all-{condition}-structure-index.jsonl>", 2250, "jsonl", generation="mixed"
                ),
                "audit_overlays": [
                    {
                        "plan": artifact_template(
                            "<clean-structure-overlay-plan.jsonl>", 14, "jsonl"
                        ),
                        "index": artifact_template(
                            "<clean-structure-overlay-index.jsonl>",
                            14,
                            "jsonl",
                            generation="preexisting",
                        ),
                    }
                ]
                if clean
                else [],
            },
            "trajectory": {"mode": "eff_revision_v3", "overrides": []},
        }
    return {
        "schema_version": "core50_final_release_publish_config_v1",
        "repo_root": "<repo-root>",
        "output_root": "AAAI_experiments/stage5_metric_calculation_0831/work/final_release_20260913/release_v2/publication_candidate",
        "max_workers": 4,
        "llm_transport_policy": {
            "new_evidence_required": "routify",
            "accepted_preexisting": ["routify", "claude_cli", "yapi"],
            "rejudge_preexisting_by_transport": False,
        },
        "mixed_index_row_contract": {
            "required_field": "evidence_generation",
            "allowed_values": ["new", "preexisting"],
            "note": "new 行强制 Routify；preexisting 行保留既有合法 CLI/Yapi/Routify 证据",
        },
        "artifacts": {
            "dataset_manifest": artifact_template("<core50-manifest.csv>", 50, "csv"),
            "semantic_runs": artifact_template("<semantic-runs.jsonl>", 6750, "jsonl"),
            "downstream_composition_manifest": artifact_template(
                "<publication-inputs-v3/compose_manifest.json>", 1, "json"
            ),
            "eff_revision": {
                "manifest": artifact_template("<eff-revision-v3/manifest.json>", 1, "json"),
                "validation_report": artifact_template(
                    "<eff-revision-v3/validation_report.json>", 1, "json"
                ),
                "checksums": artifact_template("<eff-revision-v3/SHA256SUMS>", 28, "text"),
                "conditions": {
                    condition: {
                        "run_status": artifact_template(
                            f"<eff-revision-v3/{condition}/run_status.csv>", 2250, "csv"
                        ),
                        "native_trajectories": artifact_template(
                            f"<eff-revision-v3/{condition}/native_trajectories.jsonl.gz>",
                            2250,
                            "jsonl.gz",
                        ),
                        "minute_metrics": artifact_template(
                            f"<eff-revision-v3/{condition}/id_ood_eff_minute.csv.gz>",
                            405000,
                            "csv.gz",
                        ),
                        "algorithm_curves": artifact_template(
                            f"<eff-revision-v3/{condition}/algorithm_180min.csv>",
                            2700,
                            "csv",
                        ),
                        "algorithm_summary": artifact_template(
                            f"<eff-revision-v3/{condition}/algorithm_summary.csv>", 15, "csv"
                        ),
                        "unavailable": artifact_template(
                            f"<eff-revision-v3/{condition}/unavailable.csv>", 0, "csv"
                        ),
                        "manifest": artifact_template(
                            f"<eff-revision-v3/{condition}/manifest.json>", 1, "json"
                        ),
                    }
                    for condition in CONDITIONS
                },
            },
            "ground_truth": {
                "plan": artifact_template("<all-gt-plan.jsonl>", 50, "jsonl"),
                "index": artifact_template("<all-gt-index.jsonl>", 50, "jsonl", generation="mixed"),
            },
            "conditions": conditions,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-config-template", action="store_true")
    parser.add_argument("--compose-downstream", action="store_true")
    parser.add_argument("--final-opus-status", type=Path)
    parser.add_argument("--compose-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.print_config_template:
            print(json.dumps(config_template(), ensure_ascii=False, indent=2))
            return 0
        if args.compose_downstream:
            if args.final_opus_status is None or args.compose_output is None:
                raise PublicationError(
                    "--compose-downstream需要--final-opus-status和--compose-output"
                )
            print(
                json.dumps(
                    compose_downstream_indexes(
                        final_status_path=args.final_opus_status,
                        output_root=args.compose_output,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.config is None:
            raise PublicationError("必须指定 --config，或使用 --print-config-template")
        config = load_config(args.config.resolve())
        if args.check_only:
            print(json.dumps(check_config_inputs(config), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(publish(config, config_path=args.config), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
