"""从 canonical artifact 统一重放分钟快照与最终结果的性能。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_build_canonical_artifact,
)
from scientific_intelligent_modelling.benchmarks.runner import (
    LoadedDataset,
    _predict_from_canonical_artifact,
    load_canonical_dataset,
)

from .metrics import phi_nmse


EVALUATION_PATH = "canonical_replay.v1"


class PerformanceReplayError(ValueError):
    """冻结公式无法按统一 canonical 路径重放。"""


class _CanonicalOutputInvalid(ValueError):
    """canonical 工件可执行，但其数值输出不能作为有效预测。"""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PARAMETER_REFERENCE_RE = re.compile(r"\bparams\s*\[|\bc\d+\b")
_FORCED_RAW_REBUILD_TOOLS = {
    "qlattice",
    "qlattice_wrapper",
    "drsr",
    "llmsr",
    "imcts",
    "imcts_wrapper",
}
_DIRECT_AST_REPLAY_TOOLS = {"pyoperon", "operon"}


@dataclass(frozen=True)
class FormulaRecoveryManifest:
    """已校验的公式参数恢复清单及其文件指纹。"""

    path: Path
    sha256: str
    condition: str
    entries: Mapping[str, Mapping[str, Any]]

    def params_for(
        self,
        *,
        task_id: str,
        condition: str,
        result_sha256: str,
        equation: str,
    ) -> list[float] | None:
        if condition != self.condition:
            raise PerformanceReplayError(
                f"公式恢复 manifest condition={self.condition!r}，请求为 {condition!r}"
            )
        entry = self.entries.get(task_id)
        if entry is None:
            return None
        if result_sha256 != entry["frozen_result_sha256"]:
            raise PerformanceReplayError(f"{task_id}: 公式恢复记录的 result SHA 漂移")
        equation_sha256 = hashlib.sha256(equation.encode("utf-8")).hexdigest()
        if equation_sha256 != entry["equation_sha256"]:
            raise PerformanceReplayError(f"{task_id}: 公式恢复记录的 equation SHA 漂移")
        if entry["resolution"] == "unavailable":
            raise PerformanceReplayError(
                f"{task_id}: 公式恢复 resolution=unavailable: {entry['reason']}"
            )
        # 清单加载器保证只有 recovered_params 分支含 params。
        return list(entry["params"])

    def require_params_for(
        self,
        *,
        task_id: str,
        condition: str,
        result_sha256: str,
        equation: str,
    ) -> list[float]:
        params = self.params_for(
            task_id=task_id,
            condition=condition,
            result_sha256=result_sha256,
            equation=equation,
        )
        if params is None:
            raise PerformanceReplayError(f"{task_id}: 未实例化参数公式缺少恢复记录")
        return params


@dataclass
class PerformanceReplayCache:
    """限制重复解析和预测，同时避免把预测数组长期留在内存里。"""

    datasets: dict[Path, LoadedDataset] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _validated_parameter_values(value: object, *, context: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise PerformanceReplayError(f"{context} 必须是数组")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise PerformanceReplayError(f"{context} 包含布尔值")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PerformanceReplayError(f"{context} 包含非数值") from exc
        if not math.isfinite(number):
            raise PerformanceReplayError(f"{context} 包含非有限值")
        normalized.append(number)
    return normalized


def load_formula_recovery_manifest(
    path: Path,
    *,
    expected_condition: str,
) -> FormulaRecoveryManifest:
    """严格加载恢复清单；真正使用参数时还会绑定冻结结果和公式 SHA。"""

    resolved = path.resolve()
    try:
        raw_bytes = resolved.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerformanceReplayError(f"无法读取公式恢复 manifest: {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PerformanceReplayError("公式恢复 manifest 顶层必须是 object")
    if payload.get("schema_version") != "formula_recovery.v1":
        raise PerformanceReplayError("公式恢复 manifest schema_version 不匹配")
    if payload.get("condition") != expected_condition:
        raise PerformanceReplayError(
            f"公式恢复 manifest condition 不是 {expected_condition!r}"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise PerformanceReplayError("公式恢复 manifest.entries 必须是数组")
    entries: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise PerformanceReplayError(f"公式恢复 entries[{index}] 必须是 object")
        entry = dict(raw_entry)
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise PerformanceReplayError(f"公式恢复 entries[{index}].task_id 非法")
        if f"_{expected_condition}_" not in task_id:
            raise PerformanceReplayError(f"{task_id}: task_id condition 与 manifest 不一致")
        if task_id in entries:
            raise PerformanceReplayError(f"公式恢复 manifest 存在重复 task_id: {task_id}")
        for field_name in ("frozen_result_sha256", "equation_sha256"):
            value = entry.get(field_name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise PerformanceReplayError(f"{task_id}: {field_name} 非法")
        resolution = entry.get("resolution")
        if resolution == "recovered_params":
            entry["params"] = _validated_parameter_values(
                entry.get("params"), context=f"{task_id}.params"
            )
            if entry["params"] is None:
                raise PerformanceReplayError(f"{task_id}: recovered_params 缺少 params")
        elif resolution == "unavailable":
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise PerformanceReplayError(f"{task_id}: unavailable 缺少 reason")
            if "params" in entry:
                raise PerformanceReplayError(f"{task_id}: unavailable 不得提供 params")
        else:
            raise PerformanceReplayError(f"{task_id}: resolution 非法: {resolution!r}")
        entries[task_id] = entry
    return FormulaRecoveryManifest(
        path=resolved,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        condition=expected_condition,
        entries=entries,
    )


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _finite_nonnegative(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _split_nmse(payload: Mapping[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        return None
    return _finite_nonnegative(block.get("nmse"))


def _candidate_local_paths(raw_path: object, *, repo_root: Path) -> list[Path]:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return []
    path = Path(str(raw_path).strip())
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(repo_root / path)

    parts = path.parts
    if "sim-datasets-data" in parts:
        index = parts.index("sim-datasets-data")
        candidates.append(repo_root.joinpath(*parts[index:]))
    return candidates


def resolve_local_dataset_dir(payload: Mapping[str, Any], *, repo_root: Path) -> Path:
    """把冻结结果中的远端数据路径确定性映射到当前仓库数据镜像。"""

    root = repo_root.resolve()
    candidates: list[Path] = []
    for field_name in ("expected_dataset_rel", "expected_dataset_dir", "dataset_dir"):
        candidates.extend(_candidate_local_paths(payload.get(field_name), repo_root=root))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir() and (resolved / "metadata.yaml").is_file():
            return resolved
    rendered = ", ".join(str(path) for path in seen) or "<无候选>"
    raise PerformanceReplayError(f"无法解析本地数据集目录，已检查: {rendered}")


def _corrected_artifact(
    payload: Mapping[str, Any],
    *,
    algorithm: str,
    expected_n_features: int,
    recovery_manifest: FormulaRecoveryManifest | None = None,
    task_id: str | None = None,
    condition: str | None = None,
    result_sha256: str | None = None,
) -> tuple[dict[str, Any], bool]:
    artifact_raw = payload.get("canonical_artifact")
    artifact = dict(artifact_raw) if isinstance(artifact_raw, Mapping) else None
    tool = str(algorithm).strip().lower()

    # 这四类历史工件存在变量平移、参数丢失或导出路径差异，统一从冻结原始公式
    # 重新走当前 normalizer，绝不复用旧 normalized/instantiated 表达式。
    if tool in _FORCED_RAW_REBUILD_TOOLS:
        raw_equation = payload.get("equation")
        if not isinstance(raw_equation, str) or not raw_equation.strip():
            raise PerformanceReplayError(f"{algorithm} 冻结 payload 缺少原始 equation")
        parameter_values = None
        if tool in {"drsr", "llmsr"}:
            if artifact is not None:
                parameter_values = _validated_parameter_values(
                    artifact.get("parameter_values"),
                    context=f"{algorithm}.canonical_artifact.parameter_values",
                )
            if _PARAMETER_REFERENCE_RE.search(raw_equation) and not parameter_values:
                if recovery_manifest is None or not task_id or not condition or not result_sha256:
                    raise PerformanceReplayError(
                        f"{task_id or algorithm}: 未实例化参数公式缺少绑定的恢复 manifest"
                    )
                parameter_values = recovery_manifest.require_params_for(
                    task_id=task_id,
                    condition=condition,
                    result_sha256=result_sha256,
                    equation=raw_equation,
                )
        rebuilt, error = safe_build_canonical_artifact(
            tool_name=algorithm,
            equation=raw_equation,
            expected_n_features=expected_n_features,
            parameter_values=parameter_values,
        )
        if rebuilt is None:
            raise PerformanceReplayError(f"{algorithm} canonical artifact 重建失败: {error}")
        return rebuilt, True

    if artifact is None:
        raise PerformanceReplayError("payload 缺少 canonical_artifact")
    executable_fields = (
        "instantiated_expression",
        "normalized_expression",
        "return_expression_source",
    )
    has_expression = any(
        isinstance(artifact.get(field), str) and str(artifact[field]).strip()
        for field in executable_fields
    )
    has_gplearn_prefix = (
        str(artifact.get("tool_name") or "").strip().lower() == "gplearn"
        and isinstance(artifact.get("raw_equation"), str)
        and bool(str(artifact["raw_equation"]).strip())
    )
    if not has_expression and not has_gplearn_prefix:
        raise PerformanceReplayError("canonical_artifact 缺少可执行表达式")

    # PyOperon 的冻结树已经是受限 Python 数值表达式。若再次交给 SymPy
    # ``sympify``，某些嵌套浮点幂/双曲函数会在自动求值阶段发生组合爆炸；这既
    # 不改变算法候选，也不应成为性能评估的一部分。显式补入同一 canonical
    # 执行器支持的 AST 提示，保留原始结合顺序，并继续由 runner 的 AST 白名单
    # 做安全校验。这里只修改当前 replay 的副本，不篡改冻结工件。
    if tool in _DIRECT_AST_REPLAY_TOOLS and not artifact.get("executable_expression"):
        for field_name in executable_fields:
            expression = artifact.get(field_name)
            if isinstance(expression, str) and expression.strip():
                artifact["executable_expression"] = expression.strip()
                break
    return artifact, False


def _evaluate_split(artifact: dict[str, Any], split: Any, *, split_name: str) -> dict[str, float]:
    if split is None or int(split.rows) <= 0:
        raise PerformanceReplayError(f"数据集缺少 {split_name} split")
    try:
        with np.errstate(all="ignore"):
            prediction = np.asarray(
                _predict_from_canonical_artifact(artifact, split.X),
                dtype=float,
            ).reshape(-1)
    except Exception as exc:
        raise _CanonicalOutputInvalid(
            f"{split_name} canonical prediction 失败: {exc!r}"
        ) from exc
    if prediction.shape != np.asarray(split.y).reshape(-1).shape:
        raise _CanonicalOutputInvalid(
            f"{split_name} prediction shape 不一致: {prediction.shape} != {np.asarray(split.y).shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise _CanonicalOutputInvalid(f"{split_name} canonical prediction 含 NaN/Inf")
    try:
        metrics = regression_metrics(split.y, prediction, acc_threshold=0.1)
    except Exception as exc:
        raise _CanonicalOutputInvalid(f"{split_name} 数值指标计算失败: {exc!r}") from exc
    normalized: dict[str, float] = {}
    for output_name, source_name in (
        ("rmse", "rmse"),
        ("r2", "r2"),
        ("nmse", "nmse"),
        ("acc_0_1", "acc_tau"),
    ):
        value = metrics.get(source_name)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _CanonicalOutputInvalid(f"{split_name}.{source_name} 不可用") from exc
        if not math.isfinite(number) or (source_name == "nmse" and number < 0.0):
            raise _CanonicalOutputInvalid(f"{split_name}.{source_name} 非法: {value!r}")
        normalized[output_name] = number
    return normalized


def replay_payload_performance(
    payload: Mapping[str, Any],
    *,
    algorithm: str,
    repo_root: Path,
    cache: PerformanceReplayCache | None = None,
    recovery_manifest: FormulaRecoveryManifest | None = None,
    task_id: str | None = None,
    condition: str | None = None,
    result_sha256: str | None = None,
) -> dict[str, Any]:
    """用同一执行器重放任意 periodic/final payload 的 ID/OOD 性能。"""

    replay_cache = cache if cache is not None else PerformanceReplayCache()
    dataset_dir = resolve_local_dataset_dir(payload, repo_root=repo_root)
    dataset = replay_cache.datasets.get(dataset_dir)
    if dataset is None:
        try:
            dataset = load_canonical_dataset(dataset_dir)
        except Exception as exc:
            raise PerformanceReplayError(f"加载 canonical 数据集失败: {exc!r}") from exc
        replay_cache.datasets[dataset_dir] = dataset
    artifact, rebuilt = _corrected_artifact(
        payload,
        algorithm=algorithm,
        expected_n_features=len(dataset.feature_names),
        recovery_manifest=recovery_manifest,
        task_id=task_id,
        condition=condition,
        result_sha256=result_sha256,
    )
    artifact_sha256 = _sha256_json(artifact)
    cache_key = hashlib.sha256(
        f"{EVALUATION_PATH}|{dataset_dir}|{artifact_sha256}".encode("utf-8")
    ).hexdigest()
    evaluated = replay_cache.results.get(cache_key)
    if evaluated is None:
        try:
            evaluated = {
                "id_test": _evaluate_split(artifact, dataset.id_test, split_name="id_test"),
                "ood_test": _evaluate_split(artifact, dataset.ood_test, split_name="ood_test"),
                "invalid_reason": None,
            }
        except _CanonicalOutputInvalid as exc:
            evaluated = {
                "id_test": None,
                "ood_test": None,
                "invalid_reason": str(exc),
            }
        replay_cache.results[cache_key] = evaluated

    invalid_reason = evaluated.get("invalid_reason")
    if invalid_reason:
        id_metrics = None
        ood_metrics = None
        id_quality = 0.0
        ood_quality = 0.0
        valid_output = False
    else:
        id_metrics = dict(evaluated["id_test"])
        ood_metrics = dict(evaluated["ood_test"])
        id_quality = phi_nmse(id_metrics["nmse"])
        ood_quality = phi_nmse(ood_metrics["nmse"])
        valid_output = True
    return {
        "evaluation_path": EVALUATION_PATH,
        "dataset_dir": str(dataset_dir),
        "canonical_artifact": artifact,
        "canonical_artifact_sha256": artifact_sha256,
        "artifact_rebuilt": rebuilt,
        "id_test": id_metrics,
        "ood_test": ood_metrics,
        "id_quality": id_quality,
        "ood_quality": ood_quality,
        "valid_output": valid_output,
        "invalid_reason": invalid_reason,
        "native_id_nmse": _split_nmse(payload, "id_test"),
        "native_ood_nmse": _split_nmse(payload, "ood_test"),
        "error": None,
    }
