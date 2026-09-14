#!/usr/bin/env python3
"""用三个独立模型盲审 Core50 中 Opus5 已化简的表达式。

该脚本只从 profile 文件加载凭据。输出仅保留脱敏后的请求元数据、结构化裁决和
token usage，绝不落盘请求头、鉴权 token 或原始 profile 内容。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = REPO_ROOT / "AAAI_experiments/Core50_final_20260914"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/audits"
    / "three_model_simplification_review_100_20260914"
)
DEFAULT_OUTPUT_ROOT_V2 = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/audits"
    / "three_model_simplification_review_100_common_domain_approx_20260914"
)
DEFAULT_ANTHROPIC_PROFILE = Path.home() / ".claude/settings-jyh.json"
DEFAULT_OPENAI_PROFILE = Path.home() / ".codex/routify-openai.toml"

ANTHROPIC_BASE_URL = "https://routify-pub.alibaba-inc.com/protocol/anthropic"
OPENAI_BASE_URL = "https://routify-pub.alibaba-inc.com/protocol/openai/v1"
CONDITIONS = ("clean", "noise001", "noise005")
SAMPLE_SIZE = 100
SAMPLE_SEED = 20260914
CONCURRENCY_PER_MODEL = 32
MAX_ATTEMPTS_PER_TASK = 2
REVIEW_POLICIES = ("strict", "common_domain_approx")
DEFAULT_REVIEW_POLICY = "common_domain_approx"
PROMPT_VERSIONS = {
    "strict": "three_model_simplification_review.strict.v1",
    "common_domain_approx": "three_model_simplification_review.common_domain_approx.v2",
}
PROMPT_VERSION = PROMPT_VERSIONS["strict"]
COMMON_DOMAIN_ABS_TOL = 1e-9
COMMON_DOMAIN_REL_TOL = 1e-6


class ReviewError(ValueError):
    """评审输入、配置或模型输出不满足冻结合同。"""


class RequestCapExceeded(ReviewError):
    """物理请求总上限已经耗尽。"""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    protocol: str


MODELS: dict[str, ModelSpec] = {
    "kimi-k3": ModelSpec("kimi-k3", "anthropic"),
    "glm-5.2": ModelSpec("glm-5.2", "anthropic"),
    "gpt-5.6-sol": ModelSpec("gpt-5.6-sol", "openai_responses"),
}


@dataclass(frozen=True)
class ApiProfile:
    protocol: str
    base_url: str
    headers: Mapping[str, str]
    safe_metadata: Mapping[str, object]


@dataclass(frozen=True)
class HttpRequest:
    url: str
    headers: Mapping[str, str]
    body: str
    safe_metadata: Mapping[str, object]


@dataclass(frozen=True)
class HttpResponse:
    status: int
    payload: Mapping[str, Any]


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "equivalence",
        "domain_preserved",
        "variable_mapping_preserved",
        "operator_semantics_preserved",
        "simplicity",
        "verdict",
        "confidence",
        "brief_reason",
        "counterexample",
    ],
    "properties": {
        "equivalence": {"type": "string", "enum": ["pass", "fail", "undetermined"]},
        "domain_preserved": {"type": "boolean"},
        "variable_mapping_preserved": {"type": "boolean"},
        "operator_semantics_preserved": {"type": "boolean"},
        "simplicity": {"type": "string", "enum": ["simpler", "same", "more_complex"]},
        "verdict": {"type": "string", "enum": ["pass", "fail", "undetermined"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "brief_reason": {"type": "string"},
        "counterexample": {"type": ["string", "null"]},
    },
}
REVIEW_FIELDS = tuple(REVIEW_SCHEMA["required"])


STRICT_SYSTEM_PROMPT = """You are an independent mathematical audit reviewer. Compare the original
expression with the proposed simplification under real-valued semantics. Check exact mathematical
equivalence, domain preservation, fixed variable-to-input-column identity, and operator semantics.
Do not infer that a transformation is correct merely because it looks simpler. A concrete domain,
variable, or numerical counterexample requires a fail verdict. If protected-operator semantics are
material but unavailable, return undetermined rather than guessing. Judge simplicity separately.
Return exactly one JSON object matching the supplied schema, with no Markdown or extra fields."""

COMMON_DOMAIN_APPROX_SYSTEM_PROMPT = """You are an independent mathematical audit reviewer. Compare
the original expression with the proposed simplification only on their common real-valued domain.
A domain expansion or contraction alone is not a failure. Treat fixed decimal and fitted constants
as numerically equivalent when their induced values agree within abs_tol=1e-9 and rel_tol=1e-6;
do not refit constants. Variable-to-input-column identity and operator value semantics must remain
unchanged. For protected operators, ignore a difference that only changes the domain, but fail if
the expressions change values anywhere in their common real-valued domain. A counterexample in the
common domain, including sqrt(x**2) -> x at x<0, requires fail. A passing simplification must also be
same complexity or simpler. Return undetermined rather than guessing. Return exactly one JSON object
matching the supplied schema, with no Markdown or extra fields."""

# 保留旧名称，避免导入方读取严格策略提示词时失效。
SYSTEM_PROMPT = STRICT_SYSTEM_PROMPT


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_review_policy(review_policy: str) -> str:
    if review_policy not in REVIEW_POLICIES:
        raise ReviewError(f"review_policy 必须是 {REVIEW_POLICIES} 之一")
    return review_policy


def prompt_version_for_policy(review_policy: str) -> str:
    return PROMPT_VERSIONS[_validate_review_policy(review_policy)]


def system_prompt_for_policy(review_policy: str) -> str:
    policy = _validate_review_policy(review_policy)
    return (
        STRICT_SYSTEM_PROMPT
        if policy == "strict"
        else COMMON_DOMAIN_APPROX_SYSTEM_PROMPT
    )


def apply_review_policy(
    sample: Mapping[str, Any], review_policy: str
) -> dict[str, Any]:
    """给冻结样本绑定策略专属 key，阻止跨策略复用历史裁决。"""

    policy = _validate_review_policy(review_policy)
    row = dict(sample)
    existing_key = row.get("base_review_key") or row.get("review_key")
    if not isinstance(existing_key, str) or not existing_key:
        raise ReviewError("样本缺少 review_key")
    prompt_version = prompt_version_for_policy(policy)
    row["base_review_key"] = existing_key
    row["review_policy"] = policy
    row["prompt_version"] = prompt_version
    row["review_key"] = _sha256_bytes(
        canonical_json(
            {
                "base_review_key": existing_key,
                "review_policy": policy,
                "prompt_version": prompt_version,
            }
        ).encode("utf-8")
    )
    return row


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise ReviewError(f"缺少发布输入: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReviewError(f"{path}:{line_number} 不是合法 JSON") from error
            if not isinstance(value, dict):
                raise ReviewError(f"{path}:{line_number} 必须是 JSON object")
            yield line_number, value


def _prediction_identity(logical_id: str, condition: str) -> tuple[str, str, str]:
    parts = logical_id.split("::")
    if len(parts) < 5 or parts[0] != "pred_simplify":
        raise ReviewError(f"无法解析 prediction logical_id: {logical_id}")
    algorithm, dataset_id, seed = parts[1], parts[2], parts[3]
    if parts[4] not in CONDITIONS and condition not in CONDITIONS:
        raise ReviewError(f"logical_id condition 非法: {logical_id}")
    return algorithm, dataset_id, seed


def _ground_truth_identity(logical_id: str) -> str:
    parts = logical_id.split("::")
    if len(parts) < 2 or parts[0] != "gt_simplify":
        raise ReviewError(f"无法解析 ground-truth logical_id: {logical_id}")
    return parts[1]


def _risk_features(original: str, effective: str) -> tuple[int, list[str]]:
    joined = f"{original}\n{effective}"
    flags: list[str] = []
    patterns = {
        "domain_sensitive": r"\b(?:log|sqrt|div|asin|acos|atanh)\s*\(",
        "absolute_or_piecewise": r"\b(?:abs|Abs|Piecewise|where)\s*\(",
        "power": r"\*\*|\^",
        "multiple_variables": r"\bx\d+\b.*\bx\d+\b",
        "large_rewrite": r".{500,}",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, joined, flags=re.DOTALL):
            flags.append(name)
    relative_change = abs(len(original) - len(effective)) / max(1, len(original))
    if relative_change >= 0.35:
        flags.append("large_length_change")
    return len(flags), flags


def build_review_universe(release_root: str | Path) -> list[dict[str, Any]]:
    """构造可送审全集；review key 保留任务身份，不做跨任务裸表达式去重。"""

    root = Path(release_root).expanduser().resolve()
    sources: list[tuple[str, str, Path]] = [
        ("ground_truth", "clean", root / "ground_truth/opus5_ground_truth.jsonl")
    ]
    sources.extend(
        ("prediction", condition, root / f"results/{condition}/opus5_prediction.jsonl")
        for condition in CONDITIONS
    )
    tasks: dict[str, dict[str, Any]] = {}
    for source_kind, condition, path in sources:
        source_sha256 = _sha256_file(path) if path.is_file() else ""
        relative_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        for line_number, row in _read_jsonl(path):
            structured = row.get("structured_output")
            if not isinstance(structured, Mapping):
                continue
            outcome = structured.get("outcome")
            if outcome not in {"simplified", "unchanged"}:
                continue
            original = row.get("input_expression")
            effective = row.get("effective_expression") or structured.get("simplified_expression")
            if not isinstance(original, str) or not original.strip():
                raise ReviewError(f"{path}:{line_number} 缺少 input_expression")
            if not isinstance(effective, str) or not effective.strip():
                raise ReviewError(f"{path}:{line_number} 缺少 effective_expression")
            logical_id = str(row.get("logical_id") or "")
            if source_kind == "prediction":
                algorithm, dataset_id, seed = _prediction_identity(logical_id, condition)
            else:
                algorithm, dataset_id, seed = "ground_truth", _ground_truth_identity(logical_id), ""
            pair_sha256 = _sha256_bytes(canonical_json([original, effective]).encode("utf-8"))
            evaluation_key = str(row.get("evaluation_key") or "")
            review_key = _sha256_bytes(
                canonical_json(
                    {
                        "source_kind": source_kind,
                        "condition": condition,
                        "logical_id": logical_id,
                        "evaluation_key": evaluation_key,
                        "task_type": str(row.get("task_type") or ""),
                        "original_expression": original,
                        "effective_expression": effective,
                    }
                ).encode("utf-8")
            )
            occurrence = {
                "source_kind": source_kind,
                "task_type": str(row.get("task_type") or ""),
                "algorithm": algorithm,
                "condition": condition,
                "dataset_id": dataset_id,
                "seed": seed,
                "logical_id": logical_id,
                "evaluation_key": evaluation_key,
                "source_file": relative_path,
                "source_file_sha256": source_sha256,
                "source_line": line_number,
                "source_record_sha256": _sha256_bytes(canonical_json(row).encode("utf-8")),
            }
            risk_score, risk_flags = _risk_features(original, effective)
            item = tasks.setdefault(
                review_key,
                {
                    "review_key": review_key,
                    "pair_sha256": pair_sha256,
                    "original_expression": original,
                    "effective_expression": effective,
                    "outcome": outcome,
                    "risk_score": risk_score,
                    "risk_flags": risk_flags,
                    "occurrences": [],
                },
            )
            item["occurrences"].append(occurrence)
    universe = sorted(tasks.values(), key=lambda item: item["review_key"])
    for item in universe:
        item["occurrences"].sort(
            key=lambda row: (
                row["source_kind"], row["condition"], row["algorithm"], row["logical_id"]
            )
        )
    if not universe:
        raise ReviewError("没有找到 simplified/unchanged 表达式")
    return universe


def count_release_outcomes(release_root: str | Path) -> dict[str, int]:
    root = Path(release_root).expanduser().resolve()
    paths = [root / "ground_truth/opus5_ground_truth.jsonl"] + [
        root / f"results/{condition}/opus5_prediction.jsonl" for condition in CONDITIONS
    ]
    counts: Counter[str] = Counter()
    for path in paths:
        for _, row in _read_jsonl(path):
            structured = row.get("structured_output")
            outcome = structured.get("outcome") if isinstance(structured, Mapping) else "missing"
            counts[str(outcome or "missing")] += 1
    return dict(sorted(counts.items()))


def _strata(item: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(occurrence["condition"]), str(occurrence["algorithm"]))
        for occurrence in item["occurrences"]
    }


def _stratified_take(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[tuple[str, tuple[str, str]]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    index = {str(item["review_key"]): item for item in candidates}
    for review_key, item in index.items():
        for stratum in _strata(item):
            buckets[stratum].append(review_key)
    if count > len(index):
        raise ReviewError(f"分层配额 {count} 超过候选数 {len(index)}")
    for stratum, review_keys in buckets.items():
        review_keys.sort(
            key=lambda review_key: (
                -int(index[review_key].get("risk_score", 0)),
                _sha256_bytes(f"{seed}|{stratum[0]}|{stratum[1]}|{review_key}".encode("utf-8")),
            )
        )
    cursors = {stratum: 0 for stratum in buckets}
    selected: list[tuple[str, tuple[str, str]]] = []
    selected_keys: set[str] = set()
    while len(selected) < count:
        progressed = False
        for stratum in sorted(buckets):
            bucket = buckets[stratum]
            cursor = cursors[stratum]
            while cursor < len(bucket) and bucket[cursor] in selected_keys:
                cursor += 1
            cursors[stratum] = cursor
            if cursor >= len(bucket):
                continue
            review_key = bucket[cursor]
            cursors[stratum] += 1
            selected_keys.add(review_key)
            selected.append((review_key, stratum))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ReviewError("分层抽样提前耗尽")
    return selected


def stratified_sample(
    universe: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> list[dict[str, Any]]:
    """按 70% changed prediction、20% unchanged、10% GT 分层风险抽样。"""

    if sample_size <= 0:
        raise ReviewError("sample_size 必须是正整数")
    unique = {str(item["review_key"]): dict(item) for item in universe}
    if sample_size > len(unique):
        raise ReviewError(f"sample_size={sample_size} 超过送审任务数 {len(unique)}")
    gt_total = max(2, round(sample_size * 0.10))
    gt_changed = gt_total // 2
    gt_unchanged = gt_total - gt_changed
    pred_unchanged = round(sample_size * 0.20)
    pred_changed = sample_size - gt_total - pred_unchanged
    groups = [
        ("prediction_simplified", [item for item in unique.values() if item["outcome"] == "simplified" and any(o["source_kind"] == "prediction" for o in item["occurrences"])], pred_changed),
        ("prediction_unchanged", [item for item in unique.values() if item["outcome"] == "unchanged" and any(o["source_kind"] == "prediction" for o in item["occurrences"])], pred_unchanged),
        ("ground_truth_simplified", [item for item in unique.values() if item["outcome"] == "simplified" and any(o["source_kind"] == "ground_truth" for o in item["occurrences"])], gt_changed),
        ("ground_truth_unchanged", [item for item in unique.values() if item["outcome"] == "unchanged" and any(o["source_kind"] == "ground_truth" for o in item["occurrences"])], gt_unchanged),
    ]
    selected: list[tuple[str, tuple[str, str], str]] = []
    for group_name, candidates, count in groups:
        selected.extend(
            (review_key, stratum, group_name)
            for review_key, stratum in _stratified_take(candidates, count=count, seed=seed)
        )
    output: list[dict[str, Any]] = []
    for index, (review_key, stratum, group_name) in enumerate(selected, start=1):
        row = dict(unique[review_key])
        row["sample_id"] = f"sample_{index:04d}"
        row["selection_stratum"] = {"condition": stratum[0], "algorithm": stratum[1]}
        row["selection_group"] = group_name
        row["sample_seed"] = seed
        output.append(row)
    return output


def logical_task_count(sample_size: int) -> int:
    if sample_size <= 0:
        raise ReviewError("sample_size 必须是正整数")
    return sample_size * len(MODELS)


def physical_request_cap(sample_size: int) -> int:
    return math.ceil(1.5 * logical_task_count(sample_size))


def _safe_endpoint_metadata(base_url: str, protocol: str) -> dict[str, object]:
    parsed = urlparse(base_url)
    return {
        "protocol": protocol,
        "api_host": parsed.hostname,
        "api_path": parsed.path.rstrip("/"),
        "credential_loaded": True,
    }


def _validate_base_url(base_url: str, expected: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized != expected:
        raise ReviewError(f"profile endpoint 必须是 {expected}")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        raise ReviewError("profile endpoint 必须使用无用户信息的标准 HTTPS 地址")
    return normalized


def load_anthropic_profile(path: str | Path) -> ApiProfile:
    profile_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"无法读取 Anthropic profile: {profile_path}") from error
    env = value.get("env") if isinstance(value, Mapping) else None
    if not isinstance(env, Mapping):
        raise ReviewError("Anthropic profile 缺少 env")
    base_url = _validate_base_url(str(env.get("ANTHROPIC_BASE_URL") or ""), ANTHROPIC_BASE_URL)
    token = env.get("ANTHROPIC_AUTH_TOKEN")
    if not isinstance(token, str) or not token:
        raise ReviewError("Anthropic profile 缺少 ANTHROPIC_AUTH_TOKEN")
    return ApiProfile(
        protocol="anthropic",
        base_url=base_url,
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": token,
        },
        safe_metadata=_safe_endpoint_metadata(base_url, "anthropic_messages"),
    )


def load_openai_profile(path: str | Path, *, provider: str = "custom") -> ApiProfile:
    profile_path = Path(path).expanduser().resolve()
    try:
        value = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReviewError(f"无法读取 OpenAI profile: {profile_path}") from error
    providers = value.get("model_providers")
    config = providers.get(provider) if isinstance(providers, Mapping) else None
    if not isinstance(config, Mapping):
        raise ReviewError(f"OpenAI profile 缺少 model_providers.{provider}")
    if config.get("wire_api") != "responses":
        raise ReviewError("OpenAI profile 的 wire_api 必须为 responses")
    base_url = _validate_base_url(str(config.get("base_url") or ""), OPENAI_BASE_URL)
    configured_headers = config.get("http_headers")
    authorization = configured_headers.get("Authorization") if isinstance(configured_headers, Mapping) else None
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise ReviewError("OpenAI profile 缺少 Bearer Authorization")
    return ApiProfile(
        protocol="openai_responses",
        base_url=base_url,
        headers={"content-type": "application/json", "Authorization": authorization},
        safe_metadata=_safe_endpoint_metadata(base_url, "openai_responses"),
    )


def derive_openai_profile_from_anthropic(anthropic: ApiProfile) -> ApiProfile:
    """复用同一 Routify 凭据，但固定切换到 OpenAI Responses 端口。"""

    token = anthropic.headers.get("x-api-key")
    if not isinstance(token, str) or not token:
        raise ReviewError("Anthropic profile 未提供可复用的 Routify 凭据")
    return ApiProfile(
        protocol="openai_responses",
        base_url=OPENAI_BASE_URL,
        headers={"content-type": "application/json", "Authorization": f"Bearer {token}"},
        safe_metadata=_safe_endpoint_metadata(OPENAI_BASE_URL, "openai_responses"),
    )


def load_review_profiles(args: argparse.Namespace) -> dict[str, ApiProfile]:
    anthropic = load_anthropic_profile(args.anthropic_profile)
    openai = (
        derive_openai_profile_from_anthropic(anthropic)
        if args.reuse_anthropic_token_for_openai
        else load_openai_profile(args.openai_profile, provider=args.openai_provider)
    )
    return {"anthropic": anthropic, "openai_responses": openai}


def render_review_prompt(
    sample: Mapping[str, Any], *, review_policy: str = "strict"
) -> str:
    policy = _validate_review_policy(review_policy)
    occurrence = sample["occurrences"][0] if sample.get("occurrences") else {}
    algorithm = str(occurrence.get("algorithm") or "ground_truth")
    operator_semantics = (
        "gplearn native protected semantics: div(a,b)=a/b when abs(b)>0.001 else 1; "
        "sqrt(a)=sqrt(abs(a)); log(a)=log(abs(a)) when abs(a)>0.001 else 0."
        if algorithm == "gplearn"
        else "No additional algorithm-specific protected-operator contract is asserted; return "
        "undetermined if the written expressions do not preserve required semantics."
    )
    if policy == "strict":
        number_semantics = "Decimal literals are fixed fitted constants; do not refit them."
        domain_semantics = "Ordinary real semantics; exact domain preservation is required."
        pass_contract = "Require exact equivalence and preservation of the real-valued domain."
    else:
        number_semantics = (
            "Decimal literals are fixed fitted constants; do not refit them. Treat their induced "
            "values as matching when abs(a-b) <= abs_tol + rel_tol*max(abs(a),abs(b)), with "
            "abs_tol=1e-9 and rel_tol=1e-6."
        )
        domain_semantics = (
            "Compare values only on the common real-valued domain. A domain expansion or "
            "contraction alone is not a failure; still report domain_preserved accurately."
        )
        pass_contract = (
            "Pass only when common-domain values agree within tolerance, variable mapping and "
            "operator value semantics are preserved, and simplicity is same or simpler. For a "
            "protected operator, ignore domain-only differences but fail value differences on "
            "the common domain."
        )
    payload = {
        "audit_contract": {
            "review_policy": policy,
            "prompt_version": prompt_version_for_policy(policy),
            "number_semantics": number_semantics,
            "variable_semantics": "Variable names denote fixed input columns and may not be renamed or permuted.",
            "domain": domain_semantics,
            "pass_contract": pass_contract,
            "uncertainty": "Return undetermined when available evidence cannot establish the required claim.",
            "operator_semantics": operator_semantics,
        },
        "task_context": {
            "source_kind": occurrence.get("source_kind"),
            "task_type": occurrence.get("task_type"),
            "condition": occurrence.get("condition"),
            "algorithm": algorithm,
            "dataset_id": occurrence.get("dataset_id"),
            "seed": occurrence.get("seed"),
            "logical_id": occurrence.get("logical_id"),
            "evaluation_key": occurrence.get("evaluation_key"),
            "review_key": sample.get("review_key"),
        },
        "original_expression": sample["original_expression"],
        "proposed_simplification": sample["effective_expression"],
        "output_schema": REVIEW_SCHEMA,
    }
    return system_prompt_for_policy(policy) + "\n\nAUDIT INPUT:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def build_http_request(
    model: ModelSpec,
    profile: ApiProfile,
    prompt: str,
    *,
    review_policy: str = "strict",
) -> HttpRequest:
    system_prompt = system_prompt_for_policy(review_policy)
    if model.protocol != profile.protocol:
        raise ReviewError(f"模型 {model.name} 与 profile 协议不匹配")
    if model.protocol == "anthropic":
        url = profile.base_url + "/v1/messages"
        payload = {
            "model": model.name,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1200,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif model.protocol == "openai_responses":
        url = profile.base_url + "/responses"
        payload = {
            "model": model.name,
            "stream": False,
            "store": False,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "simplification_review",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                }
            },
        }
    else:
        raise ReviewError(f"未知 API 协议: {model.protocol}")
    safe = dict(profile.safe_metadata)
    safe.update(
        {
            "model": model.name,
            "stream": False,
            "review_policy": review_policy,
            "prompt_version": prompt_version_for_policy(review_policy),
        }
    )
    if model.protocol == "openai_responses":
        safe["requested_store"] = False
    return HttpRequest(
        url=url,
        headers=dict(profile.headers),
        body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        safe_metadata=safe,
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def parse_review_json(
    text: str, *, review_policy: str = "strict"
) -> dict[str, Any]:
    policy = _validate_review_policy(review_policy)
    try:
        value = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as error:
        raise ReviewError("模型输出不是单个 JSON object") from error
    if not isinstance(value, dict):
        raise ReviewError("模型输出必须是 JSON object")
    if set(value) != set(REVIEW_FIELDS):
        raise ReviewError("模型输出字段与严格 schema 不一致")
    for field in ("domain_preserved", "variable_mapping_preserved", "operator_semantics_preserved"):
        if type(value[field]) is not bool:
            raise ReviewError(f"{field} 必须为 boolean")
    if value["equivalence"] not in {"pass", "fail", "undetermined"}:
        raise ReviewError("equivalence 枚举非法")
    if value["simplicity"] not in {"simpler", "same", "more_complex"}:
        raise ReviewError("simplicity 枚举非法")
    if value["verdict"] not in {"pass", "fail", "undetermined"}:
        raise ReviewError("verdict 枚举非法")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ReviewError("confidence 必须位于 [0,1]")
    if not isinstance(value["brief_reason"], str) or not value["brief_reason"].strip():
        raise ReviewError("brief_reason 不能为空")
    if value["counterexample"] is not None and not isinstance(value["counterexample"], str):
        raise ReviewError("counterexample 必须为 string 或 null")
    required_semantics = ["variable_mapping_preserved", "operator_semantics_preserved"]
    if policy == "strict":
        required_semantics.insert(0, "domain_preserved")
    semantics_ok = all(value[field] for field in required_semantics)
    simplicity_ok = policy == "strict" or value["simplicity"] in {"same", "simpler"}
    if value["verdict"] == "pass" and (
        value["equivalence"] != "pass" or not semantics_ok or not simplicity_ok
    ):
        raise ReviewError("pass verdict 与等价性、语义保持或简洁性字段冲突")
    hard_failure = (
        value["equivalence"] == "fail"
        or not semantics_ok
        or (policy == "common_domain_approx" and not simplicity_ok)
    )
    if hard_failure and value["verdict"] != "fail":
        raise ReviewError("确定失败证据必须给出 fail verdict")
    if value["equivalence"] == "undetermined" and not hard_failure and value["verdict"] != "undetermined":
        raise ReviewError("等价性未确定时 verdict 必须为 undetermined")
    return value


def _response_text(model: ModelSpec, payload: Mapping[str, Any]) -> str:
    if model.protocol == "anthropic":
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ReviewError("Anthropic 响应缺少 content")
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        ).strip()
    else:
        text = str(payload.get("output_text") or "").strip()
        if not text:
            chunks: list[str] = []
            output = payload.get("output")
            if isinstance(output, list):
                for message in output:
                    content = message.get("content") if isinstance(message, Mapping) else None
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if isinstance(block, Mapping) and block.get("type") in {"output_text", "text"}:
                            chunks.append(str(block.get("text") or ""))
            text = "".join(chunks).strip()
    if not text:
        raise ReviewError("API 响应没有文本输出")
    return text


def _usage(model: ModelSpec, payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if model.protocol == "anthropic":
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
    else:
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


def default_transport(request: HttpRequest, timeout: float) -> HttpResponse:
    wire_request = urllib.request.Request(
        request.url,
        data=request.body.encode("utf-8"),
        headers=dict(request.headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(wire_request, timeout=timeout) as response:
            status = int(response.status)
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        status = int(error.code)
        body = error.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {"error": {"type": "non_json_response"}}
    if not isinstance(payload, Mapping):
        payload = {"error": {"type": "non_object_response"}}
    return HttpResponse(status=status, payload=payload)


class RequestBudget:
    """跨断点持久化的保守请求预算；预留即计入物理请求。"""

    def __init__(self, path: str | Path, *, cap: int) -> None:
        if cap <= 0:
            raise ReviewError("request cap 必须是正整数")
        self.path = Path(path)
        self.cap = cap
        self._lock = threading.Lock()
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if int(value.get("cap", -1)) != cap:
                raise ReviewError("已存在 request budget 的 cap 与当前参数不同")
            reservations = value.get("reservations")
            if not isinstance(reservations, list):
                raise ReviewError("request budget 已损坏")
            self._reservations = reservations
        else:
            self._reservations: list[dict[str, object]] = []

    @property
    def used(self) -> int:
        with self._lock:
            return len(self._reservations)

    def reserve(self, logical_task_id: str, attempt_number: int) -> int:
        key = f"{logical_task_id}::{attempt_number}"
        with self._lock:
            for reservation in self._reservations:
                if reservation["reservation_key"] == key:
                    return int(reservation["ordinal"])
            if len(self._reservations) >= self.cap:
                raise RequestCapExceeded(f"物理请求达到硬上限 {self.cap}")
            ordinal = len(self._reservations) + 1
            self._reservations.append(
                {"ordinal": ordinal, "reservation_key": key, "logical_task_id": logical_task_id}
            )
            _atomic_write_json(
                self.path,
                {"cap": self.cap, "used": len(self._reservations), "reservations": self._reservations},
            )
            return ordinal


def _task_path(output_root: Path, model: str, sample_id: str) -> Path:
    return output_root / "per_model" / model / "tasks" / f"{sample_id}.json"


def _load_task_record(
    path: Path,
    *,
    model: str,
    sample: Mapping[str, Any],
    review_policy: str = "strict",
) -> dict[str, Any]:
    policy = _validate_review_policy(review_policy)
    prompt_version = prompt_version_for_policy(policy)
    if not path.exists():
        return {
            "schema_version": 1,
            "prompt_version": prompt_version,
            "review_policy": policy,
            "logical_task_id": f"{policy}::{model}::{sample['sample_id']}",
            "sample_id": sample["sample_id"],
            "review_key": sample["review_key"],
            "pair_sha256": sample["pair_sha256"],
            "model": model,
            "status": "pending",
            "attempts": [],
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        prompt_version,
        policy,
        model,
        sample["sample_id"],
        sample["review_key"],
        sample["pair_sha256"],
    )
    actual = (
        value.get("prompt_version"),
        value.get("review_policy"),
        value.get("model"),
        value.get("sample_id"),
        value.get("review_key"),
        value.get("pair_sha256"),
    )
    if actual != expected:
        raise ReviewError(f"断点记录与当前样本/提示词不一致: {path}")
    if not isinstance(value.get("attempts"), list):
        raise ReviewError(f"断点记录 attempts 非法: {path}")
    return value


def _safe_error(response: HttpResponse | None, error: BaseException | None) -> dict[str, object]:
    if response is not None:
        error_payload = response.payload.get("error")
        error_type = error_payload.get("type") if isinstance(error_payload, Mapping) else None
        return {"class": "http_error", "http_status": response.status, "provider_error_type": error_type}
    return {"class": type(error).__name__ if error is not None else "unknown_error"}


def review_one_task(
    *,
    sample: Mapping[str, Any],
    model: ModelSpec,
    profile: ApiProfile,
    output_root: str | Path,
    budget: RequestBudget,
    transport: Callable[[HttpRequest, float], HttpResponse] = default_transport,
    timeout: float = 3000.0,
    retry_delay_seconds: float = 2.0,
    review_policy: str = "strict",
) -> dict[str, Any]:
    root = Path(output_root)
    path = _task_path(root, model.name, str(sample["sample_id"]))
    record = _load_task_record(
        path, model=model.name, sample=sample, review_policy=review_policy
    )
    if record.get("status") == "completed":
        return record
    prompt = render_review_prompt(sample, review_policy=review_policy)
    request = build_http_request(
        model, profile, prompt, review_policy=review_policy
    )
    attempts: list[dict[str, Any]] = record["attempts"]
    while len(attempts) < MAX_ATTEMPTS_PER_TASK:
        attempt_number = len(attempts) + 1
        ordinal = budget.reserve(record["logical_task_id"], attempt_number)
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "physical_request_ordinal": ordinal,
            "status": "reserved",
            "request": dict(request.safe_metadata),
        }
        attempts.append(attempt)
        record["status"] = "running"
        _atomic_write_json(path, record)
        response: HttpResponse | None = None
        caught: BaseException | None = None
        try:
            response = transport(request, timeout)
            if not 200 <= response.status < 300:
                raise ReviewError(f"HTTP {response.status}")
            parsed = parse_review_json(
                _response_text(model, response.payload), review_policy=review_policy
            )
            attempt.update(
                {
                    "status": "accepted",
                    "http_status": response.status,
                    "response_id": str(response.payload.get("id") or ""),
                    "response_model": str(response.payload.get("model") or ""),
                    "response_store_echo": response.payload.get("store", "not_returned"),
                    "usage": _usage(model, response.payload),
                }
            )
            record["review"] = parsed
            record["status"] = "completed"
            _atomic_write_json(path, record)
            return record
        except BaseException as error:  # 每种网络/结构化失败都由同一预算约束。
            caught = error
            attempt.update({"status": "failed", "error": _safe_error(response, caught)})
            record["status"] = "failed"
            _atomic_write_json(path, record)
            if attempt_number < MAX_ATTEMPTS_PER_TASK and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
    return record


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _consensus_for(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, dict[str, int]]:
    verdicts = [
        str(record.get("review", {}).get("verdict", "unavailable"))
        if record.get("status") == "completed"
        else "unavailable"
        for record in records.values()
    ]
    counts = Counter(verdicts)
    if len(verdicts) == len(MODELS) and counts["pass"] == len(MODELS):
        label = "accepted_unanimous"
    elif counts["pass"] >= 2:
        label = "accepted_with_dissent"
    elif counts["fail"] >= 2:
        label = "rejected"
    else:
        label = "unresolved"
    return label, {name: counts[name] for name in ("pass", "fail", "undetermined", "unavailable")}


def write_consensus_artifacts(
    output_root: str | Path,
    samples: Sequence[Mapping[str, Any]],
    records_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, object]]:
    root = Path(output_root)
    rows: list[dict[str, object]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        records = records_by_sample.get(sample_id, {})
        complete_records = {model: records.get(model, {"status": "missing"}) for model in MODELS}
        consensus, counts = _consensus_for(complete_records)
        row: dict[str, object] = {
            "sample_id": sample_id,
            "pair_sha256": sample["pair_sha256"],
            "selection_condition": sample.get("selection_stratum", {}).get("condition", ""),
            "selection_algorithm": sample.get("selection_stratum", {}).get("algorithm", ""),
            "consensus": consensus,
            "pass_count": counts["pass"],
            "fail_count": counts["fail"],
            "undetermined_count": counts["undetermined"],
            "unavailable_count": counts["unavailable"],
        }
        for model_name, record in complete_records.items():
            row[f"{model_name}_status"] = record.get("status", "missing")
            row[f"{model_name}_verdict"] = record.get("review", {}).get("verdict", "")
        rows.append(row)
    fieldnames = [
        "sample_id", "pair_sha256", "selection_condition", "selection_algorithm",
        "consensus", "pass_count", "fail_count", "undetermined_count", "unavailable_count",
    ] + [field for model in MODELS for field in (f"{model}_status", f"{model}_verdict")]
    _write_csv(root / "consensus.csv", fieldnames, rows)
    disagreements = [
        row for row in rows
        if row["consensus"] != "accepted_unanimous"
        or len({row[f"{model}_verdict"] for model in MODELS}) > 1
    ]
    _write_csv(root / "disagreements.csv", fieldnames, disagreements)
    return rows


def _write_sample(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(canonical_json(sample) + "\n" for sample in samples)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ReviewError("sample.jsonl 已存在且与固定抽样结果不同")
    _atomic_write_text(path, content)


def _load_all_task_records(
    output_root: Path,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sample in samples:
        for model_name in MODELS:
            path = _task_path(output_root, model_name, str(sample["sample_id"]))
            if path.exists():
                output[str(sample["sample_id"])][model_name] = json.loads(path.read_text(encoding="utf-8"))
    return output


def _write_indexes(output_root: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    for model_name in MODELS:
        rows: list[dict[str, object]] = []
        for sample in samples:
            path = _task_path(output_root, model_name, str(sample["sample_id"]))
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "pair_sha256": sample["pair_sha256"],
                    "status": record.get("status"),
                    "attempt_count": len(record.get("attempts", [])),
                    "verdict": record.get("review", {}).get("verdict"),
                    "record_path": str(path.relative_to(output_root)),
                    "record_sha256": _sha256_file(path),
                }
            )
        _atomic_write_text(
            output_root / "per_model" / model_name / "index.jsonl",
            "".join(canonical_json(row) + "\n" for row in rows),
        )


def _write_cost_ledger(
    output_root: Path,
    samples: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    rows: list[dict[str, object]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for model_name in MODELS:
            record = records.get(sample_id, {}).get(model_name, {})
            for attempt in record.get("attempts", []):
                usage = attempt.get("usage") or {}
                rows.append(
                    {
                        "model": model_name,
                        "sample_id": sample_id,
                        "attempt_number": attempt.get("attempt_number"),
                        "physical_request_ordinal": attempt.get("physical_request_ordinal"),
                        "status": attempt.get("status"),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "estimated_cost": "",
                        "cost_currency": "",
                        "pricing_basis": "token_usage_recorded; provider_price_not_configured",
                    }
                )
    _write_csv(
        output_root / "cost_ledger.csv",
        [
            "model", "sample_id", "attempt_number", "physical_request_ordinal", "status",
            "input_tokens", "output_tokens", "total_tokens", "estimated_cost", "cost_currency",
            "pricing_basis",
        ],
        rows,
    )


def _write_checksums(output_root: Path) -> None:
    checksum_path = output_root / "SHA256SUMS"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256_file(path)}  {path.relative_to(output_root)}\n")
    _atomic_write_text(checksum_path, "".join(rows))


def _load_frozen_sample(path: str | Path) -> list[dict[str, Any]]:
    rows = [row for _, row in _read_jsonl(Path(path).expanduser().resolve())]
    if not rows:
        raise ReviewError("--sample-jsonl 不包含样本")
    required = {
        "sample_id",
        "review_key",
        "pair_sha256",
        "original_expression",
        "effective_expression",
        "occurrences",
    }
    for index, row in enumerate(rows, start=1):
        if not required.issubset(row):
            raise ReviewError(f"--sample-jsonl 第 {index} 条缺少必填字段")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ReviewError("--sample-jsonl 包含重复 sample_id")
    return rows


def prepare_sample(
    *,
    release_root: str | Path,
    output_root: str | Path,
    sample_size: int,
    seed: int,
    review_policy: str = "strict",
    sample_jsonl: str | Path | None = None,
) -> list[dict[str, Any]]:
    if sample_jsonl is None:
        universe = build_review_universe(release_root)
        samples = stratified_sample(universe, sample_size=sample_size, seed=seed)
    else:
        samples = _load_frozen_sample(sample_jsonl)
        if len(samples) != sample_size:
            raise ReviewError(
                f"--sample-jsonl 含 {len(samples)} 条，但 --sample-size={sample_size}"
            )
    samples = [apply_review_policy(sample, review_policy) for sample in samples]
    root = Path(output_root).expanduser().resolve()
    _write_sample(root / "sample.jsonl", samples)
    return samples


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return Path(args.output_root).expanduser().resolve()
    return (
        DEFAULT_OUTPUT_ROOT
        if args.review_policy == "strict"
        else DEFAULT_OUTPUT_ROOT_V2
    ).resolve()


def run_preflight(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args)
    samples = prepare_sample(
        release_root=args.release_root,
        output_root=output_root,
        sample_size=args.sample_size,
        seed=args.seed,
        review_policy=args.review_policy,
        sample_jsonl=args.sample_jsonl,
    )
    profiles = load_review_profiles(args)
    anthropic = profiles["anthropic"]
    openai = profiles["openai_responses"]
    root = output_root
    report = {
        "status": "preflight_passed_no_network",
        "sample_size": len(samples),
        "sample_seed": args.seed,
        "sample_source": str(args.sample_jsonl) if args.sample_jsonl else "stratified_release",
        "review_policy": args.review_policy,
        "prompt_version": prompt_version_for_policy(args.review_policy),
        "models": list(MODELS),
        "concurrency_per_model": args.concurrency_per_model,
        "logical_tasks": logical_task_count(len(samples)),
        "physical_request_cap": physical_request_cap(len(samples)),
        "max_attempts_per_task": MAX_ATTEMPTS_PER_TASK,
        "stream": False,
        "source_outcomes": count_release_outcomes(args.release_root),
        "profiles": {
            "anthropic": dict(anthropic.safe_metadata),
            "openai": dict(openai.safe_metadata),
        },
    }
    _atomic_write_json(root / "preflight.json", report)
    _write_checksums(root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def run_batch(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args)
    samples = prepare_sample(
        release_root=args.release_root,
        output_root=output_root,
        sample_size=args.sample_size,
        seed=args.seed,
        review_policy=args.review_policy,
        sample_jsonl=args.sample_jsonl,
    )
    if not 1 <= args.concurrency_per_model <= CONCURRENCY_PER_MODEL:
        raise ReviewError(f"concurrency_per_model 必须位于 [1,{CONCURRENCY_PER_MODEL}]")
    profiles = load_review_profiles(args)
    root = output_root
    budget = RequestBudget(root / "request_budget.json", cap=physical_request_cap(len(samples)))
    errors: list[str] = []
    executors = {
        model.name: concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency_per_model,
            thread_name_prefix=f"review-{model.name}",
        )
        for model in MODELS.values()
    }
    try:
        futures = {
            executors[model.name].submit(
                review_one_task,
                sample=sample,
                model=model,
                profile=profiles[model.protocol],
                output_root=root,
                budget=budget,
                timeout=args.timeout,
                retry_delay_seconds=args.retry_delay,
                review_policy=args.review_policy,
            ): (sample["sample_id"], model.name)
            for model in MODELS.values()
            for sample in samples
        }
        for future in concurrent.futures.as_completed(futures):
            sample_id, model_name = futures[future]
            try:
                future.result()
            except BaseException as error:
                errors.append(f"{model_name}::{sample_id}::{type(error).__name__}")
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
    _write_indexes(root, samples)
    records = _load_all_task_records(root, samples)
    consensus_rows = write_consensus_artifacts(root, samples, records)
    _write_cost_ledger(root, samples, records)
    model_summary = {}
    for model_name in MODELS:
        statuses = Counter(
            records.get(str(sample["sample_id"]), {}).get(model_name, {}).get("status", "missing")
            for sample in samples
        )
        verdicts = Counter(
            records.get(str(sample["sample_id"]), {}).get(model_name, {}).get("review", {}).get(
                "verdict", "unavailable"
            )
            for sample in samples
        )
        model_summary[model_name] = {"statuses": dict(statuses), "verdicts": dict(verdicts)}
    summary = {
        "status": "completed" if not errors and all(row["unavailable_count"] == 0 for row in consensus_rows) else "partial",
        "sample_size": len(samples),
        "sample_seed": args.seed,
        "sample_source": str(args.sample_jsonl) if args.sample_jsonl else "stratified_release",
        "review_policy": args.review_policy,
        "prompt_version": prompt_version_for_policy(args.review_policy),
        "logical_tasks": logical_task_count(len(samples)),
        "physical_request_cap": physical_request_cap(len(samples)),
        "physical_requests_reserved": budget.used,
        "max_attempts_per_task": MAX_ATTEMPTS_PER_TASK,
        "concurrency_per_model": args.concurrency_per_model,
        "stream": False,
        "source_outcomes": count_release_outcomes(args.release_root),
        "model_summary": model_summary,
        "consensus": dict(Counter(str(row["consensus"]) for row in consensus_rows)),
        "execution_errors": errors,
    }
    _atomic_write_json(root / "summary.json", summary)
    _write_checksums(root)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "completed" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="三模型、每模型最多 32 并发的 Core50 化简盲审控制器",
        epilog=(
            "先运行 preflight（不发网络请求），确认后再运行 run。示例：\n"
            "  python check/run_three_model_simplification_review.py preflight --openai-profile ~/.codex/routify-openai.toml\n"
            "  python check/run_three_model_simplification_review.py run --openai-profile ~/.codex/routify-openai.toml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
        subparser.add_argument(
            "--review-policy",
            choices=REVIEW_POLICIES,
            default=DEFAULT_REVIEW_POLICY,
            help="strict 为原严格合同；common_domain_approx 为共同实数域近似等价合同",
        )
        subparser.add_argument(
            "--output-root",
            type=Path,
            default=None,
            help="省略时按 review policy 写入互相隔离的默认目录",
        )
        subparser.add_argument(
            "--sample-jsonl",
            type=Path,
            default=None,
            help="复用已冻结样本；表达式和 sample_id 不变，但按当前 policy 重新生成 review_key",
        )
        subparser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
        subparser.add_argument("--seed", type=int, default=SAMPLE_SEED)
        subparser.add_argument("--concurrency-per-model", type=int, default=CONCURRENCY_PER_MODEL)
        subparser.add_argument("--anthropic-profile", type=Path, default=DEFAULT_ANTHROPIC_PROFILE)
        subparser.add_argument("--openai-profile", type=Path, default=DEFAULT_OPENAI_PROFILE)
        subparser.add_argument("--openai-provider", default="custom")
        subparser.add_argument(
            "--reuse-anthropic-token-for-openai",
            action="store_true",
            help="复用已加载的 Routify 凭据并固定路由到 OpenAI Responses 端口",
        )
        if command == "run":
            subparser.add_argument("--timeout", type=float, default=3000.0)
            subparser.add_argument("--retry-delay", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return run_preflight(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
