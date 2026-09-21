"""动态构建 LLMSR 规格（spec）的工具。"""

from __future__ import annotations

from typing import List, Optional

from scientific_intelligent_modelling.srkit.spec_builder import build_specification as build_shared_specification
from scientific_intelligent_modelling.srkit.spec_builder import dedup_names as _dedup_names
from scientific_intelligent_modelling.srkit.spec_builder import sanitize_name


def build_specification(
    background: str,
    features: List[str],
    target: str,
    max_params: int = 10,
    problem: Optional[str] = None,
    feature_descriptions: Optional[List[Optional[str]]] = None,
    target_description: Optional[str] = None,
) -> str:
    """构建完整的规范化规格文本。"""
    if not features:
        raise ValueError("features 不能为空")

    cleaned_features = _dedup_names([sanitize_name(n) for n in features])
    return build_shared_specification(
        background=background,
        features=cleaned_features,
        target=sanitize_name(target),
        max_params=max_params,
        problem=problem,
        evaluate_style="llmsr",
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )
