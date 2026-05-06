"""结果归档阶段使用的统一符号工件辅助函数。"""

from __future__ import annotations

from typing import Any

from .artifact_schema import validate_canonical_symbolic_program
from .normalizers import (
    normalize_dso_artifact,
    normalize_drsr_artifact,
    normalize_e2esr_artifact,
    normalize_gplearn_artifact,
    normalize_imcts_artifact,
    normalize_llmsr_artifact,
    normalize_operon_artifact,
    normalize_pysr_artifact,
    normalize_qlattice_artifact,
    normalize_ragsr_artifact,
    normalize_tpsr_artifact,
    normalize_udsr_artifact,
)


def _format_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def safe_export_canonical_artifact(regressor: Any) -> tuple[dict[str, Any] | None, str | None]:
    """从统一回归器安全导出 canonical artifact。"""
    try:
        artifact = regressor.export_canonical_symbolic_program()
        artifact = validate_canonical_symbolic_program(artifact)
        return artifact, None
    except Exception as exc:
        return None, _format_error(exc)


def safe_build_canonical_artifact(
    *,
    tool_name: str,
    equation: str | None,
    expected_n_features: int | None = None,
    parameter_values: list[float] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """从已落盘公式文本反向构造 canonical artifact。

    主要用于 timeout 恢复等场景，此时没有可直接调用的 regressor 实例。
    """
    text = "" if equation is None else str(equation).strip()
    if not text:
        return None, "ValueError: 空公式，无法构造 canonical_artifact"

    tool = str(tool_name).strip().lower()
    try:
        if tool == "pysr":
            artifact = normalize_pysr_artifact(text, expected_n_features=expected_n_features)
        elif tool == "gplearn":
            artifact = normalize_gplearn_artifact(text, expected_n_features=expected_n_features)
        elif tool in {"pyoperon", "operon"}:
            artifact = normalize_operon_artifact(text, expected_n_features=expected_n_features)
        elif tool == "ragsr":
            artifact = normalize_ragsr_artifact(text, expected_n_features=expected_n_features)
        elif tool == "llmsr":
            artifact = normalize_llmsr_artifact(
                text,
                parameter_values=parameter_values,
                expected_n_features=expected_n_features,
            )
        elif tool == "drsr":
            artifact = normalize_drsr_artifact(
                text,
                parameter_values=parameter_values,
                expected_n_features=expected_n_features,
            )
        elif tool == "dso":
            artifact = normalize_dso_artifact(text, expected_n_features=expected_n_features)
        elif tool == "udsr":
            artifact = normalize_udsr_artifact(text, expected_n_features=expected_n_features)
        elif tool == "tpsr":
            artifact = normalize_tpsr_artifact(text, expected_n_features=expected_n_features)
        elif tool == "e2esr":
            artifact = normalize_e2esr_artifact(text, expected_n_features=expected_n_features)
        elif tool in {"qlattice", "qlattice_wrapper"}:
            artifact = normalize_qlattice_artifact(text, expected_n_features=expected_n_features)
        elif tool in {"imcts", "imcts_wrapper"}:
            artifact = normalize_imcts_artifact(text, expected_n_features=expected_n_features)
        else:
            raise ValueError(f"暂不支持的工具名: {tool_name!r}")
        artifact = validate_canonical_symbolic_program(artifact)
        return artifact, None
    except Exception as exc:
        return None, _format_error(exc)
