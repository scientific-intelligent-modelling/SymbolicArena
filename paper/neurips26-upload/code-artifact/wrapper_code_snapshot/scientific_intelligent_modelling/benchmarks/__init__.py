"""符号回归 benchmark 指标工具集。"""

from .artifact_schema import (
    CSP_VERSION,
    build_canonical_symbolic_program,
    extract_return_expression_from_python_function,
    infer_raw_equation_kind,
    validate_canonical_symbolic_program,
)
from .normalizers import (
    normalize_drsr_artifact,
    normalize_e2esr_artifact,
    normalize_gplearn_artifact,
    normalize_llmsr_artifact,
    normalize_operon_artifact,
    normalize_pysr_artifact,
    normalize_qlattice_artifact,
    normalize_ragsr_artifact,
    normalize_tpsr_artifact,
    normalize_udsr_artifact,
)
from .metrics import (
    acc_within_threshold,
    llm_srbench_acc_tau,
    llm_srbench_nmse,
    llm_srbench_numeric_metrics,
    normalized_tree_edit_distance,
    regression_metrics,
    srbench_model_size,
    srbench_symbolic_solution,
)
from .profiles import BENCHMARK_PROFILES
from .runner import (
    DatasetSplit,
    LoadedDataset,
    build_result_payload,
    build_runner_params,
    load_canonical_dataset,
    run_benchmark_task,
)

try:
    from .judges import LLMSymbolicJudge, llm_srbench_symbolic_accuracy
except ModuleNotFoundError:  # pragma: no cover - 非 LLM 环境允许跳过 judge 依赖
    LLMSymbolicJudge = None  # type: ignore
    llm_srbench_symbolic_accuracy = None  # type: ignore

__all__ = [
    "BENCHMARK_PROFILES",
    "CSP_VERSION",
    "LLMSymbolicJudge",
    "acc_within_threshold",
    "build_canonical_symbolic_program",
    "extract_return_expression_from_python_function",
    "infer_raw_equation_kind",
    "normalize_drsr_artifact",
    "normalize_e2esr_artifact",
    "normalize_gplearn_artifact",
    "normalize_llmsr_artifact",
    "normalize_operon_artifact",
    "normalize_pysr_artifact",
    "normalize_qlattice_artifact",
    "normalize_ragsr_artifact",
    "normalize_tpsr_artifact",
    "normalize_udsr_artifact",
    "llm_srbench_acc_tau",
    "llm_srbench_nmse",
    "llm_srbench_numeric_metrics",
    "llm_srbench_symbolic_accuracy",
    "normalized_tree_edit_distance",
    "regression_metrics",
    "DatasetSplit",
    "LoadedDataset",
    "build_result_payload",
    "build_runner_params",
    "load_canonical_dataset",
    "run_benchmark_task",
    "srbench_model_size",
    "srbench_symbolic_solution",
    "validate_canonical_symbolic_program",
]
