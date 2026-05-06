from __future__ import annotations

import re
from typing import List, Literal, Optional


def sanitize_name(name: str) -> str:
    """将任意字符串清洗为稳定的 Python 变量名。"""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "f"
    if not re.match(r"^[a-z]", s):
        s = "f_" + s
    return s


def dedup_names(names: List[str]) -> List[str]:
    """对清洗后的变量名去重，冲突时追加序号。"""
    seen = {}
    result = []
    for n in names:
        base = n
        idx = seen.get(base, 0)
        out = base if idx == 0 else f"{base}_{idx}"
        while out in seen:
            idx += 1
            out = f"{base}_{idx}"
        seen[base] = idx + 1
        seen[out] = 1
        result.append(out)
    return result


def _build_header(problem: str, background_text: str, feature_doc: str, target_clean: str) -> str:
    return f'''"""
Find the mathematical function skeleton that represents {problem}.

Background:
{background_text}

Variables:
- Independents: {feature_doc}
- Dependent: {target_clean}
"""
'''


def _format_variables_block(
    cleaned_features: List[str],
    target_clean: str,
    feature_descriptions: Optional[List[Optional[str]]] = None,
    target_description: Optional[str] = None,
) -> str:
    if feature_descriptions and len(feature_descriptions) == len(cleaned_features):
        lines = ["- Independents:"]
        for name, desc in zip(cleaned_features, feature_descriptions):
            if desc and str(desc).strip():
                lines.append(f"  - {name}: {str(desc).strip()}")
            else:
                lines.append(f"  - {name}")
        lines.append("- Dependent:")
        if target_description and str(target_description).strip():
            lines.append(f"  - {target_clean}: {str(target_description).strip()}")
        else:
            lines.append(f"  - {target_clean}")
        return "\n".join(lines)
    feature_doc = ", ".join(cleaned_features)
    return f"- Independents: {feature_doc}\n- Dependent: {target_clean}"


def _build_equation_block(
    feature_sig: str,
    background_text: str,
    variables_block: str,
    linear_seed: str,
) -> str:
    return f'''
@equation.evolve
def equation({feature_sig}, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    {background_text}

    Variables:
    {variables_block}

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.

    Output requirement:
    - Write the final formula as a single-line return statement.
    - Do not split the final formula across multiple lines.
    - Example: return {linear_seed}
    """
    return {linear_seed}
'''


def build_specification(
    background: str,
    features: List[str],
    target: str,
    max_params: int = 10,
    problem: Optional[str] = None,
    evaluate_style: Literal["llmsr", "drsr"] = "llmsr",
    feature_descriptions: Optional[List[Optional[str]]] = None,
    target_description: Optional[str] = None,
) -> str:
    """
    统一构建 llmsr/drsr 的 spec 文本。

    设计原则：
    - 文本语义、变量命名、参数槽、线性 seed 对齐；
    - evaluate() 的底层执行策略允许按算法保留差异。
    """
    if not features:
        raise ValueError("features 不能为空")

    cleaned_features = dedup_names([sanitize_name(n) for n in features])
    # 线性 seed 至少需要：常数项 params[0] + 每个特征一个系数。
    # 如果调用方把 max_params 设得更小，会导致模板里引用了不存在的 params[k]。
    effective_max_params = max(int(max_params), len(cleaned_features) + 1)
    target_clean = sanitize_name(target)
    background_text = background.strip() if background else ""
    variables_block = _format_variables_block(
        cleaned_features=cleaned_features,
        target_clean=target_clean,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )
    feature_sig = ", ".join([f"{n}: np.ndarray" for n in cleaned_features])
    linear_terms = ["params[0]"] + [f"params[{i}] * {n}" for i, n in enumerate(cleaned_features, start=1)]
    linear_seed = " + ".join(linear_terms)

    header = f'''"""
Find the mathematical function skeleton that fits the data.

Background:
{background_text}

Variables:
{variables_block}
"""
'''

    if evaluate_style == "llmsr":
        evaluate_block = f'''
import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = {effective_max_params}
params = [1.0]*MAX_NPARAMS
# 全局变量用于在沙箱中读取 BFGS 结果参数
BFGS_PARAMS = None

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    global BFGS_PARAMS
    try:
        BFGS_PARAMS = result.x
    except Exception:
        BFGS_PARAMS = None
    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return None
    return -loss_val
'''
    elif evaluate_style == "drsr":
        evaluate_block = f'''
import numpy as np

# Initialize parameters
MAX_NPARAMS = {effective_max_params}
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    cols = [inputs[:, i] for i in range(inputs.shape[1])]
    try:
        y_pred = equation(*cols, params)
        loss = np.mean((y_pred - outputs) ** 2)
        if np.isnan(loss) or np.isinf(loss):
            return None
        return -loss
    except Exception:
        return None
'''
    else:
        raise ValueError(f"不支持的 evaluate_style: {evaluate_style}")

    equation_block = _build_equation_block(
        feature_sig=feature_sig,
        background_text=background_text,
        variables_block=variables_block,
        linear_seed=linear_seed,
    )
    return header + evaluate_block + equation_block
