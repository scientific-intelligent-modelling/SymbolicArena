from __future__ import annotations

import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from scientific_intelligent_modelling.benchmarks.artifact_schema import build_canonical_symbolic_program


class DGPDiv(sp.Function):
    nargs = 2


class DGPLog(sp.Function):
    nargs = 1


class DGPExp(sp.Function):
    nargs = 1


class DGPPrinter(NumPyPrinter):
    def _print_DGPDiv(self, expression):
        left, right = (self._print(value) for value in expression.args)
        return "np.divide(" + left + ", np.where(np.abs(" + right + ") > 0.0001, " + right + ", 0.0001))"

    def _print_DGPLog(self, expression):
        value = self._print(expression.args[0])
        return "np.log(np.abs(np.where(" + value + " == 0, 1.0, " + value + ")))"

    def _print_DGPExp(self, expression):
        return "np.exp(np.minimum(" + self._print(expression.args[0]) + ", 10.0))"


def normalize_dgp_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict:
    operations = {
        "add": lambda a, b, **_kwargs: sp.Add(a, b, evaluate=False),
        "sub": lambda a, b, **_kwargs: sp.Add(a, -b, evaluate=False),
        "mul": lambda a, b, **_kwargs: sp.Mul(a, b, evaluate=False),
        "div": DGPDiv,
        "log": DGPLog,
        "exp": DGPExp,
        "sin": lambda a, **_kwargs: sp.sin(a, evaluate=False),
        "cos": lambda a, **_kwargs: sp.cos(a, evaluate=False),
    }
    expression = sp.sympify(raw_equation, locals=operations, evaluate=False)
    variables = sorted((str(symbol) for symbol in expression.free_symbols), key=lambda name: int(name[1:]) if name.startswith("x") and name[1:].isdigit() else -1)
    for name in variables:
        if not name.startswith("x") or not name[1:].isdigit():
            raise ValueError("Unknown DGP variable: " + name)
        if expected_n_features is not None and int(name[1:]) >= expected_n_features:
            raise ValueError("DGP variable is outside the dataset: " + name)
    executable = DGPPrinter({"fully_qualified_modules": False}).doprint(expression)
    symbolic = expression.replace(DGPDiv, lambda a, b: a / sp.Piecewise((b, sp.Abs(b) > sp.Float("0.0001")), (sp.Float("0.0001"), True)))
    symbolic = symbolic.replace(DGPLog, lambda a: sp.log(sp.Abs(sp.Piecewise((sp.Integer(1), sp.Eq(a, 0)), (a, True)))))
    symbolic = symbolic.replace(DGPExp, lambda a: sp.exp(sp.Min(a, sp.Integer(10))))
    artifact = build_canonical_symbolic_program(
        tool_name="dgp",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        normalized_expression=str(symbolic),
        variables=variables,
        operator_set=sorted({function.func.__name__ for function in expression.atoms(sp.Function)}),
        ast_node_count=sum(1 for _ in sp.preorder_traversal(expression)),
        normalization_mode="dgp_native_protected_v1",
    )
    artifact["executable_expression"] = executable
    artifact["sympy_expression"] = str(symbolic)
    artifact["sympy_parse_ok"] = True
    artifact["native_operator_semantics"] = {"div_threshold": 0.0001, "div_small_denominator": 0.0001, "exp_max_input": 10.0, "log_zero_input": 1.0}
    return artifact
