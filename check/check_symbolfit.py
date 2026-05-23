import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(42)
    X = rng.rand(36, 1)
    y = 2.0 + 3.0 * X[:, 0] + 0.05 * rng.randn(36)

    reg = SymbolicRegressor(
        "symbolfit",
        problem_name="check_symbolfit",
        seed=42,
        niterations=1,
        maxsize=8,
        max_complexity=8,
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        y_uncertainty=1.0,
        fit_y_unc=False,
        timeout_in_seconds=600,
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:5])

    assert_ok(isinstance(eq, str) and eq.strip(), "symbolfit: 最优方程为空")
    assert_ok(isinstance(equations, list) and equations, "symbolfit: 方程列表为空")
    assert_ok(pred.shape == (5,), f"symbolfit: 预测形状异常 {pred.shape}")

    print("[check_symbolfit] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
