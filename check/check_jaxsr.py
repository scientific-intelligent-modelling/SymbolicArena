import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(42)
    X = rng.rand(16, 2)
    y = 1.5 + 2.0 * X[:, 0] - 0.7 * X[:, 1]

    reg = SymbolicRegressor(
        "jaxsr",
        problem_name="check_jaxsr",
        seed=42,
        max_terms=3,
        max_polynomial_degree=1,
        max_interaction_order=1,
        include_polynomials=False,
        include_interactions=False,
        include_transcendental=False,
        include_ratios=False,
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:5])

    assert_ok(isinstance(eq, str) and eq.strip(), "jaxsr: 最优方程为空")
    assert_ok(isinstance(equations, list) and equations, "jaxsr: 方程列表为空")
    assert_ok(pred.shape == (5,), f"jaxsr: 预测形状异常 {pred.shape}")

    print("[check_jaxsr] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
