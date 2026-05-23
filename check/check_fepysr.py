import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(42)
    X = rng.rand(16, 2)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1]

    reg = SymbolicRegressor(
        "fepysr",
        problem_name="check_fepysr",
        seed=42,
        num_workers=1,
        num_experiments=1,
        fmn_epochs=1,
        fmn_batch_size=16,
        fea_num=2,
        pysr_num=1,
        niterations=1,
        population_size=16,
        populations=1,
        ncycles_per_iteration=5,
        maxsize=12,
        maxdepth=6,
        timeout_in_seconds=600,
        unary_operators=["sin", "cos", "exp"],
        binary_operators=["+", "-", "*", "/"],
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:5])

    assert_ok(isinstance(eq, str) and eq.strip(), "fepysr: 最优方程为空")
    assert_ok(isinstance(equations, list) and equations, "fepysr: 方程列表为空")
    assert_ok(pred.shape == (5,), f"fepysr: 预测形状异常 {pred.shape}")

    print("[check_fepysr] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
