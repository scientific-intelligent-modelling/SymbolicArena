import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_tpsr_wrapper_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda *args, **kwargs: ("device", args, kwargs)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.set_num_threads = lambda *args, **kwargs: None
    torch_stub.set_num_interop_threads = lambda *args, **kwargs: None

    normalizers_stub = types.ModuleType("scientific_intelligent_modelling.benchmarks.normalizers")
    normalizers_stub.normalize_tpsr_artifact = lambda *args, **kwargs: {}

    base_wrapper_stub = types.ModuleType(
        "scientific_intelligent_modelling.algorithms.base_wrapper"
    )

    class _BaseWrapper:
        pass

    base_wrapper_stub.BaseWrapper = _BaseWrapper

    previous_modules = {}
    stubs = {
        "torch": torch_stub,
        "scientific_intelligent_modelling.benchmarks.normalizers": normalizers_stub,
        "scientific_intelligent_modelling.algorithms.base_wrapper": base_wrapper_stub,
    }
    for name, module in stubs.items():
        previous_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        module_name = "scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper"
        module_path = str(
            Path(__file__).resolve().parents[1]
            / "scientific_intelligent_modelling"
            / "algorithms"
            / "tpsr_wrapper"
            / "wrapper.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        source = spec.loader.get_source(module_name)
        exec(compile(source, module_path, "exec"), module.__dict__)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_downsample_reward_arrays_caps_rows_and_preserves_alignment():
    module = _load_tpsr_wrapper_module()
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)

    sampled_X, sampled_y = module.TPSRRegressor._downsample_reward_arrays(X, y, 4)

    assert sampled_X.shape == (4, 3)
    assert sampled_y.shape == (4,)
    np.testing.assert_array_equal(sampled_y, sampled_X[:, 0] / 3.0)


def test_downsample_reward_arrays_keeps_small_inputs_unchanged():
    module = _load_tpsr_wrapper_module()
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.arange(4, dtype=float)

    sampled_X, sampled_y = module.TPSRRegressor._downsample_reward_arrays(X, y, 16)

    np.testing.assert_array_equal(sampled_X, X)
    np.testing.assert_array_equal(sampled_y, y)


def test_tpsr_wrapper_defaults_align_official_bagging_config():
    module = _load_tpsr_wrapper_module()

    reg = module.TPSRRegressor()

    assert reg.params["max_input_points"] == 200
    assert reg.params["max_number_bags"] == 10
    assert reg.params["n_trees_to_refine"] == 10
    assert reg.params["no_seq_cache"] is False
    assert reg.params["no_prefix_cache"] is True
    assert reg.params["width"] == 3
    assert reg.params["num_beams"] == 1
    assert reg.params["rollout"] == 3
    assert reg.params["horizon"] == 200
    assert reg.params["lam"] == 0.1


def test_tpsr_runtime_feature_context_tracks_current_dataset():
    module = _load_tpsr_wrapper_module()

    reg = module.TPSRRegressor(max_input_dimension=10)
    X = np.arange(20, dtype=float).reshape(5, 4)

    n_features = reg._capture_runtime_feature_context(X)

    assert n_features == 4
    assert reg._n_features == 4
    assert reg._predict_variable_names == ["x_0", "x_1", "x_2", "x_3"]


def test_tpsr_progress_state_projects_out_of_range_variables_to_zero():
    module = _load_tpsr_wrapper_module()

    reg = module.TPSRRegressor()
    reg._n_features = 4
    written = []
    reg._write_progress_state = written.append

    reg._emit_progress_equation(
        equation="x_0 + x_9",
        score=1.0,
        complexity=3,
        source="unit_test",
    )
    assert len(written) == 1
    assert written[0]["equation"] == "x_0 + 0"

    written.clear()
    reg._emit_progress_equation(
        equation="x_0 + x_3",
        score=1.0,
        complexity=3,
        source="unit_test",
    )
    assert len(written) == 1
    assert written[0]["equation"] == "x_0 + x_3"


def test_tpsr_variable_budget_accepts_one_based_tokens():
    module = _load_tpsr_wrapper_module()

    assert module.TPSRRegressor._equation_within_feature_budget("x_1 + x_4", 4) is True
    assert module.TPSRRegressor._equation_within_feature_budget("x_1 + x_5", 4) is False


def test_tpsr_projects_out_of_range_one_based_tokens_to_zero():
    module = _load_tpsr_wrapper_module()

    assert module.TPSRRegressor._project_equation_to_feature_budget("x_1 + x_5", 4) == "x_1 + 0"


class _FakeTree:
    def __init__(self, expression):
        self.expression = expression
        self.rescale_count = 0

    def infix(self):
        return self.expression


class _MutatingScaler:
    def __init__(self, *, fail=False):
        self.fail = fail

    def rescale_function(self, equation_env, tree, a, b):
        tree.rescale_count += 1
        if self.fail:
            raise ValueError("cannot rescale tree")
        tree.expression = f"({a[0]}) * x_0 + ({b[0]})"
        return tree


def _install_e2e_model_stub(
    monkeypatch,
    *,
    refined_expr,
    refined_trees,
    raw_expr,
    raw_trees,
):
    package = types.ModuleType("symbolicregression")
    package.__path__ = []
    module = types.ModuleType("symbolicregression.e2e_model")
    module.refine_for_sample = lambda *args, **kwargs: (None, refined_expr, refined_trees)
    module.pred_for_sample_no_refine = lambda *args, **kwargs: (None, raw_expr, raw_trees)
    monkeypatch.setitem(sys.modules, "symbolicregression", package)
    monkeypatch.setitem(sys.modules, "symbolicregression.e2e_model", module)


def test_sequence_expression_is_inverse_scaled_and_does_not_mutate_candidate(monkeypatch):
    module = _load_tpsr_wrapper_module()
    refined_tree = _FakeTree("x_0")
    _install_e2e_model_stub(
        monkeypatch,
        refined_expr="x_0",
        refined_trees=[refined_tree],
        raw_expr=None,
        raw_trees=[],
    )
    reg = module.TPSRRegressor(rescale=True)
    reg._tpsr_scaler = _MutatingScaler()
    reg._tpsr_scale_params = (np.array([0.5]), np.array([-5.0]))

    expression = reg._sequence_to_e2e_expression(
        types.SimpleNamespace(),
        object(),
        object(),
        [0, 1],
        {"x_to_fit": [None], "y_to_fit": [None]},
    )

    assert expression == "(0.5) * x_0 + (-5.0)"
    assert refined_tree.expression == "x_0"
    assert refined_tree.rescale_count == 0


def test_scaled_string_fallback_is_structurally_inverse_scaled(monkeypatch):
    module = _load_tpsr_wrapper_module()
    _install_e2e_model_stub(
        monkeypatch,
        refined_expr="x_0 + x_1",
        refined_trees=[_FakeTree("x_0 + x_1")],
        raw_expr=None,
        raw_trees=[],
    )
    reg = module.TPSRRegressor(rescale=True)
    reg._tpsr_scaler = _MutatingScaler(fail=True)
    reg._tpsr_scale_params = (np.array([2.0, 3.0]), np.array([5.0, 7.0]))

    expression = reg._sequence_to_e2e_expression(
        types.SimpleNamespace(),
        object(),
        object(),
        [0, 1],
        {"x_to_fit": [None], "y_to_fit": [None]},
    )

    import sympy as sp

    expected = 2 * sp.Symbol("x_0") + 3 * sp.Symbol("x_1") + 12
    assert sp.simplify(sp.sympify(expression) - expected) == 0
    assert expression != "x_0 + x_1"


def test_inverse_scale_failure_never_returns_scaled_expression(monkeypatch):
    module = _load_tpsr_wrapper_module()
    _install_e2e_model_stub(
        monkeypatch,
        refined_expr="x_0 +",
        refined_trees=[_FakeTree("x_0")],
        raw_expr=None,
        raw_trees=[],
    )
    reg = module.TPSRRegressor(rescale=True)
    reg._tpsr_scaler = _MutatingScaler(fail=True)
    reg._tpsr_scale_params = (np.array([0.5]), np.array([-5.0]))

    expression = reg._sequence_to_e2e_expression(
        types.SimpleNamespace(),
        object(),
        object(),
        [0, 1],
        {"x_to_fit": [None], "y_to_fit": [None]},
    )

    assert expression is None


def test_missing_inverse_scale_state_fails_closed(monkeypatch):
    module = _load_tpsr_wrapper_module()
    _install_e2e_model_stub(
        monkeypatch,
        refined_expr="x_0",
        refined_trees=[_FakeTree("x_0")],
        raw_expr=None,
        raw_trees=[],
    )
    reg = module.TPSRRegressor(rescale=True)

    expression = reg._sequence_to_e2e_expression(
        types.SimpleNamespace(),
        object(),
        object(),
        [0, 1],
        {"x_to_fit": [None], "y_to_fit": [None]},
    )

    assert expression is None


def test_e2e_final_and_progress_candidates_share_original_coordinates(monkeypatch):
    module = _load_tpsr_wrapper_module()
    refined_tree = _FakeTree("x_0")
    _install_e2e_model_stub(
        monkeypatch,
        refined_expr="x_0",
        refined_trees=[refined_tree],
        raw_expr="2 * x_0",
        raw_trees=[_FakeTree("2 * x_0")],
    )
    reg = module.TPSRRegressor(rescale=True)
    reg._tpsr_scaler = _MutatingScaler()
    reg._tpsr_scale_params = (np.array([0.5]), np.array([-5.0]))

    progress_expression = reg._sequence_to_e2e_expression(
        types.SimpleNamespace(),
        object(),
        object(),
        [0, 1],
        {"x_to_fit": [None], "y_to_fit": [None]},
    )
    final_candidates = reg._e2e_expression_candidates(
        object(),
        (("x_0", [refined_tree]), ("2 * x_0", [_FakeTree("2 * x_0")])),
    )
    emitted = []
    reg._write_progress_state = emitted.append
    for source, expression in (
        ("e2e_candidate", progress_expression),
        ("e2e_terminal", progress_expression),
        ("e2e_final", final_candidates[0]),
    ):
        reg._emit_progress_equation(equation=expression, source=source)

    assert final_candidates[0] == progress_expression
    assert final_candidates == ["(0.5) * x_0 + (-5.0)", "(0.5) * x_0 + (-5.0)"]
    assert [item["source"] for item in emitted] == [
        "e2e_candidate",
        "e2e_terminal",
        "e2e_final",
    ]
    assert [item["equation"] for item in emitted] == [progress_expression] * 3
    assert refined_tree.expression == "x_0"
    assert refined_tree.rescale_count == 0


def test_e2e_expression_is_unchanged_when_rescale_is_disabled(monkeypatch):
    module = _load_tpsr_wrapper_module()
    tree = _FakeTree("x_0 + 1")
    _install_e2e_model_stub(
        monkeypatch,
        refined_expr="x_0 + 1",
        refined_trees=[tree],
        raw_expr=None,
        raw_trees=[],
    )
    reg = module.TPSRRegressor(rescale=False)

    expression = reg._sequence_to_e2e_expression(
        types.SimpleNamespace(),
        object(),
        object(),
        [0, 1],
        {"x_to_fit": [None], "y_to_fit": [None]},
    )

    assert expression == "x_0 + 1"
    assert tree.expression == "x_0 + 1"
    assert tree.rescale_count == 0
