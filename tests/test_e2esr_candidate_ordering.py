import importlib.util
import sys
import types
from pathlib import Path


def _load_sklearn_wrapper_module():
    module_name = "test_e2esr_sklearn_wrapper"
    module_path = Path(
        "scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/sklearn_wrapper.py"
    ).resolve()

    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda fn: fn)
    metrics_stub = types.ModuleType("symbolicregression.metrics")
    metrics_stub.compute_metrics = lambda *args, **kwargs: {}
    utils_wrapper_stub = types.ModuleType("symbolicregression.model.utils_wrapper")

    symbolicregression_pkg = types.ModuleType("symbolicregression")
    symbolicregression_model_pkg = types.ModuleType("symbolicregression.model")
    symbolicregression_model_pkg.utils_wrapper = utils_wrapper_stub

    sklearn_pkg = types.ModuleType("sklearn")
    sklearn_base_stub = types.ModuleType("sklearn.base")
    sklearn_base_stub.BaseEstimator = object
    sklearn_feature_selection_stub = types.ModuleType("sklearn.feature_selection")
    sklearn_feature_selection_stub.SelectKBest = object
    sklearn_feature_selection_stub.r_regression = object()
    sklearn_pkg.feature_selection = sklearn_feature_selection_stub

    previous_modules = {}
    stubs = {
        "torch": torch_stub,
        "symbolicregression": symbolicregression_pkg,
        "symbolicregression.metrics": metrics_stub,
        "symbolicregression.model": symbolicregression_model_pkg,
        "symbolicregression.model.utils_wrapper": utils_wrapper_stub,
        "sklearn": sklearn_pkg,
        "sklearn.base": sklearn_base_stub,
        "sklearn.feature_selection": sklearn_feature_selection_stub,
    }
    for name, module in stubs.items():
        previous_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_order_candidates_puts_invalid_mse_scores_at_end():
    module = _load_sklearn_wrapper_module()
    reg = module.SymbolicTransformerRegressor()
    score_map = {"a": None, "b": 0.2, "c": 0.1, "d": float("nan")}
    reg._safe_tree_metric = lambda tree, X, y, metric: score_map[tree]  # type: ignore[method-assign]

    candidates = [
        {"predicted_tree": "a"},
        {"predicted_tree": "b"},
        {"predicted_tree": "c"},
        {"predicted_tree": "d"},
    ]

    ordered = reg.order_candidates(None, None, candidates, metric="_mse")

    assert [item["predicted_tree"] for item in ordered] == ["c", "b", "a", "d"]


def test_order_candidates_puts_invalid_r2_scores_at_end():
    module = _load_sklearn_wrapper_module()
    reg = module.SymbolicTransformerRegressor()
    score_map = {"a": None, "b": 0.2, "c": 0.9, "d": float("inf")}
    reg._safe_tree_metric = lambda tree, X, y, metric: score_map[tree]  # type: ignore[method-assign]

    candidates = [
        {"predicted_tree": "a"},
        {"predicted_tree": "b"},
        {"predicted_tree": "c"},
        {"predicted_tree": "d"},
    ]

    ordered = reg.order_candidates(None, None, candidates, metric="r2")

    assert [item["predicted_tree"] for item in ordered] == ["c", "b", "a", "d"]


def test_emit_progress_candidate_can_skip_metric_computation(tmp_path):
    module = _load_sklearn_wrapper_module()
    reg = module.SymbolicTransformerRegressor(progress_state_path=str(tmp_path / "state.json"))
    captured = {}

    class FakeTree:
        def infix(self):
            return "x_0 + x_1"

    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]
    reg._safe_tree_metric = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应计算指标"))  # type: ignore[method-assign]

    reg._emit_progress_candidate(
        0,
        {
            "predicted_tree": FakeTree(),
            "refinement_type": "ForwardRaw",
            "native_model_score": -0.25,
            "bag_index": 2,
            "candidate_rank": 1,
            "generation_source": "beam_search",
        },
        None,
        None,
        stage="forward_partial",
        compute_metrics=False,
    )

    assert captured["equation"] == "x_0 + x_1"
    assert captured["native_model_score"] == -0.25
    assert captured["score"] == -0.25
    assert captured["internal_objective"] == "decoder_length_normalized_log_likelihood"
    assert captured["objective_direction"] == "max"
    assert captured["bag_index"] == 2
    assert captured["candidate_rank"] == 1
    assert captured["stage"] == "forward_partial"
