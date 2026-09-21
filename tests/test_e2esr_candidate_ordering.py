import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


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


class _FakeTree:
    def __init__(self, expression):
        self.expression = expression

    def infix(self):
        return self.expression

    def prefix(self):
        return self.expression

    def replace_node_value(self, old, new):
        self.expression = self.expression.replace(old, new)


class _RecordingScaler:
    def __init__(self, expression=None, error=None):
        self.expression = expression
        self.error = error
        self.calls = []

    def rescale_function(self, env, tree, a, b):
        self.calls.append((tree, list(a), list(b)))
        tree.expression = self.expression or tree.expression
        if self.error is not None:
            raise self.error
        return tree


def _configured_regressor(module, tmp_path, scaler, *, rescale=True, top_k_features=None):
    env = types.SimpleNamespace()
    model = types.SimpleNamespace(env=env)
    reg = module.SymbolicTransformerRegressor(
        model=model,
        rescale=rescale,
        progress_state_path=str(tmp_path / "state.json"),
    )
    reg.scalers = [scaler if rescale else None]
    reg.scale_params = [([0.5, 0.25], [-5.0, 1.5]) if rescale else None]
    reg.top_k_features = [top_k_features or [0, 1]]
    return reg


def test_emit_scaled_candidate_inverse_transforms_before_feature_relabel(tmp_path):
    module = _load_sklearn_wrapper_module()
    scaler = _RecordingScaler("0.5*x_0 - 5.0 + 0.25*x_1 + 1.5")
    reg = _configured_regressor(module, tmp_path, scaler, top_k_features=[2, 0])
    captured = {}
    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("x_0 + x_1")},
        None,
        None,
        stage="refine_final",
        compute_metrics=False,
        tree_is_scaled=True,
    )

    assert scaler.calls[0][1:] == ([0.5, 0.25], [-5.0, 1.5])
    assert captured["equation"] == "0.25*x_0 + 0.5*x_2 - 3.5"


def test_emit_candidate_with_rescale_disabled_only_relabels_features(tmp_path):
    module = _load_sklearn_wrapper_module()
    scaler = _RecordingScaler("wrong")
    reg = _configured_regressor(
        module,
        tmp_path,
        scaler,
        rescale=False,
        top_k_features=[3, 1],
    )
    captured = {}
    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("x_0 - x_1")},
        None,
        None,
        stage="noref_best",
        compute_metrics=False,
        tree_is_scaled=False,
    )

    assert captured["equation"] == "-x_1 + x_3"
    assert scaler.calls == []


def test_emit_final_candidate_does_not_inverse_transform_twice(tmp_path):
    module = _load_sklearn_wrapper_module()
    scaler = _RecordingScaler("wrong")
    reg = _configured_regressor(module, tmp_path, scaler, top_k_features=[2, 0])
    captured = {}
    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("0.5*x_0 - 5.0")},
        None,
        None,
        stage="fit_final",
        compute_metrics=False,
        tree_is_scaled=False,
    )

    assert captured["equation"] == "0.5*x_2 - 5.0"
    assert scaler.calls == []


def test_emit_scaled_candidate_does_not_mutate_candidate_tree(tmp_path):
    module = _load_sklearn_wrapper_module()
    scaler = _RecordingScaler("0.5*x_0 - 5.0")
    reg = _configured_regressor(module, tmp_path, scaler, top_k_features=[2, 0])
    reg._write_progress_state = lambda payload: None  # type: ignore[method-assign]
    tree = _FakeTree("x_0")

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": tree},
        None,
        None,
        stage="forward_partial",
        compute_metrics=False,
        tree_is_scaled=True,
    )

    assert tree.expression == "x_0"
    assert scaler.calls[0][0] is not tree


def test_emit_scaled_candidate_fails_closed_when_inverse_transform_fails(tmp_path):
    module = _load_sklearn_wrapper_module()
    scaler = _RecordingScaler(error=ValueError("cannot inverse transform"))
    reg = _configured_regressor(module, tmp_path, scaler)
    emitted = []
    reg.progress_callback = lambda **payload: emitted.append(payload)

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("x_0"), "native_model_score": -0.1},
        None,
        None,
        stage="forward_partial",
        compute_metrics=False,
        tree_is_scaled=True,
    )

    assert emitted == []


def test_fit_tracks_scaler_and_params_per_dataset_and_marks_final_unscaled(tmp_path):
    module = _load_sklearn_wrapper_module()

    class FitScaler:
        instances = []

        def __init__(self):
            self.index = len(self.instances)
            self.rescale_calls = []
            self.instances.append(self)

        def fit_transform(self, X):
            return X

        def get_params(self):
            return [self.index + 1.0], [-(self.index + 1.0)]

        def rescale_function(self, env, tree, a, b):
            self.rescale_calls.append((list(a), list(b)))
            result = _FakeTree(tree.expression)
            result.expression = f"{result.expression} + {self.index}"
            return result

    class FakeModel:
        def __init__(self):
            self.env = types.SimpleNamespace(
                params=types.SimpleNamespace(max_input_dimension=10)
            )
            self.last_generation_metadata = [[]]

        def __call__(self, inputs):
            return [[_FakeTree("x_0")]]

    module.utils_wrapper.StandardScaler = FitScaler
    module.get_top_k_features = lambda X, y, k: list(range(X.shape[1]))
    reg = module.SymbolicTransformerRegressor(
        model=FakeModel(),
        progress_state_path=str(tmp_path / "state.json"),
    )
    reg.refine = lambda dataset_idx, X, y, candidates, verbose: [  # type: ignore[method-assign]
        {"predicted_tree": candidates[0], "refinement_type": "NoRef"}
    ]
    emissions = []
    reg._emit_progress_candidate = (  # type: ignore[method-assign]
        lambda dataset_idx, candidate, X, y, *, stage, compute_metrics=True, tree_is_scaled: emissions.append(
            (dataset_idx, stage, tree_is_scaled, candidate["predicted_tree"].expression)
        )
    )

    reg.fit(
        [np.array([[10.0, 1.0], [12.0, 3.0]]), np.array([[20.0, 2.0], [24.0, 6.0]])],
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    )

    assert reg.scalers == FitScaler.instances
    assert reg.scale_params == [([1.0], [-1.0]), ([2.0], [-2.0])]
    assert FitScaler.instances[0].rescale_calls == [([1.0], [-1.0])]
    assert FitScaler.instances[1].rescale_calls == [([2.0], [-2.0])]
    assert [entry[2] for entry in emissions if entry[1] == "forward_partial"] == [True, True]
    assert [entry[2] for entry in emissions if entry[1] == "fit_final"] == [False, False]


def test_refine_marks_every_progress_candidate_as_scaled(tmp_path):
    module = _load_sklearn_wrapper_module()
    original_tree = _FakeTree("x_0")
    refined_tree = _FakeTree("2*x_0")

    class Refinement:
        def go(self, **kwargs):
            return refined_tree

    generator = types.SimpleNamespace(
        function_to_skeleton=lambda tree, constants_with_idx: (tree, [])
    )
    model = types.SimpleNamespace(env=types.SimpleNamespace(generator=generator))
    module.utils_wrapper.BFGSRefinement = Refinement
    reg = module.SymbolicTransformerRegressor(
        model=model,
        progress_state_path=str(tmp_path / "state.json"),
        rescale=True,
    )
    reg.start_fit = 0.0
    reg.order_candidates = lambda X, y, candidates, metric, verbose=False: candidates  # type: ignore[method-assign]
    reg._safe_tree_metric = (  # type: ignore[method-assign]
        lambda tree, X, y, metric: 1.0 if tree is refined_tree else 0.0
    )
    emissions = []
    reg._emit_progress_candidate = (  # type: ignore[method-assign]
        lambda dataset_idx, candidate, X, y, *, stage, compute_metrics=True, tree_is_scaled: emissions.append(
            (stage, tree_is_scaled)
        )
    )

    reg.refine(0, np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), [original_tree], verbose=False)

    assert emissions == [
        ("noref_best", True),
        ("bfgs_best", True),
        ("refine_final", True),
    ]
