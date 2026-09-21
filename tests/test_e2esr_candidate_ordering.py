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


def test_emit_progress_candidate_always_uses_training_mse(tmp_path):
    module = _load_sklearn_wrapper_module()
    reg = module.SymbolicTransformerRegressor(progress_state_path=str(tmp_path / "state.json"))
    captured = {}
    metric_calls = []

    class FakeTree:
        def infix(self):
            return "x_0 + x_1"

    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]
    reg._safe_tree_metric = (  # type: ignore[method-assign]
        lambda tree, X, y, metric: metric_calls.append((X, y, metric)) or 0.125
    )
    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 3.0])

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
        X,
        y,
        stage="forward_partial",
        compute_metrics=False,
    )

    assert metric_calls == [(X, y, "_mse")]
    assert captured["equation"] == "x_0 + x_1"
    assert captured["native_model_score"] == -0.25
    assert captured["score"] == 0.125
    assert captured["internal_loss"] == 0.125
    assert captured["training_mse"] == 0.125
    assert captured["internal_objective"] == "native_training_mse"
    assert captured["objective_direction"] == "min"
    assert captured["selection_policy"] == "e2esr_training_mse_v1"
    assert captured["bag_index"] == 2
    assert captured["candidate_rank"] == 1
    assert captured["stage"] == "forward_partial"


def test_emit_progress_candidate_rejects_non_finite_training_mse(tmp_path):
    module = _load_sklearn_wrapper_module()
    reg = module.SymbolicTransformerRegressor(progress_state_path=str(tmp_path / "state.json"))
    captured = {}
    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]
    reg._safe_tree_metric = lambda tree, X, y, metric: float("nan")  # type: ignore[method-assign]

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("x_0")},
        np.array([[0.0], [1.0]]),
        np.array([0.0, 1.0]),
        stage="forward_partial",
        compute_metrics=False,
    )

    assert captured == {}


class _FakeTree:
    def __init__(self, expression):
        self.expression = expression

    def infix(self):
        return self.expression

    def prefix(self):
        return self.expression

    def replace_node_value(self, old, new):
        self.expression = self.expression.replace(old, new)


def test_feature_relabel_is_simultaneous_for_swapped_columns():
    module = _load_sklearn_wrapper_module()
    tree = _FakeTree('x_0 + 2*x_1')
    converted = module.exchange_node_values(tree, {'x_0': 'x_1', 'x_1': 'x_0'})
    assert converted.infix() == 'x_1 + 2*x_0'
    assert tree.infix() == 'x_0 + 2*x_1'


def test_negative_mse_is_not_a_valid_training_score():
    module = _load_sklearn_wrapper_module()
    assert module.SymbolicTransformerRegressor._finite_mse(-1.0) is None


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
        {"predicted_tree": _FakeTree("x_0 + x_1"), "_mse": 0.1},
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
        {"predicted_tree": _FakeTree("x_0 - x_1"), "_mse": 0.1},
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
        {"predicted_tree": _FakeTree("0.5*x_0 - 5.0"), "_mse": 0.1},
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
        {"predicted_tree": tree, "_mse": 0.1},
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
        {"predicted_tree": _FakeTree("x_0"), "native_model_score": -0.1, "_mse": 0.1},
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
        {**candidates[0], "refinement_type": "NoRef", "_mse": 0.1}
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
        lambda tree, X, y, metric: (
            0.0 if tree is refined_tree else 1.0
        ) if metric == "_mse" else None
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


def test_refine_uses_mse_and_bfgs_without_decode_score_can_become_best(tmp_path):
    module = _load_sklearn_wrapper_module()
    raw_tree = _FakeTree("x_0")
    refined_tree = _FakeTree("2*x_0")

    class Refinement:
        def go(self, **kwargs):
            return refined_tree

    generator = types.SimpleNamespace(
        function_to_skeleton=lambda tree, constants_with_idx: (tree, [])
    )
    model = types.SimpleNamespace(env=types.SimpleNamespace(generator=generator))
    module.utils_wrapper.BFGSRefinement = Refinement
    emitted = []
    reg = module.SymbolicTransformerRegressor(
        model=model,
        progress_callback=lambda **payload: emitted.append(payload),
        rescale=False,
    )
    reg.start_fit = 0.0
    reg._safe_tree_metric = (  # type: ignore[method-assign]
        lambda tree, X, y, metric: {
            (raw_tree, "_mse"): 1.0,
            (refined_tree, "_mse"): 0.25,
        }.get((tree, metric))
    )

    candidates = reg.refine(
        0,
        np.array([[0.0], [1.0]]),
        np.array([2.0, 2.0]),
        [{"predicted_tree": raw_tree, "native_model_score": -0.4}],
        verbose=False,
    )

    assert candidates[0]["predicted_tree"] is refined_tree
    assert candidates[0]["_mse"] == 0.25
    bfgs_events = [payload for payload in emitted if payload["stage"] == "bfgs_best"]
    assert len(bfgs_events) == 1
    assert bfgs_events[0]["native_model_score"] is None
    assert bfgs_events[0]["training_mse"] == 0.25
    assert bfgs_events[0]["refinement_type"] == "BFGS"


def test_refine_does_not_start_bfgs_after_deadline_and_keeps_scored_raw_candidate():
    module = _load_sklearn_wrapper_module()
    raw_tree = _FakeTree("x_0")
    bfgs_calls = []

    class Refinement:
        def go(self, **kwargs):
            bfgs_calls.append(kwargs)
            return _FakeTree("2*x_0")

    generator = types.SimpleNamespace(
        function_to_skeleton=lambda tree, constants_with_idx: (tree, [])
    )
    model = types.SimpleNamespace(env=types.SimpleNamespace(generator=generator))
    module.utils_wrapper.BFGSRefinement = Refinement
    reg = module.SymbolicTransformerRegressor(model=model, rescale=False)
    reg.start_fit = 0.0
    reg._time_budget_exhausted = lambda: True  # type: ignore[method-assign]
    reg._safe_tree_metric = lambda tree, X, y, metric: 0.5 if metric == "_mse" else None  # type: ignore[method-assign]

    candidates = reg.refine(
        0,
        np.array([[0.0], [1.0]]),
        np.array([0.0, 1.0]),
        [raw_tree],
        verbose=False,
    )

    assert bfgs_calls == []
    assert len(candidates) == 1
    assert candidates[0]["predicted_tree"] is raw_tree
    assert candidates[0]["_mse"] == 0.5


def test_time_budget_honors_finite_bags_refines_early_and_keeps_global_best():
    module = _load_sklearn_wrapper_module()
    events = []

    class FakeModel:
        def __init__(self):
            self.env = types.SimpleNamespace(
                params=types.SimpleNamespace(max_input_dimension=10)
            )
            self.last_generation_metadata = [[]]
            self.forward_count = 0

        def __call__(self, inputs):
            self.forward_count += 1
            events.append(("forward", self.forward_count))
            return [[_FakeTree("x_0" if self.forward_count == 1 else "2*x_0")]]

    model = FakeModel()
    reg = module.SymbolicTransformerRegressor(
        model=model,
        max_input_points=10,
        max_number_bags=2,
        timeout_in_seconds=180,
        rescale=False,
    )
    module.get_top_k_features = lambda X, y, k: list(range(X.shape[1]))
    reg._time_budget_exhausted = lambda: model.forward_count >= 2  # type: ignore[method-assign]

    def fake_refine(dataset_idx, X, y, candidates, verbose):
        events.append(("refine", model.forward_count))
        candidate = dict(candidates[0])
        candidate["refinement_type"] = "NoRef"
        candidate["_mse"] = 0.1 if model.forward_count == 1 else 1.0
        return [candidate]

    reg.refine = fake_refine  # type: ignore[method-assign]
    reg._emit_progress_candidate = lambda *args, **kwargs: None  # type: ignore[method-assign]

    reg.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    assert events == [
        ("forward", 1),
        ("refine", 1),
        ("forward", 2),
        ("refine", 2),
    ]
    assert model.forward_count == 2
    assert reg.tree[0][0]["predicted_tree"].expression == "x_0"
    assert reg.tree[0][0]["_mse"] == 0.1


def test_fit_without_progress_path_still_produces_valid_final_candidate():
    module = _load_sklearn_wrapper_module()
    raw_tree = _FakeTree("x_0")

    class FakeModel:
        def __init__(self):
            generator = types.SimpleNamespace(
                function_to_skeleton=lambda tree, constants_with_idx: (tree, [])
            )
            self.env = types.SimpleNamespace(
                params=types.SimpleNamespace(max_input_dimension=10),
                generator=generator,
            )
            self.last_generation_metadata = [[]]

        def __call__(self, inputs):
            return [[raw_tree] for _ in inputs]

    class NoOpRefinement:
        def go(self, **kwargs):
            return None

    module.utils_wrapper.BFGSRefinement = NoOpRefinement
    module.get_top_k_features = lambda X, y, k: list(range(X.shape[1]))
    reg = module.SymbolicTransformerRegressor(model=FakeModel(), rescale=False)
    reg._safe_tree_metric = lambda tree, X, y, metric: 0.2 if metric == "_mse" else None  # type: ignore[method-assign]

    reg.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    assert reg.tree[0][0]["predicted_tree"] is raw_tree
    assert reg.tree[0][0]["_mse"] == 0.2


def test_worse_progress_candidate_does_not_overwrite_best_mse(tmp_path):
    module = _load_sklearn_wrapper_module()
    reg = module.SymbolicTransformerRegressor(progress_state_path=str(tmp_path / "state.json"))
    captured = {}
    reg._write_progress_state = lambda payload: captured.update(payload)  # type: ignore[method-assign]

    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("x_0"), "_mse": 0.1},
        None,
        None,
        stage="noref_best",
    )
    reg._emit_progress_candidate(
        0,
        {"predicted_tree": _FakeTree("2*x_0"), "_mse": 1.0},
        None,
        None,
        stage="refine_final",
    )

    assert captured["equation"] == "x_0"
    assert captured["training_mse"] == 0.1
    assert captured["stage"] == "noref_best"
