from pathlib import Path

import numpy as np
import pytest
import torch
from deap import gp

from scientific_intelligent_modelling.algorithms.dgp_wrapper.artifacts import normalize_dgp_artifact
from scientific_intelligent_modelling.algorithms.dgp_wrapper.wrapper import DGPRegressor, native_modules, primitive_set
from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact
from scientific_intelligent_modelling.onboarding.acceptance import validate_integration


def test_protected_native_operators_and_zero_based_variables():
    X = np.array([[0.0, -1.0, 2.0], [0.00001, 0.0, 3.0], [-0.00001, 2.0, -4.0], [2.0, 20.0, 0.25]])
    pset = primitive_set(3)
    for raw in ("add(div(x2, x0), log(x1))", "exp(x1)", "mul(x2, 1.2345678912345678)", "1.2345678912345678"):
        tree = gp.PrimitiveTree.from_string(raw, pset)
        native = np.broadcast_to(np.asarray(gp.compile(tree, pset)(*X.T), dtype=float), (len(X),))
        artifact = normalize_dgp_artifact(raw, expected_n_features=3)
        replay = _predict_from_canonical_artifact(artifact, X)
        np.testing.assert_allclose(replay, native, rtol=1e-13, atol=1e-13)
        assert replay.shape == (len(X),)
    with pytest.raises(ValueError):
        normalize_dgp_artifact("add(x3, x0)", expected_n_features=3)


def test_gradient_through_native_log_and_exp():
    _, _, _, operations = native_modules()
    for operation in (operations.Log(), operations.Exp()):
        values = torch.tensor([1.25, 2.0], requires_grad=True)
        operation.forward(values, None).sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) == 2


def test_explicit_dimension_validation():
    regressor = DGPRegressor(n_features=3, feature_names=["x0", "x1", "x2"], target_name="y", epochs=1)
    with pytest.raises(ValueError, match="n_features"):
        regressor.fit(np.ones((4, 2)), np.arange(4, dtype=float))


def test_source_and_registry_acceptance():
    root = Path(__file__).resolve().parents[1]
    result = validate_integration(root / "tools/sr_onboarder/manifests/dgp.json")
    assert result["passed"]
    assert result["source_revision"] == "dc8a8634363399270291576ec575c379dff1f9bf"
