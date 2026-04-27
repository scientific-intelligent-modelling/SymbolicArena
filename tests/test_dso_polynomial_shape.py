import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("tensorflow")


DSO_SRC = (
    Path(__file__).resolve().parents[1]
    / "scientific_intelligent_modelling"
    / "algorithms"
    / "dso_wrapper"
    / "dso"
    / "dso"
)
if str(DSO_SRC) not in sys.path:
    sys.path.insert(0, str(DSO_SRC))

from dso.library import Polynomial


def test_polynomial_eval_poly_returns_one_dimensional_output_for_column_coef():
    X = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    poly = Polynomial(
        exponents=[(1, 0, 0), (0, 1, 0)],
        coef=np.asarray([[2.0], [3.0]]),
    )

    y = poly.eval_poly(X)

    assert y.shape == (3,)
    assert np.multiply(y, X[:, 0]).shape == (3,)


def test_polynomial_eval_poly_zero_terms_returns_one_dimensional_output():
    X = np.ones((4, 2))
    poly = Polynomial(exponents=[], coef=np.ones(0))

    y = poly.eval_poly(X)

    assert y.shape == (4,)
