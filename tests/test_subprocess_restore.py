import numpy as np
import pytest

from scientific_intelligent_modelling.srkit import subprocess_runner as runner
from scientific_intelligent_modelling.benchmarks.artifact_schema import build_canonical_symbolic_program


class RestoredModel:
    _DESERIALIZE_WITHOUT_INIT = True

    def __init__(self):
        raise AssertionError('Must not construct an unused model before deserialization')

    @classmethod
    def deserialize(cls, payload):
        assert payload == 'saved'
        return cls.__new__(cls)

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def get_optimal_equation(self):
        return 'x0'

    def get_total_equations(self, n=None):
        return ['x0']

    def get_fitted_params(self):
        return []

    def get_total_equations_with_params(self, n=None):
        return [{'equation': 'x0', 'params': []}]

    def export_canonical_symbolic_program(self):
        return build_canonical_symbolic_program(tool_name='e2esr', raw_equation='x0')


@pytest.mark.parametrize('handler', [runner.handle_predict, runner.handle_get_optimal_equation,
                                    runner.handle_get_total_equations, runner.handle_get_fitted_params,
                                    runner.handle_get_total_equations_with_params,
                                    runner.handle_export_canonical_symbolic_program])
def test_restore_skips_throwaway_model_construction(handler):
    result = handler(RestoredModel, {'serialized_model': 'saved', 'data': {'X': [[1.0], [2.0]]}})
    assert result['success']


def test_instance_deserializer_is_still_supported():
    class Model:
        def deserialize(self, payload):
            self.coefficient = float(payload)
            return self

        def predict(self, X):
            return np.asarray(X)[:, 0] * self.coefficient

    assert runner.handle_predict(Model, {'serialized_model': '2', 'data': {'X': [[1.0]]}})['predictions'] == [2.0]


def test_non_opted_in_wrapper_keeps_constructor_setup():
    class Model:
        prepared = False

        def __init__(self):
            type(self).prepared = True

        @classmethod
        def deserialize(cls, payload):
            assert cls.prepared
            return cls.__new__(cls)

        def predict(self, X):
            return np.asarray(X)[:, 0]

    assert runner.handle_predict(Model, {'serialized_model': 'saved', 'data': {'X': [[1.0]]}})['predictions'] == [1.0]
