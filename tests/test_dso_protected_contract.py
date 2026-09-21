import unittest

from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.algorithms.udsr_wrapper.wrapper import UDSRRegressor


class DSOProtectedContractTest(unittest.TestCase):
    def test_protected_operators_require_a_lossless_export(self):
        for wrapper in (DSORegressor, UDSRRegressor):
            self.assertIs(wrapper().params['task']['protected'], False)
            for params in ({'protected': True}, {'task': {'protected': True}}):
                with self.subTest(wrapper=wrapper.__name__, params=params):
                    with self.assertRaisesRegex(ValueError, 'protected=True'):
                        wrapper(**params)


if __name__ == '__main__':
    unittest.main()
