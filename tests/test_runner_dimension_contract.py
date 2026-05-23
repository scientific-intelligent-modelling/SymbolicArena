import unittest

import numpy as np

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor
from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor
from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor
from scientific_intelligent_modelling.algorithms.jaxsr_wrapper.wrapper import JAXSRRegressor
from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import LLMSRRegressor
from scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper import OperonRegressor
from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor
from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor


class RunnerDimensionContractTest(unittest.TestCase):
    def test_strict_wrappers_accept_runner_dataset_contract_meta(self):
        shared = {
            "exp_path": "/tmp/bench",
            "exp_name": "demo",
            "seed": 1314,
            "n_features": 2,
            "feature_names": ["x0", "x1"],
            "target_name": "y",
        }

        pysr = PySRRegressor(**shared)
        self.assertNotIn("n_features", pysr.params)
        self.assertNotIn("feature_names", pysr.params)
        self.assertNotIn("target_name", pysr.params)

        gplearn = GPLearnRegressor(**shared)
        self.assertNotIn("n_features", gplearn.params)
        self.assertNotIn("feature_names", gplearn.params)
        self.assertNotIn("target_name", gplearn.params)

        operon = OperonRegressor(**shared)
        self.assertNotIn("n_features", operon.params)
        self.assertNotIn("feature_names", operon.params)
        self.assertNotIn("target_name", operon.params)

        dso = DSORegressor(**shared)
        self.assertNotIn("n_features", dso.params)
        self.assertNotIn("feature_names", dso.params)
        self.assertNotIn("target_name", dso.params)

        fepysr = FePySRRegressor(**shared)
        self.assertNotIn("n_features", fepysr.params)
        self.assertNotIn("feature_names", fepysr.params)
        self.assertNotIn("target_name", fepysr.params)

        jaxsr = JAXSRRegressor(**shared)
        self.assertNotIn("n_features", jaxsr.params)
        self.assertNotIn("feature_names", jaxsr.params)
        self.assertNotIn("target_name", jaxsr.params)

        symbolfit = SymbolFitRegressor(**shared)
        self.assertNotIn("n_features", symbolfit.params)
        self.assertNotIn("feature_names", symbolfit.params)
        self.assertNotIn("target_name", symbolfit.params)

    def test_llm_wrappers_capture_runner_dataset_contract_meta(self):
        shared = {
            "exp_path": "/tmp/bench",
            "exp_name": "demo",
            "seed": 1314,
            "n_features": 3,
            "feature_names": ["t", "A", "B"],
            "target_name": "target",
            "prompt_feature_names": ["x0", "x1", "x2"],
            "prompt_target_name": "y",
        }

        llmsr = LLMSRRegressor(**shared)
        self.assertEqual(llmsr._n_features, 3)
        self.assertEqual(llmsr._feature_names, ["t", "A", "B"])
        self.assertEqual(llmsr._target_name, "target")
        self.assertEqual(llmsr._resolve_prompt_columns(3), (["x0", "x1", "x2"], "y"))

        drsr = DRSRRegressor(**shared)
        self.assertEqual(drsr._n_features, 3)
        self.assertEqual(drsr._feature_names, ["t", "A", "B"])
        self.assertEqual(drsr._target_name, "target")
        names, _, _ = drsr._resolve_prompt_semantics(3)
        self.assertEqual(names, ["x0", "x1", "x2"])
        self.assertEqual(drsr._resolve_prompt_target_name(), "y")

    def test_explicit_contract_rejects_n_features_mismatch(self):
        X = np.ones((4, 2))
        y = np.ones(4)
        reg = LLMSRRegressor(
            exp_path="/tmp/bench",
            exp_name="demo",
            n_features=3,
            feature_names=["x0", "x1", "x2"],
            target_name="y",
        )
        with self.assertRaisesRegex(ValueError, "显式维度契约不一致"):
            reg.fit(X, y)

    def test_explicit_contract_rejects_feature_name_length_mismatch(self):
        X = np.ones((4, 2))
        y = np.ones(4)
        reg = DRSRRegressor(
            exp_path="/tmp/bench",
            exp_name="demo",
            n_features=2,
            feature_names=["x0"],
            target_name="y",
        )
        with self.assertRaisesRegex(ValueError, "feature_names 长度与输入维度不一致"):
            reg.fit(X, y)


if __name__ == "__main__":
    unittest.main()
