from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from jaxsr import SymbolicRegressor

from scientific_intelligent_modelling.algorithms.jaxsr_wrapper.wrapper import (
    JAXSRRegressor,
)
from scientific_intelligent_modelling.srkit import subprocess_runner


class JAXSRNativeRestoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work_dir = Path(".agent/work/FIX-002")
        work_dir.mkdir(parents=True, exist_ok=True)
        cls._temporary = tempfile.TemporaryDirectory(
            prefix="jaxsr_native_restore_",
            dir=work_dir,
        )
        root = Path(cls._temporary.name)
        cls.experiment_dir = root / "experiment"
        cls.X = np.linspace(-2.0, 2.0, 24, dtype=float).reshape(-1, 1)
        cls.y = 1.5 + 2.25 * cls.X[:, 0]
        started_at = time.monotonic()
        cls.trained = JAXSRRegressor(
            seed=520,
            n_features=1,
            feature_names=["input_feature"],
            target_name="target",
            max_terms=2,
            max_polynomial_degree=1,
            max_interaction_order=1,
            cv_folds=2,
            exp_path=str(root),
            exp_name=cls.experiment_dir.name,
        ).fit(cls.X, cls.y)
        if time.monotonic() - started_at >= 60:
            raise AssertionError("真实 JAXSR 测试训练超过 60 秒")

        cls.state_path = cls.experiment_dir / ".jaxsr_current_best.json"
        cls.snapshot = json.loads(cls.state_path.read_text(encoding="utf-8"))
        native = SymbolicRegressor._from_dict(cls.snapshot["model_state"])
        np.testing.assert_allclose(
            native.predict(cls.X),
            cls.trained.predict(cls.X),
            rtol=0.0,
            atol=0.0,
        )
        if cls.snapshot["fidelity"]["status"] != "verified":
            raise AssertionError("真实 JAXSR 快照未通过 fidelity 验证")

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def tearDown(self):
        self.state_path.write_text(json.dumps(self.snapshot), encoding="utf-8")

    def test_timeout_recovery_restores_native_model_and_contract(self):
        recovered = JAXSRRegressor(existing_exp_dir=str(self.experiment_dir))
        self.assertIsNotNone(recovered.model)
        self.assertEqual(recovered.get_optimal_equation(), self.snapshot["equation"])
        self.assertEqual(recovered._fidelity_evidence, self.snapshot["fidelity"])
        self.assertEqual(recovered._contract_n_features, 1)
        self.assertEqual(recovered._contract_feature_names, ["input_feature"])
        self.assertEqual(recovered._contract_target_name, "target")
        np.testing.assert_allclose(
            recovered.predict(self.X),
            self.trained.predict(self.X),
            rtol=0.0,
            atol=0.0,
        )

        result = subprocess_runner.handle_recover_from_timeout(
            JAXSRRegressor,
            {
                "tool_name": "jaxsr",
                "experiment_dir": str(self.experiment_dir),
                "params": {
                    "n_features": 1,
                    "feature_names": ["input_feature"],
                    "target_name": "target",
                },
                "data": {"X": self.X.tolist(), "y": self.y.tolist()},
            },
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["recovered_from_timeout"])
        self.assertEqual(result["equation"], self.snapshot["equation"])
        roundtrip = JAXSRRegressor.deserialize(result["serialized_model"])
        np.testing.assert_allclose(
            roundtrip.predict(self.X),
            self.trained.predict(self.X),
            rtol=0.0,
            atol=0.0,
        )

    def test_timeout_recovery_rejects_unverified_or_tampered_evidence(self):
        mutations = [
            lambda item: item["fidelity"].update(status="pending"),
            lambda item: item["fidelity"].update(status="failed"),
            lambda item: item["fidelity"].update(model_state_sha256="0" * 64),
            lambda item: item["fidelity"].update(equation_sha256="0" * 64),
            lambda item: item["fidelity"].update(probe_input_sha256="0" * 64),
            lambda item: item["fidelity"].update(native_prediction_sha256="0" * 64),
            lambda item: item["fidelity"].update(replay_prediction_sha256="0" * 64),
        ]
        for mutate in mutations:
            tampered = deepcopy(self.snapshot)
            mutate(tampered)
            self.state_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fidelity|哈希"):
                JAXSRRegressor(exp_dir=str(self.experiment_dir))


if __name__ == "__main__":
    unittest.main()
