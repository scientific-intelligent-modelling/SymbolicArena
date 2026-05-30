import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import sympy as sp

from scientific_intelligent_modelling.benchmarks import runner
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_external_infix_artifact


def _write_dataset(dataset_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata.yaml").write_text(
        """
dataset:
  description: progress snapshot demo
  target:
    name: target
    description: regression target
  features:
    - name: x0
      description: feature 0
    - name: x1
      description: feature 1
""".strip(),
        encoding="utf-8",
    )
    rows = {
        "train.csv": "x0,x1,target\n1,2,9\n2,1,8\n3,4,19\n",
        "valid.csv": "x0,x1,target\n4,5,24\n",
        "id_test.csv": "x0,x1,target\n5,6,29\n",
        "ood_test.csv": "x0,x1,target\n6,7,34\n",
    }
    for name, content in rows.items():
        (dataset_dir / name).write_text(content, encoding="utf-8")


def _equation_function() -> str:
    return (
        "def equation(x0: float, x1: float, params):\n"
        "    \"\"\"demo\"\"\"\n"
        "    return (\n"
        "        params[0]\n"
        "        + params[1] * x0\n"
        "        + params[2] * x1\n"
        "    )\n"
    )


class BenchmarkProgressSnapshotsTest(unittest.TestCase):
    def test_snapshot_capable_tools_default_to_one_minute_interval(self):
        for tool in runner._SNAPSHOT_CAPABLE_TOOLS:
            with self.subTest(tool=tool):
                params = {}
                interval = runner._resolve_progress_snapshot_interval_seconds(tool.lower(), params)
                self.assertEqual(interval, 60)
                self.assertNotIn("progress_snapshot_interval_seconds", params)

        self.assertIsNone(
            runner._resolve_progress_snapshot_interval_seconds("not_snapshot_capable", {})
        )

    def test_snapshot_interval_override_still_takes_precedence(self):
        params = {"progress_snapshot_interval_seconds": 30}
        interval = runner._resolve_progress_snapshot_interval_seconds("pysr", params)
        self.assertEqual(interval, 30)
        self.assertNotIn("progress_snapshot_interval_seconds", params)

    def test_build_periodic_snapshot_payload_for_llmsr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            (exp_dir / "samples").mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / "samples" / "top01_demo.json").write_text(
                json.dumps(
                    {
                        "iteration": 3,
                        "sample_order": 12,
                        "nmse": 0.01,
                        "mse": 0.01,
                        "function": _equation_function(),
                        "params": [1.0, 2.0, 3.0],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="llmsr",
                dataset=dataset,
                params={"niterations": 100},
                seed=1314,
                started_at=time.time() - 600,
                experiment_dir=exp_dir,
                checkpoint_index=1,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["record_type"], "periodic_best")
            self.assertEqual(payload["tool"], "llmsr")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["checkpoint_index"], 1)
            self.assertEqual(payload["source_iteration"], 3)
            self.assertEqual(payload["source_sample_order"], 12)
            self.assertIsNotNone(payload["canonical_artifact"])
            expr = sp.sympify(payload["canonical_artifact"]["instantiated_expression"])
            self.assertEqual(
                str(sp.expand(expr)),
                str(sp.expand(sp.sympify("2.0*x0 + 3.0*x1 + 1.0"))),
            )
            self.assertAlmostEqual(payload["train"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_pysr_from_hall_of_fame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n"
                "1,100.0,0.0\n"
                "3,0.0,2*x0 + 3*x1 + 1\n",
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="pysr",
                dataset=dataset,
                params={"niterations": 100},
                seed=1314,
                started_at=time.time() - 1200,
                experiment_dir=exp_dir,
                checkpoint_index=2,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "pysr")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], 0.0)
            self.assertEqual(payload["source_complexity"], 3)
            self.assertEqual(payload["elapsed_minutes"], 20)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_dso_from_hof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / "dso_dataset2d_1314_hof.csv").write_text(
                "r,on_policy_count,off_policy_count,expression,traversal\n"
                "-10.0,0,0,0.0,0.0\n"
                "1.0,1,0,2*x0 + 3*x1 + 1,2*x0 + 3*x1 + 1\n",
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="dso",
                dataset=dataset,
                params={"training": {"n_samples": 20}},
                seed=1314,
                started_at=time.time() - 1800,
                experiment_dir=exp_dir,
                checkpoint_index=3,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "dso")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_score"], 1.0)
            self.assertEqual(payload["elapsed_minutes"], 30)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_dso_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".dso_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x0 + 3*x1 + 1",
                        "score": 1.0,
                        "complexity": 3,
                        "iteration": 9,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="dso",
                dataset=dataset,
                params={"training": {"n_samples": 20}},
                seed=1314,
                started_at=time.time() - 1800,
                experiment_dir=exp_dir,
                checkpoint_index=3,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "dso")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_score"], 1.0)
            self.assertEqual(payload["source_iteration"], 9)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_pyoperon_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".pyoperon_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*X1 + 3*X2 + 1",
                        "loss": 0.0,
                        "complexity": 5,
                        "generation": 7,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="pyoperon",
                dataset=dataset,
                params={"generations": 100},
                seed=1314,
                started_at=time.time() - 2400,
                experiment_dir=exp_dir,
                checkpoint_index=4,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "pyoperon")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], 0.0)
            self.assertEqual(payload["source_complexity"], 5)
            self.assertEqual(payload["elapsed_minutes"], 40)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_gplearn_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".gplearn_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "add(add(mul(2.0, X0), mul(3.0, X1)), 1.0)",
                        "loss": 0.0,
                        "complexity": 7,
                        "generation": 4,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="gplearn",
                dataset=dataset,
                params={"generations": 50},
                seed=1314,
                started_at=time.time() - 3000,
                experiment_dir=exp_dir,
                checkpoint_index=5,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "gplearn")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], 0.0)
            self.assertEqual(payload["source_complexity"], 7)
            self.assertEqual(payload["elapsed_minutes"], 50)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2.0*x0 + 3.0*x1 + 1.0",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_e2esr_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".e2esr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2.0*x_0 + 3.0*x_1 + 1.0",
                        "loss": 0.0,
                        "score": 1.0,
                        "complexity": 7,
                        "refinement_type": "BFGS",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="e2esr",
                dataset=dataset,
                params={"n_trees_to_refine": 1},
                seed=1314,
                started_at=time.time() - 3600,
                experiment_dir=exp_dir,
                checkpoint_index=6,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "e2esr")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], 0.0)
            self.assertEqual(payload["source_score"], 1.0)
            self.assertEqual(payload["source_complexity"], 7)
            self.assertEqual(payload["elapsed_minutes"], 60)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2.0*x0 + 3.0*x1 + 1.0",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_imcts_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".imcts_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x0 + 3*x1 + 1",
                        "score": 1.0,
                        "evaluations": 42,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="iMCTS",
                dataset=dataset,
                params={"max_expressions": 100},
                seed=1314,
                started_at=time.time() - 4200,
                experiment_dir=exp_dir,
                checkpoint_index=7,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "iMCTS")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_score"], 1.0)
            self.assertEqual(payload["elapsed_minutes"], 70)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_tpsr_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".tpsr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x_0 + 3*x_1 + 1",
                        "score": 1.0,
                        "complexity": 5,
                        "source": "e2e_candidate",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="tpsr",
                dataset=dataset,
                params={"width": 1},
                seed=1314,
                started_at=time.time() - 4800,
                experiment_dir=exp_dir,
                checkpoint_index=8,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "tpsr")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_score"], 1.0)
            self.assertEqual(payload["source_complexity"], 5)
            self.assertEqual(payload["elapsed_minutes"], 80)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_qlattice_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".qlattice_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x0 + 3*x1 + 1",
                        "loss": -12.0,
                        "criterion": "bic",
                        "complexity": 5,
                        "epoch": 3,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="QLattice",
                dataset=dataset,
                params={"n_epochs": 10},
                seed=1314,
                started_at=time.time() - 5400,
                experiment_dir=exp_dir,
                checkpoint_index=9,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "QLattice")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], -12.0)
            self.assertEqual(payload["source_complexity"], 5)
            self.assertEqual(payload["elapsed_minutes"], 90)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_jaxsr_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / ".jaxsr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x0 + 3*x1 + 1",
                        "loss": 0.0,
                        "complexity": 5,
                        "iteration": 3,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="jaxsr",
                dataset=dataset,
                params={"max_terms": 5},
                seed=1314,
                started_at=time.time() - 600,
                experiment_dir=exp_dir,
                checkpoint_index=1,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "jaxsr")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], 0.0)
            self.assertEqual(payload["source_complexity"], 5)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_write_progress_payload_writes_outer_and_experiment_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outer"
            exp_dir = root / "exp"
            payload = {
                "record_type": "periodic_best",
                "tool": "drsr",
                "dataset": "demo",
                "status": "ok",
                "checkpoint_index": 2,
                "elapsed_seconds": 600.0,
                "elapsed_minutes": 10,
            }

            written = runner._write_progress_payload(
                payload,
                primary_dir=out_dir / "progress",
                experiment_dir=exp_dir,
            )

            self.assertEqual(len(written), 2)
            for path in written:
                self.assertEqual(path.name, "minute_0010.json")
                self.assertIn("progress", str(path))
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["checkpoint_index"], 2)

    def test_run_benchmark_task_writes_final_progress_for_non_snapshot_tool(self):
        class FakeSymbolicRegressor:
            def __init__(self, tool_name, problem_name=None, seed=1314, **params):
                del problem_name, seed
                self.tool_name = tool_name
                self.params = params
                self.experiment_dir = str(Path(params["exp_path"]) / params["exp_name"])
                Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

            def fit(self, X, y):
                self.X = X
                self.y = y
                return self

            def predict(self, X):
                arr = np.asarray(X, dtype=float)
                return 1.0 + 2.0 * arr[:, 0] + 3.0 * arr[:, 1]

            def get_optimal_equation(self):
                return "1 + 2*x0 + 3*x1"

            def get_total_equations(self):
                return [self.get_optimal_equation()]

            def export_canonical_symbolic_program(self):
                return normalize_external_infix_artifact(
                    self.get_optimal_equation(),
                    tool_name=self.tool_name,
                    expected_n_features=2,
                    shift_one_based=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            output_root = root / "out"
            _write_dataset(dataset_dir)

            old_symbolic_regressor = runner.SymbolicRegressor
            try:
                runner.SymbolicRegressor = FakeSymbolicRegressor
                result_path = runner.run_benchmark_task(
                    tool_name="symbolfit",
                    dataset_dir=dataset_dir,
                    output_root=output_root,
                    seed=520,
                    params_override={"progress_snapshot_interval_seconds": 60},
                )
            finally:
                runner.SymbolicRegressor = old_symbolic_regressor

            progress_files = sorted(result_path.parent.glob("progress/minute_*.json"))
            self.assertEqual(len(progress_files), 1)
            payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["record_type"], "final_best")
            self.assertEqual(payload["tool"], "symbolfit")
            self.assertEqual(payload["status"], "ok")
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
