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


def _jaxsr_fidelity(status: str = "verified") -> dict:
    return {
        "version": "jaxsr_export_fidelity_v1",
        "status": status,
        "reason": None if status == "verified" else "native predict 无忠实表达式",
        "equation_sha256": "a" * 64,
        "probe_input_sha256": "b" * 64,
        "native_prediction_sha256": "c" * 64,
        "replay_prediction_sha256": "d" * 64 if status == "verified" else None,
        "model_state_sha256": "e" * 64,
        "max_abs_error": 0.0 if status == "verified" else 1.0,
        "allowed_max_abs_error": 1e-6,
        "probe_definition": {
            "version": "jaxsr_export_fidelity_v1",
            "strategy": "fit_X_stratified_rows_plus_quantile_anchors",
            "values": [[1.0, 2.0]],
        },
    }


class BenchmarkProgressSnapshotsTest(unittest.TestCase):
    def test_snapshot_capable_tools_default_to_one_minute_interval(self):
        self.assertIn("symbolfit", runner._SNAPSHOT_CAPABLE_TOOL_KEYS)
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

    def test_write_progress_payload_can_use_fixed_checkpoint_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "tool": "ragsr",
                "elapsed_seconds": 125.0,
                "record_type": "periodic_best",
            }

            written = runner._write_progress_payload(
                payload,
                primary_dir=root / "progress",
                snapshot_minute_index=1,
            )

            self.assertEqual(written, [root / "progress" / "minute_0001.json"])
            self.assertTrue((root / "progress" / "minute_0001.json").exists())
            self.assertFalse((root / "progress" / "minute_0002.json").exists())

    def test_final_progress_payload_uses_budget_minute_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "exp"
            fidelity = _jaxsr_fidelity()
            result = {
                "status": "ok",
                "equation": "x0 + 1",
                "canonical_artifact": {
                    "normalized_expression": "x0 + 1",
                    "fidelity_check": fidelity,
                },
                "seconds": 10845.3,
                "params": {"timeout_in_seconds": 10800},
            }

            runner._write_final_progress_payload_if_requested(
                result=result,
                progress_snapshot_interval_seconds=60,
                output_dir=root / "out",
                experiment_dir=exp_dir,
            )

            self.assertTrue((root / "out" / "progress" / "minute_0180.json").exists())
            self.assertTrue((exp_dir / "progress" / "minute_0180.json").exists())
            self.assertFalse((root / "out" / "progress" / "minute_0181.json").exists())
            payload = json.loads(
                (root / "out" / "progress" / "minute_0180.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["record_type"], "final_best")
            self.assertEqual(payload["canonical_artifact"]["fidelity_check"], fidelity)

    def test_final_progress_prefers_budget_end_internal_best_for_snapshot_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "exp"
            result = {
                "status": "ok",
                "equation": "x0 + 99",
                "canonical_artifact": {"normalized_expression": "x0 + 99"},
                "seconds": 10801.0,
                "params": {"timeout_in_seconds": 10800},
            }
            internal = {
                "record_type": "periodic_best",
                "checkpoint_index": 180,
                "status": "ok",
                "equation": "x0 + 1",
                "canonical_artifact": {"normalized_expression": "x0 + 1"},
                "source_internal_loss": 0.125,
            }
            calls = []

            def fake_build(**kwargs):
                calls.append(kwargs)
                return dict(internal)

            old_build = runner._build_periodic_snapshot_payload
            try:
                runner._build_periodic_snapshot_payload = fake_build
                runner._write_final_progress_payload_if_requested(
                    result=result,
                    progress_snapshot_interval_seconds=60,
                    output_dir=root / "out",
                    experiment_dir=exp_dir,
                    tool_name="symbolfit",
                    dataset=object(),
                    params={"timeout_in_seconds": 10800},
                    seed=520,
                    started_at=1.0,
                )
            finally:
                runner._build_periodic_snapshot_payload = old_build

            payload = json.loads(
                (root / "out" / "progress" / "minute_0180.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["record_type"], "budget_end_internal_best")
            self.assertEqual(payload["checkpoint_index"], 180)
            self.assertEqual(payload["equation"], "x0 + 1")
            self.assertEqual(payload["source_internal_loss"], 0.125)
            self.assertEqual(len(calls), 1)

    def test_periodic_snapshot_loop_schedules_against_absolute_targets(self):
        class FakeStopEvent:
            def __init__(self):
                self.now = 100.0
                self.waits = []
                self.write_count = 0

            def wait(self, seconds):
                if self.write_count >= 2:
                    return True
                self.waits.append(round(seconds, 3))
                self.now += max(0.0, float(seconds))
                return False

        fake_stop = FakeStopEvent()
        captured_indexes = []

        def fake_time():
            return fake_stop.now

        def fake_build_payload(**kwargs):
            fake_stop.now += 5.0
            return {
                "tool": kwargs["tool_name"],
                "elapsed_seconds": fake_stop.now - kwargs["started_at"],
                "checkpoint_index": kwargs["checkpoint_index"],
            }

        def fake_write_payload(payload, *, primary_dir, experiment_dir=None, snapshot_minute_index=None):
            captured_indexes.append(snapshot_minute_index)
            fake_stop.write_count += 1
            return [Path(primary_dir) / f"minute_{snapshot_minute_index:04d}.json"]

        old_time = runner.time.time
        old_build = runner._build_periodic_snapshot_payload
        old_write = runner._write_progress_payload
        try:
            runner.time.time = fake_time
            runner._build_periodic_snapshot_payload = fake_build_payload
            runner._write_progress_payload = fake_write_payload

            runner._periodic_snapshot_loop(
                stop_event=fake_stop,
                interval_seconds=60,
                tool_name="ragsr",
                dataset=None,
                params={},
                seed=1,
                started_at=100.0,
                output_dir=Path("/tmp/out"),
                experiment_dir=Path("/tmp/exp"),
            )
        finally:
            runner.time.time = old_time
            runner._build_periodic_snapshot_payload = old_build
            runner._write_progress_payload = old_write

        self.assertEqual(captured_indexes, [1, 2])
        self.assertEqual(fake_stop.waits, [60.0, 55.0])

    def test_periodic_snapshot_loop_backfills_overdue_minutes(self):
        class FakeStopEvent:
            def __init__(self):
                self.now = 100.0
                self.write_count = 0

            def wait(self, seconds):
                if self.write_count >= 2:
                    return True
                self.now += max(0.0, float(seconds))
                return False

        fake_stop = FakeStopEvent()
        writes = []
        train_label_noise = {
            "enabled": True,
            "requested": True,
            "sigma": 0.01,
            "y_std": 2.0,
            "scale": 0.02,
            "rng_seed": 123,
            "protocol": "frozen-noise-protocol",
        }

        def fake_time():
            return fake_stop.now

        def fake_build_payload(**kwargs):
            fake_stop.now += 125.0
            return {
                "tool": kwargs["tool_name"],
                "elapsed_seconds": fake_stop.now - kwargs["started_at"],
                "checkpoint_index": kwargs["checkpoint_index"],
                "record_type": "periodic_best",
            }

        def fake_write_payload(payload, *, primary_dir, experiment_dir=None, snapshot_minute_index=None):
            writes.append(
                (
                    snapshot_minute_index,
                    payload["record_type"],
                    payload.get("backfilled_from_minute"),
                    payload.get("condition"),
                    payload.get("train_label_noise"),
                )
            )
            fake_stop.write_count += 1
            return [Path(primary_dir) / f"minute_{snapshot_minute_index:04d}.json"]

        old_time = runner.time.time
        old_build = runner._build_periodic_snapshot_payload
        old_write = runner._write_progress_payload
        try:
            runner.time.time = fake_time
            runner._build_periodic_snapshot_payload = fake_build_payload
            runner._write_progress_payload = fake_write_payload

            runner._periodic_snapshot_loop(
                stop_event=fake_stop,
                interval_seconds=60,
                tool_name="ragsr",
                dataset=None,
                params={},
                seed=1,
                started_at=100.0,
                output_dir=Path("/tmp/out"),
                experiment_dir=Path("/tmp/exp"),
                train_label_noise=train_label_noise,
            )
        finally:
            runner.time.time = old_time
            runner._build_periodic_snapshot_payload = old_build
            runner._write_progress_payload = old_write

        self.assertEqual(
            [(minute, record_type, source, condition) for minute, record_type, source, condition, _ in writes],
            [
                (1, "periodic_backfill", 3, "noise001"),
                (2, "periodic_backfill", 3, "noise001"),
                (3, "periodic_best", None, "noise001"),
            ],
        )
        self.assertTrue(all(evidence == train_label_noise for *_, evidence in writes))
        self.assertTrue(all(evidence is not train_label_noise for *_, evidence in writes))

    def test_periodic_heartbeat_records_clean_condition_and_frozen_noise_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            train_label_noise = {
                "enabled": False,
                "requested": False,
                "sigma": 0.0,
                "y_std": 3.0,
                "scale": 0.0,
                "rng_seed": 456,
                "protocol": "frozen-clean-protocol",
            }

            payload = runner._build_periodic_snapshot_payload(
                tool_name="fepysr",
                dataset=runner.load_canonical_dataset(dataset_dir),
                params={"timeout_in_seconds": 180},
                seed=520,
                started_at=time.time() - 60,
                experiment_dir=exp_dir,
                checkpoint_index=1,
                train_label_noise=train_label_noise,
            )

            self.assertEqual(payload["record_type"], "periodic_heartbeat")
            self.assertEqual(payload["condition"], "clean")
            self.assertEqual(payload["train_label_noise"], train_label_noise)
            self.assertIsNot(payload["train_label_noise"], train_label_noise)

    def test_budget_end_and_final_best_record_noise_condition(self):
        noise = {
            "enabled": True,
            "requested": True,
            "sigma": 0.05,
            "y_std": 2.0,
            "scale": 0.1,
            "rng_seed": 789,
            "protocol": "frozen-noise-protocol",
        }
        for snapshot_tool in (False, True):
            with self.subTest(snapshot_tool=snapshot_tool), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                exp_dir = root / "exp"
                result = {
                    "status": "ok",
                    "equation": "x0 + 1",
                    "canonical_artifact": {"normalized_expression": "x0 + 1"},
                    "seconds": 180.0,
                    "params": {"timeout_in_seconds": 180},
                }
                old_build = runner._build_periodic_snapshot_payload
                try:
                    if snapshot_tool:
                        runner._build_periodic_snapshot_payload = lambda **_: {
                            "record_type": "periodic_best",
                            "checkpoint_index": 3,
                            "status": "ok",
                            "equation": "x0 + 1",
                            "canonical_artifact": {"normalized_expression": "x0 + 1"},
                            "source_internal_loss": 0.1,
                        }
                    runner._write_final_progress_payload_if_requested(
                        result=result,
                        progress_snapshot_interval_seconds=60,
                        output_dir=root / "out",
                        experiment_dir=exp_dir,
                        tool_name="symbolfit" if snapshot_tool else None,
                        dataset=object() if snapshot_tool else None,
                        params={"timeout_in_seconds": 180} if snapshot_tool else None,
                        seed=520 if snapshot_tool else None,
                        started_at=time.time() - 180 if snapshot_tool else None,
                        train_label_noise=noise,
                    )
                finally:
                    runner._build_periodic_snapshot_payload = old_build

                payload = json.loads(
                    (root / "out" / "progress" / "minute_0003.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    payload["record_type"],
                    "budget_end_internal_best" if snapshot_tool else "final_best",
                )
                self.assertEqual(payload["condition"], "noise005")
                self.assertEqual(payload["train_label_noise"], noise)

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

    def test_pysr_native_hof_incumbent_uses_finite_loss_and_keeps_tie_earliest(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            hof = exp_dir / "hall_of_fame.csv"
            hof.write_text(
                "Complexity,Loss,Equation,Iteration\n"
                "1,Inf,x0,1\n"
                "3,0.5,x0 + 1,7\n",
                encoding="utf-8",
            )
            first = runner._extract_pysr_periodic_candidate(
                exp_dir, snapshot_minute=2, snapshot_elapsed_seconds=120.0
            )
            self.assertEqual(first["equation"], "x0 + 1")
            self.assertEqual(first["internal_objective"], "hof_loss")
            self.assertEqual(first["objective_direction"], "min")
            self.assertEqual(first["iteration"], 7)
            self.assertEqual(first["first_discovered_minute"], 2)

            # loss 改善必须更新，即使独立 ID/OOD 质量可能下降。
            hof.write_text(
                "Complexity,Loss,Equation,Iteration\n3,0.4,x0 - 100,9\n",
                encoding="utf-8",
            )
            improved = runner._extract_pysr_periodic_candidate(
                exp_dir, snapshot_minute=3, snapshot_elapsed_seconds=180.0
            )
            self.assertEqual(improved["equation"], "x0 - 100")
            self.assertEqual(improved["first_discovered_minute"], 3)

            hof.write_text(
                "Complexity,Loss,Equation,Iteration\n3,0.4,x0 + 999,10\n",
                encoding="utf-8",
            )
            tied = runner._extract_pysr_periodic_candidate(
                exp_dir, snapshot_minute=4, snapshot_elapsed_seconds=240.0
            )
            self.assertEqual(tied["equation"], "x0 - 100")
            self.assertEqual(tied["first_discovered_minute"], 3)

    def test_pysr_invalid_objective_cannot_create_future_final_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n1,Inf,x0\n1,NaN,x1\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                runner._extract_pysr_periodic_candidate(
                    exp_dir, snapshot_minute=180, snapshot_elapsed_seconds=10800.0
                )
            )
            self.assertFalse((exp_dir / ".pysr_native_incumbent.json").exists())

    def test_native_heartbeat_marks_objective_unavailable_without_final_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="pysr",
                dataset=runner.load_canonical_dataset(dataset_dir),
                params={"niterations": 100},
                seed=520,
                started_at=time.time() - 60,
                experiment_dir=exp_dir,
                checkpoint_index=1,
                task_label="g0001_dataset2d",
                task_global_index=1,
            )
            self.assertEqual(payload["record_type"], "periodic_heartbeat")
            self.assertFalse(payload["algorithm_native_incumbent"])
            self.assertTrue(payload["native_objective_unavailable"])
            self.assertEqual(payload["internal_objective"], "hof_loss")
            self.assertIsNone(payload["equation"])

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
                        "native_model_score": 1.0,
                        "internal_objective": "decoder_length_normalized_log_likelihood",
                        "objective_direction": "max",
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
            self.assertTrue(payload["algorithm_native_incumbent"])
            self.assertEqual(
                payload["internal_objective"],
                "decoder_length_normalized_log_likelihood",
            )
            self.assertEqual(payload["internal_objective_direction"], "max")
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
                        "equation": "2*x[0] + 3*x[1] + 1",
                        "score": 1.0,
                        "evaluations": 42,
                        "first_discovered_minute": 3,
                        "first_discovered_elapsed_seconds": 125.0,
                        "source": "imcts_native_reward",
                        "internal_objective": "native_reward",
                        "objective_direction": "max",
                        "expression_vector": "2*x[0] + 3*x[1] + 1",
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
            self.assertEqual(payload["candidate_first_discovered_minute"], 3)
            self.assertEqual(payload["candidate_source"], "imcts_native_reward")
            self.assertTrue(payload["algorithm_native_incumbent"])
            self.assertEqual(payload["internal_objective"], "native_reward")
            self.assertEqual(payload["internal_objective_direction"], "max")
            self.assertEqual(payload["expression_vector"], "2*x[0] + 3*x[1] + 1")
            self.assertEqual(payload["elapsed_minutes"], 70)
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_imcts_timeout_recovery_preserves_internal_discovery_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            (exp_dir / ".imcts_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x[0] + 3*x[1] + 1",
                        "score": 0.95,
                        "evaluations": 42,
                        "first_discovered_minute": 2,
                        "first_discovered_elapsed_seconds": 93.0,
                        "source": "imcts_native_reward",
                    }
                ),
                encoding="utf-8",
            )

            recovered = runner._recover_timeout_payload_from_candidate(
                tool_name="iMCTS",
                dataset=runner.load_canonical_dataset(dataset_dir),
                experiment_dir=exp_dir,
            )

            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["source_score"], 0.95)
            self.assertEqual(recovered["candidate_first_discovered_minute"], 2)
            self.assertEqual(recovered["candidate_source"], "imcts_native_reward")

    def test_imcts_timeout_snapshot_fallback_preserves_first_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            progress_dir = exp_dir / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            dataset = runner.load_canonical_dataset(dataset_dir)
            valid_equation = "2*x[0] + 3*x[1] + 1"
            artifact, error = runner.safe_build_canonical_artifact(
                tool_name="iMCTS",
                equation=valid_equation,
                expected_n_features=2,
            )
            self.assertIsNone(error)
            metrics = {"rmse": 0.0, "r2": 1.0, "nmse": 0.0, "acc_0_1": 1.0}
            (progress_dir / "minute_0002.json").write_text(
                json.dumps(
                    {
                        "tool": "iMCTS",
                        "equation": valid_equation,
                        "canonical_artifact": artifact,
                        "valid": metrics,
                        "id_test": metrics,
                        "ood_test": metrics,
                        "source_score": 0.9,
                        "source_evaluations": 40,
                        "candidate_first_discovered_minute": 2,
                        "candidate_first_discovered_elapsed_seconds": 91.0,
                        "candidate_source": "imcts_native_reward",
                    }
                ),
                encoding="utf-8",
            )
            # 当前 state 的表达式越界，恢复器必须回退到最近的可评估快照。
            (exp_dir / ".imcts_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "x[9]",
                        "score": 0.95,
                        "evaluations": 50,
                        "first_discovered_minute": 3,
                        "first_discovered_elapsed_seconds": 151.0,
                        "source": "imcts_native_reward",
                    }
                ),
                encoding="utf-8",
            )

            recovered = runner._recover_timeout_payload_from_candidate(
                tool_name="iMCTS",
                dataset=dataset,
                experiment_dir=exp_dir,
            )

            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["equation"], valid_equation)
            self.assertEqual(recovered["source_score"], 0.9)
            self.assertEqual(recovered["candidate_first_discovered_minute"], 2)
            self.assertEqual(
                recovered["candidate_first_discovered_elapsed_seconds"], 91.0
            )

    def test_imcts_budget_endpoint_preserves_internal_discovery_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            output_dir = root / "out"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            dataset = runner.load_canonical_dataset(dataset_dir)
            equation = "2*x[0] + 3*x[1] + 1"
            artifact, error = runner.safe_build_canonical_artifact(
                tool_name="iMCTS",
                equation=equation,
                expected_n_features=2,
            )
            self.assertIsNone(error)
            (exp_dir / ".imcts_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": equation,
                        "score": 0.95,
                        "evaluations": 42,
                        "first_discovered_minute": 2,
                        "first_discovered_elapsed_seconds": 93.0,
                        "source": "imcts_native_reward",
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "status": "ok",
                "equation": equation,
                "canonical_artifact": artifact,
                "seconds": 180.0,
                "params": {"timeout_in_seconds": 180},
            }

            runner._write_final_progress_payload_if_requested(
                result=result,
                progress_snapshot_interval_seconds=60,
                output_dir=output_dir,
                experiment_dir=exp_dir,
                tool_name="iMCTS",
                dataset=dataset,
                params={"timeout_in_seconds": 180},
                seed=520,
                started_at=time.time() - 180,
            )

            payload = json.loads(
                (output_dir / "progress" / "minute_0003.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["record_type"], "budget_end_internal_best")
            self.assertEqual(payload["source_score"], 0.95)
            self.assertEqual(payload["candidate_first_discovered_minute"], 2)
            self.assertEqual(result["source_score"], 0.95)
            self.assertEqual(result["candidate_first_discovered_minute"], 2)

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
                        "internal_loss": -12.0,
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
                task_label="g0001_demo",
                task_global_index=1,
                expected_dataset_dir=str(dataset_dir),
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["tool"], "QLattice")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], -12.0)
            self.assertEqual(payload["source_internal_loss"], -12.0)
            self.assertEqual(payload["source_iteration"], 3)
            self.assertEqual(payload["source_complexity"], 5)
            self.assertEqual(payload["elapsed_minutes"], 90)
            self.assertEqual(payload["task_label"], "g0001_demo")
            self.assertEqual(payload["task_global_index"], 1)
            self.assertTrue(payload["dataset_identity_check"]["match"])
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
                        "internal_loss": 0.0,
                        "complexity": 5,
                        "iteration": 3,
                        "first_discovered_attempt": 3,
                        "first_discovered_minute": 1,
                        "first_discovered_elapsed_seconds": 42.0,
                        "source": ".jaxsr_current_best.json",
                        "fidelity": _jaxsr_fidelity(),
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
            self.assertEqual(payload["source_internal_loss"], 0.0)
            self.assertEqual(payload["source_complexity"], 5)
            self.assertEqual(payload["candidate_first_discovered_attempt"], 3)
            self.assertEqual(payload["candidate_first_discovered_minute"], 1)
            self.assertEqual(
                payload["candidate_first_discovered_elapsed_seconds"], 42.0
            )
            self.assertEqual(payload["candidate_source"], ".jaxsr_current_best.json")
            self.assertEqual(
                payload["canonical_artifact"]["instantiated_expression"],
                "2*x0 + 3*x1 + 1",
            )
            self.assertEqual(
                payload["canonical_artifact"]["fidelity_check"],
                _jaxsr_fidelity(),
            )
            self.assertEqual(payload["candidate_fidelity"], _jaxsr_fidelity())
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_jaxsr_failed_fidelity_is_explicit_export_error_not_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            failed_fidelity = _jaxsr_fidelity("failed")
            (exp_dir / ".jaxsr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": None,
                        "loss": 0.0,
                        "complexity": None,
                        "iteration": 4,
                        "fidelity": failed_fidelity,
                        "model_state": {"result": {"coefficients": [1.0]}},
                    }
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="jaxsr",
                dataset=dataset,
                params={"max_terms": 5},
                seed=1314,
                started_at=time.time() - 120,
                experiment_dir=exp_dir,
                checkpoint_index=2,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["record_type"], "periodic_best")
            self.assertEqual(payload["status"], "error")
            self.assertTrue(payload["candidate_available"])
            self.assertIsNone(payload["equation"])
            self.assertIsNone(payload["canonical_artifact"])
            self.assertIn("export fidelity failed", payload["error"])
            self.assertEqual(payload["candidate_fidelity"], failed_fidelity)

    def test_jaxsr_budget_end_internal_best_keeps_fidelity_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            output_dir = root / "out"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            fidelity = _jaxsr_fidelity()
            (exp_dir / ".jaxsr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x0 + 3*x1 + 1",
                        "loss": 0.0,
                        "complexity": 5,
                        "iteration": 3,
                        "fidelity": fidelity,
                    }
                ),
                encoding="utf-8",
            )
            dataset = runner.load_canonical_dataset(dataset_dir)
            final_artifact = normalize_external_infix_artifact(
                "2*x0 + 3*x1 + 1",
                tool_name="jaxsr",
                expected_n_features=2,
                shift_one_based=False,
            )
            final_artifact["fidelity_check"] = fidelity
            result = {
                "status": "ok",
                "tool": "jaxsr",
                "equation": "2*x0 + 3*x1 + 1",
                "canonical_artifact": final_artifact,
                "seconds": 600.0,
                "params": {"timeout_in_seconds": 600},
            }

            runner._write_final_progress_payload_if_requested(
                result=result,
                progress_snapshot_interval_seconds=60,
                output_dir=output_dir,
                experiment_dir=exp_dir,
                tool_name="jaxsr",
                dataset=dataset,
                params={"timeout_in_seconds": 600},
                seed=1314,
                started_at=time.time() - 600,
            )

            payload = json.loads(
                (output_dir / "progress" / "minute_0010.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["record_type"], "budget_end_internal_best")
            self.assertEqual(payload["canonical_artifact"]["fidelity_check"], fidelity)
            self.assertEqual(payload["candidate_fidelity"], fidelity)

    def test_jaxsr_failed_current_best_blocks_stale_timeout_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            progress_dir = exp_dir / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            dataset = runner.load_canonical_dataset(dataset_dir)

            (exp_dir / ".jaxsr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": "2*x0 + 3*x1 + 1",
                        "loss": 0.0,
                        "complexity": 5,
                        "iteration": 1,
                        "fidelity": _jaxsr_fidelity(),
                    }
                ),
                encoding="utf-8",
            )
            stale_payload = runner._build_periodic_snapshot_payload(
                tool_name="jaxsr",
                dataset=dataset,
                params={"max_terms": 5},
                seed=1314,
                started_at=time.time() - 60,
                experiment_dir=exp_dir,
                checkpoint_index=1,
            )
            (progress_dir / "minute_0001.json").write_text(
                json.dumps(stale_payload),
                encoding="utf-8",
            )
            (exp_dir / ".jaxsr_current_best.json").write_text(
                json.dumps(
                    {
                        "equation": None,
                        "loss": -1.0,
                        "iteration": 2,
                        "fidelity": _jaxsr_fidelity("failed"),
                    }
                ),
                encoding="utf-8",
            )

            recovered = runner._recover_timeout_payload_from_candidate(
                tool_name="jaxsr",
                dataset=dataset,
                experiment_dir=exp_dir,
            )

            self.assertIsNone(recovered)

    def test_build_periodic_snapshot_payload_records_heartbeat_without_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sim-datasets-data" / "demo"
            exp_dir = root / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="fepysr",
                dataset=dataset,
                params={"timeout_in_seconds": 86400},
                seed=520,
                started_at=time.time() - 120,
                experiment_dir=exp_dir,
                checkpoint_index=1,
                task_label="g0001_demo",
                task_global_index=1,
                expected_dataset_rel="sim-datasets-data/demo",
                expected_dataset_dir=str(dataset_dir),
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["record_type"], "periodic_heartbeat")
            self.assertEqual(payload["tool"], "fepysr")
            self.assertEqual(payload["status"], "running")
            self.assertFalse(payload["candidate_available"])
            self.assertIsNone(payload["equation"])
            self.assertEqual(payload["task_label"], "g0001_demo")
            self.assertEqual(payload["task_global_index"], 1)
            self.assertEqual(payload["expected_dataset_rel"], "sim-datasets-data/demo")
            self.assertEqual(payload["expected_dataset_dir"], str(dataset_dir))
            self.assertTrue(payload["dataset_identity_check"]["match"])

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
                    params_override={
                        "progress_snapshot_interval_seconds": 60,
                        "train_label_noise_enabled": True,
                        "train_label_noise_sigma": 0.01,
                    },
                )
            finally:
                runner.SymbolicRegressor = old_symbolic_regressor

            progress_files = sorted(result_path.parent.glob("progress/minute_*.json"))
            self.assertEqual(len(progress_files), 1)
            payload = json.loads(progress_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["record_type"], "final_best")
            self.assertEqual(payload["tool"], "symbolfit")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["condition"], "noise001")
            self.assertTrue(payload["train_label_noise"]["requested"])
            self.assertEqual(payload["train_label_noise"]["sigma"], 0.01)
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)

    def test_build_periodic_snapshot_payload_for_symbolfit_active_pysr_hof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            work_dir = exp_dir / "symbolfit_work" / "attempt_0001"
            hof_dir = work_dir / "outputs_tmp" / "20260601_demo"
            hof_dir.mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)
            (hof_dir / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n"
                "1,10.0,1.0\n"
                "5,0.0,1 + 2*x0 + 3*x1\n",
                encoding="utf-8",
            )
            (exp_dir / ".symbolfit_active_run.json").write_text(
                json.dumps({"tool": "symbolfit", "attempt": 1, "work_dir": str(work_dir)}),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="symbolfit",
                dataset=dataset,
                params={"niterations": 100},
                seed=520,
                started_at=time.time() - 59,
                experiment_dir=exp_dir,
                checkpoint_index=1,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["record_type"], "periodic_best")
            self.assertEqual(payload["tool"], "symbolfit")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source_loss"], 0.0)
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)

    def test_symbolfit_active_hof_is_unscaled_before_canonical_replay(self):
        """active PySR HOF 的缩放公式必须先还原到原始坐标。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            work_dir = exp_dir / "symbolfit_work" / "attempt_0001"
            hof_dir = work_dir / "outputs_tmp" / "scaled_run"
            hof_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: target
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name in ("train.csv", "valid.csv", "id_test.csv", "ood_test.csv"):
                (dataset_dir / name).write_text(
                    "x0,target\n10,10\n20,20\n30,30\n",
                    encoding="utf-8",
                )
            # SymbolFit scales x0 to [0, 1], and y by y_scale=0.05.  The
            # scaled candidate X0+0.5 therefore means y=x0 on original data.
            (hof_dir / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n"
                "3,0.25,x0+0.5\n",
                encoding="utf-8",
            )
            (exp_dir / ".symbolfit_active_run.json").write_text(
                json.dumps(
                    {
                        "tool": "symbolfit",
                        "attempt": 1,
                        "work_dir": str(work_dir),
                        "input_rescale": True,
                        "x_min": [10.0],
                        "x_max": [30.0],
                        "y_scale": 0.05,
                    }
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="symbolfit",
                dataset=dataset,
                params={"niterations": 100},
                seed=520,
                started_at=time.time() - 419,
                experiment_dir=exp_dir,
                checkpoint_index=7,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["source_internal_loss"], 0.25)
            self.assertEqual(payload["candidate_first_discovered_minute"], 7)
            self.assertEqual(payload["candidate_first_discovered_attempt"], 1)
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)

    def test_symbolfit_active_hof_keeps_global_internal_best_across_attempts(self):
        """新的 attempt 不能按 ID/OOD 或局部 HOF 覆盖历史内部最优。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "exp"
            work1 = exp_dir / "symbolfit_work" / "attempt_0001"
            work2 = exp_dir / "symbolfit_work" / "attempt_0002"
            for work in (work1, work2):
                (work / "outputs_tmp" / "run").mkdir(parents=True, exist_ok=True)
            (work1 / "outputs_tmp" / "run" / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n3,0.10,x0+1\n", encoding="utf-8"
            )
            (work2 / "outputs_tmp" / "run" / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n3,0.50,x0+2\n", encoding="utf-8"
            )
            active = exp_dir / ".symbolfit_active_run.json"
            active.write_text(
                json.dumps({"tool": "symbolfit", "attempt": 1, "work_dir": str(work1)}),
                encoding="utf-8",
            )
            first = runner._extract_symbolfit_periodic_candidate(
                exp_dir,
                snapshot_minute=3,
            )
            assert first["equation"] == "x0+1"
            assert first["internal_loss"] == 0.10
            assert first["first_discovered_minute"] == 3
            active.write_text(
                json.dumps({"tool": "symbolfit", "attempt": 2, "work_dir": str(work2)}),
                encoding="utf-8",
            )
            second = runner._extract_symbolfit_periodic_candidate(
                exp_dir,
                snapshot_minute=8,
            )
            assert second["equation"] == "x0+1"
            assert second["internal_loss"] == 0.10
            assert second["first_discovered_minute"] == 3
            assert second["first_discovered_attempt"] == 1

    def test_symbolfit_candidate_first_seen_after_budget_is_not_used_at_endpoint(self):
        """180 分钟后才观察到的 HOF 候选不能倒灌进第 180 分钟。"""
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "exp"
            work_dir = exp_dir / "symbolfit_work" / "attempt_0001"
            hof_dir = work_dir / "outputs_tmp" / "run"
            hof_dir.mkdir(parents=True, exist_ok=True)
            (hof_dir / "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n3,0.10,x0+1\n",
                encoding="utf-8",
            )
            (exp_dir / ".symbolfit_active_run.json").write_text(
                json.dumps({"tool": "symbolfit", "attempt": 1, "work_dir": str(work_dir)}),
                encoding="utf-8",
            )

            candidate = runner._extract_symbolfit_periodic_candidate(
                exp_dir,
                snapshot_minute=180,
                snapshot_elapsed_seconds=10800.25,
            )

            self.assertIsNone(candidate)
            history = [
                json.loads(line)
                for line in (exp_dir / ".symbolfit_pysr_candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(history[0]["first_discovered_minute"], 181)
            self.assertEqual(
                history[0]["first_discovered_elapsed_seconds"],
                10800.25,
            )

    def test_symbolfit_search_best_sidecar_is_unscaled_before_replay(self):
        """仅有 wrapper sidecar 时也必须按保存的坐标变换回放。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: target
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name in ("train.csv", "valid.csv", "id_test.csv", "ood_test.csv"):
                (dataset_dir / name).write_text(
                    "x0,target\n10,10\n20,20\n30,30\n",
                    encoding="utf-8",
                )
            (exp_dir / ".symbolfit_search_best.json").parent.mkdir(parents=True, exist_ok=True)
            (exp_dir / ".symbolfit_search_best.json").write_text(
                json.dumps(
                    {
                        "tool": "symbolfit",
                        "candidate_key": "sidecar-only",
                        "scaled_equation": "X0+0.5",
                        "internal_loss": 0.25,
                        "attempt": 2,
                        "first_discovered_attempt": 1,
                        "first_discovered_minute": 4,
                        "coordinate_transform": {
                            "input_rescale": True,
                            "x_min": [10.0],
                            "x_max": [30.0],
                            "x_range": [20.0],
                            "y_scale": 0.05,
                            "y_unscale_factor": 20.0,
                            "equation_space": "scaled_input_scaled_target",
                        },
                        "source": "symbolfit_pysr_hof_postfit",
                    }
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="symbolfit",
                dataset=dataset,
                params={"niterations": 100},
                seed=520,
                started_at=time.time() - 420,
                experiment_dir=exp_dir,
                checkpoint_index=7,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["source_internal_loss"], 0.25)
            self.assertEqual(payload["candidate_first_discovered_minute"], 4)
            self.assertEqual(payload["candidate_first_discovered_attempt"], 1)
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
