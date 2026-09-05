from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.freeze_results import (
    _build_index_row,
    _copy_remote_file,
    _local_result_path,
    load_source_runs,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.source_provenance import (
    SourceProvenanceError,
)


class FreezeResultsTest(unittest.TestCase):
    def test_load_source_runs_rejects_aborted_fullcpu_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_csv = Path(tmp_dir) / "source_runs.csv"
            row = {
                "logical_key": "demo::D1::s520::clean",
                "batch": "all_15alg_fullcpu_v1_formal",
                "algorithm": "demo",
                "dataset_id": "D1",
                "seed": "520",
                "noise_tag": "clean",
                "task_id": "demo_s520_clean_g0001",
                "host": "iaaccn22",
                "status": "ok",
                "id_nmse": "0.1",
                "ood_nmse": "0.2",
                "path": "/experiments/all_15alg_fullcpu_v1_formal/result.json",
            }
            with source_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(SourceProvenanceError, "已中止"):
                load_source_runs(source_csv, noise_tag="clean")

    def test_local_result_path_is_deterministic(self) -> None:
        row = {
            "noise_tag": "clean",
            "algorithm": "llmsr",
            "task_id": "llmsr_s520_clean_g0043",
        }
        target = _local_result_path(Path("/tmp/stage5"), row)
        self.assertEqual(
            str(target),
            "/tmp/stage5/results/clean/llmsr/llmsr_s520_clean_g0043/result.json",
        )

    def test_build_index_row_extracts_presence_and_metrics(self) -> None:
        row = {
            "logical_key": "llmsr::demo::s520::clean",
            "batch": "batch_x",
            "noise_tag": "clean",
            "algorithm": "llmsr",
            "dataset_id": "demo",
            "seed": 520,
            "task_id": "llmsr_s520_clean_g0001",
            "host": "iaaccn24",
            "remote_result_path": "/remote/result.json",
            "id_nmse_source": 0.125,
            "ood_nmse_source": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "equation": "x0 + x1",
                        "canonical_artifact": {"normalized_expression": "x0 + x1"},
                        "id_test": {"nmse": 0.125},
                        "ood_test": {"nmse": 0.25},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            index_row = _build_index_row(row, result_path)
            self.assertTrue(index_row["equation_present"])
            self.assertTrue(index_row["canonical_artifact_present"])
            self.assertTrue(index_row["metrics_match"])
            self.assertEqual(index_row["id_nmse_result"], "0.125")

    def test_copy_remote_file_retries_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "result.json"
            tmp_path = local_path.with_suffix(".tmp")
            calls: list[int] = []

            def fake_run(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    return mock.Mock(returncode=1, stderr="transient scp error", stdout="")
                tmp_path.write_text("{}", encoding="utf-8")
                return mock.Mock(returncode=0, stderr="", stdout="")

            with mock.patch(
                "AAAI_experiments.stage5_metric_calculation_0831.pipeline.freeze_results.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "AAAI_experiments.stage5_metric_calculation_0831.pipeline.freeze_results.time.sleep"
            ):
                _copy_remote_file("iaaccn22", "/remote/result.json", local_path, max_attempts=3)

            self.assertEqual(len(calls), 2)
            self.assertTrue(local_path.exists())


if __name__ == "__main__":
    unittest.main()
