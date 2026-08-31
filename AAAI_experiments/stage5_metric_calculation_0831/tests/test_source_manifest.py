from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.source_manifest import (
    NOISE_ORDER,
    build_source_preflight,
    validate_source_ground_truth_alignment,
)


STAGE4_CSV = REPO_ROOT / "AAAI_experiments/stage4_ssr50_15algs_3seeds_3noise_3h/selected_runs_with_fepysr_rerun.csv"
GT_CSV = REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"


class SourceManifestTest(unittest.TestCase):
    def test_build_source_preflight_on_real_stage4_inputs(self) -> None:
        source_rows, gt_rows, report = build_source_preflight(
            stage4_csv=STAGE4_CSV,
            gt_csv=GT_CSV,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(len(source_rows), 6750)
        self.assertEqual(len(gt_rows), 50)
        self.assertEqual(report["stage4"]["row_count"], 6750)
        self.assertEqual(report["ground_truth"]["row_count"], 50)
        self.assertEqual(source_rows[0]["noise_tag"], "clean")
        self.assertEqual(source_rows[-1]["noise_tag"], "noise005")
        noise_orders = [NOISE_ORDER[row["noise_tag"]] for row in source_rows]
        self.assertEqual(noise_orders, sorted(noise_orders))
        self.assertEqual(
            {row["noise_tag"] for row in source_rows},
            {"clean", "noise001", "noise005"},
        )
        self.assertTrue(all(len(row["formula_py_sha256"]) == 64 for row in gt_rows))
        self.assertTrue(report["source_ground_truth_alignment"]["ok"])

    def test_source_and_ground_truth_dataset_sets_must_match_exactly(self) -> None:
        matched = validate_source_ground_truth_alignment(
            [{"dataset_id": "g1"}, {"dataset_id": "g2"}],
            [{"basename": "g2"}, {"basename": "g1"}],
        )
        self.assertTrue(matched["ok"])

        mismatched = validate_source_ground_truth_alignment(
            [{"dataset_id": "g1"}, {"dataset_id": "unexpected"}],
            [{"basename": "g1"}, {"basename": "g2"}],
        )
        self.assertFalse(mismatched["ok"])
        self.assertEqual(mismatched["missing_from_source"], ["g2"])
        self.assertEqual(mismatched["missing_from_ground_truth"], ["unexpected"])

    def test_cli_writes_manifest_and_report(self) -> None:
        tmp_dir = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/tests/.tmp_source_manifest"
        if tmp_dir.exists():
            for path in sorted(tmp_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = tmp_dir / "manifests"
        report_json = tmp_dir / "reports/source_preflight.json"
        cmd = [
            sys.executable,
            "-m",
            "AAAI_experiments.stage5_metric_calculation_0831.pipeline.source_manifest",
            "--stage4-csv",
            str(STAGE4_CSV),
            "--gt-csv",
            str(GT_CSV),
            "--manifest-dir",
            str(manifest_dir),
            "--report-json",
            str(report_json),
        ]
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('"row_count": 6750', completed.stdout)
        self.assertTrue((manifest_dir / "source_runs.csv").exists())
        self.assertTrue((manifest_dir / "ground_truth.csv").exists())
        self.assertTrue(report_json.exists())
        with (manifest_dir / "source_runs.csv").open("r", encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        self.assertEqual(len(source_rows), 6750)
        with report_json.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["stage4"]["row_count"], 6750)
        self.assertEqual(report["ground_truth"]["row_count"], 50)


if __name__ == "__main__":
    unittest.main()
