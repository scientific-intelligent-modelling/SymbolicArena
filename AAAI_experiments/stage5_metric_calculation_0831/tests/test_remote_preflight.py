from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.remote_preflight import (
    REMOTE_HELPER,
    build_host_payloads,
    load_source_rows,
)


SOURCE_RUNS_CSV = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/manifests/source_runs.csv"


class RemotePreflightTest(unittest.TestCase):
    def test_build_host_payloads_for_clean_rows(self) -> None:
        rows = load_source_rows(SOURCE_RUNS_CSV, noise_tag="clean")
        self.assertEqual(len(rows), 2250)
        payloads = build_host_payloads(rows, limit_per_host=2)
        self.assertTrue(payloads)
        self.assertTrue(all(len(host_rows) <= 2 for host_rows in payloads.values()))

    def test_remote_helper_local_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "demo_task"
            progress_dir = task_dir / "progress"
            progress_dir.mkdir(parents=True)
            result_path = task_dir / "result.json"
            result_path.write_text("{}", encoding="utf-8")
            for minute in (1, 2, 3):
                (progress_dir / f"minute_{minute:04d}.json").write_text("{}", encoding="utf-8")
            payload_path = root / "payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "host": "local",
                        "entries": [
                            {
                                "logical_key": "alg::demo::s520::clean",
                                "task_id": "alg_s520_clean_g0001",
                                "dataset_id": "demo",
                                "algorithm": "alg",
                                "seed": 520,
                                "noise_tag": "clean",
                                "host": "local",
                                "result_path": str(result_path),
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            helper_path = root / "helper.py"
            helper_path.write_text(REMOTE_HELPER, encoding="utf-8")
            completed = __import__("subprocess").run(
                [sys.executable, str(helper_path), str(payload_path)],
                text=True,
                capture_output=True,
                check=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["summary"]["entry_count"], 1)
            self.assertEqual(report["summary"]["result_exists"], 1)
            self.assertEqual(report["summary"]["has_progress"], 1)
            self.assertEqual(report["rows"][0]["progress_max_count"], 3)


if __name__ == "__main__":
    unittest.main()
