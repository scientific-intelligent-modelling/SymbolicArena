from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.ground_truth_extract import (
    GroundTruthExtractionError,
    build_ground_truth_records,
    extract_ground_truth_record,
    load_manifest_rows,
)


MANIFEST_CSV = (
    REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/manifests/ground_truth.csv"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GroundTruthExtractTest(unittest.TestCase):
    def test_build_ground_truth_records_on_real_core50_manifest(self) -> None:
        records, report = build_ground_truth_records(MANIFEST_CSV, repo_root=REPO_ROOT)

        self.assertEqual(report["dataset_count"], 50)
        self.assertEqual(report["success_count"], 50)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(len(records), 50)
        self.assertEqual(report["selection_counts"]["unique_feature_signature"], 5)
        self.assertEqual(report["selection_counts"]["metadata_target_name"], 45)

        by_dataset = {record["dataset_id"]: record for record in records}
        self.assertEqual(by_dataset["first_principles_hubble"]["normalized_expression_input"], "73.3 * D")
        self.assertEqual(by_dataset["feynman_I_27_6"]["function_name"], "foc")
        self.assertEqual(by_dataset["feynman_I_27_6"]["ordered_variables"], ["d1", "d2", "n"])
        self.assertIn("np.where", by_dataset["Korns-2"]["normalized_expression_input"])
        self.assertTrue(all(len(record["evidence_sha256"]) == 64 for record in records))

    def test_rejects_ambiguous_and_multi_return_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            ambiguous_manifest = self._build_single_row_fixture(
                root / "ambiguous",
                feature_names=["x"],
                target_name="target",
                formula_source="""
def first(x):
    return x

def second(x):
    return x + 1
""".strip(),
                dataset_name="ambiguous_case",
            )
            ambiguous_row = load_manifest_rows(ambiguous_manifest, repo_root=root / "ambiguous")[0]
            with self.assertRaisesRegex(GroundTruthExtractionError, "候选函数不唯一"):
                extract_ground_truth_record(ambiguous_row)

            multi_return_manifest = self._build_single_row_fixture(
                root / "multi_return",
                feature_names=["x"],
                target_name="target",
                formula_source="""
def target(x):
    if x > 0:
        return x
    return -x
""".strip(),
                dataset_name="multi_return_case",
            )
            multi_return_row = load_manifest_rows(
                multi_return_manifest,
                repo_root=root / "multi_return",
            )[0]
            with self.assertRaisesRegex(GroundTruthExtractionError, "含多个 return"):
                extract_ground_truth_record(multi_return_row)

    def test_cli_writes_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            output_jsonl = tmp_root / "ground_truth_extract.jsonl"
            report_json = tmp_root / "ground_truth_extract.report.json"
            cmd = [
                sys.executable,
                "-m",
                "AAAI_experiments.stage5_metric_calculation_0831.pipeline.ground_truth_extract",
                "--manifest-csv",
                str(MANIFEST_CSV),
                "--output-jsonl",
                str(output_jsonl),
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
            report = json.loads(completed.stdout)
            self.assertEqual(report["success_count"], 50)
            self.assertTrue(output_jsonl.exists())
            self.assertTrue(report_json.exists())
            with output_jsonl.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 50)

    def _build_single_row_fixture(
        self,
        root: Path,
        *,
        feature_names: list[str],
        target_name: str,
        formula_source: str,
        dataset_name: str,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        dataset_dir = root / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = dataset_dir / "metadata.yaml"
        formula_path = dataset_dir / "formula.py"
        split_paths = {
            "train_csv": dataset_dir / "train.csv",
            "valid_csv": dataset_dir / "valid.csv",
            "id_test_csv": dataset_dir / "id_test.csv",
            "ood_test_csv": dataset_dir / "ood_test.csv",
        }

        metadata = {
            "dataset": {
                "name": dataset_name,
                "features": [{"name": name, "type": "continuous"} for name in feature_names],
                "target": {"name": target_name, "type": "continuous"},
                "ground_truth_formula": {"file": "formula.py"},
            }
        }
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        formula_path.write_text(formula_source + "\n", encoding="utf-8")

        header = feature_names + [target_name]
        sample_row = ["1" for _ in header]
        for path in split_paths.values():
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerow(sample_row)

        manifest_path = root / "ground_truth.csv"
        row = {
            "core50_index": "1",
            "dataset_name": dataset_name,
            "basename": dataset_name,
            "target_name": target_name,
            "feature_count": str(len(feature_names)),
            "metadata_yaml": str(metadata_path.relative_to(root)),
            "metadata_yaml_sha256": _sha256_file(metadata_path),
            "train_csv": str(split_paths["train_csv"].relative_to(root)),
            "train_csv_sha256": _sha256_file(split_paths["train_csv"]),
            "valid_csv": str(split_paths["valid_csv"].relative_to(root)),
            "valid_csv_sha256": _sha256_file(split_paths["valid_csv"]),
            "id_test_csv": str(split_paths["id_test_csv"].relative_to(root)),
            "id_test_csv_sha256": _sha256_file(split_paths["id_test_csv"]),
            "ood_test_csv": str(split_paths["ood_test_csv"].relative_to(root)),
            "ood_test_csv_sha256": _sha256_file(split_paths["ood_test_csv"]),
            "formula_py": str(formula_path.relative_to(root)),
            "formula_py_sha256": _sha256_file(formula_path),
        }
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return manifest_path


if __name__ == "__main__":
    unittest.main()
