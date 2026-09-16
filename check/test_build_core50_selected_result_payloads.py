"""Focused tests for exact selected-result and terminal-snapshot binding."""

import csv
import gzip
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_core50_selected_result_payloads.py")
SPEC = importlib.util.spec_from_file_location("build_core50_selected_result_payloads", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def raw_payload(dataset, seed, condition):
    return json.dumps({"dataset": dataset, "seed": seed, "condition": condition})


class SelectedPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.collection = self.root / "collection"
        self.collection.mkdir()
        self.formal = self.root / "formal"
        self.archive = self.root / "snapshots.tar"
        self.rows = []
        self.terminals = []
        self.supplements = []
        self.archive_contents = {}
        self.formal_rows = []

    def tearDown(self):
        self.tmp.cleanup()

    def add_run(self, dataset, seed, *, source):
        key = f"example::{dataset}::s{seed}::clean"
        task_id = f"example_s{seed}_clean_{dataset}"
        old_raw = raw_payload(dataset, seed, "clean")
        old_sha = MODULE.sha256_text(old_raw)
        new_raw = json.dumps({"dataset": dataset, "seed": seed, "condition": "clean", "new": True})
        new_sha = MODULE.sha256_text(new_raw)
        selected_raw = old_raw if source == "formal_fallback" else new_raw
        selected_sha = old_sha if source == "formal_fallback" else new_sha
        result_member = f"runs/clean/example/{dataset}/seed_{seed}/{task_id}/result.json" if source == "archive" else ""
        snapshot_member = f"runs/clean/example/{dataset}/seed_{seed}/{task_id}/progress/minute_0002.json"
        snapshot_raw = raw_payload(dataset, seed, "clean")
        if result_member:
            self.archive_contents[result_member] = selected_raw
        self.archive_contents[snapshot_member] = snapshot_raw
        selected_path = f"/new/{task_id}/result.json" if source != "formal_fallback" else f"/old/{task_id}/result.json"
        locator = (f"archive:{result_member}" if source == "archive" else
                   f"supplemental_result_payloads.jsonl.gz:{key}" if source == "supplement" else
                   f"formal_raw_results:{key}")
        self.rows.append({
            "condition": "clean", "algorithm": "example", "algorithm_slug": "example",
            "dataset_id": dataset, "seed": seed, "task_id": task_id, "logical_key": key,
            "selected_result_sha256": selected_sha, "selected_result_archive_member": result_member,
            "selected_result_payload_locator": locator, "selected_result_source_path": selected_path,
            "superseded": source != "formal_fallback",
        })
        expression = f"x0 + {seed}"
        self.terminals.append({
            **self.rows[-1], "terminal_expression": expression,
            "terminal_expression_sha256": MODULE.sha256_text(expression),
            "terminal_archive_member": snapshot_member,
            "terminal_source_sha256": MODULE.sha256_text(snapshot_raw),
            "terminal_source_path": f"/new/{task_id}/progress/minute_0002.json",
            "terminal_incumbent_source_minute": 2,
        })
        self.formal_rows.append({
            "source": {"algorithm": "example", "dataset_id": dataset, "seed": seed,
                       "task_id": task_id, "path": f"/old/{task_id}/result.json"},
            "result": {"sha256": old_sha, "raw_text": old_raw},
        })
        if source == "supplement":
            self.supplements.append({"logical_key": key, "result_sha256": selected_sha,
                                     "result_source_path": selected_path, "raw_text": selected_raw})

    def write(self):
        with (self.collection / "latest_run_selection.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        write_jsonl_gz(self.collection / "terminal_inputs.jsonl.gz", self.terminals)
        write_jsonl_gz(self.collection / "supplemental_result_payloads.jsonl.gz", self.supplements)
        for condition in MODULE.CONDITIONS:
            write_jsonl_gz(self.formal / condition / "raw_results.jsonl.gz",
                           self.formal_rows if condition == "clean" else [])
        with tarfile.open(self.archive, "w") as handle:
            for member, raw in self.archive_contents.items():
                data = raw.encode("utf-8")
                info = tarfile.TarInfo(member)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))

    def test_collects_archive_formal_and_supplement_without_old_replacement(self):
        self.add_run("A", 520, source="archive")
        self.add_run("B", 521, source="formal_fallback")
        self.add_run("C", 522, source="supplement")
        self.write()
        report = MODULE.build(self.collection, self.archive, self.formal, 3, 2)
        self.assertEqual(report["result_payload_source_counts"],
                         {"archive": 1, "formal_fallback": 1, "supplement": 1})
        results = list(MODULE.read_jsonl_gz(self.collection / "selected_result_payloads.jsonl.gz"))
        snapshots = list(MODULE.read_jsonl_gz(self.collection / "terminal_snapshot_payloads.jsonl.gz"))
        self.assertEqual(len(results), 3)
        self.assertEqual(len(snapshots), 3)
        self.assertIn('"new": true', next(row["raw_text"] for row in results if row["dataset_id"] == "A"))
        self.assertTrue(all("minute_0002.json" in row["archive_member"] for row in snapshots))

    def test_rejects_snapshot_sha_mismatch_without_export(self):
        self.add_run("A", 520, source="archive")
        self.terminals[0]["terminal_source_sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "terminal snapshot SHA mismatch"):
            MODULE.build(self.collection, self.archive, self.formal, 1, 1)
        self.assertFalse((self.collection / "selected_result_payloads.jsonl.gz").exists())

    def test_accepts_recovery_snapshot_without_repeated_identity(self):
        self.add_run("A", 520, source="archive")
        member = self.terminals[0]["terminal_archive_member"]
        recovery_raw = json.dumps({"function": "x0 + 1", "iteration": 12})
        self.archive_contents[member] = recovery_raw
        self.terminals[0]["terminal_source_sha256"] = MODULE.sha256_text(recovery_raw)
        self.write()
        report = MODULE.build(self.collection, self.archive, self.formal, 1, 1)
        self.assertEqual(report["terminal_snapshot_count"], 1)

    def test_never_uses_old_formal_payload_for_missing_supersession(self):
        self.add_run("A", 520, source="supplement")
        self.supplements.clear()
        self.write()
        with self.assertRaisesRegex(ValueError, "Missing/mismatched superseding supplement"):
            MODULE.build(self.collection, self.archive, self.formal, 1, 1)

    def test_rejects_archive_result_sha_mismatch(self):
        self.add_run("A", 520, source="archive")
        member = self.rows[0]["selected_result_archive_member"]
        self.archive_contents[member] = raw_payload("A", 520, "clean")
        self.write()
        with self.assertRaisesRegex(ValueError, "result SHA mismatch"):
            MODULE.build(self.collection, self.archive, self.formal, 1, 1)


if __name__ == "__main__":
    unittest.main()
