import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("assemble_core50_terminal_release.py")
SPEC = importlib.util.spec_from_file_location("assemble_core50_terminal_release", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
ReleaseError = module.ReleaseError
sha256_file = module.sha256_file
validate_collection = module.validate_collection
validate_numerical = module.validate_numerical
validate_payloads = module.validate_payloads


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _fixture(tmp_path: Path):
    collection = tmp_path / "collection"
    eff = tmp_path / "eff"
    collection.mkdir()
    (eff / "clean").mkdir(parents=True)
    rows = []
    terminals = []
    result_payloads = []
    snapshot_payloads = []
    native = []
    minutes = []
    for seed, expression in ((520, "x0"), (521, "x0+1")):
        key = f"algorithm::task::s{seed}::clean"
        result_text = json.dumps({"seed": seed})
        snapshot_text = json.dumps({"expression": expression})
        result_sha = hashlib.sha256(result_text.encode()).hexdigest()
        snapshot_sha = hashlib.sha256(snapshot_text.encode()).hexdigest()
        expression_sha = hashlib.sha256(expression.encode()).hexdigest()
        rows.append(
            {
                "condition": "clean", "algorithm": "algorithm", "algorithm_slug": "algorithm", "dataset_id": "task",
                "seed": seed, "logical_key": key, "selected_result_sha256": result_sha,
                "superseded": seed == 521,
            }
        )
        terminals.append(
            {
                "logical_key": key, "condition": "clean", "algorithm": "algorithm",
                "dataset_id": "task", "seed": seed, "selected_result_sha256": result_sha,
                "terminal_expression": expression, "terminal_expression_sha256": expression_sha,
                "terminal_source_sha256": snapshot_sha, "minute180_id_quality": 0.5,
                "minute180_ood_quality": 0.25, "minute180_valid_output": True,
            }
        )
        common = {
            "logical_key": key, "condition": "clean", "algorithm": "algorithm",
            "dataset_id": "task", "seed": seed, "selected_result_sha256": result_sha,
            "terminal_expression": expression, "terminal_expression_sha256": expression_sha,
        }
        result_payloads.append({**common, "raw_text": result_text, "result_sha256": result_sha})
        snapshot_payloads.append({**common, "raw_text": snapshot_text, "terminal_source_sha256": snapshot_sha})
        native.append(
            {"logical_key": key, "expression": [None, expression], "id_quality": [0, 0.5],
             "ood_quality": [0, 0.25], "valid_output": [False, True]}
        )
        for minute in (1, 2):
            minutes.append(
                {"logical_key": key, "minute": minute, "expression": expression if minute == 2 else "",
                 "id_quality": 0.5 if minute == 2 else 0, "ood_quality": 0.25 if minute == 2 else 0}
            )
    _write_csv(collection / "latest_run_selection.csv", rows)
    _write_csv(collection / "supersessions.csv", [rows[1]])
    _write_jsonl_gz(collection / "terminal_inputs.jsonl.gz", terminals)
    _write_jsonl_gz(collection / "selected_result_payloads.jsonl.gz", result_payloads)
    _write_jsonl_gz(collection / "terminal_snapshot_payloads.jsonl.gz", snapshot_payloads)
    _write_jsonl_gz(eff / "clean/native_trajectories.jsonl.gz", native)
    with gzip.open(eff / "clean/id_ood_eff_minute.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=minutes[0])
        writer.writeheader()
        writer.writerows(minutes)
    outputs = {
        name: sha256_file(collection / name)
        for name in ("latest_run_selection.csv", "supersessions.csv", "terminal_inputs.jsonl.gz")
    }
    (collection / "manifest.json").write_text(
        json.dumps({"outputs": outputs, "run_count": 2, "supersession_count": 1}), encoding="utf-8"
    )
    return collection, eff


def _validated(tmp_path: Path):
    collection, eff = _fixture(tmp_path)
    _, selected, terminals = validate_collection(
        collection, expected_runs=2, expected_supersessions=1, expected_datasets=1,
        conditions=("clean",), seeds={520, 521},
    )
    return collection, eff, selected, terminals


def test_small_collection_and_terminal_binding(tmp_path: Path) -> None:
    collection, eff, selected, terminals = _validated(tmp_path)
    assert validate_payloads(collection, selected, terminals) == {
        "selected_result": 2, "terminal_snapshot": 2,
    }
    assert validate_numerical(eff, terminals, horizon=2, conditions=("clean",)) == {
        "clean": {"runs": 2, "logical_minutes": 4, "terminal_matches": 2},
    }


def test_rejects_stale_terminal_snapshot_payload(tmp_path: Path) -> None:
    collection, _, selected, terminals = _validated(tmp_path)
    path = collection / "terminal_snapshot_payloads.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows[1]["terminal_expression"] = "stale expression"
    _write_jsonl_gz(path, rows)
    with pytest.raises(ReleaseError, match="expression mismatch"):
        validate_payloads(collection, selected, terminals)


def test_rejects_missing_minute(tmp_path: Path) -> None:
    _, eff, _, terminals = _validated(tmp_path)
    path = eff / "clean/id_ood_eff_minute.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ReleaseError, match="numerical minute gap"):
        validate_numerical(eff, terminals, horizon=2, conditions=("clean",))
