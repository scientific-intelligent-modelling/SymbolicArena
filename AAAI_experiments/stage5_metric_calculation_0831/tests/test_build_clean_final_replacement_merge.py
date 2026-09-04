from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_clean_final_replacement_merge import (  # noqa: E402
    CleanFinalReplacementError,
    build_clean_final_replacement_merge,
)


FIELDS = [
    "batch",
    "noise_order",
    "algorithm",
    "dataset_id",
    "seed",
    "noise_tag",
    "task_id",
    "host",
    "status",
    "seconds",
    "id_nmse",
    "ood_nmse",
    "id_r2",
    "ood_r2",
    "id_acc",
    "ood_acc",
    "path",
    "logical_key",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(algorithm: str, dataset: str, seed: int, *, nmse: float) -> dict[str, object]:
    return {
        "tool": algorithm,
        "dataset": dataset,
        "seed": seed,
        "status": "ok",
        "seconds": 10800.0,
        "equation": "x0 + 1",
        "canonical_artifact": {
            "artifact_valid": True,
            "instantiated_expression": "x0 + 1",
        },
        "id_test": {"nmse": nmse, "r2": 0.9, "acc_0_1": 1.0},
        "ood_test": {"nmse": nmse * 2, "r2": 0.8, "acc_0_1": 0.75},
    }


def _source_row(algorithm: str, dataset: str, seed: int, index: int) -> dict[str, str]:
    key = f"{algorithm}::{dataset}::s{seed}::clean"
    return {
        "batch": "base",
        "noise_order": "0",
        "algorithm": algorithm,
        "dataset_id": dataset,
        "seed": str(seed),
        "noise_tag": "clean",
        "task_id": f"{algorithm.lower()}_s{seed}_clean_g{index:04d}",
        "host": "iaaccn22",
        "status": "ok",
        "seconds": "10",
        "id_nmse": "0.9",
        "ood_nmse": "1.8",
        "id_r2": "0.1",
        "ood_r2": "0.2",
        "id_acc": "0",
        "ood_acc": "0",
        "path": f"/base/{key}/result.json",
        "logical_key": key,
    }


def _record(row: dict[str, str], *, nmse: float, replacement: bool, horizon: int) -> dict[str, object]:
    payload = _result(row["algorithm"], row["dataset_id"], int(row["seed"]), nmse=nmse)
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    source = {
        "batch": "rerun" if replacement else row["batch"],
        "algorithm": row["algorithm"],
        "dataset_id": row["dataset_id"],
        "seed": int(row["seed"]),
        "noise_tag": "clean",
        "task_id": row["task_id"],
        "host": "iaaccn23" if replacement else row["host"],
        "path": f"/rerun/{row['task_id']}/result.json" if replacement else row["path"],
    }
    record: dict[str, object] = {
        "source": source,
        "result": {
            "status": "ok",
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_text": raw,
        },
    }
    if replacement:
        snapshots = []
        for minute in range(1, horizon + 1):
            checkpoint = dict(payload)
            checkpoint.update(
                checkpoint_index=minute,
                elapsed_minutes=minute,
                record_type="budget_end_internal_best" if minute == horizon else "periodic_best",
            )
            checkpoint_raw = json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n"
            snapshots.append(
                {
                    "minute": minute,
                    "status": "ok",
                    "conflict": False,
                    "selected_sha256": hashlib.sha256(checkpoint_raw.encode()).hexdigest(),
                    "raw_text": checkpoint_raw,
                }
            )
        record["snapshots"] = snapshots
    return record


def _write_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fixture(tmp_path: Path, *, duplicate: bool = False, missing_minute: bool = False) -> dict[str, Path]:
    rows = [
        _source_row("JAXSR", "d1", 520, 1),
        _source_row("iMCTS", "d2", 521, 2),
        _source_row("gplearn", "d1", 520, 1),
        _source_row("gplearn", "d2", 521, 2),
    ]
    source_csv = tmp_path / "source_runs.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    base_bundle = tmp_path / "base.jsonl.gz"
    _write_gzip(base_bundle, [_record(row, nmse=0.9, replacement=False, horizon=3) for row in rows])
    base_binding = tmp_path / "base_binding.json"
    base_binding.write_text(
        json.dumps(
            {
                "contract_ok": True,
                "freeze_counts": {"tasks": 4},
                "input_files": {
                    "freeze_records": [
                        {"path": str(base_bundle), "sha256": _sha(base_bundle), "size_bytes": base_bundle.stat().st_size}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    replacement_root = tmp_path / "replacement"
    replacement_dir = replacement_root / "collected"
    replacements = [
        _record(rows[0], nmse=0.1, replacement=True, horizon=3),
        _record(rows[1], nmse=0.2, replacement=True, horizon=3),
    ]
    if duplicate:
        replacements.append(replacements[0])
    if missing_minute:
        replacements[0]["snapshots"] = replacements[0]["snapshots"][:-1]  # type: ignore[index]
    replacement_bundle = replacement_dir / "freeze.jsonl.gz"
    _write_gzip(replacement_bundle, replacements)
    (replacement_root / "SHA256SUMS").write_text(
        f"{_sha(replacement_bundle)}  collected/{replacement_bundle.name}\n",
        encoding="utf-8",
    )
    return {
        "source": source_csv,
        "binding": base_binding,
        "base_bundle": base_bundle,
        "replacement_dir": replacement_dir,
    }


def _run(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return build_clean_final_replacement_merge(
        base_source_runs_csv=paths["source"],
        base_freeze_binding_json=paths["binding"],
        base_freeze_paths=[paths["base_bundle"]],
        replacement_freeze_dir=paths["replacement_dir"],
        output_bundle=output / "merged.jsonl.gz",
        output_composite_csv=output / "source.csv",
        output_binding_manifest=output / "binding.json",
        horizon=3,
        expected_clean_rows=4,
        expected_replacement_counts={"jaxsr": 1, "imcts": 1},
    )


def test_merge_closes_grid_and_replaces_exact_scope(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = _run(paths, tmp_path / "out")

    assert manifest["contract_ok"] is True
    assert manifest["freeze_counts"]["tasks"] == 4
    assert manifest["replacement_contract"]["replacement_count"] == 2
    assert manifest["replacement_contract"]["algorithm_counts"] == {"imcts": 1, "jaxsr": 1}
    assert manifest["replacement_contract"]["result_binding_changed_count"] == 2
    assert manifest["aggregation_readiness"]["numeric_canonical_replay_ready"] is True
    assert manifest["aggregation_readiness"]["formal_six_axis_ready"] is False
    assert manifest["aggregation_readiness"]["stale_symbolic_run_count"] == 2
    assert manifest["aggregation_entry"]["numeric_prepare_inputs"]["freeze_binding_json"].endswith(
        "/out/binding.json"
    )

    with (tmp_path / "out/source.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({row["logical_key"] for row in rows}) == 4
    by_key = {row["logical_key"]: row for row in rows}
    assert by_key["JAXSR::d1::s520::clean"]["id_nmse"] == "0.10000000000000001"
    assert by_key["gplearn::d1::s520::clean"]["id_nmse"] == "0.9"

    with gzip.open(tmp_path / "out/merged.jsonl.gz", "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert len(records) == 4
    assert len({record["source"]["task_id"] for record in records}) == 4
    assert all("snapshots" not in record for record in records)


@pytest.mark.parametrize("case", ["duplicate", "missing_minute"])
def test_merge_rejects_invalid_replacement_evidence(tmp_path: Path, case: str) -> None:
    paths = _fixture(
        tmp_path,
        duplicate=case == "duplicate",
        missing_minute=case == "missing_minute",
    )
    with pytest.raises(CleanFinalReplacementError):
        _run(paths, tmp_path / "out")


def test_merge_is_deterministic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "out"
    first = _run(paths, output)
    first_hashes = (_sha(output / "merged.jsonl.gz"), _sha(output / "source.csv"))
    second = _run(paths, output)
    assert first_hashes == (_sha(output / "merged.jsonl.gz"), _sha(output / "source.csv"))
    assert first == second
