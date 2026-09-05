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

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_clean_rerun_overlay import (  # noqa: E402
    CleanRerunOverlayError,
    _load_clean_source_rows,
    _parse_checksum_manifest,
    _symbolfit_record,
    _verify_manifest_file,
    build_clean_rerun_overlay,
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


def _payload(
    algorithm: str,
    dataset: str,
    seed: int,
    *,
    nmse: float,
    minute: int | None = None,
    horizon: int = 3,
) -> dict[str, object]:
    result: dict[str, object] = {
        "tool": algorithm,
        "dataset": dataset,
        "seed": seed,
        "status": "ok",
        "seconds": 10800.0 + seed,
        "equation": "x0 + 1",
        "canonical_artifact": {
            "artifact_valid": True,
            "instantiated_expression": "x0 + 1",
        },
        "id_test": {"nmse": nmse, "r2": 0.9, "acc_0_1": 1.0},
        "ood_test": {"nmse": nmse * 2, "r2": 0.8, "acc_0_1": 0.75},
    }
    if minute is not None:
        result.update(
            record_type=(
                "budget_end_internal_best" if minute == horizon else "periodic_best"
            ),
            checkpoint_index=minute,
            elapsed_minutes=minute,
            source_internal_loss=float(horizon - minute + 1),
            candidate_source="symbolfit_active_pysr_hall_of_fame",
        )
    return result


def _source_row(algorithm: str, dataset: str, seed: int, *, index: int) -> dict[str, str]:
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
        "seconds": "1",
        "id_nmse": "99",
        "ood_nmse": "99",
        "id_r2": "-1",
        "ood_r2": "-1",
        "id_acc": "0",
        "ood_acc": "0",
        "path": f"/base/{key}/result.json",
        "logical_key": key,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _frozen_record(
    algorithm: str, dataset: str, seed: int, *, task_id: str, horizon: int = 3
) -> dict[str, object]:
    result_payload = _payload(algorithm, dataset, seed, nmse=0.1)
    result_raw = json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n"
    snapshots = []
    for minute in range(1, horizon + 1):
        payload = _payload(
            algorithm,
            dataset,
            seed,
            nmse=1.0 / minute,
            minute=minute,
            horizon=horizon,
        )
        raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        snapshots.append(
            {
                "minute": minute,
                "status": "ok",
                "conflict": False,
                "selected_path": f"/remote/{task_id}/progress/minute_{minute:04d}.json",
                "selected_sha256": digest,
                "raw_text": raw,
            }
        )
    return {
        "source": {
            "batch": "rerun",
            "algorithm": algorithm,
            "dataset_id": dataset,
            "seed": seed,
            "noise_tag": "clean",
            "task_id": task_id,
            "host": "iaaccn23",
            "path": f"/remote/{task_id}/result.json",
        },
        "result": {
            "status": "ok",
            "sha256": hashlib.sha256(result_raw.encode()).hexdigest(),
            "raw_text": result_raw,
        },
        "snapshots": snapshots,
        "summary": {
            "expected_snapshots": horizon,
            "available_snapshots": horizon,
            "missing_snapshots": 0,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
        },
    }


def _write_gzip_records(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_checksums(root: Path, manifest: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(f"{_sha(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def _write_symbolfit_task(
    task_dir: Path,
    *,
    task_id: str,
    dataset: str,
    seed: int,
    nmse: float,
    horizon: int = 3,
) -> Path:
    dataset_root = task_dir / "iaaccn24" / "symbolfit" / f"g0001_{dataset}"
    if "strict" in task_dir.parts:
        dataset_root = task_dir / "symbolfit" / f"g0039_{dataset}"
    _write_json(dataset_root / "result.json", _payload("symbolfit", dataset, seed, nmse=nmse))
    for minute in range(1, horizon + 1):
        _write_json(
            dataset_root / "progress" / f"minute_{minute:04d}.json",
            _payload(
                "symbolfit",
                dataset,
                seed,
                nmse=nmse,
                minute=minute,
                horizon=horizon,
            ),
        )
    return dataset_root


def _fixture(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "source_runs.csv"
    base_rows = [
        _source_row("jaxsr", "jax-demo", 520, index=1),
        _source_row("iMCTS", "imcts-demo", 520, index=2),
        _source_row("symbolfit", "sym-demo", 520, index=2),
        _source_row("symbolfit", "strict-demo", 521, index=39),
    ]
    with base.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(base_rows)

    freeze_parent = tmp_path / "freeze"
    freeze_dir = freeze_parent / "collected"
    freeze_path = freeze_dir / "freeze.jsonl.gz"
    _write_gzip_records(
        freeze_path,
        [
            _frozen_record(
                "jaxsr", "jax-demo", 520, task_id="jaxsr_s520_clean_g0001"
            ),
            _frozen_record(
                "iMCTS", "imcts-demo", 520, task_id="imcts_s520_clean_g0002"
            ),
        ],
    )
    (freeze_parent / "SHA256SUMS").write_text(
        f"{_sha(freeze_path)}  collected/{freeze_path.name}\n", encoding="utf-8"
    )

    full = tmp_path / "full"
    _write_symbolfit_task(
        full / "symbolfit/seed520/tasks/symbolfit_s520_clean_g0002",
        task_id="symbolfit_s520_clean_g0002",
        dataset="sym-demo",
        seed=520,
        nmse=0.2,
    )
    old_strict = _write_symbolfit_task(
        full / "symbolfit/seed521/tasks/symbolfit_s521_clean_g0039",
        task_id="symbolfit_s521_clean_g0039",
        dataset="strict-demo",
        seed=521,
        nmse=0.3,
    )
    _write_checksums(full, full / "SHA256SUMS")

    strict_parent = tmp_path / "strict"
    strict = strict_parent / "collected"
    strict_dataset = _write_symbolfit_task(
        strict,
        task_id="symbolfit_s521_clean_g0039",
        dataset="strict-demo",
        seed=521,
        nmse=0.004,
    )
    _write_checksums(strict, strict_parent / "collected.SHA256SUMS")
    return {
        "base": base,
        "freeze_dir": freeze_dir,
        "freeze_path": freeze_path,
        "freeze_manifest": freeze_parent / "SHA256SUMS",
        "full": full,
        "old_strict_result": old_strict / "result.json",
        "strict": strict,
        "strict_result": strict_dataset / "result.json",
    }


def test_base_source_rejects_aborted_fullcpu_batch(tmp_path: Path) -> None:
    source_csv = tmp_path / "source_runs.csv"
    row = _source_row("jaxsr", "demo", 520, index=1)
    row["batch"] = "all_15alg_fullcpu_v1_formal"
    row["path"] = "/experiments/all_15alg_fullcpu_v1_formal/result.json"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(CleanRerunOverlayError, match="已中止"):
        _load_clean_source_rows(source_csv, expected_clean_rows=1)


def _build(paths: dict[str, Path], output_dir: Path) -> dict[str, object]:
    return build_clean_rerun_overlay(
        base_source_runs_csv=paths["base"],
        jax_imcts_freeze_dir=paths["freeze_dir"],
        symbolfit_full_root=paths["full"],
        symbolfit_strict_root=paths["strict"],
        output_bundle=output_dir / "overlay.jsonl.gz",
        output_manifest=output_dir / "manifest.json",
        output_composite_csv=output_dir / "composite.csv",
        horizon=3,
        expected_clean_rows=4,
        expected_counts={"jaxsr": 1, "imcts": 1, "symbolfit": 2},
    )


def test_strict_replacement_scope_composite_and_determinism(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = tmp_path / "out-a"
    second = tmp_path / "out-b"
    manifest = _build(paths, first)
    _build(paths, second)

    assert (first / "overlay.jsonl.gz").read_bytes() == (
        second / "overlay.jsonl.gz"
    ).read_bytes()
    assert (first / "composite.csv").read_bytes() == (second / "composite.csv").read_bytes()
    assert (first / "overlay.jsonl.gz").read_bytes()[4:8] == b"\x00\x00\x00\x00"
    assert manifest["outputs"]["overlay_bundle"]["sha256"] == _sha(
        first / "overlay.jsonl.gz"
    )
    assert manifest["outputs"]["diagnostic_composite_source_runs_csv"][
        "sha256"
    ] == _sha(first / "composite.csv")

    assert manifest["eff_replacement_count"] == 4
    assert manifest["final_replacement_count"] == 2
    assert manifest["strict_replacement"]["forced"] is True
    assert manifest["strict_replacement"]["full150_result_sha256"] == _sha(
        paths["old_strict_result"]
    )
    assert manifest["strict_replacement"]["strict_result_sha256"] == _sha(
        paths["strict_result"]
    )

    with gzip.open(first / "overlay.jsonl.gz", "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    by_key = {row["overlay"]["logical_key"]: row for row in records}
    strict = by_key["symbolfit::strict-demo::s521::clean"]
    assert strict["overlay"]["origin"] == "symbolfit_strict_replacement"
    assert strict["overlay"]["replacement_scope"] == "eff_only"
    assert strict["result"]["sha256"] == _sha(paths["strict_result"])
    assert by_key["jaxsr::jax-demo::s520::clean"]["overlay"]["replacement_scope"] == (
        "final_and_eff"
    )

    with (first / "composite.csv").open(encoding="utf-8", newline="") as handle:
        composite = {row["logical_key"]: row for row in csv.DictReader(handle)}
    assert composite["jaxsr::jax-demo::s520::clean"]["id_nmse"] == "0.10000000000000001"
    assert composite["iMCTS::imcts-demo::s520::clean"]["id_nmse"] == (
        "0.10000000000000001"
    )
    assert composite["symbolfit::sym-demo::s520::clean"]["id_nmse"] == "99"
    assert composite["symbolfit::strict-demo::s521::clean"]["id_nmse"] == "99"


def test_checksum_lookup_resolves_once_per_file_with_many_same_basenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    selected = data_root / "nested" / "result.json"
    _write_json(selected, {"status": "ok"})
    digest = _sha(selected)
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        "".join(
            [
                f"{digest}  nested/result.json\n",
                f"{digest}  /remote/archive/nested/result.json\n",
                f"{digest}  {selected.resolve().as_posix()}\n",
            ]
            + [
                f"{hashlib.sha256(str(index).encode()).hexdigest()}  "
                f"decoy-{index}/result.json\n"
                for index in range(300)
            ]
        ),
        encoding="utf-8",
    )
    entries = _parse_checksum_manifest(manifest)
    real_resolve = Path.resolve
    resolve_calls = {"selected": 0, "root": 0}

    def counted_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == selected:
            resolve_calls["selected"] += 1
        elif path == data_root:
            resolve_calls["root"] += 1
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    for _ in range(4):
        assert _verify_manifest_file(
            selected,
            data_root=data_root,
            checksum_entries=entries,
        ) == digest
    assert resolve_calls == {"selected": 4, "root": 4}

    conflicting = dict(entries)
    conflicting["result.json"] = [
        *entries["result.json"],
        ("nested/result.json", "f" * 64),
    ]
    with pytest.raises(CleanRerunOverlayError, match="冲突哈希"):
        _verify_manifest_file(
            selected,
            data_root=data_root,
            checksum_entries=conflicting,
        )


def test_symbolfit_record_resolves_each_selected_path_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    task_dir = paths["full"] / "symbolfit/seed520/tasks/symbolfit_s520_clean_g0002"
    result_path = task_dir / "iaaccn24/symbolfit/g0001_sym-demo/result.json"
    selected_paths = {
        result_path,
        *(result_path.parent / "progress" / f"minute_{minute:04d}.json" for minute in range(1, 4)),
    }
    resolve_calls = {path: 0 for path in selected_paths}
    real_resolve = Path.resolve

    def counted_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path in resolve_calls:
            resolve_calls[path] += 1
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    base_row = _source_row("symbolfit", "sym-demo", 520, index=2)
    _symbolfit_record(
        task_dir,
        task_id="symbolfit_s520_clean_g0002",
        batch="rerun",
        checksum_root=paths["full"],
        checksum_entries=_parse_checksum_manifest(paths["full"] / "SHA256SUMS"),
        horizon=3,
        base_by_normal_key={
            ("symbolfit", "sym-demo", 520): (base_row["logical_key"], base_row)
        },
        input_files=[],
        origin="symbolfit_full150",
    )
    assert set(resolve_calls.values()) == {1}


def test_duplicate_overlay_key_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with gzip.open(paths["freeze_path"], "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows.append(rows[0])
    _write_gzip_records(paths["freeze_path"], rows)
    paths["freeze_manifest"].write_text(
        f"{_sha(paths['freeze_path'])}  collected/{paths['freeze_path'].name}\n",
        encoding="utf-8",
    )
    with pytest.raises(CleanRerunOverlayError, match="重复 logical_key"):
        _build(paths, tmp_path / "out")


@pytest.mark.parametrize("damage", ["missing_minute", "raw_sha", "bundle_sha"])
def test_missing_minute_or_bad_sha_is_rejected(tmp_path: Path, damage: str) -> None:
    paths = _fixture(tmp_path)
    with gzip.open(paths["freeze_path"], "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if damage == "missing_minute":
        rows[0]["snapshots"].pop()
        _write_gzip_records(paths["freeze_path"], rows)
        paths["freeze_manifest"].write_text(
            f"{_sha(paths['freeze_path'])}  collected/{paths['freeze_path'].name}\n",
            encoding="utf-8",
        )
        expected = "恰有 3 个快照"
    elif damage == "raw_sha":
        rows[0]["snapshots"][0]["raw_text"] += " "
        _write_gzip_records(paths["freeze_path"], rows)
        paths["freeze_manifest"].write_text(
            f"{_sha(paths['freeze_path'])}  collected/{paths['freeze_path'].name}\n",
            encoding="utf-8",
        )
        expected = "raw SHA256 不匹配"
    else:
        paths["freeze_path"].write_bytes(paths["freeze_path"].read_bytes() + b"x")
        expected = "输入文件 SHA256 不匹配"
    with pytest.raises(CleanRerunOverlayError, match=expected):
        _build(paths, tmp_path / "out")
