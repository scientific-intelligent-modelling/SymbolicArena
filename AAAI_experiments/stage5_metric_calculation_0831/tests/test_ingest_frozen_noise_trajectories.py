from __future__ import annotations

import gzip
import gc
import hashlib
import json
import sys
import tracemalloc
import csv
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.ingest_frozen_noise_trajectories import (  # noqa: E402
    FrozenNoiseTrajectoryError,
    ingest_frozen_noise_trajectories,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    phi_nmse,
)  # noqa: E402


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload(
    minute: int,
    *,
    expression: str | None = None,
    nmse: float | None = None,
    source_loss: float | None = None,
    record_type: str = "periodic_heartbeat",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": "qlattice",
        "record_type": record_type,
        "checkpoint_index": "final" if record_type == "final_best" else minute,
        "elapsed_minutes": minute,
        "status": "ok" if expression is not None else "running",
        "equation": expression,
        "id_test": {"nmse": nmse} if nmse is not None else None,
        "ood_test": {"nmse": nmse} if nmse is not None else None,
    }
    if expression is not None:
        payload["canonical_artifact"] = {"instantiated_expression": expression}
    if source_loss is not None:
        payload["source_loss"] = source_loss
    return payload


def _frozen_snapshot(minute: int, payload: dict[str, object]) -> dict[str, object]:
    raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = _sha256_text(raw_text)
    return {
        "minute": minute,
        "status": "ok",
        "conflict": False,
        "selected_path": f"/remote/progress/minute_{minute:04d}.json",
        "selected_sha256": digest,
        "raw_text": raw_text,
    }


def _record(
    *,
    dataset_index: int = 1,
    missing_minute: int | None = None,
    include_overtime: bool = True,
) -> dict[str, object]:
    source: dict[str, object] = {
        "batch": "batch1",
        "algorithm": "QLattice",
        "host": "iaaccn22",
        "seed": "520",
        "noise_tag": "noise001",
        "task_id": f"qlattice_s520_noise001_g{dataset_index:04d}",
        "dataset_id": f"Demo-{dataset_index}",
        "path": "/remote/result.json",
    }
    source["source_row_sha256"] = _source_row_sha256(source)

    def payload(*args: object, **kwargs: object) -> dict[str, object]:
        value = _payload(*args, **kwargs)  # type: ignore[arg-type]
        value.update(
            {
                "tool": source["algorithm"],
                "dataset": source["dataset_id"],
                "seed": int(source["seed"]),
                "noise_tag": source["noise_tag"],
                "task_id": source["task_id"],
                "train_label_noise": {
                    "enabled": True,
                    "requested": True,
                    "sigma": 0.01,
                },
            }
        )
        return value

    snapshots = [
        _frozen_snapshot(minute, payload(minute))
        for minute in range(1, 181)
        if minute != missing_minute
    ]
    snapshots[0] = _frozen_snapshot(
        1,
        payload(
            1, expression="x0", nmse=1.0, source_loss=1.0, record_type="periodic_best"
        ),
    )
    snapshots[1] = _frozen_snapshot(
        2,
        payload(
            2,
            expression="x0 + 1",
            nmse=1e-12,
            source_loss=2.0,
            record_type="periodic_best",
        ),
    )
    if missing_minute != 180:
        snapshots[-1] = _frozen_snapshot(
            180,
            payload(180, expression="x0 + 2", nmse=1e-12, record_type="final_best"),
        )
    if include_overtime:
        snapshots.append(
            _frozen_snapshot(
                181,
                payload(
                    181,
                    expression="x0 + 3",
                    nmse=1e-12,
                    source_loss=0.01,
                    record_type="periodic_best",
                ),
            )
        )
    result_payload = payload(180, expression="x0", nmse=1e-12, record_type="final_best")
    result_raw = json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
    return {
        "source": source,
        "result": {
            "status": "ok",
            "sha256": _sha256_text(result_raw),
            "raw_text": result_raw,
        },
        "snapshots": snapshots,
        # 故意放入不可信汇总，解析器必须从冻结点本身重新统计。
        "summary": {"expected_snapshots": 999, "missing_snapshots": 999},
    }


def _source_row_sha256(source: dict[str, object]) -> str:
    without_sha = {
        key: value for key, value in source.items() if key != "source_row_sha256"
    }
    without_sha["seed"] = int(without_sha["seed"])
    canonical = json.dumps(
        without_sha, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_source_csv(path: Path, records: list[dict[str, object]]) -> None:
    rows: dict[str, dict[str, str]] = {}
    for record in records:
        source = dict(record["source"])
        source.pop("source_row_sha256", None)
        row = {key: str(value) for key, value in source.items()}
        row["logical_key"] = (
            f"{source['algorithm']}::{source['dataset_id']}::"
            f"s{source['seed']}::{source['noise_tag']}"
        )
        key = f"{source['algorithm']}::{source['dataset_id']}::s{source['seed']}::{source['noise_tag']}"
        rows[key] = row
    fieldnames = sorted({field for row in rows.values() for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())


def _source_csv_for(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "source_runs.csv"
    _write_source_csv(path, records)
    return path


def _write_bundle(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_output(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_ingest_truncates_overtime_and_reuses_internal_best_policy(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "noise_freeze_iaaccn22.jsonl.gz"
    output = tmp_path / "prepared/noise_trajectory_input.jsonl.gz"
    manifest = tmp_path / "prepared/noise_trajectory_manifest.json"
    records = [_record()]
    _write_bundle(bundle, records)
    source_csv = _source_csv_for(tmp_path, records)

    report = ingest_frozen_noise_trajectories(
        input_paths=[bundle],
        output_jsonl_gz=output,
        manifest_json=manifest,
        source_runs_csv=source_csv,
        expected_tasks=1,
    )

    rows = _read_output(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["logical_key"] == "QLattice::Demo-1::s520::noise001"
    assert sorted(map(int, row["snapshots"])) == list(range(1, 181))
    assert len(row["best_so_far_trajectory"]) == 180
    assert row["best_so_far_trajectory"][1]["quality"] == phi_nmse(1.0)
    assert row["best_so_far_trajectory"][1]["source"] == "internal_best_carry_forward:1"
    assert row["best_so_far_trajectory"][-1]["expression"] == "x0"
    assert report["summary"]["over_budget_snapshots_truncated"] == 1
    assert report["summary"]["missing_snapshot_count"] == 0
    assert report["summary"]["ready_task_count"] == 1
    assert report["summary"]["strict_1_to_180_ready"] is True
    assert (
        report["outputs"]["jsonl_gz"]["sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert json.loads(manifest.read_text(encoding="utf-8")) == report

    first_output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    ingest_frozen_noise_trajectories(
        input_paths=[bundle],
        output_jsonl_gz=output,
        manifest_json=manifest,
        source_runs_csv=source_csv,
        expected_tasks=1,
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first_output_sha


def test_ingest_reports_missing_minute_without_synthesizing_data(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "noise_freeze_iaaccn22.jsonl.gz"
    output = tmp_path / "prepared/noise_trajectory_input.jsonl.gz"
    manifest = tmp_path / "prepared/noise_trajectory_manifest.json"
    records = [_record(missing_minute=77, include_overtime=False)]
    _write_bundle(bundle, records)
    source_csv = _source_csv_for(tmp_path, records)

    report = ingest_frozen_noise_trajectories(
        input_paths=[bundle],
        output_jsonl_gz=output,
        manifest_json=manifest,
        source_runs_csv=source_csv,
        expected_tasks=1,
    )

    assert _read_output(output) == []
    assert report["summary"]["ready_task_count"] == 0
    assert report["summary"]["unresolved_task_count"] == 1
    assert report["summary"]["missing_snapshot_count"] == 1
    assert report["summary"]["strict_1_to_180_ready"] is False
    assert report["unresolved"] == [
        {
            "logical_key": "QLattice::Demo-1::s520::noise001",
            "missing_minutes": [77],
            "reason": "missing_or_unusable_snapshots",
        }
    ]


def test_duplicate_logical_keys_are_excluded_and_audited(tmp_path: Path) -> None:
    bundle = tmp_path / "noise_freeze_iaaccn22.jsonl.gz"
    output = tmp_path / "prepared/noise_trajectory_input.jsonl.gz"
    manifest = tmp_path / "prepared/noise_trajectory_manifest.json"
    records = [_record(), _record()]
    _write_bundle(bundle, records)
    source_csv = _source_csv_for(tmp_path, records)

    report = ingest_frozen_noise_trajectories(
        input_paths=[bundle],
        output_jsonl_gz=output,
        manifest_json=manifest,
        source_runs_csv=source_csv,
        expected_tasks=2,
    )

    assert _read_output(output) == []
    assert report["summary"]["duplicate_logical_key_count"] == 1
    assert report["summary"]["duplicate_record_count"] == 2
    assert report["summary"]["ready_task_count"] == 0
    assert report["unresolved"] == [
        {
            "logical_key": "QLattice::Demo-1::s520::noise001",
            "reason": "duplicate_logical_key",
            "occurrences": [
                {
                    "input_bundle_path": str(bundle.resolve()),
                    "input_line_number": 1,
                },
                {
                    "input_bundle_path": str(bundle.resolve()),
                    "input_line_number": 2,
                },
            ],
        }
    ]


def _run_with_peak_memory(
    tmp_path: Path, *, run_count: int, label: str
) -> tuple[int, str]:
    bundle = tmp_path / f"{label}.jsonl.gz"
    records = [
        _record(dataset_index=index, include_overtime=False)
        for index in range(1, run_count + 1)
    ]
    _write_bundle(bundle, records)
    source_csv = _source_csv_for(tmp_path, records)
    output = tmp_path / f"{label}.output.jsonl.gz"
    manifest = tmp_path / f"{label}.manifest.json"
    gc.collect()
    tracemalloc.start()
    try:
        report = ingest_frozen_noise_trajectories(
            input_paths=[bundle],
            output_jsonl_gz=output,
            manifest_json=manifest,
            source_runs_csv=source_csv,
            expected_tasks=run_count,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert report["summary"]["ready_task_count"] == run_count
    return peak, hashlib.sha256(output.read_bytes()).hexdigest()


def test_many_runs_are_streamed_with_bounded_peak_memory_and_stable_hash(
    tmp_path: Path,
) -> None:
    one_peak, _ = _run_with_peak_memory(tmp_path, run_count=1, label="one")
    many_peak, first_hash = _run_with_peak_memory(tmp_path, run_count=32, label="many")

    # 任务数扩大 32 倍时，只允许轻量 key 索引增长，不能常驻全部 180 分钟 payload。
    assert many_peak < one_peak * 4

    output = tmp_path / "many.output.jsonl.gz"
    manifest = tmp_path / "many.manifest.json"
    bundle = tmp_path / "many.jsonl.gz"
    source_csv = _source_csv_for(
        tmp_path,
        [
            _record(dataset_index=index, include_overtime=False)
            for index in range(1, 33)
        ],
    )
    ingest_frozen_noise_trajectories(
        input_paths=[bundle],
        output_jsonl_gz=output,
        manifest_json=manifest,
        source_runs_csv=source_csv,
        expected_tasks=32,
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first_hash


def test_rejects_seed_outside_formal_seed_set(tmp_path: Path) -> None:
    record = _record()
    record["source"]["seed"] = "519"
    record["source"]["task_id"] = "qlattice_s519_noise001_g0001"
    record["source"]["source_row_sha256"] = _source_row_sha256(record["source"])
    bundle = tmp_path / "noise_freeze.jsonl.gz"
    _write_bundle(bundle, [record])
    source_csv = _source_csv_for(tmp_path, [record])

    with pytest.raises(FrozenNoiseTrajectoryError, match="seed"):
        ingest_frozen_noise_trajectories(
            input_paths=[bundle],
            output_jsonl_gz=tmp_path / "output.jsonl.gz",
            manifest_json=tmp_path / "manifest.json",
            source_runs_csv=source_csv,
            expected_tasks=1,
        )


def test_rejects_source_row_sha256_not_matching_canonical_source_csv_row(
    tmp_path: Path,
) -> None:
    record = _record()
    source_csv = _source_csv_for(tmp_path, [record])
    record["source"]["source_row_sha256"] = "0" * 64
    bundle = tmp_path / "noise_freeze.jsonl.gz"
    _write_bundle(bundle, [record])

    with pytest.raises(FrozenNoiseTrajectoryError, match="source_row_sha256"):
        ingest_frozen_noise_trajectories(
            input_paths=[bundle],
            output_jsonl_gz=tmp_path / "output.jsonl.gz",
            manifest_json=tmp_path / "manifest.json",
            source_runs_csv=source_csv,
            expected_tasks=1,
        )


def test_rejects_result_raw_payload_from_another_task_even_with_valid_sha(
    tmp_path: Path,
) -> None:
    first = _record(dataset_index=1)
    second = _record(dataset_index=2)
    first["result"] = dict(second["result"])
    source_csv = _source_csv_for(tmp_path, [first])
    bundle = tmp_path / "noise_freeze.jsonl.gz"
    _write_bundle(bundle, [first])

    with pytest.raises(FrozenNoiseTrajectoryError, match="result.*(dataset|task|身份)"):
        ingest_frozen_noise_trajectories(
            input_paths=[bundle],
            output_jsonl_gz=tmp_path / "output.jsonl.gz",
            manifest_json=tmp_path / "manifest.json",
            source_runs_csv=source_csv,
            expected_tasks=1,
        )


def test_rejects_legal_snapshot_borrowed_from_another_task(tmp_path: Path) -> None:
    first = _record(dataset_index=1)
    second = _record(dataset_index=2)
    first["snapshots"][0] = second["snapshots"][0]
    source_csv = _source_csv_for(tmp_path, [first])
    bundle = tmp_path / "noise_freeze.jsonl.gz"
    _write_bundle(bundle, [first])

    with pytest.raises(FrozenNoiseTrajectoryError, match="snapshot.*(dataset|task|身份)"):
        ingest_frozen_noise_trajectories(
            input_paths=[bundle],
            output_jsonl_gz=tmp_path / "output.jsonl.gz",
            manifest_json=tmp_path / "manifest.json",
            source_runs_csv=source_csv,
            expected_tasks=1,
        )


@pytest.mark.parametrize(
    "field", ["tool", "seed", "noise_tag", "checkpoint_index", "record_type"]
)
def test_rejects_snapshot_payload_identity_drift(tmp_path: Path, field: str) -> None:
    record = _record()
    snapshot = dict(record["snapshots"][0])
    payload = json.loads(str(snapshot["raw_text"]))
    payload[field] = {
        "tool": "dso",
        "seed": 521,
        "noise_tag": "noise005",
        "checkpoint_index": 2,
        "record_type": "foreign_record_type",
    }[field]
    snapshot["raw_text"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    snapshot["selected_sha256"] = _sha256_text(str(snapshot["raw_text"]))
    record["snapshots"][0] = snapshot
    source_csv = _source_csv_for(tmp_path, [record])
    bundle = tmp_path / "noise_freeze.jsonl.gz"
    _write_bundle(bundle, [record])

    with pytest.raises(FrozenNoiseTrajectoryError, match="snapshot"):
        ingest_frozen_noise_trajectories(
            input_paths=[bundle],
            output_jsonl_gz=tmp_path / "output.jsonl.gz",
            manifest_json=tmp_path / "manifest.json",
            source_runs_csv=source_csv,
            expected_tasks=1,
        )
