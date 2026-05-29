from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_dataset(root: Path, name: str) -> None:
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.yaml").write_text(
        "target:\n  name: y\nfeatures:\n  - name: x0\n",
        encoding="utf-8",
    )


def test_manifest_generation_writes_750_stage1_tasks(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("manifest")
    ssr50_root = tmp_path / "ssr50"
    for idx in range(50):
        _write_dataset(ssr50_root, f"dataset_{idx:04d}")

    config_path = tmp_path / "toolbox_config.json"
    config_path.write_text(
        json.dumps(
            {
                "tool_mapping": {
                    f"alg{idx:02d}": {"env": f"env{idx:02d}", "regressor": f"Reg{idx:02d}"}
                    for idx in range(15)
                }
            }
        ),
        encoding="utf-8",
    )
    batch_dir = tmp_path / "benchmark-runs" / "compliance" / "batch"

    summary = module.generate_manifest(
        toolbox_config_path=config_path,
        ssr50_root=ssr50_root,
        batch_dir=batch_dir,
        git_revision="abc123",
    )

    assert summary["total_algorithms"] == 15
    assert summary["total_datasets"] == 50
    assert summary["total_tasks"] == 750
    tasks_path = batch_dir / "manifest" / "tasks.csv"
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 750
    assert rows[0]["seed"] == "520"
    assert rows[0]["timeout_in_seconds"] == "3600"
    assert rows[0]["progress_snapshot_interval_seconds"] == "60"
    assert "__seed520__" in rows[0]["task_id"]
