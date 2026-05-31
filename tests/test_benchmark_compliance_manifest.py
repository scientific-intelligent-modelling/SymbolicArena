from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


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


def test_manifest_generation_preserves_relative_dataset_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    monkeypatch.chdir(tmp_path)
    batch_dir = tmp_path / "benchmark-runs" / "compliance" / "batch"

    module.generate_manifest(
        toolbox_config_path=config_path,
        ssr50_root=Path("ssr50"),
        batch_dir=batch_dir,
        git_revision="abc123",
    )

    tasks_path = batch_dir / "manifest" / "tasks.csv"
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset_dir"] == "ssr50/dataset_0000"


def test_manifest_generation_reads_nested_ssr50_dataset_layout(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("manifest")
    ssr50_root = tmp_path / "ssr50"
    for idx in range(50):
        family = f"family_{idx % 5:02d}"
        _write_dataset(ssr50_root / "datasets" / family, f"dataset_{idx:04d}")
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
        dataset_dir_base=tmp_path,
    )

    assert summary == {"total_algorithms": 15, "total_datasets": 50, "total_tasks": 750}
    with (batch_dir / "manifest" / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset_dir"].startswith("ssr50/datasets/")


def test_full24h_manifest_generation_writes_5850_noise_aware_tasks(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("manifest")
    ssr50_root = tmp_path / "ssr50"
    for index in range(50):
        _write_dataset(ssr50_root, f"d{index:02d}")

    selected_tools = (
        "gplearn",
        "pyoperon",
        "pysr",
        "dso",
        "tpsr",
        "e2esr",
        "fepysr",
        "jaxsr",
        "QLattice",
        "iMCTS",
        "udsr",
        "ragsr",
        "symbolfit",
    )
    toolbox_config = tmp_path / "toolbox_config.json"
    toolbox_config.write_text(
        json.dumps(
            {
                "tool_mapping": {
                    name: {"env": "sim_base", "regressor": name}
                    for name in (*selected_tools, "llmsr", "drsr")
                }
            }
        ),
        encoding="utf-8",
    )

    batch_dir = tmp_path / "benchmark-runs" / "formal24h" / "batch"
    summary = module.generate_manifest(
        toolbox_config_path=toolbox_config,
        ssr50_root=ssr50_root,
        batch_dir=batch_dir,
        git_revision="abc123",
        dataset_dir_base=tmp_path,
        algorithms=selected_tools,
        seeds=(520, 521, 522),
        noise_sigmas=(0.0, 0.01, 0.05),
        timeout_in_seconds=86400,
        progress_snapshot_interval_seconds=60,
        min_runtime_seconds=82800,
    )

    assert summary == {"total_algorithms": 13, "total_datasets": 50, "total_tasks": 5850}
    with (batch_dir / "manifest" / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5850
    assert {row["seed"] for row in rows} == {"520", "521", "522"}
    assert {row["noise_tag"] for row in rows} == {"clean", "noise001", "noise005"}
    assert {row["noise_sigma"] for row in rows} == {"0", "0.01", "0.05"}
    assert {row["timeout_in_seconds"] for row in rows} == {"86400"}
    assert {row["min_runtime_seconds"] for row in rows} == {"82800"}
    assert "pysr__seed520__noise001__d00" in {row["task_id"] for row in rows}


def test_load_for_test_restores_sys_path() -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    before = list(sys.path)
    load_for_test("models")
    assert sys.path == before


def test_prepare_batch_resolves_repo_relative_paths_from_any_cwd(tmp_path: Path, monkeypatch) -> None:
    launcher_path = Path(__file__).resolve().parents[1] / "benchmark-control" / "compliance" / "launchers" / "prepare_batch.py"
    spec = importlib.util.spec_from_file_location("benchmark_compliance_prepare_batch", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.chdir(tmp_path)
    resolved = module._resolve_repo_path("scientific_intelligent_modelling/config/toolbox_config.json")

    assert resolved == launcher_path.parents[3] / "scientific_intelligent_modelling" / "config" / "toolbox_config.json"


def test_prepare_batch_git_revision_runs_from_repo_root(monkeypatch) -> None:
    launcher_path = Path(__file__).resolve().parents[1] / "benchmark-control" / "compliance" / "launchers" / "prepare_batch.py"
    spec = importlib.util.spec_from_file_location("benchmark_compliance_prepare_batch_git", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured: dict[str, object] = {}

    def _fake_check_output(cmd: list[str], *, text: bool, cwd: Path | None = None) -> str:
        captured["cmd"] = cmd
        captured["text"] = text
        captured["cwd"] = cwd
        return "deadbeef\n"

    monkeypatch.setattr(module.subprocess, "check_output", _fake_check_output)

    revision = module._git_revision()

    assert revision == "deadbeef"
    assert captured["cmd"] == ["git", "rev-parse", "HEAD"]
    assert captured["text"] is True
    assert captured["cwd"] == module.ROOT


def test_prepare_batch_main_writes_repo_relative_dataset_paths(tmp_path: Path, monkeypatch) -> None:
    launcher_path = Path(__file__).resolve().parents[1] / "benchmark-control" / "compliance" / "launchers" / "prepare_batch.py"
    spec = importlib.util.spec_from_file_location("benchmark_compliance_prepare_batch_main", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake_root = tmp_path / "repo"
    toolbox_dir = fake_root / "scientific_intelligent_modelling" / "config"
    toolbox_dir.mkdir(parents=True)
    (toolbox_dir / "toolbox_config.json").write_text(
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
    ssr50_root = fake_root / "sim-datasets-data" / "ssr50"
    for idx in range(50):
        _write_dataset(ssr50_root, f"dataset_{idx:04d}")
    batch_dir = fake_root / "benchmark-runs" / "compliance" / "batch"
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()

    monkeypatch.setattr(module, "ROOT", fake_root)
    monkeypatch.setattr(module, "_git_revision", lambda: "abc123")
    monkeypatch.chdir(outside_cwd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_batch.py",
            "--batch-dir",
            "benchmark-runs/compliance/batch",
        ],
    )

    assert module.main() == 0

    tasks_path = batch_dir / "manifest" / "tasks.csv"
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert not Path(rows[0]["dataset_dir"]).is_absolute()
    assert rows[0]["dataset_dir"].startswith("sim-datasets-data/ssr50/")
