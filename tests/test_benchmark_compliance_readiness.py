from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from benchmark_control_compliance_manifest_import import load_for_test


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_ready_batch(batch_dir: Path) -> None:
    tasks = [
        {
            "task_id": f"alg{tool_idx:02d}__seed520__dataset_{dataset_idx:04d}",
            "algorithm": f"alg{tool_idx:02d}",
            "dataset_id": f"dataset_{dataset_idx:04d}",
            "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            "seed": "520",
            "timeout_in_seconds": "3600",
            "progress_snapshot_interval_seconds": "60",
        }
        for tool_idx in range(15)
        for dataset_idx in range(50)
    ]
    datasets = [
        {
            "dataset_id": f"dataset_{dataset_idx:04d}",
            "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
        }
        for dataset_idx in range(50)
    ]
    algorithms = {f"alg{tool_idx:02d}": {"env": "sim_base", "regressor": "Reg"} for tool_idx in range(15)}
    _write_csv(batch_dir / "manifest" / "tasks.csv", tasks)
    _write_csv(batch_dir / "manifest" / "datasets.csv", datasets)
    (batch_dir / "manifest" / "algorithms.json").write_text(json.dumps(algorithms), encoding="utf-8")
    _write_csv(
        batch_dir / "queues" / "ssr50_source.csv",
        [
            {
                "global_index": str(dataset_idx + 1),
                "dataset_id": f"dataset_{dataset_idx:04d}",
                "dataset_name": f"dataset_{dataset_idx:04d}",
                "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
                "dataset_rel": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            }
            for dataset_idx in range(50)
        ],
    )
    _write_csv(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        [
            {
                "global_index": str(dataset_idx + 1),
                "dataset_id": f"dataset_{dataset_idx:04d}",
                "dataset_name": f"dataset_{dataset_idx:04d}",
                "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
                "dataset_rel": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            }
            for dataset_idx in range(2)
        ],
    )
    for script in (
        "00_sync_code_and_batch_to_iaaccn22.sh",
        "01_preflight_from_iaaccn22.sh",
        "02_smoke_dispatch_from_iaaccn22.sh",
        "03_full_dispatch_from_iaaccn22.sh",
    ):
        path = batch_dir / "deploy" / script
        path.parent.mkdir(parents=True, exist_ok=True)
        if script.startswith("00_sync"):
            path.write_text(
                "#!/usr/bin/env bash\n"
                "RSYNC_FILTERS=(\"--exclude=.git/\" \"--exclude=__pycache__/\" \"--exclude=*.pyc\" \"--exclude=*.pyo\")\n"
                "SYNC_ITEMS=(\n"
                "  \"check/run_e1_candidate200_12alg_load_queue.py\"\n"
                "  \"check/launch_e1_benchmark.py\"\n"
                "  \"scientific_intelligent_modelling/\"\n"
                "  \"benchmark-control/compliance/\"\n"
                "  \"exp-planning/02.E1选择验证/generated/params/\"\n"
                ")\n"
                'rsync -aR "${RSYNC_FILTERS[@]}" "${SYNC_ITEMS[@]}" dest\n',
                encoding="utf-8",
            )
        elif script.startswith("01_preflight"):
            path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python check/run_e1_candidate200_12alg_load_queue.py\n"
                "python benchmark-control/compliance/launchers/check_preflight_report.py\n",
                encoding="utf-8",
            )
        elif script.startswith("02_smoke"):
            path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python - <<'PY'\n"
                "path = 'benchmark-runs/compliance/latest/preflight/preflight_gate_summary.json'\n"
                "ready_for_smoke = True\n"
                "PY\n"
                "python check/run_e1_candidate200_12alg_load_queue.py\n",
                encoding="utf-8",
            )
        else:
            path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python - <<'PY'\n"
                "path = 'benchmark-runs/compliance/latest/smoke/audit/audit_gate_summary.json'\n"
                "audit_passed = True\n"
                "PY\n"
                "python check/run_e1_candidate200_12alg_load_queue.py\n",
                encoding="utf-8",
            )


def test_readiness_passes_for_complete_stage1_batch(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    _write_ready_batch(batch_dir)

    summary = readiness.check_readiness(batch_dir=batch_dir)

    assert summary["ready"] is True
    assert summary["total_tasks"] == 750
    assert summary["total_algorithms"] == 15
    assert summary["total_datasets"] == 50
    assert summary["issues"] == []
    saved = json.loads((batch_dir / "readiness" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert saved["ready"] is True


def test_readiness_passes_for_formal24h_batch(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    tools = (
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
    noises = (("clean", "0"), ("noise001", "0.01"), ("noise005", "0.05"))
    tasks = [
        {
            "task_id": f"{tool}__seed{seed}__{noise_tag}__dataset_{dataset_idx:04d}",
            "algorithm": tool,
            "dataset_id": f"dataset_{dataset_idx:04d}",
            "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            "seed": str(seed),
            "noise_tag": noise_tag,
            "noise_sigma": noise_sigma,
            "timeout_in_seconds": "86400",
            "min_runtime_seconds": "82800",
            "progress_snapshot_interval_seconds": "60",
        }
        for tool in tools
        for seed in (520, 521, 522)
        for noise_tag, noise_sigma in noises
        for dataset_idx in range(50)
    ]
    datasets = [
        {
            "dataset_id": f"dataset_{dataset_idx:04d}",
            "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
        }
        for dataset_idx in range(50)
    ]
    algorithms = {tool: {"env": "sim_base", "regressor": tool} for tool in tools}
    _write_csv(batch_dir / "manifest" / "tasks.csv", tasks)
    _write_csv(batch_dir / "manifest" / "datasets.csv", datasets)
    _write_csv(
        batch_dir / "manifest" / "noise_levels.csv",
        [{"noise_tag": noise_tag, "noise_sigma": noise_sigma} for noise_tag, noise_sigma in noises],
    )
    (batch_dir / "manifest" / "algorithms.json").write_text(json.dumps(algorithms), encoding="utf-8")
    _write_csv(
        batch_dir / "queues" / "ssr50_source.csv",
        [
            {
                "global_index": str(dataset_idx + 1),
                "dataset_id": f"dataset_{dataset_idx:04d}",
                "dataset_name": f"dataset_{dataset_idx:04d}",
                "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
                "dataset_rel": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            }
            for dataset_idx in range(50)
        ],
    )
    _write_csv(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        [
            {
                "global_index": str(dataset_idx + 1),
                "dataset_id": f"dataset_{dataset_idx:04d}",
                "dataset_name": f"dataset_{dataset_idx:04d}",
                "dataset_dir": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
                "dataset_rel": f"sim-datasets-data/ssr50/dataset_{dataset_idx:04d}",
            }
            for dataset_idx in range(2)
        ],
    )
    _write_ready_batch(batch_dir)
    _write_csv(batch_dir / "manifest" / "tasks.csv", tasks)
    _write_csv(batch_dir / "manifest" / "datasets.csv", datasets)
    _write_csv(
        batch_dir / "manifest" / "noise_levels.csv",
        [{"noise_tag": noise_tag, "noise_sigma": noise_sigma} for noise_tag, noise_sigma in noises],
    )
    (batch_dir / "manifest" / "algorithms.json").write_text(json.dumps(algorithms), encoding="utf-8")
    for tool in tools:
        params_tool = {"QLattice": "qlattice", "iMCTS": "imcts"}.get(tool, tool)
        for noise_tag, noise_sigma in noises:
            (batch_dir / "params").mkdir(parents=True, exist_ok=True)
            (batch_dir / "params" / f"{params_tool}__{noise_tag}.json").write_text(
                json.dumps(
                    {
                        "timeout_in_seconds": 86400,
                        "progress_snapshot_interval_seconds": 60,
                        "train_label_noise_sigma": float(noise_sigma),
                    }
                ),
                encoding="utf-8",
            )

    summary = readiness.check_readiness(batch_dir=batch_dir, profile="formal24h_13alg_3seed_3noise")

    assert summary["ready"] is True
    assert summary["total_tasks"] == 5850
    assert summary["total_algorithms"] == 13
    assert summary["total_noise_levels"] == 3
    assert summary["issues"] == []


def test_readiness_reports_missing_full_queue_and_scripts(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    _write_ready_batch(batch_dir)
    (batch_dir / "queues" / "ssr50_source.csv").unlink()
    (batch_dir / "deploy" / "02_smoke_dispatch_from_iaaccn22.sh").unlink()

    summary = readiness.check_readiness(batch_dir=batch_dir)

    assert summary["ready"] is False
    assert "queues/ssr50_source.csv missing" in summary["issues"]
    assert "deploy/02_smoke_dispatch_from_iaaccn22.sh missing" in summary["issues"]


def test_readiness_reports_incomplete_sync_script_contract(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    _write_ready_batch(batch_dir)
    (batch_dir / "deploy" / "00_sync_code_and_batch_to_iaaccn22.sh").write_text(
        "#!/usr/bin/env bash\nrsync -aR check/run_e1_candidate200_12alg_load_queue.py dest\n",
        encoding="utf-8",
    )

    summary = readiness.check_readiness(batch_dir=batch_dir)

    assert summary["ready"] is False
    assert "deploy/00_sync_code_and_batch_to_iaaccn22.sh missing sync item scientific_intelligent_modelling/" in summary["issues"]
    assert "deploy/00_sync_code_and_batch_to_iaaccn22.sh missing rsync filter --exclude=.git/" in summary["issues"]
    assert "deploy/00_sync_code_and_batch_to_iaaccn22.sh does not use RSYNC_FILTERS array" in summary["issues"]


def test_readiness_reports_missing_stage_gate_contracts(tmp_path: Path) -> None:
    readiness = load_for_test("readiness")
    batch_dir = tmp_path / "batch"
    _write_ready_batch(batch_dir)
    for script in ("02_smoke_dispatch_from_iaaccn22.sh", "03_full_dispatch_from_iaaccn22.sh"):
        (batch_dir / "deploy" / script).write_text(
            "#!/usr/bin/env bash\npython check/run_e1_candidate200_12alg_load_queue.py\n",
            encoding="utf-8",
        )

    summary = readiness.check_readiness(batch_dir=batch_dir)

    assert summary["ready"] is False
    assert "deploy/02_smoke_dispatch_from_iaaccn22.sh missing preflight gate check" in summary["issues"]
    assert "deploy/03_full_dispatch_from_iaaccn22.sh missing smoke audit gate check" in summary["issues"]


def test_readiness_launcher_resolves_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "check_stage1_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_check_stage1_readiness", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    fake_root = tmp_path / "repo"
    calls: list[Path] = []

    def fake_check_readiness(*, batch_dir: Path, profile: str = "stage1_1h") -> dict[str, object]:
        calls.append(batch_dir)
        assert profile == "stage1_1h"
        return {"ready": True, "issues": []}

    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(launcher, "check_readiness", fake_check_readiness)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_stage1_readiness.py",
            "--batch-dir",
            "benchmark-runs/compliance/latest",
        ],
    )

    assert launcher.main() == 0
    assert calls == [fake_root / "benchmark-runs" / "compliance" / "latest"]
