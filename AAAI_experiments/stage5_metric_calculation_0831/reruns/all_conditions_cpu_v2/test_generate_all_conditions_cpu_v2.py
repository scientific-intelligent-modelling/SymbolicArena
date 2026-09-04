from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_all_conditions_cpu_v2.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("all_conditions_cpu_v2_generator", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generator: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_merges_exactly_143_noise_tasks(tmp_path):
    module = _load_module()
    asset_root = tmp_path / "all_conditions_cpu_v2"

    result = module.generate_assets(asset_root=asset_root, run_dry_run=True)

    assert result["task_count"] == 143
    assert result["algorithm_counts"] == {"drsr": 1, "imcts": 79, "jaxsr": 63}
    assert result["condition_counts"] == {"noise001": 79, "noise005": 64}
    assert result["task_ids_unique"] is True
    assert result["max_cpu_used_ratio"] == 0.95
    assert result["max_jobs_per_host"] == 18
    assert result["max_new_jobs_per_host_per_poll"] == 18
    assert result["first_round_capacity"] == 144
    assert result["first_round_tasks_fit"] is True

    allowlist = asset_root / "manifests/all_conditions_noise_task_allowlist.csv"
    with allowlist.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 143
    assert len({row["task_id"] for row in rows}) == 143

    dry_run = json.loads(
        (asset_root / "reports/dry_run_validation.json").read_text(encoding="utf-8")
    )
    assert dry_run["status"] == "ok"
    assert dry_run["future_flag_excluded"] is None
    assert dry_run["task_count"] == 143
    assert dry_run["algorithm_counts"] == {"drsr": 1, "imcts": 79, "jaxsr": 63}
    assert dry_run["condition_counts"] == {"noise001": 79, "noise005": 64}


def test_generated_commands_are_single_queue_and_cpu_guarded(tmp_path):
    module = _load_module()
    asset_root = tmp_path / "all_conditions_cpu_v2"
    module.generate_assets(asset_root=asset_root, run_dry_run=False)

    expected_hosts = "iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29"
    for name in (
        "run_all_conditions_preflight.sh",
        "run_all_conditions_dry_run.sh",
        "run_all_conditions_formal.sh",
    ):
        content = (asset_root / "commands" / name).read_text(encoding="utf-8")
        assert "--hosts " + expected_hosts in content
        assert "--tools jaxsr imcts drsr" in content
        assert "--noise-sigmas 0.01 0.05" in content
        assert "--task-id-allowlist-csv" in content
        assert "--max-cpu-used-ratio 0.95" in content
        assert "--max-jobs-per-host 18" in content
        assert "--max-new-jobs-per-host-per-poll 18" in content
        assert '"0.50:18,0.70:18,0.85:18,0.95:18"' in content
        assert "export SIM_QUEUE_CONTROLLER_IS_LOCAL=1" in content
        assert "/tmp" not in content

    formal_params = asset_root / "params"
    for name in (
        "jaxsr__noise001.json",
        "jaxsr__noise005.json",
        "imcts__noise001.json",
        "imcts__noise005.json",
        "drsr__noise005.json",
        "drsr.json",
    ):
        payload = json.loads((formal_params / name).read_text(encoding="utf-8"))
        assert payload["timeout_in_seconds"] == 10800
        assert payload["progress_snapshot_interval_seconds"] == 60

    readme = (asset_root / "README.md").read_text(encoding="utf-8")
    assert "4500" in readme
    assert "无需全重跑" in readme
    assert "8 x 18 = 144" in readme
    assert "不代表物理占满 2048 核" in readme
