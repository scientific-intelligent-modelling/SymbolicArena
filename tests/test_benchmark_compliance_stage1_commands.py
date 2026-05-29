from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_launcher():
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "write_stage1_queue_commands.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_stage1_commands", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_stage1_queue_commands_generates_preflight_smoke_and_full(tmp_path: Path) -> None:
    launcher = _load_launcher()
    batch_dir = tmp_path / "benchmark-runs" / "compliance" / "batch"
    (batch_dir / "queues").mkdir(parents=True)
    (batch_dir / "queues" / "smoke_2datasets_source.csv").write_text("header\n", encoding="utf-8")
    (batch_dir / "queues" / "ssr50_source.csv").write_text("header\n", encoding="utf-8")

    paths = launcher.write_stage1_queue_commands(batch_dir=batch_dir)

    assert [path.name for path in paths] == [
        "01_preflight_from_iaaccn22.sh",
        "02_smoke_dispatch_from_iaaccn22.sh",
        "03_full_dispatch_from_iaaccn22.sh",
    ]
    preflight = paths[0].read_text(encoding="utf-8")
    smoke = paths[1].read_text(encoding="utf-8")
    full = paths[2].read_text(encoding="utf-8")
    assert "--preflight-only" in preflight
    assert "--source-csv benchmark-runs/compliance/latest/queues/smoke_2datasets_source.csv" in preflight
    assert "check_preflight_report.py" in preflight
    assert "preflight_gate_summary.json" in smoke
    assert "ready_for_smoke" in smoke
    assert "--batch-name \"${BATCH_ID}_smoke\"" in smoke
    assert "--expected-rows 2" in smoke
    assert "prepare_smoke_batch.py" in smoke
    assert "--batch-id \"${BATCH_ID}_smoke\"" in smoke
    assert "check_audit_success.py" in smoke
    assert "--expected-total-tasks 30" in smoke
    assert "smoke/audit/audit_gate_summary.json" in full
    assert "audit_passed" in full
    assert "--batch-name \"${BATCH_ID}\"" in full
    assert "--expected-rows 50" in full
    assert "--source-csv benchmark-runs/compliance/latest/queues/ssr50_source.csv" in full
    assert "--batch-id \"${BATCH_ID}\"" in full
    assert "collect_remote_batch.py" in full
    assert "harvest_batch.py" in full
    assert "audit_batch.py" in full
    assert "check_audit_success.py" in full
    assert "--expected-total-tasks 750" in full
    assert "SIM_QUEUE_CONTROLLER_IS_LOCAL=1" in preflight


def test_stage1_queue_commands_launcher_resolves_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    launcher = _load_launcher()
    fake_root = tmp_path / "repo"
    calls: list[Path] = []

    def fake_write_stage1_queue_commands(*, batch_dir: Path) -> list[Path]:
        calls.append(batch_dir)
        return [batch_dir / "deploy" / "01_preflight_from_iaaccn22.sh"]

    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(launcher, "write_stage1_queue_commands", fake_write_stage1_queue_commands)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_stage1_queue_commands.py",
            "--batch-dir",
            "benchmark-runs/compliance/latest",
        ],
    )

    assert launcher.main() == 0
    assert calls == [fake_root / "benchmark-runs" / "compliance" / "latest"]
