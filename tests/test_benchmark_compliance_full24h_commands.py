from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_launcher():
    path = Path("benchmark-control/compliance/launchers/write_full24h_queue_commands.py")
    spec = importlib.util.spec_from_file_location("full24h_commands", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_full24h_queue_commands_use_13_tools_3_seeds_3_noise(tmp_path: Path) -> None:
    launcher = _load_launcher()
    batch_dir = tmp_path / "benchmark-runs" / "formal24h" / "batch"
    paths = launcher.write_full24h_queue_commands(batch_dir=batch_dir)
    assert {path.name for path in paths} == {
        "00_sync_code_and_batch_to_iaaccn22.sh",
        "01_preflight_from_iaaccn22.sh",
        "02_smoke_dispatch_from_iaaccn22.sh",
        "03_full_dispatch_from_iaaccn22.sh",
    }

    sync = (batch_dir / "deploy" / "00_sync_code_and_batch_to_iaaccn22.sh").read_text(encoding="utf-8")
    smoke = (batch_dir / "deploy" / "02_smoke_dispatch_from_iaaccn22.sh").read_text(encoding="utf-8")
    full = (batch_dir / "deploy" / "03_full_dispatch_from_iaaccn22.sh").read_text(encoding="utf-8")
    assert "benchmark-runs/formal24h/latest" in sync
    assert "--expected-total-tasks 234" in smoke
    assert "--tools gplearn pyoperon pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit" in full
    assert "--seeds 520 521 522" in full
    assert "--noise-sigmas 0 0.01 0.05" in full
    assert "--expected-total-tasks 5850" in full
    assert "compliance_1h_" not in full
