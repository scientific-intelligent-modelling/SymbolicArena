from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_launcher():
    path = Path("benchmark-control/compliance/launchers/write_full3h_queue_commands.py")
    spec = importlib.util.spec_from_file_location("full3h_commands", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_full3h_commands_use_formal3h_paths(tmp_path: Path) -> None:
    launcher = _load_launcher()
    batch_dir = tmp_path / "benchmark-runs" / "formal3h" / "batch"
    paths = launcher.write_full3h_queue_commands(batch_dir=batch_dir)
    assert {path.name for path in paths} == {
        "00_sync_code_and_batch_to_iaaccn22.sh",
        "01_preflight_from_iaaccn22.sh",
        "02_recover_from_formal24h_from_iaaccn22.sh",
        "03_smoke_dispatch_from_iaaccn22.sh",
        "04_full_dispatch_from_iaaccn22.sh",
    }

    scripts = {path.name: path.read_text(encoding="utf-8") for path in paths}
    assert "benchmark-runs/formal3h/latest" in scripts["00_sync_code_and_batch_to_iaaccn22.sh"]
    assert "--exclude=benchmark-runs/formal3h/*/remote-experiments/" in scripts["00_sync_code_and_batch_to_iaaccn22.sh"]
    assert "recover_formal3h_from_24h.py" in scripts["02_recover_from_formal24h_from_iaaccn22.sh"]
    assert "--source-batch formal24h_13alg_ssr50_seed520-522_noise0-001-005_20260531-230535" in scripts[
        "02_recover_from_formal24h_from_iaaccn22.sh"
    ]
    assert "--params-root benchmark-runs/formal3h/latest/params_smoke" in scripts[
        "03_smoke_dispatch_from_iaaccn22.sh"
    ]
    assert "--params-root benchmark-runs/formal3h/latest/params" in scripts["04_full_dispatch_from_iaaccn22.sh"]
    assert "--task-id-allowlist-csv benchmark-runs/formal3h/latest/recovery/missing_tasks.csv" in scripts[
        "04_full_dispatch_from_iaaccn22.sh"
    ]
    assert "--session-prefix formal3h_full_" in scripts["04_full_dispatch_from_iaaccn22.sh"]
