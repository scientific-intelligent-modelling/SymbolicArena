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
        / "write_remote_sync_commands.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_remote_sync_commands", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_remote_sync_commands_generates_reviewable_sync_script(tmp_path: Path) -> None:
    launcher = _load_launcher()
    batch_dir = tmp_path / "benchmark-runs" / "compliance" / "batch"
    batch_dir.mkdir(parents=True)

    path = launcher.write_remote_sync_commands(batch_dir=batch_dir)

    assert path.name == "00_sync_code_and_batch_to_iaaccn22.sh"
    content = path.read_text(encoding="utf-8")
    assert "rsync -aR" in content
    assert "--delete-after" not in content
    assert "iaaccn22:/home/zhangziwen/workplace/scientific-intelligent-modelling/" in content
    assert "10.10.100.23" in content
    assert "10.10.100.29" in content
    assert "benchmark-runs/compliance/" in content
    assert 'BATCH_ID="batch"' in content
    assert 'ln -sfn \\"$BATCH_ID\\" benchmark-runs/compliance/latest' in content
    assert "check/run_e1_candidate200_12alg_load_queue.py" in content
    assert "benchmark-control/compliance/" in content
    assert "tmux new-session" not in content
    assert "run_e1_candidate200_12alg_load_queue.py \\" not in content


def test_remote_sync_commands_launcher_resolves_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    launcher = _load_launcher()
    fake_root = tmp_path / "repo"
    calls: list[Path] = []

    def fake_write_remote_sync_commands(*, batch_dir: Path) -> Path:
        calls.append(batch_dir)
        return batch_dir / "deploy" / "00_sync_code_and_batch_to_iaaccn22.sh"

    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(launcher, "write_remote_sync_commands", fake_write_remote_sync_commands)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_remote_sync_commands.py",
            "--batch-dir",
            "benchmark-runs/compliance/latest",
        ],
    )

    assert launcher.main() == 0
    assert calls == [fake_root / "benchmark-runs" / "compliance" / "latest"]
