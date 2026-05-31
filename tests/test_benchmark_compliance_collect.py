from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from benchmark_control_compliance_manifest_import import load_for_test


def test_collect_uses_controller_local_path_and_internal_ips(tmp_path: Path) -> None:
    collect = load_for_test("remote_collect")
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    batch_dir = tmp_path / "batch"
    summary = collect.collect_remote_batch(
        batch_dir=batch_dir,
        batch_id="compliance_batch",
        hosts=["iaaccn22", "iaaccn23"],
        remote_root=Path("/home/zhangziwen/workplace/scientific-intelligent-modelling"),
        controller_host="iaaccn22",
        use_internal_ips=True,
        runner=fake_run,
    )

    assert summary == {"total_hosts": 2, "succeeded": 2, "failed": 0}
    assert commands[0][-2:] == [
        "/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/compliance_batch/",
        str(batch_dir / "remote-experiments" / "iaaccn22") + "/",
    ]
    assert commands[1][-2:] == [
        "10.10.100.23:/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/compliance_batch/",
        str(batch_dir / "remote-experiments" / "iaaccn23") + "/",
    ]
    assert "--prune-empty-dirs" in commands[1]
    assert "--include" in commands[1]

    payload = json.loads((batch_dir / "collect" / "collect_summary.json").read_text(encoding="utf-8"))
    assert payload["summary"] == summary
    assert [item["host"] for item in payload["results"]] == ["iaaccn22", "iaaccn23"]


def test_collect_records_timeout_and_continues_other_hosts(tmp_path: Path) -> None:
    collect = load_for_test("remote_collect")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if len(calls) == 2:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5, output="partial", stderr="hung")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    batch_dir = tmp_path / "batch"
    summary = collect.collect_remote_batch(
        batch_dir=batch_dir,
        batch_id="compliance_batch",
        hosts=["iaaccn22", "iaaccn25", "iaaccn26"],
        remote_root=Path("/home/zhangziwen/workplace/scientific-intelligent-modelling"),
        controller_host="iaaccn22",
        use_internal_ips=True,
        timeout=5,
        runner=fake_run,
    )

    assert summary == {"total_hosts": 3, "succeeded": 2, "failed": 1}
    assert len(calls) == 3
    payload = json.loads((batch_dir / "collect" / "collect_summary.json").read_text(encoding="utf-8"))
    timeout_result = payload["results"][1]
    assert timeout_result["host"] == "iaaccn25"
    assert timeout_result["returncode"] == 124
    assert "timed out after 5 seconds" in timeout_result["stderr"]


def test_collect_launcher_resolves_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "collect_remote_batch.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_collect_remote_batch", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    fake_root = tmp_path / "repo"
    calls: list[dict[str, object]] = []

    def fake_collect_remote_batch(**kwargs: object) -> dict[str, int]:
        calls.append(kwargs)
        return {"total_hosts": 1, "succeeded": 1, "failed": 0}

    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(launcher, "collect_remote_batch", fake_collect_remote_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_remote_batch.py",
            "--batch-dir",
            "benchmark-runs/compliance/batch",
            "--batch-id",
            "compliance_batch",
            "--hosts",
            "iaaccn22",
            "--controller-host",
            "iaaccn22",
            "--use-internal-ips",
        ],
    )

    assert launcher.main() == 0
    assert calls[0]["batch_dir"] == fake_root / "benchmark-runs" / "compliance" / "batch"
    assert calls[0]["batch_id"] == "compliance_batch"
    assert calls[0]["hosts"] == ["iaaccn22"]
    assert calls[0]["controller_host"] == "iaaccn22"
    assert calls[0]["use_internal_ips"] is True
