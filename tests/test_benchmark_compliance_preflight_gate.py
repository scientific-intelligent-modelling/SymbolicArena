from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from benchmark_control_compliance_manifest_import import load_for_test


def _valid_report(hosts: list[str]) -> dict[str, object]:
    return {
        "requested_tools": ["gplearn"],
        "requested_envs": ["sim_base"],
        "hosts": [
            {
                "host": host,
                "ok": True,
                "files": {
                    "scheduler": {"matches_expected": True},
                    "launcher": {"matches_expected": True},
                    "runner": {"matches_expected": True},
                    "toolbox_config": {"matches_expected": True},
                    "source_csv": {"matches_expected": True},
                    "gplearn_wrapper": {"matches_expected": True},
                    "gplearn_params": {"matches_expected": True},
                },
                "dataset_sync": {"all_match": True},
                "params": {"gplearn": {"exists": True, "json_ok": True}},
                "envs": {"envs": {"sim_base": {"ok": True}}},
            }
            for host in hosts
        ],
    }


def test_preflight_gate_passes_when_all_hosts_match(tmp_path: Path) -> None:
    gate = load_for_test("preflight_gate")
    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(_valid_report(["iaaccn22", "iaaccn23"])), encoding="utf-8")

    summary = gate.check_preflight_report(
        report_path=report_path,
        expected_hosts=["iaaccn22", "iaaccn23"],
        batch_dir=tmp_path / "batch",
    )

    assert summary["ready_for_smoke"] is True
    assert summary["issues"] == []
    saved = json.loads((tmp_path / "batch" / "preflight" / "preflight_gate_summary.json").read_text(encoding="utf-8"))
    assert saved["ready_for_smoke"] is True


def test_preflight_gate_reports_mismatch_missing_host_and_failed_env(tmp_path: Path) -> None:
    gate = load_for_test("preflight_gate")
    report = _valid_report(["iaaccn22"])
    host = report["hosts"][0]
    host["files"]["scheduler"]["matches_expected"] = False
    host["envs"]["envs"]["sim_base"]["ok"] = False
    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = gate.check_preflight_report(
        report_path=report_path,
        expected_hosts=["iaaccn22", "iaaccn23"],
        batch_dir=tmp_path / "batch",
    )

    assert summary["ready_for_smoke"] is False
    assert "missing host in preflight report: iaaccn23" in summary["issues"]
    assert "iaaccn22 file scheduler mismatch" in summary["issues"]
    assert "iaaccn22 env sim_base import failed" in summary["issues"]


def test_preflight_gate_requires_tool_wrapper_and_param_file_hashes(tmp_path: Path) -> None:
    gate = load_for_test("preflight_gate")
    report = _valid_report(["iaaccn22"])
    host = report["hosts"][0]
    del host["files"]["gplearn_wrapper"]
    del host["files"]["gplearn_params"]
    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = gate.check_preflight_report(
        report_path=report_path,
        expected_hosts=["iaaccn22"],
        batch_dir=tmp_path / "batch",
    )

    assert summary["ready_for_smoke"] is False
    assert "iaaccn22 file gplearn_wrapper missing" in summary["issues"]
    assert "iaaccn22 file gplearn_params missing" in summary["issues"]


def test_preflight_gate_launcher_exits_nonzero_when_not_ready(tmp_path: Path, monkeypatch) -> None:
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "check_preflight_report.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_check_preflight_report", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(_valid_report(["iaaccn22"])), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_preflight_report.py",
            "--batch-dir",
            str(tmp_path / "batch"),
            "--report",
            str(report_path),
            "--expected-hosts",
            "iaaccn22",
            "iaaccn23",
        ],
    )

    assert launcher.main() == 1
