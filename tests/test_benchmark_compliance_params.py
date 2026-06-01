from __future__ import annotations

import json
from pathlib import Path


def test_generate_noise_params_writes_13_by_3_files(tmp_path: Path) -> None:
    from benchmark_control_compliance_manifest_import import load_for_test

    module = load_for_test("params")
    source = tmp_path / "source"
    source.mkdir()
    tools = (
        "gplearn",
        "pyoperon",
        "pysr",
        "dso",
        "tpsr",
        "e2esr",
        "fepysr",
        "jaxsr",
        "qlattice",
        "imcts",
        "udsr",
        "ragsr",
        "symbolfit",
    )
    for tool in tools:
        payload = {"timeout_in_seconds": 3600, "niterations": 1000000}
        if tool == "pyoperon":
            payload["max_evaluations"] = 500000
        (source / f"{tool}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    output = tmp_path / "params"
    summary = module.generate_noise_params(
        source_params_root=source,
        output_params_root=output,
        tools=tools,
        noise_sigmas=(0.0, 0.01, 0.05),
        timeout_in_seconds=86400,
        progress_snapshot_interval_seconds=60,
    )

    assert summary == {"tools": 13, "noise_levels": 3, "params_files": 39}
    clean = json.loads((output / "pysr__clean.json").read_text(encoding="utf-8"))
    noisy = json.loads((output / "pysr__noise001.json").read_text(encoding="utf-8"))
    pyoperon = json.loads((output / "pyoperon__clean.json").read_text(encoding="utf-8"))
    gplearn = json.loads((output / "gplearn__clean.json").read_text(encoding="utf-8"))
    assert clean["timeout_in_seconds"] == 86400
    assert clean["train_label_noise_enabled"] is False
    assert clean["train_label_noise_sigma"] == 0.0
    assert pyoperon["max_evaluations"] == 1152000000
    assert gplearn["n_jobs"] == 1
    assert gplearn["low_memory"] is True
    assert noisy["progress_snapshot_interval_seconds"] == 60
    assert noisy["train_label_noise_enabled"] is True
    assert noisy["train_label_noise_sigma"] == 0.01
