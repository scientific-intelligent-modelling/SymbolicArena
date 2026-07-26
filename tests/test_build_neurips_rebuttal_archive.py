from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "build_neurips_rebuttal_archive.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_neurips_rebuttal_archive",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_archive_fixture(batch_dir: Path) -> None:
    _write(batch_dir / "README.md", "readme\n")
    _write(batch_dir / "BATCH_NAME.txt", "batch\n")
    _write(batch_dir / "EXPERIMENT_PATHS.md", "paths\n")
    _write(batch_dir / "SOURCE_MAP.tsv", "path\tsource\n")
    _write(batch_dir / "manifest" / "tasks.csv", "task_id\none\n")
    _write(batch_dir / "analysis" / "summary.json", "{}\n")
    _write(
        batch_dir / "runs" / "fepysr" / "seed520" / "g0001" / "result.json",
        "{}\n",
    )
    _write(batch_dir / "runs" / ".worker.lock", "")
    _write(batch_dir / "runs" / "partial.tmp", "partial\n")
    _write(batch_dir / "analysis" / ".summary.tmp.123", "partial\n")
    _write(batch_dir / "remote-experiments" / "duplicate.json", "{}\n")
    _write(batch_dir / "runtime_queue" / "request.json", "{}\n")
    _write(
        batch_dir / "smoke" / "remote-experiments" / "duplicate.json",
        "{}\n",
    )
    _write(batch_dir / "smoke" / "runtime_queue" / "request.json", "{}\n")


def test_archive_scope_is_sorted_and_excludes_staging_and_volatile_files(
    tmp_path: Path,
) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _make_archive_fixture(batch_dir)

    paths = [
        path.relative_to(batch_dir).as_posix()
        for path in module.collect_archive_files(batch_dir)
    ]

    assert paths == sorted(paths, key=str.encode)
    assert "README.md" in paths
    assert "analysis/summary.json" in paths
    assert "runs/fepysr/seed520/g0001/result.json" in paths
    assert "runs/.worker.lock" not in paths
    assert "runs/partial.tmp" not in paths
    assert "analysis/.summary.tmp.123" not in paths
    assert not any(path.startswith("remote-experiments/") for path in paths)
    assert not any(path.startswith("runtime_queue/") for path in paths)
    assert not any("/remote-experiments/" in path for path in paths)
    assert not any("/runtime_queue/" in path for path in paths)


def test_write_and_verify_detects_tampering_and_unexpected_scope_file(
    tmp_path: Path,
) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _make_archive_fixture(batch_dir)

    summary = module.write_archive(batch_dir, validate_final=False)
    assert summary["files"] == len(module.collect_archive_files(batch_dir))
    assert module.verify_archive(
        batch_dir,
        require_exact_scope=True,
        validate_final=False,
    )["valid"]

    _write(batch_dir / "analysis" / "late.csv", "late\n")
    with pytest.raises(RuntimeError, match="归档范围"):
        module.verify_archive(
            batch_dir,
            require_exact_scope=True,
            validate_final=False,
        )
    assert module.verify_archive(
        batch_dir,
        require_exact_scope=False,
        validate_final=False,
    )["valid"]

    _write(batch_dir / "README.md", "mutate\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        module.verify_archive(
            batch_dir,
            require_exact_scope=False,
            validate_final=False,
        )


def test_archive_rejects_symlinks_inside_declared_scope(tmp_path: Path) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _make_archive_fixture(batch_dir)
    (batch_dir / "analysis" / "linked.json").symlink_to(
        batch_dir / "analysis" / "summary.json"
    )

    with pytest.raises(RuntimeError, match="软链接"):
        module.collect_archive_files(batch_dir)


def test_manifest_and_checksums_have_stable_standard_format(
    tmp_path: Path,
) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _make_archive_fixture(batch_dir)

    module.write_archive(batch_dir, validate_final=False)

    manifest_lines = (
        (batch_dir / "MANIFEST.tsv").read_text(encoding="utf-8").splitlines()
    )
    checksum_lines = (
        (batch_dir / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    )
    assert manifest_lines[0] == "relative_path\tsize_bytes"
    assert len(checksum_lines) == len(manifest_lines)
    assert checksum_lines[0].endswith("  ./MANIFEST.tsv")
    assert all(len(line.split("  ./", maxsplit=1)[0]) == 64 for line in checksum_lines)


def test_final_gate_rejects_running_finalization(tmp_path: Path) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _write(
        batch_dir / "deploy" / "finalization_status.json",
        json.dumps(
            {
                "state": "running",
                "exit_code": None,
                "ended_at": None,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="finalization"):
        module.validate_final_artifacts(batch_dir)


def test_final_gate_rejects_nonfinal_analysis(tmp_path: Path) -> None:
    module = _load_module()
    batch_dir = tmp_path / "batch"
    _write(
        batch_dir / "deploy" / "finalization_status.json",
        json.dumps(
            {
                "state": "finished",
                "exit_code": 0,
                "ended_at": "2026-07-26T13:00:00+08:00",
            }
        ),
    )
    _write(
        batch_dir / "analysis" / "analysis_summary.json",
        json.dumps({"final_ready": False}),
    )

    with pytest.raises(RuntimeError, match="final_ready"):
        module.validate_final_artifacts(batch_dir)


def test_result_gate_requires_exact_manifest_path_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "EXPECTED_TASKS", 1)
    monkeypatch.setattr(module, "EXPECTED_DATASETS", 1)
    monkeypatch.setattr(module, "EXPECTED_NEW_ALGORITHMS", {"fepysr"})
    monkeypatch.setattr(module, "EXPECTED_SEEDS", {520})
    batch_dir = tmp_path / "batch"
    _write(
        batch_dir / "manifest" / "tasks.csv",
        (
            "task_id,algorithm,dataset_id,seed,noise_tag\n"
            "fepysr__seed520__clean__g0001,fepysr,g0001,520,clean\n"
        ),
    )
    _write(
        batch_dir / "runs" / "fepysr" / "seed520" / "clean" / "g9999" / "result.json",
        "{}\n",
    )

    with pytest.raises(RuntimeError, match="一一对应"):
        module._validate_task_and_result_counts(batch_dir)


def test_verify_cli_requires_exact_scope_flag(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--batch-dir",
            str(tmp_path),
            "--verify",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--require-exact-scope" in completed.stderr
