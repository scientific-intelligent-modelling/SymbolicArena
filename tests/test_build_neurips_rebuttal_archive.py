from __future__ import annotations

import importlib.util
from pathlib import Path

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
    _write(batch_dir / "analysis" / ".summary.tmp.123", "partial\n")
    _write(batch_dir / "remote-experiments" / "duplicate.json", "{}\n")
    _write(batch_dir / "runtime_queue" / "request.json", "{}\n")


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
    assert "analysis/.summary.tmp.123" not in paths
    assert not any(path.startswith("remote-experiments/") for path in paths)
    assert not any(path.startswith("runtime_queue/") for path in paths)


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
