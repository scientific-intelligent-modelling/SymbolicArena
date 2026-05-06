#!/usr/bin/env python3
"""Build the anonymized external full artifact package.

This script intentionally writes large generated outputs only under
paper/neurips26-upload/external-artifact-hosting/{staging,dist}, which are
git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
ARTIFACT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]
STAGING_ROOT = ARTIFACT_ROOT / "staging"
DIST_ROOT = ARTIFACT_ROOT / "dist"
PAYLOAD_ROOT = STAGING_ROOT / "SymbolicArena_NeurIPS26_ED_external_full_artifact"
ARCHIVE_PATH = DIST_ROOT / "SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst"
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "EXTERNAL_ARTIFACT_SOURCE_MANIFEST.csv"
FILE_MANIFEST_PATH = DIST_ROOT / "EXTERNAL_ARTIFACT_FILE_MANIFEST.txt"
CHECKSUM_PATH = DIST_ROOT / "SHA256SUMS.txt"


@dataclass(frozen=True)
class ArtifactItem:
    artifact_id: str
    source: Path
    target: Path
    required: bool
    description: str


ITEMS = [
    ArtifactItem(
        "results_slim",
        Path("paper/neurips26-upload/results-artifact"),
        Path("results_slim"),
        True,
        "Lightweight final result tables used by the OpenReview upload package.",
    ),
    ArtifactItem(
        "dataset_metadata",
        Path("paper/neurips26-upload/dataset-metadata"),
        Path("dataset_metadata"),
        True,
        "Core-50 manifests, table CSVs, ground-truth formulas, and hyperparameter manifests.",
    ),
    ArtifactItem(
        "code_snapshot",
        Path("paper/neurips26-upload/code-artifact"),
        Path("code_snapshot"),
        True,
        "Minimal code snapshot sufficient to interpret wrappers and benchmark artifacts.",
    ),
    ArtifactItem(
        "clean_core50_raw",
        Path("experiments/core50_12alg_5seed_all_20260502-065700"),
        Path("raw_results/clean_core50_12alg_5seed"),
        True,
        "Raw Core-50 12-algorithm 5-seed clean experiment outputs.",
    ),
    ArtifactItem(
        "core50_formal_analysis",
        Path("exp-planning/04.Core50正式全量评测/analysis"),
        Path("derived_analysis/core50_formal_analysis"),
        True,
        "Derived formal metrics, symbolic fidelity, hexagon scores, and ablation analysis.",
    ),
    ArtifactItem(
        "core50_collection_manifest",
        Path("exp-planning/04.Core50正式全量评测/generated/core50_12alg_5seed_final_results"),
        Path("derived_analysis/core50_final_result_collection"),
        True,
        "Local collection summary and manifest for clean Core-50 runs.",
    ),
    ArtifactItem(
        "noise_robustness_raw",
        Path("exp-planning/05.Core50噪声鲁棒性评测/results"),
        Path("raw_results/noise_robustness"),
        True,
        "Raw noisy-train clean-test robustness outputs, excluding minute-level snapshots.",
    ),
    ArtifactItem(
        "noise_experiment_docs",
        Path("exp-planning/05.Core50噪声鲁棒性评测/README.md"),
        Path("docs/noise_experiment_README.md"),
        True,
        "Noise robustness experiment description.",
    ),
]


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".log",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
    ".toml",
    ".cfg",
    ".ini",
    ".tex",
    ".bib",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".ipynb_checkpoints",
    "controller",
    "__launcher__",
    "remote_sync",
    "orchestrator_logs",
    "load_queue",
}

EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".swp",
    ".log",
}

SENSITIVE_PATTERNS = [
    re.compile(r"/home/[^,\\s\"']+/workplace/scientific-intelligent-modelling"),
    re.compile(r"/home/[^,\\s\"']+/projects/scientific-intelligent-modelling"),
    re.compile(r"/home/[^,\\s\"']+/sim-datasets-data"),
    re.compile(r"/data[0-9]+/[^,\\s\"']+"),
    re.compile(r"/home/[^,\\s\"']+"),
    re.compile(r"i[a]accn(22|23|24|25|26|27|28|29|48|49|50|51|52|53|54|55)"),
    re.compile(r"10\\.10\\.100\\.\\d+"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)[^,\\n\\r]*"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"),
]

SENSITIVE_SCAN_PATTERNS = [
    re.compile(r"/home/[^,\\s\"']+"),
    re.compile(r"i[a]accn"),
    re.compile(r"10\\.10\\.100\\.\\d+"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)[^,\\n\\r]*"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"),
]


def rel(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    if path.name.startswith(".") and path.name not in {".gitkeep"}:
        return True
    if path.name.startswith("minute_") and path.suffix == ".json":
        return True
    return path.suffix in EXCLUDED_FILE_SUFFIXES


def iter_files(root: Path):
    if root.is_file():
        if not should_skip(root):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIR_NAMES and not name.startswith(".")]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if not should_skip(path):
                yield path


def path_for_artifact(path: Path) -> Path:
    parts = []
    for part in path.parts:
        part = re.sub(r"i[a]accn(\d+)", r"host_\1", part)
        parts.append(part)
    return Path(*parts)


def sanitize_text(text: str) -> str:
    replacements = [
        ("scientific-intelligent-modelling", "scientific-intelligent-modelling"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = re.sub(r"/home/[^,\\s\"']+/workplace/scientific-intelligent-modelling", "ANONYMIZED_REPO_ROOT", text)
    text = re.sub(r"/home/[^,\\s\"']+/projects/scientific-intelligent-modelling", "ANONYMIZED_REPO_ROOT", text)
    text = re.sub(r"/home/[^,\\s\"']+/sim-datasets-data", "ANONYMIZED_DATA_ROOT", text)
    text = re.sub(r"/data[0-9]+/[^,\\s\"']+", "ANONYMIZED_REMOTE_ROOT", text)
    text = re.sub(r"/home/[^,\\s\"']+", "ANONYMIZED_HOME_PATH", text)
    text = re.sub(r"i[a]accn(\d+)", r"host_\1", text)
    text = re.sub(r"10\\.10\\.100\\.\\d+", "ANONYMIZED_PRIVATE_IP", text)
    text = re.sub(r"(?i)(api[_-]?key|authorization|bearer)[^,\\n\\r]*", "ANONYMIZED_API_CREDENTIAL", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", "ANONYMIZED_EMAIL", text)
    return text


def source_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    count = 0
    for file_path in iter_files(path):
        try:
            total += file_path.stat().st_size
            count += 1
        except FileNotFoundError:
            continue
    return count, total


def command_inventory() -> None:
    rows = []
    for item in ITEMS:
        source = rel(item.source)
        file_count, bytes_total = source_stats(source)
        rows.append(
            {
                "artifact_id": item.artifact_id,
                "required": str(item.required).lower(),
                "exists": str(source.exists()).lower(),
                "source_path": str(item.source),
                "artifact_path": str(item.target),
                "file_count_after_exclusions": file_count,
                "bytes_after_exclusions": bytes_total,
                "size_human_after_exclusions": human_size(bytes_total),
                "description": item.description,
            }
        )
    SOURCE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        status = "OK" if row["exists"] == "true" else "MISSING"
        print(f"{status:7s} {row['artifact_id']:28s} {row['size_human_after_exclusions']:>10s} {row['source_path']}")
    print(f"wrote {SOURCE_MANIFEST_PATH}")


def copy_one_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIXES:
        data = src.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
            return
        dst.write_text(sanitize_text(text), encoding="utf-8", newline="")
        shutil.copystat(src, dst, follow_symlinks=True)
    else:
        shutil.copy2(src, dst)


def command_prepare() -> None:
    if PAYLOAD_ROOT.exists():
        shutil.rmtree(PAYLOAD_ROOT)
    PAYLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_required = []
    for item in ITEMS:
        source = rel(item.source)
        if not source.exists():
            if item.required:
                missing_required.append(str(item.source))
            print(f"MISSING {item.artifact_id}: {item.source}")
            continue
        if source.is_file():
            copy_one_file(source, PAYLOAD_ROOT / item.target)
            copied += 1
            continue
        for file_path in iter_files(source):
            rel_file = file_path.relative_to(source)
            dst = PAYLOAD_ROOT / item.target / path_for_artifact(rel_file)
            copy_one_file(file_path, dst)
            copied += 1
            if copied % 50000 == 0:
                print(f"copied {copied} files...")

    shutil.copy2(ARTIFACT_ROOT / "README_external_artifact.md", PAYLOAD_ROOT / "README_external_artifact.md")
    shutil.copy2(SOURCE_MANIFEST_PATH, PAYLOAD_ROOT / "EXTERNAL_ARTIFACT_SOURCE_MANIFEST.csv")

    if missing_required:
        raise SystemExit(f"missing required sources: {missing_required}")
    print(f"prepared {copied} files under {PAYLOAD_ROOT}")


def command_file_manifest() -> None:
    if not PAYLOAD_ROOT.exists():
        raise SystemExit(f"missing prepared payload: {PAYLOAD_ROOT}")
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(PAYLOAD_ROOT.rglob("*")):
        if path.is_file():
            rows.append(path.relative_to(PAYLOAD_ROOT).as_posix())
    FILE_MANIFEST_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {FILE_MANIFEST_PATH} ({len(rows)} files)")


def command_package() -> None:
    if not PAYLOAD_ROOT.exists():
        raise SystemExit(f"missing prepared payload: {PAYLOAD_ROOT}")
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    command_file_manifest()
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    cmd = [
        "tar",
        "--use-compress-program=zstd -T0 -10",
        "-cf",
        str(ARCHIVE_PATH),
        "-C",
        str(STAGING_ROOT),
        PAYLOAD_ROOT.name,
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {ARCHIVE_PATH} ({human_size(ARCHIVE_PATH.stat().st_size)})")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_checksum() -> None:
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in sorted(DIST_ROOT.iterdir())
        if path.is_file() and path.name not in {CHECKSUM_PATH.name, ".gitkeep"}
    ]
    with CHECKSUM_PATH.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(f"wrote {CHECKSUM_PATH}")


def scan_text_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    hits = []
    for pattern in SENSITIVE_SCAN_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def command_scan() -> None:
    roots = [PAYLOAD_ROOT]
    problems = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            hits = scan_text_file(path)
            if hits:
                problems.append((path.relative_to(root).as_posix(), hits))
                if len(problems) >= 50:
                    break
    if problems:
        print("sensitive pattern scan failed:")
        for rel_path, hits in problems:
            print(f"- {rel_path}: {', '.join(hits)}")
        raise SystemExit(1)
    print("sensitive pattern scan passed")


def command_clean() -> None:
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    if FILE_MANIFEST_PATH.exists():
        FILE_MANIFEST_PATH.unlink()
    if CHECKSUM_PATH.exists():
        CHECKSUM_PATH.unlink()
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    (STAGING_ROOT / ".gitkeep").touch()
    (DIST_ROOT / ".gitkeep").touch()
    print("cleaned staging and generated dist files")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["inventory", "prepare", "package", "checksum", "scan", "clean", "all"],
    )
    args = parser.parse_args(argv)

    if args.command == "inventory":
        command_inventory()
    elif args.command == "prepare":
        command_inventory()
        command_prepare()
    elif args.command == "package":
        command_package()
    elif args.command == "checksum":
        command_checksum()
    elif args.command == "scan":
        command_scan()
    elif args.command == "clean":
        command_clean()
    elif args.command == "all":
        command_inventory()
        command_prepare()
        command_scan()
        command_package()
        command_checksum()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
