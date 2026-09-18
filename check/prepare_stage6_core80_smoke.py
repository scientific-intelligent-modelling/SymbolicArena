"""Freeze only Core80-minus-Core50 clean inputs for a pre-dispatch smoke batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SELECTION_ZIP = ROOT / (
    "AAAI_experiments/stage6_multiCore60-70-80/"
    "SymbolicArena_Core50_60_70_80_complete.zip"
)
FROZEN_CORE50 = ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
FORMAL = ROOT / (
    "benchmark-runs/formal3h/"
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658/params"
)
DRSR = ROOT / (
    "benchmark-runs/drsr3h/"
    "drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355/params"
)
LLMSR = ROOT / (
    "benchmark-runs/llmsr3h/"
    "llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442/params"
)
TOOLS = (
    "gplearn", "llmsr", "pyoperon", "drsr", "pysr", "dso", "tpsr",
    "e2esr", "fepysr", "jaxsr", "qlattice", "imcts", "udsr", "ragsr",
    "symbolfit",
)
SMOKE_IDS = ("g0596", "g0275")
SOURCE_FIELDS = (
    "global_index", "dataset_id", "dataset_name", "dataset_dir",
    "dataset_rel", "family", "subgroup", "basename",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def new30_rows(core50: list[dict[str, str]], core80: list[dict[str, str]]) -> list[dict[str, str]]:
    old = {row["dataset_rel"] for row in core50}
    if len(core50) != 50 or len(core80) != 80 or len(old) != 50:
        raise ValueError("expected 50 unique old tasks and 80 Core80 tasks")
    if len({row["dataset_rel"] for row in core80}) != 80:
        raise ValueError("Core80 contains duplicate task paths")
    if not old <= {row["dataset_rel"] for row in core80}:
        raise ValueError("Core80 does not contain the frozen Core50")
    new = [row for row in core80 if row["dataset_rel"] not in old]
    if len(new) != 30 or len({row["dataset_id"] for row in new}) != 30:
        raise ValueError("Core80-minus-Core50 is not 30 unique tasks")
    return new


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _source_row(row: dict[str, str]) -> dict[str, str]:
    dataset_id = row["dataset_id"]
    if not (dataset_id.startswith("g") and dataset_id[1:].isdigit()):
        raise ValueError(f"invalid global dataset ID: {dataset_id}")
    return {
        "global_index": str(int(dataset_id[1:])),
        "dataset_id": dataset_id,
        "dataset_name": row["dataset_name"],
        "dataset_dir": row["dataset_rel"],
        "dataset_rel": row["dataset_rel"],
        "family": row["family"],
        "subgroup": row["subgroup"],
        "basename": row["basename"],
    }


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(SELECTION_ZIP) as archive:
        root = "SymbolicArena_Core50_60_70_80_complete/outputs/"
        core50 = _csv_bytes(archive.read(root + "core50.csv"))
        core80 = _csv_bytes(archive.read(root + "core80.csv"))
    frozen = {row["dataset_dir"] for row in _csv_bytes(FROZEN_CORE50.read_bytes())}
    if frozen != {row["dataset_rel"] for row in core50}:
        raise ValueError("selection ZIP Core50 does not equal the frozen Stage4 task set")
    new = new30_rows(core50, core80)
    source_rows = [_source_row(row) for row in sorted(new, key=lambda row: row["dataset_id"])]
    for row in source_rows:
        dataset_dir = ROOT / row["dataset_dir"]
        for name in ("metadata.yaml", "train.csv", "valid.csv", "id_test.csv",
                     "ood_test.csv", "formula.py"):
            path = dataset_dir / name
            if not path.is_file():
                raise ValueError(f"missing or LFS-pointer dataset file: {path}")
            with path.open("rb") as handle:
                if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(f"missing or LFS-pointer dataset file: {path}")
    output.mkdir(parents=True)
    _write_csv(output / "new30_source.csv", source_rows)
    smoke_rows = [row for row in source_rows if row["dataset_id"] in SMOKE_IDS]
    if len(smoke_rows) != len(SMOKE_IDS):
        raise ValueError("a smoke task is absent from the new30 task set")
    _write_csv(output / "smoke_source.csv", smoke_rows)

    param_sources = {}
    for tool in TOOLS:
        source_root = DRSR if tool == "drsr" else LLMSR if tool == "llmsr" else FORMAL
        source = source_root / f"{tool}__clean.json"
        params = json.loads(source.read_text(encoding="utf-8"))
        if (params.get("timeout_in_seconds") != 10800
                or params.get("progress_snapshot_interval_seconds") != 60
                or float(params.get("train_label_noise_sigma", -1)) != 0.0
                or params.get("train_label_noise_enabled") is not False):
            raise ValueError(f"formal clean params differ from Stage4 contract: {tool}")
        param_sources[tool] = {"source": str(source), "sha256": sha256(source)}
        formal_path = output / "formal_clean_params" / f"{tool}__clean.json"
        formal_path.parent.mkdir(parents=True, exist_ok=True)
        formal_path.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        params["timeout_in_seconds"] = 600
        if tool in {"llmsr", "drsr"}:
            params["niterations"] = 3
            params["samples_per_iteration"] = 4
        smoke_path = output / "smoke_params" / f"{tool}__clean.json"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "stage6_core80_minus_core50_smoke.v1",
        "selection_zip_sha256": sha256(SELECTION_ZIP),
        "frozen_core50_sha256": sha256(FROZEN_CORE50),
        "new_dataset_count": 30,
        "smoke_dataset_ids": list(SMOKE_IDS),
        "algorithms": list(TOOLS),
        "seeds": [520, 521, 522],
        "condition": "clean",
        "formal_expected_runs": 30 * len(TOOLS) * 3,
        "smoke_expected_runs": len(SMOKE_IDS) * len(TOOLS),
        "formal_budget_seconds": 10800,
        "smoke_budget_seconds": 600,
        "parameter_sources": param_sources,
        "formal_dispatch_authorized": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
