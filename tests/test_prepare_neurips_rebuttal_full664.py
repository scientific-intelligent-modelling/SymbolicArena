from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "prepare_neurips_rebuttal_full664.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_neurips_rebuttal_full664", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source_csv(path: Path, *, duplicate_dataset_rel: bool = False) -> None:
    rows = []
    for index in range(1, 4):
        dataset_dir = f"sim-datasets-data/family/d{index}"
        dataset_rel = "sim-datasets-data/family/shared" if duplicate_dataset_rel else dataset_dir
        rows.append(
            {
                "family": "family",
                "subgroup": "group",
                "dataset_name": "shared" if index < 3 else "unique",
                "dataset_dir": dataset_dir,
                "global_index": str(index),
                "dataset_rel": dataset_rel,
                "basename": "shared" if index < 3 else "unique",
                "formula_py": f"{dataset_rel}/formula.py",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aaai_params(root: Path) -> None:
    payloads = {
        "fepysr": {"niterations": 1_000_000, "num_workers": 4},
        "jaxsr": {"max_terms": 5, "strategy": "greedy_forward"},
        "symbolfit": {"niterations": 1_000_000, "parallelism": "serial"},
    }
    root.mkdir(parents=True)
    for tool, extra in payloads.items():
        payload = {
            "timeout_in_seconds": 10_800,
            "progress_snapshot_interval_seconds": 60,
            **extra,
            "train_label_noise_sigma": 0.0,
            "train_label_noise_enabled": False,
        }
        (root / f"{tool}__clean.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_batch_builds_clean_full_and_smoke_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    source_csv = tmp_path / "full664.csv"
    aaai_params_root = tmp_path / "aaai_params"
    output_dir = tmp_path / "batch"
    _write_source_csv(source_csv)
    _write_aaai_params(aaai_params_root)

    summary = module.prepare_batch(
        source_csv=source_csv,
        aaai_params_root=aaai_params_root,
        output_dir=output_dir,
        expected_datasets=3,
        git_revision="deadbeef",
    )

    assert summary == {
        "algorithms": 3,
        "datasets": 3,
        "seeds": 3,
        "noise_levels": 1,
        "tasks": 27,
        "smoke_datasets": 2,
        "smoke_tasks": 18,
    }
    tasks = _read_csv(output_dir / "manifest" / "tasks.csv")
    assert len(tasks) == 27
    assert {row["dataset_id"] for row in tasks} == {"g0001", "g0002", "g0003"}
    assert {row["task_id"] for row in tasks} == {
        f"{tool}__seed{seed}__clean__g{index:04d}"
        for tool in ("fepysr", "jaxsr", "symbolfit")
        for seed in (520, 521, 522)
        for index in range(1, 4)
    }
    assert {row["timeout_in_seconds"] for row in tasks} == {"3600"}
    assert {row["min_runtime_seconds"] for row in tasks} == {"3300"}
    assert {row["noise_tag"] for row in tasks} == {"clean"}
    assert {row["noise_sigma"] for row in tasks} == {"0"}

    smoke_tasks = _read_csv(output_dir / "smoke" / "manifest" / "tasks.csv")
    assert len(smoke_tasks) == 18
    assert {row["dataset_id"] for row in smoke_tasks} == {"g0001", "g0002"}
    assert {row["timeout_in_seconds"] for row in smoke_tasks} == {"600"}

    fepysr = json.loads((output_dir / "params" / "fepysr__clean.json").read_text(encoding="utf-8"))
    fepysr_smoke = json.loads(
        (output_dir / "params_smoke" / "fepysr__clean.json").read_text(encoding="utf-8")
    )
    assert fepysr["timeout_in_seconds"] == 3600
    assert fepysr["niterations"] == 1_000_000
    assert fepysr["num_workers"] == 4
    assert fepysr["train_label_noise_sigma"] == 0.0
    assert fepysr["train_label_noise_enabled"] is False
    assert fepysr_smoke == {**fepysr, "timeout_in_seconds": 600}

    for tool in ("fepysr", "jaxsr", "symbolfit"):
        aaai = json.loads(
            (aaai_params_root / f"{tool}__clean.json").read_text(encoding="utf-8")
        )
        full = json.loads(
            (output_dir / "params" / f"{tool}__clean.json").read_text(encoding="utf-8")
        )
        smoke = json.loads(
            (output_dir / "smoke" / "params" / f"{tool}__clean.json").read_text(
                encoding="utf-8"
            )
        )
        assert full == {**aaai, "timeout_in_seconds": 3600}
        assert smoke == {**aaai, "timeout_in_seconds": 600}

    queue_rows = _read_csv(output_dir / "queues" / "full664_source.csv")
    assert len(queue_rows) == 3
    assert queue_rows[0]["dataset_id"] == "g0001"
    assert queue_rows[1]["dataset_name"] == "shared"
    assert queue_rows[0]["dataset_rel"] != queue_rows[1]["dataset_rel"]

    fingerprints = json.loads(
        (output_dir / "manifest" / "source_fingerprints.json").read_text(encoding="utf-8")
    )
    assert fingerprints["git_revision"] == "deadbeef"
    assert fingerprints["schema_version"] == 1
    assert fingerprints["path_base"] == "repository_root"
    assert fingerprints["source_csv"]["path"] == "full664.csv"
    assert fingerprints["source_csv"]["sha256"]
    assert fingerprints["aaai_params"]["fepysr"] == {
        "path": "aaai_params/fepysr__clean.json",
        "sha256": module._sha256(
            aaai_params_root / "fepysr__clean.json"
        ),
        "archived_path": (
            "batch/provenance/aaai_params_3h/fepysr__clean.json"
        ),
    }
    assert fingerprints == json.loads(
        (
            output_dir
            / "smoke/manifest/source_fingerprints.json"
        ).read_text(encoding="utf-8")
    )
    path_values = [fingerprints["source_csv"]["path"]]
    for record in fingerprints["aaai_params"].values():
        path_values.extend([record["path"], record["archived_path"]])
    assert all(not Path(value).is_absolute() for value in path_values)


def test_prepare_batch_rejects_provenance_path_outside_repo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    source_csv = tmp_path / "outside.csv"
    aaai_params_root = repo_root / "aaai_params"
    _write_source_csv(source_csv)
    _write_aaai_params(aaai_params_root)

    with pytest.raises(ValueError, match="仓库根目录之外"):
        module.prepare_batch(
            source_csv=source_csv,
            aaai_params_root=aaai_params_root,
            output_dir=repo_root / "batch",
            expected_datasets=3,
            git_revision="deadbeef",
        )


def test_prepare_batch_rejects_duplicate_dataset_rel(tmp_path: Path) -> None:
    module = _load_module()
    source_csv = tmp_path / "full664.csv"
    aaai_params_root = tmp_path / "aaai_params"
    _write_source_csv(source_csv, duplicate_dataset_rel=True)
    _write_aaai_params(aaai_params_root)

    with pytest.raises(ValueError, match="dataset_rel"):
        module.prepare_batch(
            source_csv=source_csv,
            aaai_params_root=aaai_params_root,
            output_dir=tmp_path / "batch",
            expected_datasets=3,
            git_revision="deadbeef",
        )


def test_prepare_batch_rejects_non_contiguous_global_indexes(tmp_path: Path) -> None:
    module = _load_module()
    source_csv = tmp_path / "full664.csv"
    aaai_params_root = tmp_path / "aaai_params"
    _write_source_csv(source_csv)
    _write_aaai_params(aaai_params_root)
    text = source_csv.read_text(encoding="utf-8").replace(",2,", ",4,", 1)
    source_csv.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="global_index"):
        module.prepare_batch(
            source_csv=source_csv,
            aaai_params_root=aaai_params_root,
            output_dir=tmp_path / "batch",
            expected_datasets=3,
            git_revision="deadbeef",
        )
