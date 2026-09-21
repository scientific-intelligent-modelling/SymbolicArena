from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_generate_ood_samples_supports_non_feynman_dataset_keys(tmp_path: Path):
    root = tmp_path
    dataset_dir = root / "blackbox" / "demo"
    write_csv(
        dataset_dir / "train.csv",
        ["x0", "target"],
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
    )
    (dataset_dir / "formula.py").write_text(
        "def target(x0):\n    return x0 * 2\n",
        encoding="utf-8",
    )
    (root / "feature_distributions.csv").write_text(
        "dataset,feature_1,dist_1,min_1,max_1,left_ood_min_1,left_ood_max_1,right_ood_min_1,right_ood_max_1\n"
        "blackbox/demo,x0,均匀分布,1,3,0.8,1,3,3.2\n",
        encoding="utf-8",
    )

    module = load_module(
        "generate_ood_samples",
        Path(".codex/skills/format-symbolic-regression-dataset/scripts/generate_ood_samples.py").resolve(),
    )

    old_argv = sys.argv[:]
    sys.argv = [
        "generate_ood_samples.py",
        "--root",
        str(root),
        "--samples-per-dataset",
        "5",
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    out_path = dataset_dir / "ood_test.csv"
    assert out_path.exists()
    with out_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["x0", "target"]
    assert len(rows) == 6


def test_extract_ood_by_iterative_shrink_generates_ood_close_to_target(tmp_path: Path):
    input_csv = tmp_path / "all.csv"
    header = ["x0", "target"]
    rows = [[float(i), float(i * 2)] for i in range(1000)]
    write_csv(input_csv, header, rows)

    module = load_module(
        "extract_ood_by_iterative_shrink",
        Path(
            ".codex/skills/format-symbolic-regression-dataset/scripts/extract_ood_by_iterative_shrink.py"
        ).resolve(),
    )

    output_dir = tmp_path / "out"
    old_argv = sys.argv[:]
    sys.argv = [
        "extract_ood_by_iterative_shrink.py",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    ood_path = output_dir / "ood_test.csv"
    id_path = output_dir / "id_pool.csv"
    summary_path = output_dir / "ood_split_summary.json"
    assert ood_path.exists()
    assert id_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert 0.10 <= summary["ood_ratio"] <= 0.20


def test_extract_ood_by_iterative_shrink_falls_back_for_discrete_like_data(tmp_path: Path):
    input_csv = tmp_path / "all.csv"
    header = ["x0", "x1", "target"]
    rows = []
    for i in range(100):
        rows.append([float((i % 4) + 1), float((i % 3) + 1), float(i)])
    write_csv(input_csv, header, rows)

    module = load_module(
        "extract_ood_by_iterative_shrink",
        Path(
            ".codex/skills/format-symbolic-regression-dataset/scripts/extract_ood_by_iterative_shrink.py"
        ).resolve(),
    )

    output_dir = tmp_path / "out"
    old_argv = sys.argv[:]
    sys.argv = [
        "extract_ood_by_iterative_shrink.py",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_dir / "ood_split_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["method"] == "fallback_boundary_rarity_ranking"
    assert abs(summary["ood_ratio"] - 0.10) < 1e-9
    assert (output_dir / "ood_test.csv").exists()


def test_extract_ood_by_iterative_shrink_fails_when_ratio_jumps_over_threshold(tmp_path: Path):
    input_csv = tmp_path / "all.csv"
    header = ["x0", "target"]
    rows = [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]
    write_csv(input_csv, header, rows)

    module = load_module(
        "extract_ood_by_iterative_shrink",
        Path(
            ".codex/skills/format-symbolic-regression-dataset/scripts/extract_ood_by_iterative_shrink.py"
        ).resolve(),
    )

    output_dir = tmp_path / "out"
    old_argv = sys.argv[:]
    sys.argv = [
        "extract_ood_by_iterative_shrink.py",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_dir / "ood_split_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["reason"] == "ood_ratio_exceeded_max_ratio"
    assert not (output_dir / "ood_test.csv").exists()
