from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scientific_intelligent_modelling.benchmarks.runner import (
    _predict_from_canonical_artifact,
    load_canonical_dataset,
)


CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = (520, 521, 522)
SPLITS = ("train", "valid", "id_test", "ood_test")
SCHEMA = "symbolicarena-pages-v1"
SAMPLE_LIMIT = 128
LINE_GRID_SIZE = 256
SURFACE_GRID_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any], *, compressed: bool) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with temporary.open("wb") as output:
        if compressed:
            with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6) as archive:
                archive.write(data)
        else:
            output.write(data)
    os.replace(temporary, path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finite_list(values: np.ndarray) -> list[float | None]:
    with np.errstate(all="ignore"):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
    if np.isfinite(array).all():
        return array.tolist()
    return [float(value) if math.isfinite(float(value)) else None for value in array]


def sample_indices(rows: int, dataset_number: int, split_number: int) -> np.ndarray:
    if rows <= SAMPLE_LIMIT:
        return np.arange(rows, dtype=np.int64)
    random = np.random.default_rng(dataset_number * 1009 + split_number)
    return np.sort(random.choice(rows, size=SAMPLE_LIMIT, replace=False))


def grid_range(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.01, 0.99]).tolist()
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return float(lower), float(upper)


def build_dataset_plan(dataset_dir: Path, output_root: Path) -> dict[str, Any]:
    dataset_index = dataset_dir.name
    dataset_number = int(dataset_index.removeprefix("g"))
    dataset = load_canonical_dataset(dataset_dir)
    metadata = dataset.metadata.get("dataset", dataset.metadata)
    split_rows: list[str] = []
    row_indices: list[int] = []
    sample_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    train_sample_indices: np.ndarray | None = None
    train_sample_positions: np.ndarray | None = None

    for split_number, split_name in enumerate(SPLITS):
        split = getattr(dataset, split_name)
        if split is None:
            continue
        selected = sample_indices(split.rows, dataset_number, split_number)
        if split_name == "train":
            train_sample_indices = selected
            train_sample_positions = np.arange(len(selected), dtype=np.int64)
        split_rows.extend([split_name] * len(selected))
        row_indices.extend(selected.tolist())
        sample_parts.append(split.X[selected])
        target_parts.append(split.y[selected])

    if train_sample_indices is None or train_sample_positions is None:
        raise ValueError(f"缺少训练样本：{dataset_index}")
    sample_X = np.vstack(sample_parts).astype(np.float64, copy=False)
    clean_y = np.concatenate(target_parts).astype(np.float64, copy=False)
    if not np.isfinite(sample_X).all() or not np.isfinite(clean_y).all():
        raise ValueError(f"展示样本存在非有限数值：{dataset_index}")

    dimension = sample_X.shape[1]
    pca_metadata = None
    grid_X = np.empty((0, dimension), dtype=np.float64)
    grid_metadata: dict[str, Any] | None = None
    if dimension == 1:
        kind = "line"
        coordinates = sample_X[:, :1]
        x_min, x_max = grid_range(sample_X[:, 0])
        grid_axis = np.linspace(x_min, x_max, LINE_GRID_SIZE)
        grid_X = grid_axis[:, None]
        grid_metadata = {"x": finite_list(grid_axis)}
        axis_labels = [dataset.feature_names[0], dataset.target_name]
    elif dimension == 2:
        kind = "surface"
        coordinates = sample_X[:, :2]
        x_min, x_max = grid_range(sample_X[:, 0])
        y_min, y_max = grid_range(sample_X[:, 1])
        x_axis = np.linspace(x_min, x_max, SURFACE_GRID_SIZE)
        y_axis = np.linspace(y_min, y_max, SURFACE_GRID_SIZE)
        mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)
        grid_X = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
        grid_metadata = {"x": finite_list(x_axis), "y": finite_list(y_axis)}
        axis_labels = [*dataset.feature_names, dataset.target_name]
    else:
        kind = "projection"
        scaler = StandardScaler().fit(dataset.train.X)
        pca = PCA(n_components=2, svd_solver="full").fit(scaler.transform(dataset.train.X))
        coordinates = pca.transform(scaler.transform(sample_X))
        if not np.isfinite(coordinates).all() or not np.isfinite(pca.explained_variance_ratio_).all():
            raise ValueError(f"PCA 投影存在非有限数值：{dataset_index}")
        pca_metadata = {
            "mean": finite_list(scaler.mean_),
            "scale": finite_list(scaler.scale_),
            "components": [finite_list(row) for row in pca.components_],
            "explained_variance_ratio": finite_list(pca.explained_variance_ratio_),
        }
        axis_labels = ["PC1", "PC2", dataset.target_name]

    prediction_X = np.vstack((sample_X, grid_X))
    plan_path = output_root / "_internal" / "datasets" / f"{dataset_index}.npz"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        plan_path,
        prediction_X=prediction_X,
        clean_y=clean_y,
        train_y=dataset.train.y,
        train_sample_indices=train_sample_indices,
        train_sample_positions=train_sample_positions,
        sample_count=np.asarray([len(sample_X)], dtype=np.int64),
    )
    dataset_payload = {
        "schema": SCHEMA,
        "dataset_index": dataset_index,
        "name": dataset.dataset_name,
        "feature_names": dataset.feature_names,
        "feature_descriptions": dataset.feature_descriptions,
        "target_name": dataset.target_name,
        "target_description": dataset.target_description,
        "license": metadata.get("license", "unknown"),
        "citation": metadata.get("citation"),
        "plot": {
            "kind": kind,
            "axis_labels": axis_labels,
            "coordinates": [finite_list(row) for row in coordinates],
            "grid": grid_metadata,
            "pca": pca_metadata,
        },
        "samples": {
            "split": split_rows,
            "row_index": row_indices,
            "y_clean": finite_list(clean_y),
        },
        "source_sha256": {
            split_name: sha256_file(dataset_dir / metadata["splits"][split_name]["file"])
            for split_name in SPLITS
            if split_name in metadata["splits"]
        },
    }
    relative_path = Path("datasets") / f"{dataset_index}.json.gz"
    file_info = write_json(output_root / relative_path, dataset_payload, compressed=True)
    return {
        "index": dataset_index,
        "name": dataset.dataset_name,
        "feature_count": dimension,
        "plot_kind": kind,
        "license": dataset_payload["license"],
        "path": relative_path.as_posix(),
        **file_info,
    }


def metric_payload(value: Any) -> dict[str, float | None] | None:
    if not isinstance(value, dict):
        return None
    output: dict[str, float | None] = {}
    for field in ("rmse", "r2", "nmse", "acc_0_1"):
        item = value.get(field)
        if item is None:
            output[field] = None
        else:
            number = float(item)
            output[field] = number if math.isfinite(number) else None
    return output


def same_tool_identity(expected: str, recorded: Any) -> bool:
    return isinstance(recorded, str) and expected.casefold() == recorded.casefold()


def result_path_from_source(stage_root: Path, source: dict[str, Any]) -> Path:
    archived = Path(str(source["path"]))
    condition = str(source["noise_tag"])
    algorithm = str(source["algorithm"])
    dataset_id = str(source["dataset_id"])
    seed = int(source["seed"])
    if (
        len(archived.parts) < 5
        or archived.name != "result.json"
        or archived.parts[-2] != str(seed)
        or archived.parts[-3] != dataset_id
        or archived.parts[-5] != condition
        or not same_tool_identity(algorithm, archived.parts[-4])
    ):
        raise ValueError(f"来源结果路径与运行身份不一致：{archived}")
    result_path = (
        stage_root / "2.2 core50 experiments" / condition /
        archived.parts[-4] / dataset_id / str(seed) / "result.json"
    )
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    return result_path


def evaluate_candidate(artifact: Any, prediction_X: np.ndarray, sample_count: int) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {"status": "unavailable", "reason": "canonical_artifact 缺失"}
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(_predict_from_canonical_artifact(artifact, prediction_X), dtype=float).reshape(-1)
    except Exception as error:
        return {"status": "error", "reason": f"{type(error).__name__}: {error}"[:300]}
    if len(values) != len(prediction_X):
        raise ValueError("预测数量与展示输入数量不一致")
    usable = np.isfinite(values)
    return {
        "status": "ok" if usable.all() else "partial",
        "finite_count": int(usable.sum()),
        "sample_y": finite_list(values[:sample_count]),
        "grid_y": finite_list(values[sample_count:]),
    }


def observed_labels(result: dict[str, Any], plan: Any, condition: str) -> list[float]:
    values = np.asarray(plan["clean_y"], dtype=np.float64).copy()
    noise = result.get("train_label_noise")
    if not isinstance(noise, dict):
        raise ValueError("结果缺少训练噪声记录")
    if result.get("condition") != condition:
        raise ValueError("运行噪声条件与结果不一致")
    if noise.get("enabled"):
        full_train_y = np.asarray(plan["train_y"], dtype=np.float64)
        if not np.isclose(np.std(full_train_y), float(noise["y_std"]), rtol=1e-10, atol=1e-12):
            raise ValueError("噪声记录与原始训练目标不一致")
        random = np.random.default_rng(int(noise["rng_seed"]))
        generated = random.normal(loc=0.0, scale=float(noise["scale"]), size=full_train_y.shape)
        indices = np.asarray(plan["train_sample_indices"], dtype=np.int64)
        positions = np.asarray(plan["train_sample_positions"], dtype=np.int64)
        values[positions] += generated[indices]
    return finite_list(values)


def export_run(record: dict[str, Any], stage_root: Path, output_root: Path, condition: str, *, resume: bool = False) -> dict[str, Any]:
    source = record["source"]
    dataset_index = f"g{int(source['global_index']):04d}"
    algorithm = str(source["algorithm"])
    dataset_id = str(source["dataset_id"])
    seed = int(source["seed"])
    if source["noise_tag"] != condition or seed not in SEEDS:
        raise ValueError(f"运行身份不一致：{algorithm} {dataset_index} {condition} {seed}")
    relative_path = Path("runs") / condition / algorithm / dataset_index / f"{seed}.json.gz"
    output_path = output_root / relative_path
    if resume and output_path.is_file():
        with gzip.open(output_path, "rt", encoding="utf-8") as existing:
            bundle = json.load(existing)
        identity = bundle["identity"]
        if (
            identity["algorithm"] != algorithm
            or identity["dataset_index"] != dataset_index
            or identity["dataset_id"] != dataset_id
            or identity["condition"] != condition
            or identity["seed"] != seed
            or bundle["source"]["result_sha256"] != record["result"].get("sha256")
        ):
            raise ValueError(f"已有运行文件与来源不一致：{output_path}")
        return {
            **identity,
            "path": relative_path.as_posix(),
            "terminal_status": bundle["terminal"]["status"],
            "available_snapshots": bundle["source"]["available_snapshots"],
            "candidate_count": len(bundle["candidates"]),
            "plot_available_count": sum(item["plot"]["status"] in {"ok", "partial"} for item in bundle["candidates"]),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    result_path = result_path_from_source(stage_root, source)
    result = json.loads(result_path.read_text())
    if record["result"].get("sha256") and sha256_file(result_path) != record["result"]["sha256"]:
        raise ValueError(f"结束结果与来源哈希不一致：{result_path}")
    if not same_tool_identity(algorithm, result.get("tool")) or result.get("dataset") != dataset_id or int(result.get("seed")) != seed:
        raise ValueError(f"结束结果身份不一致：{result_path}")

    with np.load(output_root / "_internal" / "datasets" / f"{dataset_index}.npz", allow_pickle=False) as plan:
        prediction_X = np.asarray(plan["prediction_X"], dtype=np.float64)
        sample_count = int(plan["sample_count"][0])
        observed_y = observed_labels(result, plan, condition)

    candidate_index: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    timeline: dict[int, dict[str, Any]] = {}
    for snapshot in record["snapshots"]:
        minute = int(snapshot["minute"])
        if minute < 1 or minute > 180 or minute in timeline:
            raise ValueError(f"分钟编号重复或超出范围：{result_path} {minute}")
        raw_text = snapshot.get("raw_text")
        if not raw_text:
            timeline[minute] = {"minute": minute, "status": snapshot.get("status") or "missing", "candidate": None}
            continue
        raw = json.loads(raw_text)
        if not same_tool_identity(algorithm, raw.get("tool")) or raw.get("dataset") != dataset_id or int(raw.get("seed")) != seed:
            raise ValueError(f"分钟记录身份不一致：{result_path} {minute}")
        equation = raw.get("equation")
        artifact = raw.get("canonical_artifact")
        candidate_id = None
        if equation or artifact:
            identity_data = {"equation": equation, "artifact": artifact}
            digest = hashlib.sha256(
                json.dumps(identity_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            candidate_id = candidate_index.get(digest)
            if candidate_id is None:
                candidate_id = len(candidates)
                candidate_index[digest] = candidate_id
                candidates.append({
                    "id": candidate_id,
                    "sha256": digest,
                    "source_candidate_sha256": raw.get("candidate_sha256"),
                    "equation": equation or artifact.get("raw_equation") or artifact.get("instantiated_expression"),
                    "plot": evaluate_candidate(artifact, prediction_X, sample_count),
                })
        timeline[minute] = {
            "minute": minute,
            "status": raw.get("status") or snapshot.get("status"),
            "candidate": candidate_id,
            "metrics": {name: metric_payload(raw.get(name)) for name in SPLITS},
        }

    minutes = [timeline.get(minute, {"minute": minute, "status": "missing", "candidate": None}) for minute in range(1, 181)]
    bundle = {
        "schema": SCHEMA,
        "identity": {
            "algorithm": algorithm,
            "dataset_index": dataset_index,
            "dataset_id": dataset_id,
            "condition": condition,
            "seed": seed,
        },
        "noise": result["train_label_noise"],
        "observed_y": observed_y,
        "timeline": minutes,
        "candidates": candidates,
        "terminal": {
            "status": result.get("status"),
            "equation": result.get("equation"),
            "metrics": {name: metric_payload(result.get(name)) for name in SPLITS},
            "termination_reason": result.get("termination_reason"),
            "reason": result.get("no_valid_output_reason") or result.get("error"),
        },
        "source": {
            "result_sha256": record["result"].get("sha256"),
            "expected_snapshots": record["summary"].get("expected_snapshots"),
            "available_snapshots": record["summary"].get("available_snapshots"),
        },
    }
    file_info = write_json(output_path, bundle, compressed=True)
    return {
        **bundle["identity"],
        "path": relative_path.as_posix(),
        "terminal_status": bundle["terminal"]["status"],
        "available_snapshots": bundle["source"]["available_snapshots"],
        "candidate_count": len(candidates),
        "plot_available_count": sum(item["plot"]["status"] in {"ok", "partial"} for item in candidates),
        **file_info,
    }


def export_source_file(source_file: Path, stage_root: Path, output_root: Path, only_key: str | None, resume: bool) -> list[dict[str, Any]]:
    condition = source_file.parent.name
    entries = []
    with gzip.open(source_file, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            identity = record["source"]
            key = f"{condition}:{identity['algorithm']}:g{int(identity['global_index']):04d}:{int(identity['seed'])}"
            if only_key is not None and key != only_key:
                continue
            entries.append(export_run(record, stage_root, output_root, condition, resume=resume))
            if only_key is not None:
                break
    return entries


def build_catalog(stage_root: Path, output_root: Path, entries: list[dict[str, Any]], datasets: list[dict[str, Any]], *, complete: bool) -> None:
    keys = [(entry["condition"], entry["algorithm"], entry["dataset_index"], entry["seed"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("轨迹键重复")
    if complete:
        expected = pd.read_csv(stage_root / "3、metrics" / "terminal_run_metrics.csv", usecols=["condition", "algorithm", "dataset_index", "seed"])
        expected_keys = {(row.condition, row.algorithm, row.dataset_index, int(row.seed)) for row in expected.itertuples(index=False)}
        if len(expected_keys) != 6750 or set(keys) != expected_keys:
            raise ValueError(f"轨迹覆盖不完整：导出 {len(keys)} 条，期望 {len(expected_keys)} 条")
    verification = json.loads((stage_root / "3、metrics" / "verification.json").read_text())
    catalog = {
        "schema": SCHEMA,
        "release": output_root.name,
        "conditions": [{"id": "clean", "label": "无噪声"}, {"id": "noise001", "label": "1% 噪声"}, {"id": "noise005", "label": "5% 噪声"}],
        "seeds": list(SEEDS),
        "algorithms": sorted({entry["algorithm"] for entry in entries}),
        "datasets": sorted(datasets, key=lambda item: item["index"]),
        "runs": sorted(entries, key=lambda item: (item["algorithm"], item["dataset_index"], item["condition"], item["seed"])),
        "source_summary": {
            "run_count": verification["run_count"],
            "task_count": verification["task_count"],
            "formal_ready": verification["formal_ready"],
            "terminal_unresolved": verification["terminal_unresolved"],
        },
    }
    write_json(output_root / "catalog.json", catalog, compressed=False)
    files = [output_root / "catalog.json", *(output_root / item["path"] for item in datasets), *(output_root / item["path"] for item in entries)]
    manifest = {
        "schema": SCHEMA,
        "release": output_root.name,
        "files": [{"path": path.relative_to(output_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }
    write_json(output_root / "manifest.json", manifest, compressed=False)
    print(f"数据集 {len(datasets)}，运行 {len(entries)}，文件 {len(files) + 1}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only-run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stage_root = args.stage_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = stage_root / "3、metrics" / "inputs" / "datasets"
    datasets = [build_dataset_plan(path, output_root) for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if len(datasets) != 50:
        raise ValueError(f"数据集数量异常：{len(datasets)}")
    source_root = stage_root / "3、metrics" / "inputs" / "source_trajectory_full"
    source_files = sorted(source_root.glob("*/*.jsonl.gz"))
    if len(source_files) != 45:
        raise ValueError(f"合并轨迹文件数量异常：{len(source_files)}")

    entries: list[dict[str, Any]] = []
    if args.only_run:
        for path in source_files:
            entries.extend(export_source_file(path, stage_root, output_root, args.only_run, args.resume))
            if entries:
                break
        if len(entries) != 1:
            raise ValueError(f"指定运行匹配数量异常：{len(entries)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(export_source_file, path, stage_root, output_root, None, args.resume): path for path in source_files}
            for future in as_completed(futures):
                completed = future.result()
                entries.extend(completed)
                print(f"已处理 {futures[future].parent.name}/{futures[future].name}：{len(completed)} 条", flush=True)
    build_catalog(stage_root, output_root, entries, datasets, complete=args.only_run is None)


if __name__ == "__main__":
    main()
