#!/usr/bin/env python3
"""Appendix H 的 Probe-4 权重敏感性实验。

输入是冻结后的候选四算法组合分项表。脚本只扰动 H/F/C/V 四个权重，
不重新估计组合分项，避免把数据处理变化混入权重敏感性。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np


DEFAULT_WEIGHTS = (0.20, 0.40, 0.25, 0.15)
PERTURBATION_RADIUS = 0.05
DEFAULT_SEED = 20260831
COMPONENT_COLUMNS = ("health", "panel_fidelity", "complementarity", "coverage")
INPUT_COLUMNS = ("panel", *COMPONENT_COLUMNS)


class ContractError(RuntimeError):
    """输入或历史基线不满足正式实验合同。"""


class WeightSamples(NamedTuple):
    raw: np.ndarray
    normalized: np.ndarray


class Candidates(NamedTuple):
    panels: tuple[str, ...]
    components: np.ndarray


class BaselineContract(NamedTuple):
    selected_panel: str
    expected_best_panel: str
    expected_selected_rank: int
    expected_best_score: float | None
    expected_selected_score: float | None
    score_tolerance: float


HISTORICAL_CONTRACT = BaselineContract(
    selected_panel="dso;imcts;pyoperon;udsr",
    expected_best_panel="dso;pyoperon;qlattice;udsr",
    expected_selected_rank=2,
    expected_best_score=0.6570,
    expected_selected_score=0.6512,
    # 论文只报告四位小数，因此仅允许落在相同四舍五入区间内。
    score_tolerance=5.0000001e-5,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_panel(value: str) -> str:
    algorithms = [part.strip().lower() for part in value.split(";") if part.strip()]
    if len(algorithms) != 4 or len(set(algorithms)) != 4:
        raise ContractError(f"Probe-4 组合必须包含四个不同算法: {value!r}")
    return ";".join(sorted(algorithms))


def load_candidates(path: Path) -> Candidates:
    if not path.is_file():
        raise ContractError(f"候选分项输入不存在或不是文件: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ContractError("候选分项输入没有表头")
        missing = sorted(set(INPUT_COLUMNS) - set(reader.fieldnames))
        extra = sorted(set(reader.fieldnames) - set(INPUT_COLUMNS))
        if missing or extra:
            raise ContractError(f"候选分项表字段不匹配; missing={missing}, extra={extra}")

        rows: list[tuple[str, list[float]]] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            panel = canonical_panel(row["panel"])
            if panel in seen:
                raise ContractError(f"第 {line_number} 行出现重复候选组合: {panel}")
            seen.add(panel)
            values: list[float] = []
            for column in COMPONENT_COLUMNS:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise ContractError(f"第 {line_number} 行 {column} 不是数值") from exc
                if not math.isfinite(value):
                    raise ContractError(f"第 {line_number} 行 {column} 必须是有限数值")
                if not 0.0 <= value <= 1.0:
                    raise ContractError(f"第 {line_number} 行 {column} 必须位于 [0, 1]")
                values.append(value)
            rows.append((panel, values))

    if len(rows) < 2:
        raise ContractError("至少需要两个候选组合才能计算相对最优差距")
    rows.sort(key=lambda item: item[0])
    return Candidates(
        panels=tuple(panel for panel, _ in rows),
        components=np.asarray([values for _, values in rows], dtype=np.float64),
    )


def sample_weights(n_draws: int, seed: int) -> WeightSamples:
    if n_draws <= 0:
        raise ContractError("n_draws 必须是正整数")
    if seed < 0:
        raise ContractError("seed 必须是非负整数")
    center = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
    rng = np.random.default_rng(seed)
    raw = rng.uniform(
        low=center - PERTURBATION_RADIUS,
        high=center + PERTURBATION_RADIUS,
        size=(n_draws, len(center)),
    )
    totals = raw.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ContractError("采样权重无法安全归一化")
    normalized = raw / totals
    return WeightSamples(raw=raw, normalized=normalized)


def _ranking(scores: np.ndarray, panels: Sequence[str]) -> np.ndarray:
    if scores.ndim != 1 or len(scores) != len(panels):
        raise ContractError("分数向量与候选组合数量不一致")
    if not np.all(np.isfinite(scores)):
        raise ContractError("候选组合得分包含非有限值")
    # 相同分数时按规范化 panel 名排序，确保不同平台结果一致。
    order = np.lexsort((np.asarray(panels, dtype=str), -scores))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _assert_score(label: str, actual: float, expected: float | None, tolerance: float) -> None:
    if expected is not None and abs(actual - expected) > tolerance:
        raise ContractError(
            f"{label}不符合历史基线: actual={actual:.17g}, "
            f"expected={expected:.17g}, tolerance={tolerance:.3g}"
        )


def audit_default_baseline(candidates: Candidates, contract: BaselineContract) -> dict[str, object]:
    selected = canonical_panel(contract.selected_panel)
    expected_best = canonical_panel(contract.expected_best_panel)
    if contract.expected_selected_rank <= 0:
        raise ContractError("expected_selected_rank 必须是正整数")
    if not math.isfinite(contract.score_tolerance) or contract.score_tolerance < 0:
        raise ContractError("score_tolerance 必须是有限非负数")
    try:
        selected_index = candidates.panels.index(selected)
        expected_best_index = candidates.panels.index(expected_best)
    except ValueError as exc:
        raise ContractError(f"历史基线组合没有出现在候选表中: {exc}") from exc

    weights = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
    scores = candidates.components @ weights
    ranks = _ranking(scores, candidates.panels)
    best_index = int(np.argmin(ranks))
    best_panel = candidates.panels[best_index]
    selected_rank = int(ranks[selected_index])
    if best_panel != expected_best:
        raise ContractError(f"基线最优组合不一致: actual={best_panel}, expected={expected_best}")
    if selected_rank != contract.expected_selected_rank:
        raise ContractError(
            f"固定 Probe-4 基线排名不一致: actual={selected_rank}, "
            f"expected={contract.expected_selected_rank}"
        )
    best_score = float(scores[expected_best_index])
    selected_score = float(scores[selected_index])
    _assert_score("基线最优组合分数", best_score, contract.expected_best_score, contract.score_tolerance)
    _assert_score("固定 Probe-4 基线分数", selected_score, contract.expected_selected_score, contract.score_tolerance)

    return {
        "passed": True,
        "weights": dict(zip(("wH", "wF", "wC", "wV"), DEFAULT_WEIGHTS)),
        "selected_panel": selected,
        "selected_rank": selected_rank,
        "selected_score": selected_score,
        "best_panel": best_panel,
        "best_score": best_score,
        "candidate_count": len(candidates.panels),
        "tie_break": "score_desc_then_canonical_panel_asc",
    }


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ContractError("拒绝写出非有限浮点数")
    return format(value, ".17g")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(
    stage: Path,
    candidates: Candidates,
    samples: WeightSamples,
    scores: np.ndarray,
    baseline_audit: dict[str, object],
    input_path: Path,
    seed: int,
    contract: BaselineContract,
) -> dict[str, object]:
    weight_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    selected = canonical_panel(contract.selected_panel)
    selected_index = candidates.panels.index(selected)

    raw_names = ("wH_raw", "wF_raw", "wC_raw", "wV_raw")
    normalized_names = ("wH", "wF", "wC", "wV")
    for draw_index in range(samples.raw.shape[0]):
        config_id = draw_index + 1
        weight_row: dict[str, object] = {"configuration_id": config_id}
        weight_row.update({name: _format_float(value) for name, value in zip(raw_names, samples.raw[draw_index])})
        weight_row.update(
            {name: _format_float(value) for name, value in zip(normalized_names, samples.normalized[draw_index])}
        )
        weight_rows.append(weight_row)

        current_scores = scores[draw_index]
        ranks = _ranking(current_scores, candidates.panels)
        best_index = int(np.argmin(ranks))
        best_score = float(current_scores[best_index])
        selected_score = float(current_scores[selected_index])
        if best_score <= 0:
            raise ContractError(f"配置 {config_id} 的最优分数不为正，无法定义相对差距")
        relative_gap_pct = 100.0 * (best_score - selected_score) / best_score

        for candidate_index, panel in enumerate(candidates.panels):
            component_values = candidates.components[candidate_index]
            contributions = samples.normalized[draw_index] * component_values
            score_rows.append(
                {
                    "configuration_id": config_id,
                    "panel": panel,
                    "health": _format_float(component_values[0]),
                    "panel_fidelity": _format_float(component_values[1]),
                    "complementarity": _format_float(component_values[2]),
                    "coverage": _format_float(component_values[3]),
                    "health_contribution": _format_float(contributions[0]),
                    "fidelity_contribution": _format_float(contributions[1]),
                    "complementarity_contribution": _format_float(contributions[2]),
                    "coverage_contribution": _format_float(contributions[3]),
                    "score": _format_float(float(current_scores[candidate_index])),
                    "rank": int(ranks[candidate_index]),
                    "is_selected_panel": panel == selected,
                    "is_best_panel": candidate_index == best_index,
                }
            )
        result_rows.append(
            {
                "configuration_id": config_id,
                "selected_panel": selected,
                "selected_score": _format_float(selected_score),
                "selected_rank": int(ranks[selected_index]),
                "best_panel": candidates.panels[best_index],
                "best_score": _format_float(best_score),
                "relative_gap_pct": _format_float(relative_gap_pct),
                "selected_is_best": selected_index == best_index,
            }
        )

    weights_path = stage / "probe4_weight_configurations.csv"
    scores_path = stage / "probe4_candidate_scores.csv"
    results_path = stage / "probe4_perturbation_results.csv"
    audit_path = stage / "probe4_baseline_audit.json"
    _write_csv(weights_path, ("configuration_id", *raw_names, *normalized_names), weight_rows)
    _write_csv(
        scores_path,
        (
            "configuration_id",
            "panel",
            *COMPONENT_COLUMNS,
            "health_contribution",
            "fidelity_contribution",
            "complementarity_contribution",
            "coverage_contribution",
            "score",
            "rank",
            "is_selected_panel",
            "is_best_panel",
        ),
        score_rows,
    )
    _write_csv(
        results_path,
        (
            "configuration_id",
            "selected_panel",
            "selected_score",
            "selected_rank",
            "best_panel",
            "best_score",
            "relative_gap_pct",
            "selected_is_best",
        ),
        result_rows,
    )
    audit_path.write_text(json.dumps(baseline_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_paths = (weights_path, scores_path, results_path, audit_path)
    manifest = {
        "schema_version": "symbolicarena.appendix_h.probe4_sensitivity.v1",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "contract": {
            "default_weights": list(DEFAULT_WEIGHTS),
            "perturbation_radius": PERTURBATION_RADIUS,
            "perturbation": "independent_uniform_plus_minus_0.05_then_renormalize",
            "n_draws": samples.raw.shape[0],
            "seed": seed,
            "selected_panel": selected,
            "expected_best_panel": canonical_panel(contract.expected_best_panel),
            "expected_selected_rank": contract.expected_selected_rank,
            "tie_break": "score_desc_then_canonical_panel_asc",
        },
        "counts": {
            "candidates": len(candidates.panels),
            "configurations": samples.raw.shape[0],
            "candidate_score_rows": len(score_rows),
        },
        "baseline_audit": baseline_audit,
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_experiment(
    input_path: Path,
    output_dir: Path,
    n_draws: int = 5000,
    seed: int = DEFAULT_SEED,
    contract: BaselineContract = HISTORICAL_CONTRACT,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ContractError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    candidates = load_candidates(input_path)
    # 先审计再创建任何输出，基线不匹配时严格 fail-closed。
    baseline_audit = audit_default_baseline(candidates, contract)
    samples = sample_weights(n_draws=n_draws, seed=seed)
    scores = samples.normalized @ candidates.components.T
    if scores.shape != (n_draws, len(candidates.panels)) or not np.all(np.isfinite(scores)):
        raise ContractError("扰动得分矩阵不完整或包含非有限值")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        manifest = _write_artifacts(
            stage=stage,
            candidates=candidates,
            samples=samples,
            scores=scores,
            baseline_audit=baseline_audit,
            input_path=input_path,
            seed=seed,
            contract=contract,
        )
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "configuration_count": manifest["counts"]["configurations"],
        "candidate_count": manifest["counts"]["candidates"],
        "candidate_score_rows": manifest["counts"]["candidate_score_rows"],
        "output_dir": str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-components", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_experiment(
        input_path=args.candidate_components,
        output_dir=args.output_dir,
        n_draws=args.n_draws,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
