#!/usr/bin/env python3
"""固定 Probe-4 条件下重构 Core-50 选择并分析成员敏感性。

本脚本只读取已经完成的 Full-664 Probe-4 后处理结果，不启动算法训练。
由于仓库中不存在论文所述历史选择器的可执行实现，脚本首先执行基线复现
闸门，再把后续结果明确标记为 paper-spec reconstruction。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_LEVEL = REPO_ROOT / (
    "exp-planning/03.四探针全量664三种子验证/generated/"
    "postprocess_final_20260501-105508/probe4_postprocess_dataset_level.csv"
)
DEFAULT_DATASET_ALGORITHM = REPO_ROOT / (
    "exp-planning/03.四探针全量664三种子验证/generated/"
    "postprocess_final_20260501-105508/probe4_postprocess_dataset_algorithm.csv"
)
DEFAULT_CORE_MANIFEST = REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
DEFAULT_OUTPUT = REPO_ROOT / (
    "A_Neurips_experiments/rebuttal/"
    "04_core50_membership_sensitivity_fixed_probe4"
)

PROBE4 = ("dso", "imcts", "pyoperon", "udsr")
STRUCTURAL_FIELDS = (
    "family",
    "subgroup",
    "operator_group",
    "feature_count_bin",
    "sample_count_bin",
    "complexity_bin",
    "dummy_variable_status",
)
RESPONSE_FIELDS = (
    "difficulty_bin",
    "failure_mode",
    "winner_probe",
    "valid_output_pattern",
)
DECLARED_BUT_UNAVAILABLE_FIELDS = ("ood_type",)


@dataclass(frozen=True)
class ObjectiveSpec:
    coverage_weight: float = 0.45
    information_weight: float = 0.35
    balance_weight: float = 0.20
    structural_share: float = 0.60

    def normalized(self) -> "ObjectiveSpec":
        values = np.asarray(
            [self.coverage_weight, self.information_weight, self.balance_weight],
            dtype=float,
        )
        if np.any(values < 0) or float(values.sum()) <= 0:
            raise ValueError("目标权重必须非负且总和大于 0")
        values = values / values.sum()
        if not 0.0 <= self.structural_share <= 1.0:
            raise ValueError("structural_share 必须在 [0, 1]")
        return ObjectiveSpec(*map(float, values), float(self.structural_share))


@dataclass(frozen=True)
class ConstraintSpec:
    family_smoothing: float = 0.60
    family_lower_slack: float = 1.0
    family_upper_slack: float = 2.0
    subgroup_cap: int = 5
    require_all_difficulty_bins: bool = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_weights(
    coverage: float,
    information: float,
    balance: float,
    structural_share: float = 0.60,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        coverage,
        information,
        balance,
        structural_share,
    ).normalized()


def derive_valid_output_pattern(dataset_algorithm: pd.DataFrame) -> pd.Series:
    """把四探针各自的有效 seed 数编码成稳定的分类模式。"""
    frame = dataset_algorithm.copy()
    frame["method_norm"] = frame["method_norm"].astype(str).str.lower()
    frame = frame[frame["method_norm"].isin(PROBE4)]
    frame["n_valid_total_seeds"] = pd.to_numeric(
        frame["n_valid_total_seeds"], errors="coerce"
    ).fillna(0).astype(int)
    pivot = frame.pivot_table(
        index="dataset_id",
        columns="method_norm",
        values="n_valid_total_seeds",
        aggfunc="max",
        fill_value=0,
    )
    for method in PROBE4:
        if method not in pivot:
            pivot[method] = 0
    return pivot[list(PROBE4)].astype(str).agg("|".join, axis=1)


def prepare_dataset_table(
    dataset_level: pd.DataFrame,
    dataset_algorithm: pd.DataFrame,
) -> pd.DataFrame:
    frame = dataset_level.copy()
    required = {
        "dataset_id",
        "dataset_rel",
        "family",
        "subgroup",
        "basename",
        "semantic_duplicate_group",
        "difficulty_bin",
        "info_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset-level 输入缺少字段: {missing}")
    if frame["dataset_id"].duplicated().any():
        raise ValueError("dataset-level 输入存在重复 dataset_id")

    patterns = derive_valid_output_pattern(dataset_algorithm)
    frame["valid_output_pattern"] = frame["dataset_id"].map(patterns).fillna(
        "missing"
    )
    metadata_class = frame.get(
        "metadata_class", pd.Series("", index=frame.index, dtype=str)
    ).fillna("").astype(str).str.lower()
    family = frame["family"].fillna("").astype(str).str.lower()
    frame["dummy_variable_status"] = np.where(
        metadata_class.str.contains("non-dummy"),
        "non_dummy",
        np.where(
            metadata_class.str.contains("dummy"),
            "dummy",
            np.where(family.eq("srsd"), "non_dummy", "not_applicable"),
        ),
    )
    for field in STRUCTURAL_FIELDS + RESPONSE_FIELDS:
        if field not in frame:
            raise ValueError(f"无法构造论文目标字段: {field}")
        frame[field] = frame[field].fillna("unknown").astype(str)
    frame["info_score"] = pd.to_numeric(
        frame["info_score"], errors="coerce"
    ).fillna(0.0)
    return frame.sort_values("dataset_id").reset_index(drop=True)


def core_ids_from_manifest(
    dataset_level: pd.DataFrame,
    core_manifest: pd.DataFrame,
) -> set[str]:
    if "dataset_dir" not in core_manifest:
        raise ValueError("Core-50 manifest 缺少 dataset_dir")
    if core_manifest["dataset_dir"].isna().any():
        raise ValueError("Core-50 manifest 包含空 dataset_dir")
    mapping = dict(
        zip(
            dataset_level["dataset_rel"].astype(str),
            dataset_level["dataset_id"].astype(str),
            strict=True,
        )
    )
    missing = sorted(
        set(core_manifest["dataset_dir"].astype(str)) - set(mapping)
    )
    if missing:
        raise ValueError(f"manifest 有 {len(missing)} 个目录无法映射到 Full-664")
    ids = {mapping[path] for path in core_manifest["dataset_dir"].astype(str)}
    if len(ids) != len(core_manifest):
        raise ValueError("Core-50 manifest 映射后 dataset_id 不唯一")
    return ids


def rankdata(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and math.isclose(
            ordered[stop][1], ordered[start][1], abs_tol=1e-12
        ):
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in range(start, stop):
            ranks[ordered[index][0]] = rank
        start = stop
    return ranks


def pearson(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")
    x = np.asarray([a[key] for key in keys], dtype=float)
    y = np.asarray([b[key] for key in keys], dtype=float)
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    return pearson(
        {key: rankdata(a)[key] for key in keys},
        {key: rankdata(b)[key] for key in keys},
    )


def kendall(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    concordant = discordant = 0
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            sign_a = np.sign(a[left] - a[right])
            sign_b = np.sign(b[left] - b[right])
            if sign_a == 0 or sign_b == 0:
                continue
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def method_scores(
    dataset_algorithm: pd.DataFrame,
    dataset_ids: set[str],
) -> dict[str, float]:
    frame = dataset_algorithm[
        dataset_algorithm["dataset_id"].astype(str).isin(dataset_ids)
    ].copy()
    frame["method_norm"] = frame["method_norm"].astype(str).str.lower()
    frame = frame[frame["method_norm"].isin(PROBE4)]
    scores: dict[str, float] = {}
    for method in PROBE4:
        group = frame[frame["method_norm"].eq(method)]
        by_dataset = {
            str(row["dataset_id"]): (
                12.0
                if _finite(row.get("median_log_id_nmse")) is None
                or _finite(row.get("median_log_ood_nmse")) is None
                else 0.5 * float(row["median_log_id_nmse"])
                + 0.5 * float(row["median_log_ood_nmse"])
            )
            for _, row in group.iterrows()
        }
        values = [by_dataset.get(dataset_id, 12.0) for dataset_id in dataset_ids]
        scores[method] = float(np.mean(values))
    return scores


def ranking_metrics(
    full_scores: dict[str, float],
    subset_scores: dict[str, float],
) -> dict[str, Any]:
    keys = sorted(set(full_scores) & set(subset_scores))
    full_ranks = rankdata(full_scores)
    subset_ranks = rankdata(subset_scores)
    pair_total = pair_agree = 0
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            full_sign = np.sign(full_scores[left] - full_scores[right])
            subset_sign = np.sign(subset_scores[left] - subset_scores[right])
            if full_sign == 0:
                continue
            pair_total += 1
            pair_agree += int(full_sign == subset_sign)
    order = ",".join(
        method for method, _ in sorted(subset_scores.items(), key=lambda item: item[1])
    )
    return {
        "score_pearson": pearson(full_scores, subset_scores),
        "rank_spearman": pearson(full_ranks, subset_ranks),
        "rank_kendall": kendall(full_scores, subset_scores),
        "pairwise_rank_agreement": (
            pair_agree / pair_total if pair_total else float("nan")
        ),
        "aggregate_score_mae": float(
            np.mean([abs(full_scores[key] - subset_scores[key]) for key in keys])
        ),
        "rank_order": order,
    }


class SelectionProblem:
    """对固定 664 行表执行约束一换一局部搜索。"""

    def __init__(self, dataset_level: pd.DataFrame):
        self.frame = dataset_level.reset_index(drop=True).copy()
        self.n = len(self.frame)
        self.k = 50
        self.dataset_ids = self.frame["dataset_id"].astype(str).to_numpy()
        self.id_to_index = {
            dataset_id: index
            for index, dataset_id in enumerate(self.dataset_ids)
        }
        self.info = self.frame["info_score"].to_numpy(dtype=float)
        self.objective_fields = STRUCTURAL_FIELDS + RESPONSE_FIELDS
        self.codes: dict[str, np.ndarray] = {}
        self.n_categories: dict[str, int] = {}
        self.full_probabilities: dict[str, np.ndarray] = {}
        for field in self.objective_fields:
            codes, uniques = pd.factorize(
                self.frame[field].astype(str), sort=True
            )
            self.codes[field] = codes.astype(int)
            self.n_categories[field] = len(uniques)
            self.full_probabilities[field] = (
                np.bincount(codes, minlength=len(uniques)).astype(float) / self.n
            )

        constraint_fields = (
            "semantic_duplicate_group",
            "basename",
            "family",
            "subgroup",
            "difficulty_bin",
        )
        self.constraint_codes: dict[str, np.ndarray] = {}
        self.constraint_labels: dict[str, list[str]] = {}
        for field in constraint_fields:
            codes, uniques = pd.factorize(
                self.frame[field].fillna("unknown").astype(str), sort=True
            )
            self.constraint_codes[field] = codes.astype(int)
            self.constraint_labels[field] = list(map(str, uniques))

    def indices(self, dataset_ids: Iterable[str]) -> np.ndarray:
        ids = sorted(set(map(str, dataset_ids)))
        missing = [dataset_id for dataset_id in ids if dataset_id not in self.id_to_index]
        if missing:
            raise ValueError(f"未知 dataset_id: {missing[:5]}")
        return np.asarray([self.id_to_index[dataset_id] for dataset_id in ids])

    def ids(self, indices: Iterable[int]) -> set[str]:
        return {str(self.dataset_ids[index]) for index in indices}

    def family_bounds(
        self, spec: ConstraintSpec
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0.0 <= spec.family_smoothing <= 1.0:
            raise ValueError("family_smoothing 必须在 [0, 1]")
        codes = self.constraint_codes["family"]
        counts = np.bincount(
            codes, minlength=len(self.constraint_labels["family"])
        ).astype(float)
        target = self.k * (
            spec.family_smoothing * counts / self.n
            + (1.0 - spec.family_smoothing) / len(counts)
        )
        lower = np.maximum(
            1, np.floor(target - spec.family_lower_slack)
        ).astype(int)
        upper = np.ceil(target + spec.family_upper_slack).astype(int)
        return lower, upper, target

    def audit(
        self,
        selected: Iterable[int],
        spec: ConstraintSpec,
    ) -> dict[str, int]:
        chosen = np.asarray(sorted(set(map(int, selected))), dtype=int)
        semantic = np.bincount(
            self.constraint_codes["semantic_duplicate_group"][chosen]
        )
        basename = np.bincount(self.constraint_codes["basename"][chosen])
        family = np.bincount(
            self.constraint_codes["family"][chosen],
            minlength=len(self.constraint_labels["family"]),
        )
        subgroup = np.bincount(self.constraint_codes["subgroup"][chosen])
        difficulty = np.bincount(
            self.constraint_codes["difficulty_bin"][chosen],
            minlength=len(self.constraint_labels["difficulty_bin"]),
        )
        lower, upper, _ = self.family_bounds(spec)
        values = {
            "size_excess": abs(len(chosen) - self.k),
            "semantic_duplicate_excess": int(
                np.maximum(semantic - 1, 0).sum()
            ),
            "basename_duplicate_excess": int(
                np.maximum(basename - 1, 0).sum()
            ),
            "family_quota_excess": int(
                np.maximum(lower - family, 0).sum()
                + np.maximum(family - upper, 0).sum()
            ),
            "subgroup_cap_excess": int(
                np.maximum(subgroup - spec.subgroup_cap, 0).sum()
            ),
            "difficulty_missing": (
                int((difficulty == 0).sum())
                if spec.require_all_difficulty_bins
                else 0
            ),
        }
        values["hard_constraint_excess"] = int(sum(values.values()))
        return values

    def is_feasible(
        self,
        selected: Iterable[int],
        spec: ConstraintSpec,
    ) -> bool:
        return self.audit(selected, spec)["hard_constraint_excess"] == 0

    @staticmethod
    def _tv_similarity(
        selected_counts: np.ndarray,
        full_probabilities: np.ndarray,
        denominator: int,
    ) -> float:
        selected_probabilities = selected_counts.astype(float) / denominator
        return float(
            1.0
            - 0.5
            * np.abs(selected_probabilities - full_probabilities).sum()
        )

    def objective_terms(
        self,
        selected: Iterable[int],
        objective: ObjectiveSpec,
    ) -> dict[str, float]:
        chosen = np.asarray(sorted(set(map(int, selected))), dtype=int)
        if len(chosen) != self.k:
            raise ValueError(f"目标函数要求 {self.k} 个任务，实际 {len(chosen)}")
        objective = objective.normalized()
        coverage: dict[str, float] = {}
        balance: dict[str, float] = {}
        for field in self.objective_fields:
            counts = np.bincount(
                self.codes[field][chosen],
                minlength=self.n_categories[field],
            )
            coverage[field] = float(
                np.count_nonzero(counts) / self.n_categories[field]
            )
            balance[field] = self._tv_similarity(
                counts, self.full_probabilities[field], self.k
            )
        structural_coverage = float(
            np.mean([coverage[field] for field in STRUCTURAL_FIELDS])
        )
        response_coverage = float(
            np.mean([coverage[field] for field in RESPONSE_FIELDS])
        )
        structural_balance = float(
            np.mean([balance[field] for field in STRUCTURAL_FIELDS])
        )
        response_balance = float(
            np.mean([balance[field] for field in RESPONSE_FIELDS])
        )
        coverage_score = (
            objective.structural_share * structural_coverage
            + (1.0 - objective.structural_share) * response_coverage
        )
        balance_score = (
            objective.structural_share * structural_balance
            + (1.0 - objective.structural_share) * response_balance
        )
        mean_information = float(self.info[chosen].mean())
        score = (
            objective.coverage_weight * coverage_score
            + objective.information_weight * mean_information
            + objective.balance_weight * balance_score
        )
        return {
            "objective": score,
            "coverage": coverage_score,
            "structural_coverage": structural_coverage,
            "response_coverage": response_coverage,
            "mean_information": mean_information,
            "balance": balance_score,
            "structural_balance": structural_balance,
            "response_balance": response_balance,
        }

    def _selected_counts(
        self, selected: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        objective_counts = {
            field: np.bincount(
                self.codes[field][selected],
                minlength=self.n_categories[field],
            )
            for field in self.objective_fields
        }
        constraint_counts = {
            field: np.bincount(
                self.constraint_codes[field][selected],
                minlength=len(self.constraint_labels[field]),
            )
            for field in self.constraint_codes
        }
        return objective_counts, constraint_counts

    def _candidate_mask(
        self,
        old: int,
        candidates: np.ndarray,
        constraint_counts: dict[str, np.ndarray],
        spec: ConstraintSpec,
    ) -> np.ndarray:
        mask = np.ones(len(candidates), dtype=bool)
        for field in ("semantic_duplicate_group", "basename"):
            codes = self.constraint_codes[field]
            old_code = codes[old]
            new_codes = codes[candidates]
            remaining = constraint_counts[field][new_codes] - (
                new_codes == old_code
            )
            mask &= remaining == 0

        family_codes = self.constraint_codes["family"]
        old_family = family_codes[old]
        new_family = family_codes[candidates]
        same_family = new_family == old_family
        lower, upper, _ = self.family_bounds(spec)
        if constraint_counts["family"][old_family] - 1 < lower[old_family]:
            mask &= same_family
        mask &= same_family | (
            constraint_counts["family"][new_family] + 1 <= upper[new_family]
        )

        subgroup_codes = self.constraint_codes["subgroup"]
        old_subgroup = subgroup_codes[old]
        new_subgroup = subgroup_codes[candidates]
        mask &= (
            constraint_counts["subgroup"][new_subgroup]
            + 1
            - (new_subgroup == old_subgroup)
            <= spec.subgroup_cap
        )

        if spec.require_all_difficulty_bins:
            difficulty_codes = self.constraint_codes["difficulty_bin"]
            old_difficulty = difficulty_codes[old]
            if constraint_counts["difficulty_bin"][old_difficulty] == 1:
                mask &= difficulty_codes[candidates] == old_difficulty
        return mask

    def _objective_swap_delta(
        self,
        old: int,
        candidates: np.ndarray,
        objective_counts: dict[str, np.ndarray],
        objective: ObjectiveSpec,
    ) -> np.ndarray:
        objective = objective.normalized()
        structural_coverage_delta = np.zeros(len(candidates), dtype=float)
        response_coverage_delta = np.zeros(len(candidates), dtype=float)
        structural_balance_delta = np.zeros(len(candidates), dtype=float)
        response_balance_delta = np.zeros(len(candidates), dtype=float)

        for field in self.objective_fields:
            codes = self.codes[field]
            old_code = codes[old]
            new_codes = codes[candidates]
            same = new_codes == old_code
            counts = objective_counts[field]
            coverage_delta = (
                (counts[new_codes] == 0).astype(float)
                - float(counts[old_code] == 1)
            ) / self.n_categories[field]
            coverage_delta[same] = 0.0

            old_before = abs(
                counts[old_code] / self.k
                - self.full_probabilities[field][old_code]
            )
            old_after = abs(
                (counts[old_code] - 1) / self.k
                - self.full_probabilities[field][old_code]
            )
            new_before = np.abs(
                counts[new_codes] / self.k
                - self.full_probabilities[field][new_codes]
            )
            new_after = np.abs(
                (counts[new_codes] + 1) / self.k
                - self.full_probabilities[field][new_codes]
            )
            balance_delta = -0.5 * (
                old_after - old_before + new_after - new_before
            )
            balance_delta[same] = 0.0

            if field in STRUCTURAL_FIELDS:
                structural_coverage_delta += (
                    coverage_delta / len(STRUCTURAL_FIELDS)
                )
                structural_balance_delta += (
                    balance_delta / len(STRUCTURAL_FIELDS)
                )
            else:
                response_coverage_delta += (
                    coverage_delta / len(RESPONSE_FIELDS)
                )
                response_balance_delta += (
                    balance_delta / len(RESPONSE_FIELDS)
                )

        coverage_delta = (
            objective.structural_share * structural_coverage_delta
            + (1.0 - objective.structural_share) * response_coverage_delta
        )
        balance_delta = (
            objective.structural_share * structural_balance_delta
            + (1.0 - objective.structural_share) * response_balance_delta
        )
        information_delta = (self.info[candidates] - self.info[old]) / self.k
        return (
            objective.coverage_weight * coverage_delta
            + objective.information_weight * information_delta
            + objective.balance_weight * balance_delta
        )

    def best_swap(
        self,
        selected: Iterable[int],
        objective: ObjectiveSpec,
        spec: ConstraintSpec,
    ) -> tuple[int, int, float] | None:
        chosen = np.asarray(
            sorted(set(map(int, selected)), key=lambda index: self.dataset_ids[index]),
            dtype=int,
        )
        if not self.is_feasible(chosen, spec):
            raise ValueError("best_swap 的起点不满足硬约束")
        selected_mask = np.zeros(self.n, dtype=bool)
        selected_mask[chosen] = True
        candidates = np.flatnonzero(~selected_mask)
        candidates = candidates[np.argsort(self.dataset_ids[candidates])]
        objective_counts, constraint_counts = self._selected_counts(chosen)

        best: tuple[int, int, float] | None = None
        for old in chosen:
            mask = self._candidate_mask(
                old, candidates, constraint_counts, spec
            )
            if not mask.any():
                continue
            feasible_candidates = candidates[mask]
            deltas = self._objective_swap_delta(
                old,
                feasible_candidates,
                objective_counts,
                objective,
            )
            position = int(np.argmax(deltas))
            delta = float(deltas[position])
            candidate = int(feasible_candidates[position])
            if best is None or delta > best[2] + 1e-14:
                best = (int(old), candidate, delta)
        if best is None or best[2] <= 1e-12:
            return None
        return best

    def optimize(
        self,
        start: Iterable[int],
        objective: ObjectiveSpec,
        spec: ConstraintSpec,
        max_steps: int = 100,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        selected = set(map(int, start))
        if not self.is_feasible(selected, spec):
            raise ValueError("局部搜索起点不满足硬约束")
        history: list[dict[str, Any]] = []
        for step in range(max_steps):
            swap = self.best_swap(selected, objective, spec)
            if swap is None:
                break
            old, new, delta = swap
            selected.remove(old)
            selected.add(new)
            history.append(
                {
                    "step": step + 1,
                    "removed": str(self.dataset_ids[old]),
                    "added": str(self.dataset_ids[new]),
                    "delta": delta,
                }
            )
        result = np.asarray(sorted(selected), dtype=int)
        if not self.is_feasible(result, spec):
            raise AssertionError("局部搜索产生了不可行子集")
        return result, history

    def _target_family_counts(self, spec: ConstraintSpec) -> np.ndarray:
        lower, upper, target = self.family_bounds(spec)
        counts = lower.copy()
        while int(counts.sum()) < self.k:
            available = np.flatnonzero(counts < upper)
            if not len(available):
                raise ValueError("family quota 上界无法容纳 50 个任务")
            deficits = target[available] - counts[available]
            chosen = int(available[int(np.argmax(deficits))])
            counts[chosen] += 1
        if int(counts.sum()) != self.k:
            raise ValueError("family quota 下界总和超过 50")
        return counts

    def construct_feasible(
        self,
        spec: ConstraintSpec,
        rng: np.random.Generator,
        attempts: int = 200,
    ) -> np.ndarray:
        target = self._target_family_counts(spec)
        family_codes = self.constraint_codes["family"]
        difficulty_codes = self.constraint_codes["difficulty_bin"]
        for _ in range(attempts):
            selected: list[int] = []
            remaining = target.copy()
            semantic_used: set[int] = set()
            basename_used: set[int] = set()
            subgroup_counts = np.zeros(
                len(self.constraint_labels["subgroup"]), dtype=int
            )
            difficulties_used: set[int] = set()
            while len(selected) < self.k:
                allowed: list[int] = []
                for index in range(self.n):
                    family = family_codes[index]
                    if remaining[family] <= 0:
                        continue
                    semantic = self.constraint_codes[
                        "semantic_duplicate_group"
                    ][index]
                    basename = self.constraint_codes["basename"][index]
                    subgroup = self.constraint_codes["subgroup"][index]
                    if semantic in semantic_used or basename in basename_used:
                        continue
                    if subgroup_counts[subgroup] >= spec.subgroup_cap:
                        continue
                    allowed.append(index)
                if not allowed:
                    break
                allowed_array = np.asarray(allowed, dtype=int)
                missing_bonus = np.asarray(
                    [
                        1.0
                        if difficulty_codes[index] not in difficulties_used
                        else 0.0
                        for index in allowed_array
                    ]
                )
                jitter = rng.random(len(allowed_array))
                priority = (
                    self.info[allowed_array]
                    + 0.25 * missing_bonus
                    + 0.15 * jitter
                )
                choice = int(allowed_array[int(np.argmax(priority))])
                selected.append(choice)
                remaining[family_codes[choice]] -= 1
                semantic_used.add(
                    self.constraint_codes["semantic_duplicate_group"][choice]
                )
                basename_used.add(self.constraint_codes["basename"][choice])
                subgroup_counts[
                    self.constraint_codes["subgroup"][choice]
                ] += 1
                difficulties_used.add(difficulty_codes[choice])
            result = np.asarray(sorted(selected), dtype=int)
            if self.is_feasible(result, spec):
                return result
        raise RuntimeError(f"{attempts} 次尝试后仍无法构造可行 Core-50")


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_selection(
    problem: SelectionProblem,
    selected: np.ndarray,
    frozen_ids: set[str],
    reference_ids: set[str],
    objective: ObjectiveSpec,
    constraints: ConstraintSpec,
    dataset_algorithm: pd.DataFrame,
    full_scores: dict[str, float],
) -> dict[str, Any]:
    ids = problem.ids(selected)
    terms = problem.objective_terms(selected, objective)
    audit = problem.audit(selected, constraints)
    rank = ranking_metrics(full_scores, method_scores(dataset_algorithm, ids))
    return {
        **asdict(objective.normalized()),
        **terms,
        **audit,
        "frozen_overlap": len(ids & frozen_ids),
        "frozen_jaccard": jaccard(ids, frozen_ids),
        "reference_overlap": len(ids & reference_ids),
        "reference_jaccard": jaccard(ids, reference_ids),
        "exact_frozen_match": ids == frozen_ids,
        "exact_reference_match": ids == reference_ids,
        **rank,
    }


def unique_local_weight_grid() -> list[tuple[str, ObjectiveSpec]]:
    baseline = np.asarray([0.45, 0.35, 0.20], dtype=float)
    factors = (0.8, 0.9, 1.0, 1.1, 1.2)
    unique: dict[tuple[float, float, float], tuple[str, ObjectiveSpec]] = {}
    for cov_factor, info_factor, balance_factor in product(factors, repeat=3):
        weights = baseline * np.asarray(
            [cov_factor, info_factor, balance_factor]
        )
        spec = normalize_weights(*map(float, weights))
        key = (
            round(spec.coverage_weight, 12),
            round(spec.information_weight, 12),
            round(spec.balance_weight, 12),
        )
        label = (
            f"local_c{cov_factor:.1f}_i{info_factor:.1f}_b{balance_factor:.1f}"
        )
        unique.setdefault(key, (label, spec))
    return sorted(unique.values(), key=lambda item: item[0])


def broad_simplex_grid(step: float = 0.10) -> list[tuple[str, ObjectiveSpec]]:
    scale = round(1.0 / step)
    rows: list[tuple[str, ObjectiveSpec]] = []
    for coverage_units in range(scale + 1):
        for information_units in range(scale - coverage_units + 1):
            balance_units = scale - coverage_units - information_units
            values = np.asarray(
                [coverage_units, information_units, balance_units], dtype=float
            ) / scale
            spec = normalize_weights(*map(float, values))
            rows.append(
                (
                    "simplex_"
                    f"c{spec.coverage_weight:.2f}_"
                    f"i{spec.information_weight:.2f}_"
                    f"b{spec.balance_weight:.2f}",
                    spec,
                )
            )
    return rows


def choose_best(
    problem: SelectionProblem,
    starts: Iterable[np.ndarray],
    objective: ObjectiveSpec,
    constraints: ConstraintSpec,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    best_selection: np.ndarray | None = None
    best_history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_start = -1
    for start_index, start in enumerate(starts):
        selected, history = problem.optimize(
            start, objective, constraints
        )
        score = problem.objective_terms(selected, objective)["objective"]
        ids_key = tuple(sorted(problem.ids(selected)))
        best_ids_key = (
            tuple(sorted(problem.ids(best_selection)))
            if best_selection is not None
            else ()
        )
        if score > best_score + 1e-12 or (
            math.isclose(score, best_score, abs_tol=1e-12)
            and ids_key < best_ids_key
        ):
            best_selection = selected
            best_history = history
            best_score = score
            best_start = start_index
    if best_selection is None:
        raise RuntimeError("没有可用的局部搜索起点")
    return best_selection, best_history, best_start


def deduplicate_starts(starts: Iterable[np.ndarray]) -> list[np.ndarray]:
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for start in starts:
        key = tuple(sorted(map(int, start)))
        unique.setdefault(key, np.asarray(key, dtype=int))
    return list(unique.values())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [json_safe(value) for value in payload]
    if isinstance(payload, tuple):
        return [json_safe(value) for value in payload]
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.floating, float)):
        value = float(payload)
        return value if math.isfinite(value) else None
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    return payload


def summarize_sweep(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "configurations": int(len(frame)),
        "exact_frozen_match_rate": float(frame["exact_frozen_match"].mean()),
        "frozen_overlap_min": int(frame["frozen_overlap"].min()),
        "frozen_overlap_median": float(frame["frozen_overlap"].median()),
        "frozen_overlap_max": int(frame["frozen_overlap"].max()),
        "frozen_jaccard_min": float(frame["frozen_jaccard"].min()),
        "frozen_jaccard_median": float(frame["frozen_jaccard"].median()),
        "reference_overlap_min": int(frame["reference_overlap"].min()),
        "reference_jaccard_min": float(frame["reference_jaccard"].min()),
        "rank_spearman_min": float(frame["rank_spearman"].min()),
        "rank_kendall_min": float(frame["rank_kendall"].min()),
        "pairwise_rank_agreement_min": float(
            frame["pairwise_rank_agreement"].min()
        ),
        "aggregate_score_mae_max": float(
            frame["aggregate_score_mae"].max()
        ),
        "unique_rank_orders": int(frame["rank_order"].nunique()),
    }


def build_membership_frequency(
    problem: SelectionProblem,
    memberships: pd.DataFrame,
    frozen_ids: set[str],
    reference_ids: set[str],
) -> pd.DataFrame:
    counts = memberships.groupby("dataset_id")["config_id"].nunique()
    output = problem.frame[
        [
            "dataset_id",
            "dataset_name",
            "dataset_rel",
            "family",
            "subgroup",
            "info_score",
            "difficulty_bin",
            "failure_mode",
        ]
    ].copy()
    output["selection_count"] = output["dataset_id"].map(counts).fillna(0).astype(int)
    output["selection_frequency"] = output["selection_count"] / max(
        memberships["config_id"].nunique(), 1
    )
    output["is_frozen_core50"] = output["dataset_id"].isin(frozen_ids)
    output["is_reconstructed_reference"] = output["dataset_id"].isin(reference_ids)
    return output.sort_values(
        ["selection_frequency", "dataset_id"], ascending=[False, True]
    )


def render_readme(
    input_audit: dict[str, Any],
    baseline: dict[str, Any],
    local_summary: dict[str, Any],
    simplex_summary: dict[str, Any],
    constraint_summary: dict[str, Any],
) -> str:
    reproduction = "通过" if baseline["exact_frozen_match"] else "未通过"
    conclusion = (
        "因此可以把权重扫描解释为历史 Core-50 的直接复现敏感性。"
        if baseline["exact_frozen_match"]
        else (
            "因此不能把后续扫描写成历史 Core-50 选择器的直接复现；"
            "它只能作为论文公式的可审计反事实重构。"
        )
    )
    return f"""# 固定 Probe-4 的 Core-50 成员敏感性

本目录只复用已经完成的 Full-664 Probe-4 三种子结果，不启动任何新的
符号回归训练。固定算法为 `dso / imcts / pyoperon / udsr`。

## 结论边界

基线复现闸门：**{reproduction}**。

- 冻结 Core-50 与论文公式重构基线重合 `{baseline['frozen_overlap']}/50`，
  Jaccard 为 `{baseline['frozen_jaccard']:.4f}`。
- 冻结清单目标值为 `{baseline['frozen_objective']:.6f}`，重构局部最优为
  `{baseline['objective']:.6f}`。
- 名义基线使用 `{baseline['restart_count_unique']}` 个互异可行起点；每个
  权重配置使用重构基线加 `{input_audit['grid_auxiliary_starts_requested']}`
  个辅助起点。
- {conclusion}

这个边界很重要：仓库中没有论文所述“随机重启 + 一换一局部搜索”的历史
可执行选择器；冻结清单来自更早的 Master-100 到 Dev/Core 切分链。

## 可执行重构

目标函数为：

```text
J = w_cov * Coverage + w_info * MeanInfo + w_bal * Balance
baseline = (0.45, 0.35, 0.20)
```

已落实的硬约束：

1. 恰好 50 个任务。
2. semantic duplicate 与 basename 均至多出现一次。
3. family smoothing 为 `0.6`，下/上松弛为 `-1/+2`。
4. subgroup cap 为 `5`。
5. easy / medium / hard / extreme 均至少出现一次。

Coverage 和 Balance 使用论文列出的已物化分类字段；`dummy_variable_status`
由 SRSd metadata class 确定，`valid_output_pattern` 由四探针有效 seed 数编码。
论文声明的 `ood_type` 未出现在 Full-664 后处理表中，未伪造该字段。Balance
的结构/响应混合沿用 Coverage 的 `0.6/0.4`，因为论文没有给出另一组数值。
`limited-quota` 的具体 cap 同样未公开，所以不把事后猜测写进主约束。

## 扫描结果

局部合理扰动把三个主权重分别乘以
`{{0.8, 0.9, 1.0, 1.1, 1.2}}` 后重新归一化：

- 配置数：`{local_summary['configurations']}`
- 对冻结清单 overlap：最小 `{local_summary['frozen_overlap_min']}/50`，
  中位 `{local_summary['frozen_overlap_median']:.1f}/50`
- 对重构基线 Jaccard 最小值：
  `{local_summary['reference_jaccard_min']:.4f}`
- Probe-4 对 Full-664 的 Spearman 最小值：
  `{local_summary['rank_spearman_min']:.4f}`
- Kendall 最小值：`{local_summary['rank_kendall_min']:.4f}`
- 成对排序一致率最小值：
  `{local_summary['pairwise_rank_agreement_min']:.4f}`

这组结果的正确解释是：

1. **冻结成员资格尚不能由论文目标函数复现。** 在补齐历史选择器或修正文稿
   之前，不应声称冻结 Core-50 对这些权重已经通过成员稳定性验证。
2. **条件于重构选择器，局部权重扰动较稳定。** 最差仍保留
   `{local_summary['reference_overlap_min']}/50` 个重构基线任务。
3. **Probe-4 排序稳定只是面板内诊断。** 这里只包含参与构造的 4 个算法，
   `rho=1` 不能替代 held-out algorithm 验证，也不能单独证明 Core-50
   的外部代表性。

宽范围压力测试使用步长 `0.1` 的三权重完整单纯形：

- 配置数：`{simplex_summary['configurations']}`
- 对冻结清单 overlap 最小值：
  `{simplex_summary['frozen_overlap_min']}/50`
- 对重构基线 Jaccard 最小值：
  `{simplex_summary['reference_jaccard_min']:.4f}`
- Probe-4 排名顺序种类：`{simplex_summary['unique_rank_orders']}`
- Spearman 最小值：`{simplex_summary['rank_spearman_min']:.4f}`

约束/内部常数压力测试：

- 配置数：`{constraint_summary['configurations']}`
- 成功构造并优化：`{constraint_summary['optimized']}`
- 冻结清单在其中不可行：`{constraint_summary['frozen_infeasible']}`
- 成功配置对冻结 overlap 最小值：
  `{constraint_summary['frozen_overlap_min']}/50`

## 复现

```bash
OPENBLAS_NUM_THREADS=1 python check/analyze_core50_membership_sensitivity.py
```

关键输入 SHA-256：

```text
dataset_level     {input_audit['dataset_level_sha256']}
dataset_algorithm {input_audit['dataset_algorithm_sha256']}
core_manifest     {input_audit['core_manifest_sha256']}
```

## 输出

- `input_audit.json`：输入、字段和约束定义审计。
- `baseline_reproduction.json`：基线复现闸门。
- `weight_sweep_local.csv`：主权重 ±20% 全因子扫描。
- `weight_sweep_simplex.csv`：宽范围单纯形压力测试。
- `constraint_stress.csv`：结构/响应混合、family smoothing、subgroup cap。
- `selected_memberships.csv`：每个权重配置的 50 个成员。
- `membership_frequency.csv`：任务入选频率。
- `summary.json`：机器可读汇总。
- `rebuttal_text.md`：内部草稿；基线闸门失败时禁止直接提交。
"""


def render_rebuttal(
    baseline: dict[str, Any],
    local_summary: dict[str, Any],
    simplex_summary: dict[str, Any],
) -> str:
    caveat = (
        "The nominal reconstruction exactly reproduced the frozen Core-50."
        if baseline["exact_frozen_match"]
        else (
            "The nominal reconstruction did not exactly reproduce the frozen "
            "Core-50, so we report this as a counterfactual reconstruction "
            "rather than evidence from the historical selector."
        )
    )
    return f"""# INTERNAL DRAFT: NOT SUBMISSION-READY WHILE THE BASELINE GATE FAILS

## Core-50 objective sensitivity under fixed Probe-4

We added an offline sensitivity analysis that reuses the completed Full-664
three-seed outputs of the fixed Probe-4 panel (DSO, PyOperon, iMCTS, and uDSR);
no new SR runs were launched. We reconstructed the stated constrained
one-for-one local search and first applied an exact baseline-reproduction gate.
{caveat}

Under the local factorial perturbation, each of the three objective weights was
scaled by 0.8--1.2 and renormalized ({local_summary['configurations']}
configurations). The selected subsets retained at least
{local_summary['frozen_overlap_min']}/50 frozen tasks and had a minimum Jaccard
similarity of {local_summary['reference_jaccard_min']:.3f} to the reconstructed
nominal solution. Their Probe-4 rankings relative to Full-664 had minimum
Spearman rho {local_summary['rank_spearman_min']:.3f}, minimum Kendall tau
{local_summary['rank_kendall_min']:.3f}, and minimum pairwise-order agreement
{local_summary['pairwise_rank_agreement_min']:.3f}. A broader 0.1-simplex stress
test ({simplex_summary['configurations']} configurations) gave a minimum
reconstructed-reference Jaccard of
{simplex_summary['reference_jaccard_min']:.3f} and minimum rank Spearman rho
{simplex_summary['rank_spearman_min']:.3f}.

The rank result is an internal diagnostic over only the four construction
probes; it is not a held-out-algorithm validation and should not be presented as
one. Because the exact historical-membership reproduction gate currently
fails, this paragraph must not be submitted as a claim that the frozen Core-50
membership is weight-robust.

We also disclose two specification gaps rather than filling them post hoc:
`ood_type` is not materialized in the released dataset-level table, and the
paper does not give numerical limited-quota caps. We therefore exclude invented
values from the primary analysis and will clarify these implementation details
in the revision.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.restarts < 1 or args.grid_restarts < 1:
        raise ValueError("restarts 与 grid_restarts 必须至少为 1")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dataset_level_raw = pd.read_csv(args.dataset_level)
    dataset_algorithm = pd.read_csv(args.dataset_algorithm)
    core_manifest = pd.read_csv(args.core_manifest)
    dataset_level = prepare_dataset_table(
        dataset_level_raw, dataset_algorithm
    )
    if len(dataset_level) != 664:
        raise ValueError(f"期望 664 个任务，实际 {len(dataset_level)}")
    observed_methods = set(
        dataset_algorithm["method_norm"].astype(str).str.lower()
    )
    if set(PROBE4) != observed_methods:
        raise ValueError(
            f"Probe-4 算法集合不符: {sorted(observed_methods)}"
        )
    expected_algorithm_rows = len(dataset_level) * len(PROBE4)
    if len(dataset_algorithm) != expected_algorithm_rows:
        raise ValueError(
            f"dataset-algorithm 行数应为 {expected_algorithm_rows}，"
            f"实际 {len(dataset_algorithm)}"
        )

    problem = SelectionProblem(dataset_level)
    frozen_ids = core_ids_from_manifest(dataset_level, core_manifest)
    if len(frozen_ids) != 50:
        raise ValueError(f"冻结清单应有 50 个任务，实际 {len(frozen_ids)}")
    frozen = problem.indices(frozen_ids)
    nominal_objective = ObjectiveSpec().normalized()
    nominal_constraints = ConstraintSpec()
    frozen_audit = problem.audit(frozen, nominal_constraints)
    if frozen_audit["hard_constraint_excess"] != 0:
        raise ValueError(f"冻结 Core-50 未通过主约束: {frozen_audit}")

    input_audit = {
        "analysis_type": "offline_fixed_probe4_paper_spec_reconstruction",
        "new_sr_runs_launched": False,
        "dataset_level": str(args.dataset_level.relative_to(REPO_ROOT)),
        "dataset_algorithm": str(
            args.dataset_algorithm.relative_to(REPO_ROOT)
        ),
        "core_manifest": str(args.core_manifest.relative_to(REPO_ROOT)),
        "dataset_level_sha256": sha256(args.dataset_level),
        "dataset_algorithm_sha256": sha256(args.dataset_algorithm),
        "core_manifest_sha256": sha256(args.core_manifest),
        "dataset_count": len(dataset_level),
        "dataset_algorithm_rows": len(dataset_algorithm),
        "probe4": list(PROBE4),
        "frozen_core_count": len(frozen_ids),
        "structural_fields_used": list(STRUCTURAL_FIELDS),
        "response_fields_used": list(RESPONSE_FIELDS),
        "declared_but_unavailable_fields": list(
            DECLARED_BUT_UNAVAILABLE_FIELDS
        ),
        "balance_structural_share_assumption": 0.60,
        "limited_quota_cap": None,
        "limited_quota_note": (
            "论文与可执行代码均未提供数值，主分析不事后猜测"
        ),
        "historical_selector_available": False,
        "historical_selector_note": (
            "论文所述 constrained local search 无对应历史可执行脚本；"
            "本分析为 paper-spec reconstruction"
        ),
        "nominal_objective": asdict(nominal_objective),
        "nominal_constraints": asdict(nominal_constraints),
        "random_seed": args.seed,
        "baseline_restarts_requested": args.restarts,
        "grid_auxiliary_starts_requested": args.grid_restarts,
        "frozen_constraint_audit": frozen_audit,
    }
    write_json(output / "input_audit.json", input_audit)

    rng = np.random.default_rng(args.seed)
    starts = [frozen]
    for _ in range(args.restarts - 1):
        starts.append(problem.construct_feasible(nominal_constraints, rng))
    starts = deduplicate_starts(starts)
    reconstructed, baseline_history, best_start = choose_best(
        problem, starts, nominal_objective, nominal_constraints
    )
    reconstructed_ids = problem.ids(reconstructed)
    full_ids = set(dataset_level["dataset_id"].astype(str))
    full_scores = method_scores(dataset_algorithm, full_ids)
    baseline = evaluate_selection(
        problem,
        reconstructed,
        frozen_ids,
        reconstructed_ids,
        nominal_objective,
        nominal_constraints,
        dataset_algorithm,
        full_scores,
    )
    frozen_terms = problem.objective_terms(frozen, nominal_objective)
    baseline.update(
        {
            "gate": (
                "pass_exact_reproduction"
                if reconstructed_ids == frozen_ids
                else "fail_exact_reproduction"
            ),
            "frozen_objective": frozen_terms["objective"],
            "objective_improvement_over_frozen": (
                baseline["objective"] - frozen_terms["objective"]
            ),
            "best_start_index": best_start,
            "restart_count_requested": args.restarts,
            "restart_count_unique": len(starts),
            "accepted_swaps": len(baseline_history),
            "swap_history": baseline_history,
            "frozen_probe4_scores": method_scores(
                dataset_algorithm, frozen_ids
            ),
            "reconstructed_probe4_scores": method_scores(
                dataset_algorithm, reconstructed_ids
            ),
            "full664_probe4_scores": full_scores,
        }
    )
    write_json(
        output / "baseline_reproduction.json", json_safe(baseline)
    )

    membership_rows: list[dict[str, str]] = []
    grid_starts = deduplicate_starts(
        [reconstructed, *starts[: args.grid_restarts]]
    )

    def execute_weight_grid(
        grid: list[tuple[str, ObjectiveSpec]], sweep_name: str
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for config_id, objective in grid:
            selected, history, start_index = choose_best(
                problem,
                grid_starts,
                objective,
                nominal_constraints,
            )
            row = evaluate_selection(
                problem,
                selected,
                frozen_ids,
                reconstructed_ids,
                objective,
                nominal_constraints,
                dataset_algorithm,
                full_scores,
            )
            row.update(
                {
                    "sweep": sweep_name,
                    "config_id": config_id,
                    "accepted_swaps": len(history),
                    "best_start_index": start_index,
                    "unique_search_starts": len(grid_starts),
                }
            )
            rows.append(row)
            for dataset_id in sorted(problem.ids(selected)):
                membership_rows.append(
                    {
                        "sweep": sweep_name,
                        "config_id": config_id,
                        "dataset_id": dataset_id,
                    }
                )
        return pd.DataFrame(rows).sort_values("config_id")

    local = execute_weight_grid(
        unique_local_weight_grid(), "local_factorial"
    )
    simplex = execute_weight_grid(
        broad_simplex_grid(), "broad_simplex"
    )
    local.to_csv(output / "weight_sweep_local.csv", index=False)
    simplex.to_csv(output / "weight_sweep_simplex.csv", index=False)
    memberships = pd.DataFrame(membership_rows)
    memberships.to_csv(output / "selected_memberships.csv", index=False)
    membership_frequency = build_membership_frequency(
        problem, memberships, frozen_ids, reconstructed_ids
    )
    membership_frequency.to_csv(
        output / "membership_frequency.csv", index=False
    )

    constraint_rows: list[dict[str, Any]] = []
    for structural_share, family_smoothing, subgroup_cap in product(
        (0.40, 0.60, 0.80),
        (0.40, 0.60, 0.80),
        (4, 5, 6),
    ):
        objective = ObjectiveSpec(structural_share=structural_share).normalized()
        constraints = ConstraintSpec(
            family_smoothing=family_smoothing,
            subgroup_cap=subgroup_cap,
        )
        frozen_constraint_audit = problem.audit(frozen, constraints)
        config_id = (
            f"struct{structural_share:.1f}_"
            f"family{family_smoothing:.1f}_subcap{subgroup_cap}"
        )
        starts_for_stress: list[np.ndarray] = []
        if frozen_constraint_audit["hard_constraint_excess"] == 0:
            starts_for_stress.append(frozen)
        try:
            starts_for_stress.append(
                problem.construct_feasible(constraints, rng)
            )
        except RuntimeError:
            pass
        if not starts_for_stress:
            constraint_rows.append(
                {
                    "config_id": config_id,
                    **asdict(objective),
                    **asdict(constraints),
                    "status": "no_feasible_start",
                    "frozen_feasible": False,
                    **{
                        f"frozen_{key}": value
                        for key, value in frozen_constraint_audit.items()
                    },
                }
            )
            continue
        selected, history, start_index = choose_best(
            problem, starts_for_stress, objective, constraints
        )
        row = evaluate_selection(
            problem,
            selected,
            frozen_ids,
            reconstructed_ids,
            objective,
            constraints,
            dataset_algorithm,
            full_scores,
        )
        row.update(
            {
                "config_id": config_id,
                **asdict(constraints),
                "status": "optimized",
                "frozen_feasible": (
                    frozen_constraint_audit["hard_constraint_excess"] == 0
                ),
                "accepted_swaps": len(history),
                "best_start_index": start_index,
                **{
                    f"frozen_{key}": value
                    for key, value in frozen_constraint_audit.items()
                },
            }
        )
        constraint_rows.append(row)
    constraint_stress = pd.DataFrame(constraint_rows).sort_values("config_id")
    constraint_stress.to_csv(output / "constraint_stress.csv", index=False)

    local_summary = summarize_sweep(local)
    simplex_summary = summarize_sweep(simplex)
    optimized_constraints = constraint_stress[
        constraint_stress["status"].eq("optimized")
    ]
    constraint_summary = {
        "configurations": int(len(constraint_stress)),
        "optimized": int(len(optimized_constraints)),
        "no_feasible_start": int(
            constraint_stress["status"].eq("no_feasible_start").sum()
        ),
        "frozen_infeasible": int(
            (~constraint_stress["frozen_feasible"].astype(bool)).sum()
        ),
        "frozen_overlap_min": (
            int(optimized_constraints["frozen_overlap"].min())
            if len(optimized_constraints)
            else 0
        ),
        "reference_jaccard_min": (
            float(optimized_constraints["reference_jaccard"].min())
            if len(optimized_constraints)
            else None
        ),
        "rank_spearman_min": (
            float(optimized_constraints["rank_spearman"].min())
            if len(optimized_constraints)
            else None
        ),
    }
    summary = {
        "input_audit": input_audit,
        "baseline_reproduction": {
            key: value
            for key, value in baseline.items()
            if key not in {"swap_history"}
        },
        "local_weight_sweep": local_summary,
        "broad_simplex_sweep": simplex_summary,
        "constraint_stress": constraint_summary,
    }
    write_json(output / "summary.json", json_safe(summary))
    (output / "README.md").write_text(
        render_readme(
            input_audit,
            baseline,
            local_summary,
            simplex_summary,
            constraint_summary,
        ),
        encoding="utf-8",
    )
    (output / "rebuttal_text.md").write_text(
        render_rebuttal(baseline, local_summary, simplex_summary),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定 Probe-4 的 Core-50 成员敏感性分析"
    )
    parser.add_argument(
        "--dataset-level", type=Path, default=DEFAULT_DATASET_LEVEL
    )
    parser.add_argument(
        "--dataset-algorithm",
        type=Path,
        default=DEFAULT_DATASET_ALGORITHM,
    )
    parser.add_argument(
        "--core-manifest", type=Path, default=DEFAULT_CORE_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument(
        "--grid-restarts",
        type=int,
        default=2,
        help="每个权重配置复用的 nominal 起点数，另加重构基线起点",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    baseline = summary["baseline_reproduction"]
    local = summary["local_weight_sweep"]
    print(
        json.dumps(
            {
                "baseline_gate": baseline["gate"],
                "baseline_frozen_overlap": baseline["frozen_overlap"],
                "local_configs": local["configurations"],
                "local_min_frozen_overlap": local["frozen_overlap_min"],
                "local_min_rank_spearman": local["rank_spearman_min"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
