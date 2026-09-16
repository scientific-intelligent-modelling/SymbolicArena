"""按当前轨迹终点聚合修订六轴；缺失的符号证据只标 unresolved。

本模块不调用模型。输入的 equivalence/structure 文件必须是已冻结的有效裁决索引，
不能传计划、复用候选或旧发布包的裁决。每条裁决须绑定当前化简依赖和终点 SHA。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from .metrics import (
    RunQuality,
    efficiency_from_qualities,
    minimality_score,
    numerical_consistency,
    stability_score,
    symbolic_fidelity_score,
)
from .symbolic_evidence import (
    SymbolicEvidenceError,
    build_symbolic_artifact,
    operator_f1,
    tree_similarity,
    variable_f1,
)

CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = (520, 521, 522)
POSITIVE_STRUCTURE = {"mathematically_equivalent", "same_canonical_structure"}
DECIDED_STRUCTURE = POSITIVE_STRUCTURE | {"different_structure"}
DECIDED_EQUIVALENCE = {"equivalent", "not_equivalent"}
RunKey = tuple[str, str, str, int]
PairKey = tuple[str, str, str, int, int]


class TerminalAggregateError(ValueError):
    """输入重复或违反当前终点契约。"""


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TerminalAggregateError(f"JSONL 包含非 object: {path}")
    return rows


def _index(rows: Iterable[dict[str, Any]], fields: tuple[str, ...], name: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        try:
            key = tuple(
                int(row[field]) if field.startswith("seed") else
                str(row[field]).casefold() if field == "algorithm" else row[field]
                for field in fields
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TerminalAggregateError(f"{name} 缺少身份字段 {fields}: {row!r}") from exc
        if key in result:
            raise TerminalAggregateError(f"{name} 重复身份: {key}")
        result[key] = row
    return result


def _run_key(row: Mapping[str, Any]) -> RunKey:
    return (str(row["condition"]), str(row["algorithm"]).casefold(), str(row["dataset_id"]), int(row["seed"]))


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=4096)
def _artifact(expression: str, variables: tuple[str, ...]) -> dict[str, Any]:
    return build_symbolic_artifact(expression, allowed_variables=variables)


def _unit(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TerminalAggregateError(f"{label} 缺失或非数值") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise TerminalAggregateError(f"{label} 不在 [0,1] 内: {value!r}")
    return number


def _bool(value: object) -> bool:
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    raise TerminalAggregateError(f"布尔值非法: {value!r}")


def _read_minute_grid(numerical_dir: Path) -> tuple[dict[RunKey, dict[str, Any]], dict[RunKey, str]]:
    """逐行读取三个当前 native replay 文件，不使用旧 run_final。"""
    grids: dict[RunKey, dict[str, Any]] = {}
    failures: dict[RunKey, str] = {}
    for condition in CONDITIONS:
        path = numerical_dir / condition / "id_ood_eff_minute.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = _run_key(row)
                if key[0] != condition:
                    raise TerminalAggregateError(f"分钟条件漂移: {key}")
                minute = int(row["minute"])
                if minute not in range(1, 181):
                    raise TerminalAggregateError(f"分钟越界: {key} {minute}")
                grid = grids.setdefault(key, {"q": {}, "endpoint": None})
                if minute in grid["q"]:
                    raise TerminalAggregateError(f"分钟重复: {key} {minute}")
                qid = _unit(row["id_quality"], f"{key} id minute {minute}")
                qood = _unit(row["ood_quality"], f"{key} ood minute {minute}")
                q = _unit(row["quality"], f"{key} quality minute {minute}")
                if not math.isclose(q, (qid + qood) / 2, rel_tol=0, abs_tol=1e-10):
                    failures[key] = "minute_quality_inconsistent"
                grid["q"][minute] = q
                if minute == 180:
                    grid["endpoint"] = dict(row)
    for key, grid in grids.items():
        if set(grid["q"]) != set(range(1, 181)):
            failures[key] = "minute_grid_incomplete"
    return grids, failures


def _binding_issue(terminal: Mapping[str, Any], pred: Mapping[str, Any] | None) -> str | None:
    if pred is None:
        return "prediction_binding_missing"
    for field in ("terminal_expression", "terminal_expression_sha256", "terminal_source_sha256", "selected_result_sha256"):
        if terminal.get(field) != pred.get(field):
            return f"prediction_{field}_mismatch"
    expression = terminal.get("terminal_expression")
    if expression is not None and _sha_text(expression) != terminal.get("terminal_expression_sha256"):
        return "terminal_expression_hash_mismatch"
    if terminal.get("minute180_valid_output") and pred.get("processing_status") != "ready":
        return "prediction_not_ready"
    if terminal.get("minute180_valid_output") and not all(
        pred.get(field) for field in ("prediction_effective_expression", "prediction_frozen_evaluation_key", "prediction_response_sha256")
    ):
        return "prediction_evidence_incomplete"
    return None


def _selection_issue(terminal: Mapping[str, Any]) -> str | None:
    condition, algorithm, dataset, seed = (
        terminal.get("condition"), terminal.get("algorithm"),
        terminal.get("dataset_id"), terminal.get("seed"),
    )
    if terminal.get("logical_key") != f"{algorithm}::{dataset}::s{seed}::{condition}":
        return "selection_logical_key_mismatch"
    if not terminal.get("selected_result_sha256") or not terminal.get("terminal_source_sha256"):
        return "selection_source_hash_missing"
    status = terminal.get("selection_status")
    superseded = terminal.get("superseded")
    if status == "formal_raw_retained" and superseded is False:
        return None
    if status in {
        "eff_overlay_supersession", "symbolfit_validated_supersession",
        "symbolfit_targeted_rerun_supersession",
    } and superseded is True:
        return None
    return "selection_status_inconsistent"


def _judgment_issue(
    row: Mapping[str, Any] | None,
    *,
    dependencies: list[str],
    terminals: list[Mapping[str, Any]],
    decision_set: set[str],
) -> str | None:
    if row is None:
        return "judgment_missing"
    if row.get("state") != "frozen" or not row.get("response_sha256") or not row.get("evaluation_key"):
        return "judgment_not_frozen"
    if row.get("dependencies") != dependencies:
        return "judgment_dependencies_mismatch"
    if row.get("effective_decision") not in decision_set:
        return "judgment_undetermined"
    if len(terminals) == 1:
        terminal = terminals[0]
        if row.get("terminal_expression_sha256") != terminal.get("terminal_expression_sha256"):
            return "judgment_terminal_expression_mismatch"
        if row.get("terminal_source_sha256") != terminal.get("terminal_source_sha256"):
            return "judgment_terminal_source_mismatch"
    else:
        for side, terminal in zip(("left", "right"), terminals):
            for suffix in ("expression_sha256", "source_sha256"):
                expected = terminal.get(f"terminal_{suffix}")
                if row.get(f"{side}_terminal_{suffix}") != expected:
                    return f"judgment_{side}_terminal_{suffix}_mismatch"
    return None


def aggregate(
    *,
    terminal_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    equivalence_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    grids: dict[RunKey, dict[str, Any]],
    grid_failures: dict[RunKey, str] | None = None,
    expected_runs_per_condition: int = 2250,
) -> dict[str, Any]:
    """返回逐运行、逐任务和算法级结果；任何缺项都阻止相关六轴。"""
    terminals = _index(terminal_rows, ("condition", "algorithm", "dataset_id", "seed"), "terminal")
    preds = _index(prediction_rows, ("condition", "algorithm", "dataset_id", "seed"), "prediction")
    gts = _index(gt_rows, ("dataset_id",), "ground truth")
    eqs = _index(equivalence_rows, ("condition", "algorithm", "dataset_id", "seed"), "equivalence")
    structs = _index(structure_rows, ("condition", "algorithm", "dataset_id", "seed_left", "seed_right"), "structure")
    failures = grid_failures or {}
    run_output: list[dict[str, Any]] = []
    task_output: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    by_task: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    def mark(key: tuple[Any, ...], reason: str) -> None:
        unresolved.append({"key": "::".join(map(str, key)), "reason": reason})

    for key, terminal in sorted(terminals.items()):
        condition, _, dataset, seed = key
        algorithm = str(terminal["algorithm"])
        if condition not in CONDITIONS or seed not in SEEDS:
            raise TerminalAggregateError(f"非法运行键: {key}")
        grid = grids.get(key)
        pred = preds.get(key)
        gt = gts.get((dataset,))
        issues: list[str] = []
        selection_issue = _selection_issue(terminal)
        if selection_issue:
            issues.append(selection_issue)
        if gt is None or not all(gt.get(field) for field in (
            "fixed_reference_expression", "gt_frozen_evaluation_key",
            "gt_result_sha256", "gt_source_evidence_sha256",
        )):
            issues.append("ground_truth_missing")
        if grid is None:
            issues.append("minute_grid_missing")
        elif key in failures:
            issues.append(failures[key])
        elif len(grid["q"]) != 180 or grid.get("endpoint") is None:
            issues.append("minute_grid_incomplete")
        else:
            endpoint = grid["endpoint"]
            endpoint_expression = endpoint.get("expression") or None
            if endpoint_expression != terminal.get("terminal_expression"):
                issues.append("minute180_expression_mismatch")
            if endpoint.get("source_sha256") != terminal.get("terminal_source_sha256"):
                issues.append("minute180_source_mismatch")
            if _bool(endpoint.get("valid_output")) != _bool(terminal.get("minute180_valid_output")):
                issues.append("minute180_validity_mismatch")
            if not math.isclose(_unit(endpoint.get("id_quality"), "endpoint ID"), _unit(terminal.get("minute180_id_quality"), "terminal ID"), abs_tol=1e-10):
                issues.append("minute180_id_mismatch")
            if not math.isclose(_unit(endpoint.get("ood_quality"), "endpoint OOD"), _unit(terminal.get("minute180_ood_quality"), "terminal OOD"), abs_tol=1e-10):
                issues.append("minute180_ood_mismatch")
        numeric_ready = not issues or issues == ["ground_truth_missing"]
        issue = _binding_issue(terminal, pred)
        if issue:
            issues.append(issue)
        valid = _bool(terminal.get("minute180_valid_output"))
        if algorithm.casefold() == "gplearn":
            issues.append("gplearn_deferred")
        prediction_ready = (
            numeric_ready and issue is None and algorithm.casefold() != "gplearn"
            and "ground_truth_missing" not in issues
        )
        row: dict[str, Any] = {
            "condition": condition, "algorithm": algorithm, "dataset_id": dataset, "seed": seed,
            "terminal_expression": terminal.get("terminal_expression"),
            "terminal_expression_sha256": terminal.get("terminal_expression_sha256"),
            "terminal_source_sha256": terminal.get("terminal_source_sha256"),
            "selected_result_sha256": terminal.get("selected_result_sha256"),
            "prediction_expression": pred.get("prediction_effective_expression") if pred else None,
            "prediction_key": pred.get("prediction_frozen_evaluation_key") if pred else None,
            "gt_key": gt.get("gt_frozen_evaluation_key") if gt else None,
            "valid_output": valid,
            "id_quality": _unit(terminal.get("minute180_id_quality"), "terminal ID"),
            "ood_quality": _unit(terminal.get("minute180_ood_quality"), "terminal OOD"),
            "m_eff": None, "tree_similarity": None, "variable_f1": None, "operator_f1": None,
            "c_pred": None, "c_ref": None, "equivalence_decision": None,
            "equivalence_key": None, "m_sym": None, "m_min": None,
        }
        if grid is not None and numeric_ready:
            row["m_eff"] = efficiency_from_qualities([grid["q"][minute] for minute in range(1, 181)])
        if not valid and "gplearn_deferred" not in issues:
            row["m_sym"] = row["m_min"] = 0.0
        elif valid and prediction_ready:
            try:
                feature_names = terminal.get("feature_names")
                if not isinstance(feature_names, list) or not feature_names or any(not isinstance(name, str) for name in feature_names):
                    raise TerminalAggregateError("feature_names 缺失或非法")
                variables = tuple(feature_names)
                reference = _artifact(gt["fixed_reference_expression"], variables)
                prediction = _artifact(pred["prediction_effective_expression"], variables)
                row["variable_f1"] = variable_f1(prediction, reference)
                row["operator_f1"] = operator_f1(prediction, reference)
                row["c_pred"] = prediction["node_count"]
                row["c_ref"] = reference["node_count"]
                row["m_min"] = minimality_score(reference["node_count"], prediction["node_count"])
            except (SymbolicEvidenceError, SyntaxError, ValueError, TypeError) as exc:
                issues.append(f"deterministic_symbolic_error:{type(exc).__name__}")
            if row["m_min"] is not None:
                eq = eqs.get(key)
                eq_issue = _judgment_issue(
                    eq,
                    dependencies=[gt["gt_frozen_evaluation_key"], pred["prediction_frozen_evaluation_key"]],
                    terminals=[terminal], decision_set=DECIDED_EQUIVALENCE,
                )
                if eq_issue:
                    issues.append("equivalence_" + eq_issue)
                else:
                    row["equivalence_decision"] = eq["effective_decision"]
                    row["equivalence_key"] = eq["evaluation_key"]
                    row["tree_similarity"] = tree_similarity(prediction, reference)
                    row["m_sym"] = symbolic_fidelity_score(
                        equivalent=eq["effective_decision"] == "equivalent",
                        tree_similarity=row["tree_similarity"],
                        variable_f1=row["variable_f1"],
                        operator_f1=row["operator_f1"],
                    )
        row["numeric_ready"] = numeric_ready
        row["prediction_ready"] = prediction_ready and (not valid or row["m_min"] is not None)
        row["unresolved"] = ";".join(issues)
        row["formal_ready"] = not issues
        for reason in issues:
            mark(key, reason)
        run_output.append(row)
        by_task[(condition, algorithm, dataset)].append(row)

    for task_key, runs in sorted(by_task.items()):
        issues: list[str] = []
        runs = sorted(runs, key=lambda row: row["seed"])
        if [run["seed"] for run in runs] != list(SEEDS):
            issues.append("seed_triple_incomplete")
        pair_decisions: list[bool] = []
        pair_keys: list[str] = []
        if not issues:
            for left, right in combinations(runs, 2):
                pair_key = (task_key[0], task_key[1].casefold(), task_key[2], left["seed"], right["seed"])
                if not left["valid_output"] or not right["valid_output"]:
                    pair_decisions.append(False)
                    pair_keys.append("non_applicable_invalid_output")
                    continue
                pair = structs.get(pair_key)
                deps = [left["prediction_key"], right["prediction_key"]]
                issue = _judgment_issue(
                    pair, dependencies=deps,
                    terminals=[terminals[(task_key[0], task_key[1].casefold(), task_key[2], left["seed"])],
                               terminals[(task_key[0], task_key[1].casefold(), task_key[2], right["seed"])]],
                    decision_set=DECIDED_STRUCTURE,
                )
                if issue:
                    issues.append(f"pair_{left['seed']}_{right['seed']}_{issue}")
                    continue
                pair_decisions.append(pair["effective_decision"] in POSITIVE_STRUCTURE)
                pair_keys.append(pair["evaluation_key"])
        if len(runs) == 3:
            qualities = [RunQuality(run["id_quality"], run["ood_quality"], run["valid_output"]) for run in runs]
            numeric_n = numerical_consistency(qualities) if all(run["numeric_ready"] for run in runs) else None
            validity_v = sum(run["valid_output"] for run in runs) / 3
        else:
            qualities, numeric_n, validity_v = [], None, None
        if any(not run["prediction_ready"] or not run["numeric_ready"] for run in runs):
            issues.append("run_symbolic_or_numeric_unresolved")
        result = {
            "condition": task_key[0], "algorithm": task_key[1], "dataset_id": task_key[2],
            "seed_count": len(runs), "pair_keys": pair_keys,
            "n": numeric_n, "v": validity_v,
            "c": sum(pair_decisions) / 3 if len(pair_decisions) == 3 and not any(x.startswith("pair_") for x in issues) else None,
            "m_stab": None,
            "formal_ready": not issues, "unresolved": ";".join(issues),
        }
        if len(runs) == 3 and not issues:
            stab = stability_score(
                [RunQuality(run["id_quality"], run["ood_quality"], run["valid_output"]) for run in runs],
                structural_pair_results=pair_decisions,
            )
            result.update(n=stab.numerical_consistency, v=stab.validity,
                          c=stab.structural_consistency, m_stab=stab.score)
        for issue in issues:
            mark(task_key, issue)
        task_output.append(result)

    algorithm_output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_output:
        groups[(row["condition"], row["algorithm"])].append(row)
    for row in task_output:
        task_groups[(row["condition"], row["algorithm"])].append(row)
    for key, runs in sorted(groups.items()):
        tasks = task_groups[key]
        numeric_ready = all(run["numeric_ready"] and run["m_eff"] is not None for run in runs)
        sym_ready = all(run["prediction_ready"] and run["m_sym"] is not None for run in runs)
        min_ready = all(run["prediction_ready"] and run["m_min"] is not None for run in runs)
        stab_ready = all(task["m_stab"] is not None for task in tasks)
        complete = len(runs) == 150 and len(tasks) == 50
        algorithm_output.append({
            "condition": key[0], "algorithm": key[1], "run_count": len(runs), "task_count": len(tasks),
            "ID": 100 * sum(run["id_quality"] for run in runs) / len(runs) if numeric_ready and complete else None,
            "OOD": 100 * sum(run["ood_quality"] for run in runs) / len(runs) if numeric_ready and complete else None,
            "EFF": 100 * sum(run["m_eff"] for run in runs) / len(runs) if numeric_ready and complete else None,
            "SYM": 100 * sum(run["m_sym"] for run in runs) / len(runs) if sym_ready and complete else None,
            "MIN": 100 * sum(run["m_min"] for run in runs) / len(runs) if min_ready and complete else None,
            "STAB": 100 * sum(task["m_stab"] for task in tasks) / len(tasks) if stab_ready and complete else None,
            "formal_ready": sym_ready and min_ready and stab_ready and numeric_ready and complete
                            and key[1].casefold() != "gplearn",
        })
    counts = {condition: sum(row["condition"] == condition for row in run_output) for condition in CONDITIONS}
    cohorts = {
        condition: {
            "algorithms": {row["algorithm"].casefold() for row in run_output if row["condition"] == condition},
            "datasets": {row["dataset_id"] for row in run_output if row["condition"] == condition},
        }
        for condition in CONDITIONS
    }
    cohort_consistent = (
        len({frozenset(cohorts[condition]["algorithms"]) for condition in CONDITIONS}) == 1
        and len({frozenset(cohorts[condition]["datasets"]) for condition in CONDITIONS}) == 1
        and all(len(cohorts[condition]["algorithms"]) == 15
                and len(cohorts[condition]["datasets"]) == 50 for condition in CONDITIONS)
    )
    unused_grid_count = len(set(grids) - set(terminals))
    unused_prediction_count = len(set(preds) - set(terminals))
    global_ready = (
        all(count == expected_runs_per_condition for count in counts.values())
        and len(algorithm_output) == 45 and cohort_consistent
        and unused_grid_count == 0 and unused_prediction_count == 0
        and all(row["formal_ready"] for row in algorithm_output)
    )
    return {
        "run_metrics": run_output, "task_stability": task_output,
        "algorithm_six_axis": algorithm_output, "unresolved": unresolved,
        "manifest": {"formal_ready": global_ready, "run_counts": counts,
                     "unresolved_count": len(unresolved), "gplearn_deferred": True,
                     "cohort_consistent": cohort_consistent,
                     "unused_grid_count": unused_grid_count,
                     "unused_prediction_count": unused_prediction_count,
                     "numerical_basis": "current_native_180min_replay", "symbolic_basis": "current_terminal_frozen_judgments"},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-inputs", type=Path, required=True)
    parser.add_argument("--prediction-bindings", type=Path, required=True)
    parser.add_argument("--gt-bindings", type=Path, required=True)
    parser.add_argument("--equivalence-bindings", type=Path)
    parser.add_argument("--structure-bindings", type=Path)
    parser.add_argument("--numerical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    grids, grid_failures = _read_minute_grid(args.numerical_dir)
    output = aggregate(
        terminal_rows=_read_jsonl(args.terminal_inputs),
        prediction_rows=_read_jsonl(args.prediction_bindings),
        gt_rows=_read_jsonl(args.gt_bindings),
        equivalence_rows=_read_jsonl(args.equivalence_bindings),
        structure_rows=_read_jsonl(args.structure_bindings),
        grids=grids, grid_failures=grid_failures,
    )
    input_paths = {
        "terminal_inputs": args.terminal_inputs,
        "prediction_bindings": args.prediction_bindings,
        "gt_bindings": args.gt_bindings,
        "equivalence_bindings": args.equivalence_bindings,
        "structure_bindings": args.structure_bindings,
    }
    for condition in CONDITIONS:
        input_paths[f"minute_{condition}"] = args.numerical_dir / condition / "id_ood_eff_minute.csv.gz"
    output["manifest"]["input_sha256"] = {
        name: _file_sha256(path) for name, path in input_paths.items() if path is not None
    }
    output["manifest"]["schema_version"] = "current_terminal_six_axis.v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("run_metrics", "task_stability", "algorithm_six_axis", "unresolved"):
        _write_csv(args.output_dir / f"{name}.csv", output[name])
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output_dir, delete=False) as handle:
        json.dump(output["manifest"], handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output_dir / "manifest.json")
    print(json.dumps(output["manifest"], sort_keys=True))


if __name__ == "__main__":
    main()
