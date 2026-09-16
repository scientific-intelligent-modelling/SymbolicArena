"""Audit whether frozen symbolic decisions belong to selected trajectory endpoints."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


CONDITIONS = ("clean", "noise001", "noise005")
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))
INDEXED_VARIABLE = re.compile(r"(?:x|X|col)(\d+)")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _algorithm(value: str) -> str:
    return value.casefold().replace("-", "").replace("_", "")


class _CanonicalNames(ast.NodeTransformer):
    def __init__(self, features: list[str], *, map_indexed: bool) -> None:
        self.features = features
        self.map_indexed = map_indexed

    def visit_Name(self, node: ast.Name) -> ast.AST:
        match = INDEXED_VARIABLE.fullmatch(node.id)
        if self.map_indexed and match is not None:
            index = int(match.group(1))
            if index >= len(self.features):
                raise ValueError(f"variable {node.id} exceeds {len(self.features)} features")
            return ast.copy_location(ast.Name(id=self.features[index], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy", "math"}:
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        return node


def expression_fingerprint(expression: str, features: list[str], *, map_indexed: bool) -> str:
    """Only remove representation differences; never assume algebraic equivalence."""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("empty expression")
    tree = ast.parse(expression.strip(), mode="eval")
    tree = _CanonicalNames(features, map_indexed=map_indexed).visit(tree)
    return _sha(ast.dump(tree, annotate_fields=True, include_attributes=False))


def verify_response_artifact(row: dict[str, Any], repo_root: Path) -> bool:
    source = row.get("response_path")
    expected = row.get("response_sha256")
    if not isinstance(source, str) or not source or not isinstance(expected, str) or len(expected) != 64:
        return False
    path = Path(source)
    if not path.is_absolute():
        path = repo_root / path
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _old_rows(release: Path, condition: str) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[str, dict[str, Any]]]:
    rows: dict[tuple[str, str, int], dict[str, str]] = {}
    with (release / condition / "run_final.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (_algorithm(row["algorithm_slug"]), row["dataset_id"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate old final row: {condition}/{key}")
            rows[key] = row
    opus: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(release / condition / "opus5_prediction.jsonl"):
        key = row["evaluation_key"]
        if key in opus:
            raise ValueError(f"duplicate Opus evaluation key: {key}")
        opus[key] = row
    if len(rows) != 2250 or len(opus) != 2250:
        raise ValueError(f"old final grid incomplete: {condition}")
    return rows, opus


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(*, collection: Path, release: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    old_by_condition = {condition: _old_rows(release, condition) for condition in CONDITIONS}
    run_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    artifact_integrity: dict[str, bool] = {}
    for current in _read_jsonl(collection / "terminal_inputs.jsonl.gz"):
        condition = current["condition"]
        algorithm = _algorithm(current["algorithm_slug"])
        dataset = current["dataset_id"]
        seed = int(current["seed"])
        key = (condition, algorithm, dataset, seed)
        if key in by_key:
            raise ValueError(f"duplicate selected terminal: {key}")
        old, opus_by_key = old_by_condition[condition]
        previous = old[(algorithm, dataset, seed)]
        prior = opus_by_key[previous["pred_evaluation_key"]]
        features = current["feature_names"]
        if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
            raise ValueError(f"invalid feature names: {key}")
        terminal = current.get("terminal_expression")
        old_input = prior.get("input_expression")
        match_mode = "none"
        terminal_fp = None
        old_fp = None
        diagnostic = ""
        if isinstance(terminal, str) and terminal.strip() and isinstance(old_input, str) and old_input.strip():
            try:
                old_fp = expression_fingerprint(old_input, features, map_indexed=False)
                terminal_fp = expression_fingerprint(terminal, features, map_indexed=True)
                if terminal_fp == old_fp:
                    match_mode = "mapped_ast"
                else:
                    raw_fp = expression_fingerprint(terminal, features, map_indexed=False)
                    if raw_fp == old_fp:
                        terminal_fp = raw_fp
                        match_mode = "native_ast"
            except (SyntaxError, ValueError) as exc:
                diagnostic = str(exc)[:240]

        valid = current["minute180_valid_output"] is True
        same_result = current.get("selected_result_sha256") == current.get("old_formal_result_sha256")
        old_evaluation_key = prior["evaluation_key"]
        if match_mode != "none" and not diagnostic and algorithm != "gplearn":
            if old_evaluation_key not in artifact_integrity:
                artifact_integrity[old_evaluation_key] = verify_response_artifact(prior, repo_root)
            integrity = artifact_integrity[old_evaluation_key]
        else:
            integrity = False
        if algorithm == "gplearn":
            status = "deferred_gplearn"
        elif not valid:
            status = "invalid_final_output"
        elif not isinstance(terminal, str) or not terminal.strip():
            status = "unresolved_missing_terminal"
        elif not isinstance(old_input, str) or not old_input.strip():
            status = "needs_prediction_simplification_no_old_input"
        elif diagnostic:
            status = "needs_semantic_resolution"
        elif match_mode != "none" and not integrity:
            status = "needs_prediction_simplification_cache_artifact_invalid"
        elif match_mode != "none" and same_result:
            status = "candidate_reuse_same_result"
        elif match_mode != "none":
            status = "candidate_reuse_new_result_rebind"
        else:
            status = "needs_prediction_simplification_changed_expression"

        row = {
            "condition": condition,
            "algorithm": algorithm,
            "dataset_id": dataset,
            "seed": seed,
            "task_id": current["task_id"],
            "logical_key": current["logical_key"],
            "source_status": current["source_status"],
            "minute180_valid_output": valid,
            "terminal_expression": terminal or "",
            "terminal_expression_sha256": current.get("terminal_expression_sha256") or "",
            "terminal_source_sha256": current["terminal_source_sha256"],
            "terminal_archive_member": current["terminal_archive_member"],
            "feature_names_json": json.dumps(features, ensure_ascii=False),
            "selected_result_sha256": current.get("selected_result_sha256") or "",
            "old_formal_result_sha256": current["old_formal_result_sha256"],
            "old_opus_input_expression": old_input or "",
            "old_opus_evaluation_key": prior["evaluation_key"],
            "old_opus_response_sha256": prior.get("response_sha256") or "",
            "old_response_integrity_verified": integrity,
            "terminal_ast_fingerprint": terminal_fp or "",
            "old_opus_ast_fingerprint": old_fp or "",
            "match_mode": match_mode,
            "binding_status": status,
            "diagnostic": diagnostic,
        }
        run_rows.append(row)
        by_key[key] = row
    if len(run_rows) != 6750:
        raise ValueError(f"selected terminal grid is {len(run_rows)}, expected 6750")

    pair_rows: list[dict[str, Any]] = []
    groups = sorted({(condition, algorithm, dataset) for condition, algorithm, dataset, _ in by_key})
    for condition, algorithm, dataset in groups:
        for seed_a, seed_b in SEED_PAIRS:
            left = by_key[(condition, algorithm, dataset, seed_a)]
            right = by_key[(condition, algorithm, dataset, seed_b)]
            if algorithm == "gplearn":
                status = "deferred_gplearn"
            elif not left["minute180_valid_output"] or not right["minute180_valid_output"]:
                status = "non_applicable_invalid_seed"
            elif all(
                item["binding_status"] == "candidate_reuse_same_result"
                for item in (left, right)
            ):
                status = "candidate_reuse_old_pair"
            else:
                status = "refresh_after_prediction_binding"
            pair_rows.append({
                "condition": condition,
                "algorithm": algorithm,
                "dataset_id": dataset,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "terminal_a_sha256": left["terminal_expression_sha256"],
                "terminal_b_sha256": right["terminal_expression_sha256"],
                "prediction_a_status": left["binding_status"],
                "prediction_b_status": right["binding_status"],
                "pair_binding_status": status,
            })
    if len(pair_rows) != 6750:
        raise ValueError(f"seed-pair grid is {len(pair_rows)}, expected 6750")

    run_rows.sort(key=lambda row: (row["condition"], row["algorithm"], row["dataset_id"], row["seed"]))
    pair_rows.sort(key=lambda row: (row["condition"], row["algorithm"], row["dataset_id"], row["seed_a"], row["seed_b"]))
    run_path = output / "prediction_binding_audit.csv"
    pair_path = output / "structure_pair_binding_audit.csv"
    _write_csv(run_path, run_rows, list(run_rows[0]))
    _write_csv(pair_path, pair_rows, list(pair_rows[0]))
    summary = {
        "schema_version": "terminal_symbolic_binding_audit.v1",
        "formal_ready": False,
        "run_count": len(run_rows),
        "pair_count": len(pair_rows),
        "run_status_counts": dict(sorted(Counter(row["binding_status"] for row in run_rows).items())),
        "pair_status_counts": dict(sorted(Counter(row["pair_binding_status"] for row in pair_rows).items())),
        "run_status_by_condition": {
            condition: dict(sorted(Counter(row["binding_status"] for row in run_rows if row["condition"] == condition).items()))
            for condition in CONDITIONS
        },
        "inputs": {
            "collection_manifest": str((collection / "manifest.json").resolve()),
            "historical_release": str(release.resolve()),
        },
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (run_path, pair_path)
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    stage = repo / "AAAI_experiments/stage5_metric_calculation_0831/work"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=stage / "core50_terminal_collection_20260916_v2")
    parser.add_argument("--release", type=Path, default=repo / "AAAI_experiments/Core50_final_20260914/results")
    parser.add_argument("--output", type=Path, default=stage / "core50_terminal_collection_20260916_v2/symbolic_preflight")
    args = parser.parse_args()
    print(json.dumps(audit(collection=args.collection, release=args.release, output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
