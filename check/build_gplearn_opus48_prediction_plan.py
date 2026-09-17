"""Plan one protected-semantics Opus 4.8 simplification per gplearn endpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as builder
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import sha256_json
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.gplearn_native_prefix_evidence import (
    build_inventory_prefix_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
INVENTORY = STAGE5 / "work/core50_terminal_collection_20260916_v3/gplearn_terminal_inventory_v2/gplearn_terminal_inventory.jsonl"
PROBES = STAGE5 / "reports/dataset_probes.jsonl"
PROMPT = STAGE5 / "config/prompts/gplearn_protected_simplify.v1.txt"
SCHEMA = STAGE5 / "config/schemas/simplify.v1.json"
SEMANTICS_VERSION = "gplearn_native_protected_prefix.v1"
CONDITIONS = ("clean", "noise001", "noise005")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def request_for(row: dict, probe: dict, typed: dict) -> dict:
    features = row["feature_names"]
    if probe["basename"] != row["dataset_id"] or probe["variables"] != features:
        raise ValueError(f"probe feature mapping mismatch: {row['logical_key']}")
    values = []
    for point in probe["points"]:
        item = [float(point["values"][name]) for name in features]
        if not all(math.isfinite(number) for number in item):
            raise ValueError(f"nonfinite probe: {row['logical_key']}")
        values.append(item)
    if not values or len(values) != int(probe["point_count"]):
        raise ValueError(f"frozen probe count mismatch: {row['logical_key']}")
    match = re.search(r"g\d{4}$", row["task_id"])
    if match is None:
        raise ValueError(f"dataset index missing: {row['task_id']}")
    evidence = {
        "logical_key": row["logical_key"],
        "terminal_snapshot_sha256": row["terminal_snapshot_sha256"],
        "terminal_expression_sha256": row["terminal_expression_sha256"],
        "native_prefix_sha256": row["native_prefix_sha256"],
        "selected_result_sha256": row["selected_result_sha256"],
        "typed_exact_fingerprint": typed["exact_fingerprint"],
        "feature_names": features,
        "probe_sample_sha256": probe["sample_sha256"],
        "semantics_version": SEMANTICS_VERSION,
    }
    typed_expression = typed["typed_expression"]
    return {
        "algorithm": "gplearn", "algorithm_slug": "gplearn",
        "dataset_id": row["dataset_id"], "dataset_index": match.group(0),
        "noise_tag": row["condition"], "seed": row["seed"],
        "task_id": row["task_id"],
        "expression": typed_expression, "original_expression": typed_expression,
        "native_semantics_version": SEMANTICS_VERSION,
        "native_prefix_sha256": row["native_prefix_sha256"],
        "feature_names": features,
        "variables": [f"x{index}" for index in range(len(features))],
        "variable_mapping": {f"x{index}": name for index, name in enumerate(features)},
        "allowed_functions": sorted({node["operator"] for node in typed["nodes"]
                                     if node["kind"] == "operator"}),
        "protected_semantics": typed["protected_semantics"],
        "typed_node_count": typed["node_count"],
        "typed_exact_fingerprint": typed["exact_fingerprint"],
        "typed_structure_fingerprint": typed["constants_abstracted_structure_fingerprint"],
        "probe_points": values,
        "probe_source": probe["schema_version"],
        "probe_sample_sha256": probe["sample_sha256"],
        "model_binding": "claude-opus-4-8",
        "terminal_binding_evidence": evidence,
        "ast_source_evidence": {
            "native_prefix_sha256": row["native_prefix_sha256"],
            "terminal_snapshot_sha256": row["terminal_snapshot_sha256"],
            "selected_result_sha256": row["selected_result_sha256"],
            "canonical_artifact": {"native_prefix": row["native_prefix"]},
        },
        "evidence_hash": sha256_json(evidence),
    }


def build(inventory: Path, probes: Path, prompt: Path, schema: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    records = list(rows(inventory))
    probe_rows = {row["basename"]: row for row in rows(probes)}
    if len(records) != 450 or len(probe_rows) != 50:
        raise ValueError("expected 450 current gplearn endpoints and 50 probes")
    prompt_bytes, schema_bytes = prompt.read_bytes(), schema.read_bytes()
    contract = builder.PromptSchemaBundle(
        prompt_path=str(prompt.resolve()), prompt_version=prompt.stem,
        prompt_template=prompt_bytes.decode("utf-8"), prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        schema_path=str(schema.resolve()), schema_version=schema.stem,
        schema=json.loads(schema_bytes), schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
    )
    output.mkdir(parents=True)
    plan_path = output / "pred_simplify_plan.jsonl"
    typed_path = output / "typed_prefix_evidence.jsonl"
    counts = Counter()
    max_nodes = 0
    keys = set()
    with plan_path.open("w", encoding="utf-8") as plan_handle, typed_path.open("w", encoding="utf-8") as typed_handle:
        for row in sorted(records, key=lambda item: item["logical_key"]):
            key = row["logical_key"]
            if key in keys:
                raise ValueError(f"duplicate gplearn endpoint: {key}")
            keys.add(key)
            typed = build_inventory_prefix_evidence(row)
            request = request_for(row, probe_rows[row["dataset_id"]], typed)
            logical_id = (f"pred_simplify::gplearn::{request['dataset_index']}::s{row['seed']}::"
                          f"{row['condition']}::v3")
            task = builder._task_from_request(
                logical_id=logical_id, task_type="pred_simplify", priority=20,
                request=request, evidence_hash=request["evidence_hash"], contract=contract,
                dependencies=(), condition=row["condition"],
            )
            plan_handle.write(json.dumps(task.to_json_record(), ensure_ascii=False,
                                         sort_keys=True, separators=(",", ":")) + "\n")
            typed_handle.write(json.dumps({"logical_key": key, "evaluation_key": task.evaluation_key,
                                           "typed_evidence": typed}, ensure_ascii=False,
                                          sort_keys=True, separators=(",", ":")) + "\n")
            counts[row["condition"]] += 1
            max_nodes = max(max_nodes, typed["node_count"])
    if counts != {condition: 150 for condition in CONDITIONS}:
        raise ValueError(f"unexpected plan condition counts: {counts}")
    report = {
        "schema_version": "gplearn_opus48_prediction_plan.v1", "logical_tasks": len(keys),
        "counts": dict(counts), "max_native_typed_nodes": max_nodes,
        "model": "claude-opus-4-8", "max_output_tokens": 16384,
        "input_sha256": {"inventory": sha(inventory), "dataset_probes": sha(probes),
                         "prompt": sha(prompt), "schema": sha(schema)},
        "output_sha256": {plan_path.name: sha(plan_path), typed_path.name: sha(typed_path)},
        "api_requests_sent": 0,
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--probes", type=Path, default=PROBES)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.inventory, args.probes, args.prompt, args.schema, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
