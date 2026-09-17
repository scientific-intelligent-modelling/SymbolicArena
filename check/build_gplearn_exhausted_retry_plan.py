"""Replan only exhausted gplearn simplification calls with an explicit unable fallback prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as builder


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
PLAN = COLLECTION / "gplearn_opus48_prediction_plan_v4/pred_simplify_plan.jsonl"
UNRESOLVED = COLLECTION / "gplearn_opus48_prediction_run_v2/unresolved.json"
PROMPT = STAGE5 / "config/prompts/gplearn_protected_simplify_exhausted.v2.txt"
SCHEMA = STAGE5 / "config/schemas/simplify.v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(plan_path: Path, unresolved_path: Path, prompt_path: Path,
          schema_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    prior = json.loads(unresolved_path.read_text(encoding="utf-8"))["items"]
    source_keys = {item["source_evaluation_key"] for item in prior}
    if len(prior) != len(source_keys) or not source_keys:
        raise ValueError("unresolved source key inventory invalid")
    original = {}
    with plan_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["evaluation_key"] in source_keys:
                original[row["evaluation_key"]] = row
    if set(original) != source_keys:
        raise ValueError("unresolved gplearn plan source keys not found")
    prompt_bytes, schema_bytes = prompt_path.read_bytes(), schema_path.read_bytes()
    contract = builder.PromptSchemaBundle(
        prompt_path=str(prompt_path.resolve()), prompt_version=prompt_path.stem,
        prompt_template=prompt_bytes.decode("utf-8"),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        schema_path=str(schema_path.resolve()), schema_version=schema_path.stem,
        schema=json.loads(schema_bytes), schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
    )
    output.mkdir(parents=True)
    new_rows = []
    for item in prior:
        old = original[item["source_evaluation_key"]]
        if old["logical_id"] != item["logical_id"] or old["task_type"] != "pred_simplify":
            raise ValueError("unresolved identity drift")
        request = dict(old["request"])
        request["prior_exhausted_source_evaluation_key"] = old["evaluation_key"]
        request["prior_exhausted_model_evaluation_key"] = item["evaluation_key"]
        request["prior_attempt_count"] = item["attempts"]
        row = builder._task_from_request(
            logical_id=old["logical_id"] + "::retry2", task_type="pred_simplify",
            priority=20, request=request, evidence_hash=request["evidence_hash"],
            contract=contract, dependencies=(), condition=old["condition"],
        ).to_json_record()
        if row["evaluation_key"] == old["evaluation_key"]:
            raise ValueError("retry task reused an exhausted evaluation key")
        new_rows.append(row)
    path = output / "pred_simplify_retry_plan.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    report = {"logical_tasks": len(new_rows), "source_keys": sorted(source_keys),
              "prompt_sha256": sha(prompt_path), "source_plan_sha256": sha(plan_path),
              "output_sha256": sha(path), "api_requests_sent": 0}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--unresolved", type=Path, default=UNRESOLVED)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.unresolved, args.prompt, args.schema, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
