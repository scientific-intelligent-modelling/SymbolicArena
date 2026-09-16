"""Replan only unresolved Opus 4.8 structure tasks with a JSON-focused prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as builder


ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
COLLECTION = STAGE5 / "work/core50_terminal_collection_20260916_v3"
ORIGINAL = COLLECTION / "opus48_downstream_plan_v1/structure_plan.jsonl"
UNRESOLVED = COLLECTION / "opus48_structure_run_20260916/unresolved.json"
PROMPT = STAGE5 / "config/prompts/structure_json_repair.v2.txt"


def build(original: Path, unresolved: Path, prompt_path: Path, output: Path) -> dict:
    pending = json.loads(unresolved.read_text(encoding="utf-8"))["items"]
    keys = {item["source_evaluation_key"] for item in pending}
    if len(keys) != len(pending) or not keys:
        raise ValueError("unresolved source keys are empty or duplicated")
    base_contract = builder._load_prompt_schema(ROOT, task_kind="structure")
    prompt_bytes = prompt_path.read_bytes()
    contract = builder.PromptSchemaBundle(
        prompt_path=str(prompt_path.resolve()), prompt_version=prompt_path.stem,
        prompt_template=prompt_bytes.decode("utf-8"),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        schema_path=base_contract.schema_path, schema_version=base_contract.schema_version,
        schema=base_contract.schema, schema_sha256=base_contract.schema_sha256,
    )
    source: dict[str, dict] = {}
    with original.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["evaluation_key"] in keys:
                source[row["evaluation_key"]] = row
    if set(source) != keys:
        raise ValueError(f"missing unresolved source plans: {sorted(keys - set(source))}")
    rows = []
    for item in pending:
        old = source[item["source_evaluation_key"]]
        request = old["request"]
        if (old["logical_id"] != item["logical_id"] or old["task_type"] != "stab_structure"
                or request["noise_tag"] != "noise005"):
            raise ValueError(f"unresolved identity mismatch: {item['source_evaluation_key']}")
        new = builder._task_from_request(
            logical_id=old["logical_id"], task_type="stab_structure", priority=old["priority"],
            request=request, evidence_hash=request["evidence_hash"], contract=contract,
            dependencies=tuple(old["dependencies"]), condition=old["condition"],
        ).to_json_record()
        if new["evaluation_key"] == old["evaluation_key"]:
            raise ValueError("repair prompt did not change source evaluation key")
        rows.append(new)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    result = {"original_count": len(pending), "repair_count": len(rows),
              "conditions": sorted({row["request"]["noise_tag"] for row in rows}),
              "source_evaluation_keys": [item["source_evaluation_key"] for item in pending],
              "repair_evaluation_keys": [row["evaluation_key"] for row in rows],
              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (output.parent / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--unresolved", type=Path, default=UNRESOLVED)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.original, args.unresolved, args.prompt, args.output), indent=2))


if __name__ == "__main__":
    main()
