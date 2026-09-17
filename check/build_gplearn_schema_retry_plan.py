"""Replan only schema-invalid gplearn pair judgments with a stricter JSON prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as builder


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(*, source_plan: Path, unresolved: Path, prompt: Path,
          schema: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    missing = json.loads(unresolved.read_text(encoding="utf-8"))["items"]
    keys = {row["source_evaluation_key"] for row in missing if row["attempts"] >= 2}
    if not keys or len(keys) != len(missing):
        raise ValueError("unresolved pair task inventory is empty or not exhausted")
    source = {}
    with source_plan.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["evaluation_key"] in keys:
                source[row["evaluation_key"]] = row
    if set(source) != keys:
        raise ValueError("unresolved source key missing from pair plan")
    prompt_bytes, schema_bytes = prompt.read_bytes(), schema.read_bytes()
    contract = builder.PromptSchemaBundle(
        prompt_path=str(prompt.resolve()), prompt_version=prompt.stem,
        prompt_template=prompt_bytes.decode("utf-8"),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        schema_path=str(schema.resolve()), schema_version=schema.stem,
        schema=json.loads(schema_bytes), schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
    )
    output.mkdir(parents=True)
    rows = []
    for item in missing:
        old = source[item["source_evaluation_key"]]
        if old["logical_id"] != item["logical_id"] or old["task_type"] not in {"equivalence", "stab_structure"}:
            raise ValueError("unresolved pair identity mismatch")
        request = dict(old["request"])
        request["prior_exhausted_source_evaluation_key"] = old["evaluation_key"]
        request["prior_exhausted_model_evaluation_key"] = item["evaluation_key"]
        task = builder._task_from_request(
            logical_id=old["logical_id"] + "::json_retry", task_type=old["task_type"],
            priority=old["priority"], request=request,
            evidence_hash=request["evidence_hash"], contract=contract,
            dependencies=tuple(old["dependencies"]), condition=old["condition"],
        ).to_json_record()
        rows.append(task)
    path = output / "pair_retry_plan.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    report = {"logical_tasks": len(rows), "phase": rows[0]["task_type"],
              "source_plan_sha256": sha(source_plan), "prompt_sha256": sha(prompt),
              "output_sha256": sha(path), "api_requests_sent": 0}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(source_plan=args.source_plan, unresolved=args.unresolved,
                           prompt=args.prompt, schema=args.schema, output=args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
