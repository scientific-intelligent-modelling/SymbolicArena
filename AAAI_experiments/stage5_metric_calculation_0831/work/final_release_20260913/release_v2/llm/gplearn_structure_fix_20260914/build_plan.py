"""生成 gplearn g0029 s520-s521 的 protected-operator 重裁计划。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec


REPO_ROOT = Path(__file__).resolve().parents[7]
WORK_ROOT = Path(__file__).resolve().parent
SOURCE_PLAN = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/work"
    / "final_release_20260913/release_v2/downstream/clean_structure_refresh_plan.jsonl"
)
PROMPT_PATH = WORK_ROOT / "config/structure_gplearn_protected.v1.txt"
OUTPUT_PLAN = WORK_ROOT / "plans/gplearn_g0029_s520_s521_v4.jsonl"
REPORT_PATH = WORK_ROOT / "plans/build_report.json"
SOURCE_LOGICAL_ID = "stab_structure::gplearn::g0029::s520-s521"
TARGET_LOGICAL_ID = f"{SOURCE_LOGICAL_ID}::v4"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    source_rows = [
        json.loads(line)
        for line in SOURCE_PLAN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in source_rows if row["logical_id"] == SOURCE_LOGICAL_ID]
    if len(matches) != 1:
        raise RuntimeError(f"源任务数量异常: {len(matches)}")
    row = copy.deepcopy(matches[0])

    native_contract = {
        "algorithm": "gplearn",
        "number_system": "real",
        "protected_div": "a / b if abs(b) > 0.001 else 1",
        "protected_log": "log(abs(z)) if abs(z) > 0.001 else 0",
        "protected_sqrt": "sqrt(abs(z))",
        "source": "scientific_intelligent_modelling.benchmarks.runner._eval_gplearn_prefix_node",
    }
    native_programs = {
        "seed_520": {
            "raw_prefix": "mul(log(cos(mul(X1, 0.420))), add(add(sin(add(sin(X1), X1)), sin(X1)), cos(add(log(X1), cos(X1)))))",
            "constants_abstracted_prefix": "mul(log(cos(mul(var,const))),add(add(sin(add(sin(var),var)),sin(var)),cos(add(log(var),cos(var)))))",
            "root_operator": "mul",
        },
        "seed_521": {
            "raw_prefix": "log(sqrt(add(sin(X1), log(0.262))))",
            "constants_abstracted_prefix": "log(sqrt(add(sin(var),log(const))))",
            "root_operator": "log",
        },
    }
    request = copy.deepcopy(row["request"])
    request["frozen_dependency_evaluation_keys"] = list(row["dependencies"])
    predecessor_evidence_hash = request["evidence_hash"]
    request["algorithm_native_operator_contract"] = native_contract
    request["algorithm_native_programs"] = native_programs
    request["display_expression_warning"] = (
        "Display infix expressions omit gplearn protection; use native prefix programs "
        "for domain semantics and structural identity."
    )
    for evidence_key in ("deterministic_evidence", "deterministic_pair_evidence"):
        evidence = request.get(evidence_key)
        if not isinstance(evidence, dict):
            continue
        evidence["algorithm_native_operator_contract"] = native_contract
        evidence["algorithm_native_programs"] = native_programs
        evidence["domain_assumptions"] = {
            "lhs": native_contract,
            "rhs": native_contract,
        }
    request["evidence_hash"] = sha256_bytes(
        canonical_json(
            {
                "predecessor_evidence_hash": predecessor_evidence_hash,
                "algorithm_native_operator_contract": native_contract,
                "algorithm_native_programs": native_programs,
            }
        ).encode("utf-8")
    )

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha256 = sha256_bytes(PROMPT_PATH.read_bytes())
    schema_path = Path(row["schema_path"])
    schema_sha256 = sha256_bytes(schema_path.read_bytes())
    schema_content = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_version = "structure.gplearn_protected.v1"
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = sha256_bytes(canonical_json(normalized_input).encode("utf-8"))
    target_key = evaluation_key(
        task_type="stab_structure",
        logical_id=TARGET_LOGICAL_ID,
        prompt_version=prompt_version,
        schema_version=row["schema_version"],
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    task_spec = TaskSpec(
        evaluation_key=target_key,
        logical_id=TARGET_LOGICAL_ID,
        task_type="stab_structure",
        condition="clean",
        priority=int(row["priority"]),
        input_hash=input_hash,
        prompt_version=prompt_version,
        schema_version=row["schema_version"],
        dependencies=(),
    )
    row.update(
        {
            "evaluation_key": target_key,
            "logical_id": TARGET_LOGICAL_ID,
            "dependencies": [],
            "input_hash": input_hash,
            "prompt_path": str(PROMPT_PATH),
            "prompt_sha256": prompt_sha256,
            "prompt_template": prompt_template,
            "prompt_version": prompt_version,
            "rendered_prompt": render_prompt(prompt_template, request, schema_content),
            "request": request,
            "normalized_input": normalized_input,
            "schema_sha256": schema_sha256,
            "schema_content": schema_content,
            "task_spec": json.loads(task_spec.canonical_json()),
        }
    )
    write_text(OUTPUT_PLAN, canonical_json(row) + "\n")
    loaded = load_plan_jsonl(OUTPUT_PLAN)
    if len(loaded.entries) != 1 or loaded.entries[0].logical_id != TARGET_LOGICAL_ID:
        raise RuntimeError("计划回读校验失败")
    report = {
        "status": "passed",
        "network_request": False,
        "source_plan": str(SOURCE_PLAN),
        "source_logical_id": SOURCE_LOGICAL_ID,
        "target_logical_id": TARGET_LOGICAL_ID,
        "evaluation_key": target_key,
        "plan": str(OUTPUT_PLAN),
        "plan_sha256": loaded.plan_sha256,
        "prompt": str(PROMPT_PATH),
        "prompt_sha256": prompt_sha256,
        "predecessor_evidence_hash": predecessor_evidence_hash,
        "evidence_hash": request["evidence_hash"],
    }
    write_text(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
