"""将通过校验的 gplearn protected-operator 裁决接入完整发布配置。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[7]
RELEASE_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parent
BASE_INPUT_ROOT = RELEASE_ROOT / "publication_inputs_v3_final"
BASE_CONFIG = RELEASE_ROOT / "publication_config_v3.json"
SOURCE_PLAN = BASE_INPUT_ROOT / "clean_structure_current_plan.jsonl"
SOURCE_INDEX = BASE_INPUT_ROOT / "clean_structure_current_mixed_index.jsonl"
RESOLUTION_PLAN = WORK_ROOT / "plans/gplearn_g0029_s520_s521_v4.jsonl"
RESOLUTION_FROZEN = (
    WORK_ROOT
    / "execution_v4/frozen/33af6129a9dcc53d96b915a0786f29d33a2595da7c43068b4b2eb2979d61c4ca.json"
)
OUTPUT_ROOT = RELEASE_ROOT / "publication_inputs_v4_final"
OUTPUT_PLAN = OUTPUT_ROOT / "clean_structure_current_plan.jsonl"
OUTPUT_INDEX = OUTPUT_ROOT / "clean_structure_current_mixed_index.jsonl"
OUTPUT_CONFIG = RELEASE_ROOT / "publication_config_v4_resolved.json"
REPORT_PATH = OUTPUT_ROOT / "gplearn_structure_resolution_composition.json"
TARGET_PREFIX = "stab_structure::gplearn::g0029::s520-s521"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def replace_target(
    rows: list[dict[str, object]],
    replacement: dict[str, object],
) -> list[dict[str, object]]:
    matches = [index for index, row in enumerate(rows) if str(row["logical_id"]).startswith(TARGET_PREFIX)]
    if len(matches) != 1:
        raise RuntimeError(f"待替换目标数量异常: {len(matches)}")
    output = list(rows)
    output[matches[0]] = replacement
    return output


def main() -> None:
    resolution_plan_rows = read_jsonl(RESOLUTION_PLAN)
    if len(resolution_plan_rows) != 1:
        raise RuntimeError("protected-operator resolution plan 必须恰有一行")
    resolution_plan = resolution_plan_rows[0]
    frozen = json.loads(RESOLUTION_FROZEN.read_text(encoding="utf-8"))
    structured = frozen.get("structured_output")
    if (
        frozen.get("evaluation_key") != resolution_plan.get("evaluation_key")
        or frozen.get("logical_id") != resolution_plan.get("logical_id")
        or not isinstance(structured, dict)
        or structured.get("decision") != "different_structure"
        or frozen.get("metadata", {}).get("api_channel") != "routify"
        or frozen.get("metadata", {}).get("requested_model") != "claude-opus-5"
        or frozen.get("metadata", {}).get("requested_effort") != "xhigh"
        or frozen.get("metadata", {}).get("stream") is not False
    ):
        raise RuntimeError("protected-operator resolution frozen 结果未通过身份或传输校验")
    resolution_index = {
        "attempt_id": frozen["attempt_id"],
        "condition": "clean",
        "effective_expression": None,
        "evaluation_key": frozen["evaluation_key"],
        "evidence_generation": "new",
        "expression_resolution": None,
        "logical_id": frozen["logical_id"],
        "result_path": str(RESOLUTION_FROZEN),
        "result_sha256": sha256_file(RESOLUTION_FROZEN),
        "state": "frozen",
        "structured_output": structured,
        "task_kind": "structure",
        "task_type": "stab_structure",
    }
    plan_rows = replace_target(read_jsonl(SOURCE_PLAN), resolution_plan)
    index_rows = replace_target(read_jsonl(SOURCE_INDEX), resolution_index)
    if len(plan_rows) != 2250 or len(index_rows) != 2250:
        raise RuntimeError("完整 structure 网格行数漂移")
    if {row["evaluation_key"] for row in plan_rows} != {
        row["evaluation_key"] for row in index_rows
    }:
        raise RuntimeError("structure plan/index evaluation_key 网格不一致")
    write_text(OUTPUT_PLAN, "".join(canonical_json(row) + "\n" for row in plan_rows))
    write_text(OUTPUT_INDEX, "".join(canonical_json(row) + "\n" for row in index_rows))
    loaded = load_plan_jsonl(OUTPUT_PLAN)
    if len(loaded.entries) != 2250:
        raise RuntimeError("完整 structure plan 回读失败")

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    structure = config["artifacts"]["conditions"]["clean"]["structure"]
    structure["plan"] = {
        "format": "jsonl",
        "path": str(OUTPUT_PLAN.relative_to(REPO_ROOT)),
        "rows": 2250,
        "sha256": sha256_file(OUTPUT_PLAN),
    }
    structure["index"] = {
        "format": "jsonl",
        "generation": "mixed",
        "path": str(OUTPUT_INDEX.relative_to(REPO_ROOT)),
        "rows": 2250,
        "sha256": sha256_file(OUTPUT_INDEX),
    }
    config["output_root"] = str(
        (RELEASE_ROOT / "publication_candidate_v4_resolved").relative_to(REPO_ROOT)
    )
    write_text(OUTPUT_CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    report = {
        "status": "passed",
        "network_request": False,
        "source_plan": str(SOURCE_PLAN),
        "source_index": str(SOURCE_INDEX),
        "resolution_plan": str(RESOLUTION_PLAN),
        "resolution_plan_sha256": sha256_file(RESOLUTION_PLAN),
        "resolution_frozen": str(RESOLUTION_FROZEN),
        "resolution_frozen_sha256": sha256_file(RESOLUTION_FROZEN),
        "resolution_evaluation_key": frozen["evaluation_key"],
        "resolution_decision": structured["decision"],
        "output_plan": str(OUTPUT_PLAN),
        "output_plan_sha256": sha256_file(OUTPUT_PLAN),
        "output_index": str(OUTPUT_INDEX),
        "output_index_sha256": sha256_file(OUTPUT_INDEX),
        "output_config": str(OUTPUT_CONFIG),
        "output_config_sha256": sha256_file(OUTPUT_CONFIG),
    }
    write_text(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
