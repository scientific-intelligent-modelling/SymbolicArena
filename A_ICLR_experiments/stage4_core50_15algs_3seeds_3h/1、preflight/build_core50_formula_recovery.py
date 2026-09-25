import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import (
    PARAMETER_REFERENCE_START_PATTERN,
    select_formula_with_source,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import (
    _corrected_artifact,
)


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / ".agent/work/GOAL-CORE50"
CONDITIONS = ("clean", "noise001", "noise005")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_freeze(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: freeze row 必须为object")
            rows.append(row)
    return rows


def numeric_parameters(value: object, *, task_id: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{task_id}: canonical parameter_values 缺失")
    values = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError(f"{task_id}: canonical parameter_values 含布尔值")
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise ValueError(f"{task_id}: canonical parameter_values 含非有限值")
        values.append(parsed)
    return values


def build_entry(row: dict[str, Any], candidate_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source = row["source"]
    result = row["result"]
    task_id = str(source["task_id"])
    raw_text = str(result["raw_text"])
    raw_sha = sha256_bytes(raw_text.encode("utf-8"))
    if raw_sha != result["sha256"]:
        raise ValueError(f"{task_id}: freeze result SHA256 不一致")
    payload = json.loads(raw_text)
    expression, expression_source = select_formula_with_source(payload)
    if not expression or PARAMETER_REFERENCE_START_PATTERN.search(expression) is None:
        return None, None

    equation = payload.get("equation")
    canonical = payload.get("canonical_artifact")
    if not isinstance(equation, str) or not isinstance(canonical, dict):
        return None, {
            "task_id": task_id,
            "condition": source["noise_tag"],
            "result_sha256": raw_sha,
            "reason": "parameterized_expression_without_canonical_artifact",
            "expression_source": expression_source,
        }
    if expression != equation or canonical.get("raw_equation") != equation:
        return None, {
            "task_id": task_id,
            "condition": source["noise_tag"],
            "result_sha256": raw_sha,
            "reason": "selected_parameterized_expression_differs_from_raw_equation",
            "expression_source": expression_source,
        }
    if canonical.get("artifact_valid") is not True or canonical.get("validation_errors"):
        return None, {
            "task_id": task_id,
            "condition": source["noise_tag"],
            "result_sha256": raw_sha,
            "reason": "canonical_artifact_not_validated",
        }

    params = numeric_parameters(canonical.get("parameter_values"), task_id=task_id)
    algorithm = str(source["algorithm"])
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"{task_id}: feature_names 缺失")
    rebuilt, rebuild_changed = _corrected_artifact(
        payload,
        algorithm=algorithm,
        expected_n_features=len(feature_names),
    )
    for field in ("raw_equation", "parameter_values", "instantiated_expression", "expected_n_features"):
        if rebuilt.get(field) != canonical.get(field):
            return None, {
                "task_id": task_id,
                "condition": source["noise_tag"],
                "result_sha256": raw_sha,
                "reason": f"canonical_artifact_rebuild_mismatch:{field}",
                "rebuilt_canonical_artifact_sha256": sha256_bytes(canonical_json(rebuilt).encode("utf-8")),
                "source_canonical_artifact_sha256": sha256_bytes(canonical_json(canonical).encode("utf-8")),
            }

    candidate = {
        "function": equation,
        "params": params,
        "source": "result_bound_canonical_artifact",
        "source_result_sha256": raw_sha,
        "canonical_artifact_sha256": sha256_bytes(canonical_json(canonical).encode("utf-8")),
        "rebuilt_by_current_performance_replay": bool(rebuild_changed),
    }
    candidate_bytes = (json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    candidate_path = candidate_root / str(source["noise_tag"]) / f"{task_id}.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        if candidate_path.read_bytes() != candidate_bytes:
            raise FileExistsError(f"拒绝覆盖漂移 recovery candidate: {candidate_path}")
    else:
        candidate_path.write_bytes(candidate_bytes)

    params_sha = sha256_bytes(canonical_json(params).encode("utf-8"))
    equation_sha = sha256_bytes(equation.encode("utf-8"))
    entry = {
        "task_id": task_id,
        "resolution": "recovered_params",
        "frozen_result_sha256": raw_sha,
        "equation_sha256": equation_sha,
        "params": params,
        "source_evidence": {
            "source_kind": "result_bound_canonical_artifact_parameter_values",
            "source_result_path": source.get("path"),
            "source_result_sha256": raw_sha,
            "canonical_artifact_sha256": candidate["canonical_artifact_sha256"],
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha256_bytes(candidate_bytes),
            "params_sha256": params_sha,
            "rebuild_matches_source_artifact": True,
        },
    }
    return entry, None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.candidate")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_manifests(freeze_root: Path, output_root: Path, report_path: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"recovery output root 非空，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = output_root / "candidates"
    totals: dict[str, Any] = {}
    for condition in CONDITIONS:
        freeze_path = freeze_root / f"{condition}_runs_available.jsonl.gz"
        rows = read_freeze(freeze_path)
        entries = []
        unresolved = []
        for row in rows:
            if row["source"]["noise_tag"] != condition:
                raise ValueError(f"{condition}: freeze condition identity mismatch")
            entry, issue = build_entry(row, candidate_root)
            if entry is not None:
                entries.append(entry)
            if issue is not None:
                unresolved.append(issue)
        manifest = {"schema_version": "formula_recovery.v1", "condition": condition, "entries": entries}
        manifest_path = output_root / f"{condition}_formula_recovery.v1.json"
        write_json(manifest_path, manifest)
        unresolved_path = output_root / f"{condition}_unresolved.jsonl"
        unresolved_path.write_text(
            "".join(canonical_json(item) + "\n" for item in unresolved),
            encoding="utf-8",
        )
        totals[condition] = {
            "freeze_path": str(freeze_path.resolve()),
            "freeze_sha256": sha256_file(freeze_path),
            "freeze_rows": len(rows),
            "recovered_param_count": len(entries),
            "unresolved_param_count": len(unresolved),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "unresolved_path": str(unresolved_path),
        }
    report = {
        "schema": "core50.formula_recovery_build.v1",
        "source_basis": "current frozen result.json canonical_artifact with exact result SHA and deterministic rebuild",
        "source_freeze_root": str(freeze_root.resolve()),
        "output_root": str(output_root),
        "conditions": totals,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-root", type=Path, default=WORK / "source_freezes_available")
    parser.add_argument("--output-root", type=Path, default=WORK / "formula_recovery_available")
    parser.add_argument("--report", type=Path, default=WORK / "formula_recovery_available_report.json")
    args = parser.parse_args()
    report = build_manifests(args.freeze_root, args.output_root, args.report)
    print(json.dumps({
        "conditions": {
            condition: {
                "freeze_rows": item["freeze_rows"],
                "recovered_param_count": item["recovered_param_count"],
                "unresolved_param_count": item["unresolved_param_count"],
            }
            for condition, item in report["conditions"].items()
        }
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
