"""只为输入或依赖变化的最终公式重建等价和跨种子结构裁决。"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from . import symbolic_task_builder as builder
from ..scripts.build_core50_final_release import read_csv, read_jsonl, write_jsonl


CONDITIONS = ("clean", "noise001", "noise005")
PAIRS = ((520, 521), (520, 522), (521, 522))
FIELDS = {
    "equivalence": (
        "prediction_result_sha256", "prediction_frozen_evaluation_key",
        "ground_truth_frozen_evaluation_key", "effective_prediction_expression",
        "effective_ground_truth_expression",
    ),
    "structure": (
        "prediction_a_result_sha256", "prediction_b_result_sha256",
        "prediction_a_frozen_evaluation_key", "prediction_b_frozen_evaluation_key",
        "effective_prediction_a_expression", "effective_prediction_b_expression",
        "prediction_a_valid_output", "prediction_b_valid_output",
    ),
}


class RefreshError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_index_row(
    *, plan: Mapping[str, Any], source: Mapping[str, Any], plan_sha256: str,
) -> dict[str, Any]:
    if source.get("evaluation_key") != plan.get("evaluation_key"):
        raise RefreshError(f"结果与计划evaluation_key不一致: {plan.get('logical_id')}")
    structured = source.get("structured_output")
    if source.get("state") == "frozen" and not isinstance(structured, Mapping):
        raise RefreshError(f"frozen结果缺少structured_output: {plan.get('logical_id')}")
    result_sha = source.get("result_sha256") or source.get("frozen_response_sha256") or source.get("response_sha256")
    row = {
        "logical_id": plan["logical_id"], "evaluation_key": plan["evaluation_key"],
        "task_type": plan["task_type"], "task_kind": "simplify",
        "condition": plan["condition"], "priority": plan["priority"],
        "state": source["state"], "attempt_id": source.get("attempt_id"),
        "plan_sha256": plan_sha256, "result_path": source.get("result_path") or source.get("response_path"),
        "result_sha256": result_sha, "structured_output": structured,
        "effective_expression": source.get("effective_expression"),
        "expression_resolution": source.get("expression_resolution"),
        "non_applicable": source.get("non_applicable"), "exhausted": source.get("exhausted"),
        "evidence_generation": source.get("evidence_generation"),
    }
    outcome = structured.get("outcome") if isinstance(structured, Mapping) else None
    if row["state"] == "frozen":
        if outcome in {"simplified", "unchanged"}:
            expected = structured.get("simplified_expression")
            if row["effective_expression"] != expected:
                raise RefreshError(f"effective_expression与Opus输出不一致: {plan['logical_id']}")
        elif outcome == "unable":
            original = plan["request"].get("original_expression")
            if row["effective_expression"] != original:
                raise RefreshError(f"unable未回退原式: {plan['logical_id']}")
        else:
            raise RefreshError(f"非法simplify outcome: {plan['logical_id']}")
    return row


def compose_simplification_indexes(
    *, full_plan_paths: Mapping[str, Path], historical_index_paths: list[Path],
    refresh_index_path: Path, output: Path, repo_root: Path,
) -> dict[str, Any]:
    """按evaluation_key合并旧有效结果和本轮Opus结果，并绑定新的完整plan哈希。"""
    sources: dict[str, dict[str, Any]] = {}
    for path in historical_index_paths:
        for row in read_jsonl(path):
            key = str(row["evaluation_key"])
            previous = sources.get(key)
            tagged = {**row, "evidence_generation": "preexisting"}
            if previous is None:
                sources[key] = tagged
                continue
            previous_sha = previous.get("result_sha256") or previous.get("frozen_response_sha256") or previous.get("response_sha256")
            current_sha = row.get("result_sha256") or row.get("frozen_response_sha256") or row.get("response_sha256")
            if previous_sha and current_sha and previous_sha != current_sha:
                raise RefreshError(f"同一evaluation_key存在不同冻结响应: {key}")
            merged = dict(previous)
            for name, value in tagged.items():
                if name not in merged or merged[name] is None:
                    merged[name] = value
            sources[key] = merged
    for row in read_jsonl(refresh_index_path):
        key = str(row["evaluation_key"])
        previous = sources.get(key)
        tagged = {**row, "evidence_generation": "new"}
        if previous is not None:
            previous_sha = previous.get("result_sha256") or previous.get("frozen_response_sha256")
            current_sha = row.get("result_sha256") or row.get("frozen_response_sha256")
            if previous_sha != current_sha:
                raise RefreshError(f"同一evaluation_key存在不同结果: {key}")
        sources[key] = tagged
    outputs = {}
    for label, plan_path in full_plan_paths.items():
        plans = read_jsonl(plan_path)
        plan_sha = _sha(plan_path)
        rows = []
        for plan in plans:
            source = sources.get(str(plan["evaluation_key"]))
            if source is None:
                raise RefreshError(f"{label}缺少冻结结果: {plan['logical_id']}")
            if "state" not in source:
                raw_path = source.get("result_path")
                result_sha = source.get("result_sha256")
                if not isinstance(raw_path, str) or not isinstance(result_sha, str):
                    raise RefreshError(f"新冻结结果缺少路径或SHA: {plan['logical_id']}")
                resolved = Path(raw_path)
                if not resolved.is_absolute():
                    resolved = repo_root / resolved
                if not resolved.is_file() or _sha(resolved) != result_sha:
                    raise RefreshError(f"新冻结响应缺失或SHA漂移: {resolved}")
                container = json.loads(resolved.read_text(encoding="utf-8"))
                if container.get("evaluation_key") != plan["evaluation_key"] or container.get("logical_id") != plan["logical_id"]:
                    raise RefreshError(f"新冻结响应身份漂移: {plan['logical_id']}")
                structured = container.get("structured_output")
                if not isinstance(structured, Mapping):
                    raise RefreshError(f"新冻结响应缺少structured_output: {plan['logical_id']}")
                outcome = structured.get("outcome")
                if outcome in {"simplified", "unchanged"}:
                    effective = structured.get("simplified_expression")
                    resolution = "llm_simplified_expression"
                elif outcome == "unable":
                    effective = plan["request"].get("original_expression")
                    resolution = "original_identity_fallback_after_llm_unable"
                else:
                    raise RefreshError(f"新冻结响应outcome非法: {plan['logical_id']}")
                source = {
                    **source, "state": "frozen", "structured_output": structured,
                    "effective_expression": effective, "expression_resolution": resolution,
                    "non_applicable": None, "exhausted": None,
                }
            row = _normalized_index_row(plan=plan, source=source, plan_sha256=plan_sha)
            result_path = row.get("result_path")
            result_sha = row.get("result_sha256")
            if row["state"] == "frozen":
                if not isinstance(result_path, str) or not result_path:
                    raise RefreshError(f"frozen结果缺少result_path: {plan['logical_id']}")
                resolved = Path(result_path)
                if not resolved.is_absolute():
                    resolved = repo_root / resolved
                if not resolved.is_file() or _sha(resolved) != result_sha:
                    raise RefreshError(f"冻结响应文件缺失或SHA漂移: {resolved}")
            rows.append(row)
        path = output / f"{label}_frozen_index.jsonl"
        write_jsonl(path, rows)
        summary = {
            "status": "ok", "output_jsonl": str(path.resolve()), "output_sha256": _sha(path),
            "plan_jsonl": str(plan_path.resolve()), "plan_sha256": plan_sha,
            "row_count": len(rows), "state_counts": dict(__import__("collections").Counter(r["state"] for r in rows)),
        }
        summary_path = output / f"{label}_frozen_index_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs[label] = {"index": str(path), "summary": str(summary_path), "rows": len(rows), "sha256": summary["output_sha256"]}
    report = {"status": "ok", "source_index_count": len(historical_index_paths) + 1, "outputs": outputs}
    (output / "compose_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def materialize_effective_clean_judgments(
    *, eq_index_path: Path, compact_eq_path: Path, structure_index_path: Path,
    structure_overlay_path: Path, output: Path,
) -> dict[str, Any]:
    """保留冻结Opus原响应，另加审计后的有效decision，绝不改写响应SHA。"""
    compact = {r["logical_id"]: r for r in read_jsonl(compact_eq_path)}
    eq_rows = []
    eq_overlays = 0
    for row in read_jsonl(eq_index_path):
        effective = dict(row)
        audit = compact.get(row["logical_id"])
        if audit is None or audit.get("evaluation_key") != row.get("evaluation_key"):
            raise RefreshError(f"clean等价compact与冻结索引不一致: {row['logical_id']}")
        original = (row.get("structured_output") or {}).get("decision")
        decision = audit.get("effective_decision") or original
        if decision not in {"equivalent", "not_equivalent", "undetermined"}:
            raise RefreshError(f"clean等价有效decision非法: {row['logical_id']}")
        effective["effective_decision"] = decision
        effective["audit_overlay_applied"] = bool(audit.get("audit_overlay_applied"))
        effective["underlying_opus_decision"] = original
        if effective["audit_overlay_applied"]:
            eq_overlays += 1
        eq_rows.append(effective)
    structure_overlays = {r["logical_id"]: r for r in read_jsonl(structure_overlay_path)}
    structure_rows = []
    applied = 0
    current_structure_ids = set()
    for row in read_jsonl(structure_index_path):
        current_structure_ids.add(row["logical_id"])
        effective = dict(row)
        original = (row.get("structured_output") or {}).get("decision")
        overlay = structure_overlays.get(row["logical_id"])
        decision = (overlay.get("structured_output") or {}).get("decision") if overlay else original
        if decision not in {"mathematically_equivalent", "same_canonical_structure", "different_structure", "undetermined"}:
            raise RefreshError(f"clean结构有效decision非法: {row['logical_id']}")
        effective["effective_decision"] = decision
        effective["underlying_opus_decision"] = original
        effective["audit_overlay_applied"] = overlay is not None
        if overlay:
            effective["audit_overlay_evaluation_key"] = overlay["evaluation_key"]
            effective["audit_overlay_result_path"] = overlay.get("result_path")
            effective["audit_overlay_result_sha256"] = overlay.get("result_sha256") or overlay.get("result_container_sha256")
            applied += 1
        structure_rows.append(effective)
    output.mkdir(parents=True, exist_ok=True)
    eq_output = output / "clean_equivalence_effective_index.jsonl"
    structure_output = output / "clean_structure_effective_index.jsonl"
    write_jsonl(eq_output, eq_rows)
    write_jsonl(structure_output, structure_rows)
    unmatched_overlays = sorted(set(structure_overlays) - current_structure_ids)
    report = {"status": "ok", "equivalence_rows": len(eq_rows), "equivalence_audit_overlays": eq_overlays,
              "structure_rows": len(structure_rows), "structure_audit_overlays": applied,
              "unmatched_structure_overlay_logical_ids": unmatched_overlays,
              "unmatched_structure_overlay_note": "后续final replacement已改变对应seed表达式，不复用旧审计覆盖层",
              "underlying_opus_responses_preserved": True,
              "outputs": {"equivalence": {"path": str(eq_output), "sha256": _sha(eq_output)},
                          "structure": {"path": str(structure_output), "sha256": _sha(structure_output)}}}
    (output / "effective_judgment_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _key(request: Mapping[str, Any], condition: str):
    return str(request.get("algorithm_slug") or request["algorithm"]).casefold(), str(request["dataset_id"]), int(request["seed"]), condition


def _pair_key(request: Mapping[str, Any], condition: str):
    return str(request.get("algorithm_slug") or request["algorithm"]).casefold(), str(request["dataset_id"]), int(request["seed_a"]), int(request["seed_b"]), condition


def _bool(value: Any) -> bool:
    if value is True or str(value).lower() in {"true", "1"}:
        return True
    if value is False or str(value).lower() in {"false", "0"}:
        return False
    raise RefreshError(f"非法有效性标志: {value!r}")


def can_reuse(old_plan: Mapping[str, Any], old_index: Mapping[str, Any], context: Mapping[str, Any], dependencies, phase: str) -> bool:
    if old_index.get("state") not in {"frozen", "non_applicable"}:
        return False
    if old_index.get("evaluation_key") != old_plan.get("evaluation_key"):
        return False
    if set(old_plan.get("dependencies", [])) != set(dependencies):
        return False
    request = old_plan.get("request") or {}
    for name in FIELDS[phase]:
        if name.endswith("valid_output"):
            if _bool(request.get(name)) != _bool(context.get(name)):
                return False
        elif request.get(name) != context.get(name):
            return False
    return True


def _load_bundles(plan_path: Path, index_path: Path):
    indexes = {r["logical_id"]: r for r in read_jsonl(index_path)}
    plans = read_jsonl(plan_path)
    if len(indexes) != len(plans):
        raise RefreshError(f"plan/index行数不一致: {plan_path}")
    output = []
    for plan in plans:
        index = indexes[plan["logical_id"]]
        if index["evaluation_key"] != plan["evaluation_key"]:
            raise RefreshError(f"evaluation_key不一致: {plan['logical_id']}")
        output.append((plan, index))
    return output


def _records(plan: Mapping[str, Any], index: Mapping[str, Any]):
    structured = index.get("structured_output") or {}
    p = builder.SimplifyPlanRecord(
        logical_id=plan["logical_id"], task_type=plan["task_type"], evaluation_key=plan["evaluation_key"],
        priority=int(plan["priority"]), request=dict(plan["request"]),
    )
    f = builder.FrozenSimplifyRecord(
        evaluation_key=index["evaluation_key"], logical_id=index["logical_id"], task_type=index["task_type"],
        plan_sha256=index.get("plan_sha256", ""), state=index["state"],
        simplified_status=structured.get("outcome", "non_applicable"),
        simplified_expression=structured.get("simplified_expression"),
        effective_expression=index.get("effective_expression"), expression_resolution=index.get("expression_resolution"),
        structured_output=index.get("structured_output"), non_applicable=index.get("non_applicable"),
        result_sha256=index.get("result_sha256"),
    )
    if f.state == "frozen":
        builder._validate_effective_expression_binding(
            row=index, plan_request=p.request, logical_id=p.logical_id,
            simplified_status=f.simplified_status, simplified_expression=f.simplified_expression,
        )
    return p, f


def _new_job(job):
    phase, condition, logical_id, context, dependencies, left, right, seed, repo_root, output = job
    left_plan, left_index = left
    right_plan, right_index = right
    lp, lf = _records(left_plan, left_index)
    rp, rf = _records(right_plan, right_index)
    contract = builder._load_prompt_schema(Path(repo_root), task_kind=phase)
    task_type = "equivalence" if phase == "equivalence" else "stab_structure"
    no_call = not (builder._has_callable_expression(lf) and builder._has_callable_expression(rf))
    if phase == "structure":
        no_call = no_call or not (_bool(context["prediction_a_valid_output"]) and _bool(context["prediction_b_valid_output"]))
    if no_call:
        record = builder._materialize_non_applicable_evidence(
            evidence_dir=Path(output) / "non_applicable_evidence", logical_id=logical_id, task_type=task_type,
            phase=phase, reason="upstream_expression_or_seed_invalid", contract=contract,
            dependencies=tuple(dependencies), request_context=dict(context), write_evidence=True, condition=condition,
        )
        return {"kind": "non_applicable", "phase": phase, "condition": condition, "record": record}
    evidence = builder._build_full_pair_evidence(
        logical_id=logical_id, phase=phase, left_plan=lp, left_frozen=lf,
        right_plan=rp, right_frozen=rf, pair_seed=seed,
    )
    request = {**context, "deterministic_evidence": evidence, "evidence_hash": evidence["evidence_sha256"]}
    if phase == "structure":
        request["deterministic_pair_evidence"] = evidence
    task = builder._task_from_request(
        logical_id=logical_id, task_type=task_type, priority=30 if phase == "equivalence" else 40,
        request=request, evidence_hash=evidence["evidence_sha256"], contract=contract,
        dependencies=tuple(dependencies), condition=condition,
    )
    return {"kind": "refresh", "phase": phase, "condition": condition, "record": task.to_json_record()}


def _binding(prefix: str, plan: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    structured = index.get("structured_output") or {}
    return {
        f"{prefix}_logical_id": plan["logical_id"],
        f"{prefix}_plan_evaluation_key": plan["evaluation_key"],
        f"{prefix}_frozen_evaluation_key": index["evaluation_key"],
        f"{prefix}_frozen_plan_sha256": index.get("plan_sha256"),
        f"{prefix}_simplify_status": structured.get("outcome", "non_applicable"),
        f"simplified_{prefix}_expression": structured.get("simplified_expression"),
        f"effective_{prefix}_expression": index.get("effective_expression"),
        f"{prefix}_expression_resolution": index.get("expression_resolution"),
    }


def build_refresh(config: Mapping[str, Any], output: Path, *, workers: int = 4) -> dict[str, Any]:
    if workers < 1 or workers > 4:
        raise RefreshError("本地证据计算并发必须在1至4之间")
    output.mkdir(parents=True, exist_ok=True)
    gt = {p["request"]["dataset_id"]: (p, i) for p, i in _load_bundles(Path(config["gt_plan"]), Path(config["gt_index"]))}
    if len(gt) != 50:
        raise RefreshError("GT必须覆盖50个任务")
    repo_root = config["repo_root"]
    forced_eq = set(config.get("force_equivalence_logical_ids", []))
    jobs = []
    retained = {(phase, cond): [] for phase in ("equivalence", "structure") for cond in CONDITIONS}
    retained_index = {key: [] for key in retained}
    results = {key: [] for key in retained}
    no_call = {key: [] for key in retained}
    inputs = {}
    for condition in CONDITIONS:
        spec = config["conditions"][condition]
        pred = {_key(p["request"], condition): (p, i) for p, i in _load_bundles(Path(spec["pred_plan"]), Path(spec["pred_index"]))}
        numeric = {(r["algorithm"].casefold(), r["dataset_id"], int(r["seed"]), condition): r for r in read_csv(Path(spec["numeric"]))}
        old_eq = {_key(p["request"], condition): (p, i) for p, i in _load_bundles(Path(spec["old_eq_plan"]), Path(spec["old_eq_index"]))}
        old_struct = {_pair_key(p["request"], condition): (p, i) for p, i in _load_bundles(Path(spec["old_structure_plan"]), Path(spec["old_structure_index"]))}
        if len(pred) != 2250 or set(numeric) != set(pred) or set(old_eq) != set(pred) or len(old_struct) != 2250:
            raise RefreshError(f"{condition}网格不完整")
        for key, (plan, index) in sorted(pred.items()):
            request = plan["request"]
            gp, gi = gt[key[1]]
            number = numeric[key]
            context = {"dataset_id": key[1], "dataset_index": request["dataset_index"],
                       "algorithm": request["algorithm"], "algorithm_slug": key[0], "seed": key[2],
                       "noise_tag": condition, "variables": request["variables"],
                       "allowed_functions": sorted(set(request.get("allowed_functions", [])) | set(gp["request"].get("allowed_functions", []))),
                       **_binding("ground_truth", gp, gi), **_binding("prediction", plan, index),
                       "prediction_task_id": number["task_id"], "prediction_valid_output": _bool(number["valid_output"]),
                       "prediction_result_sha256": number["result_sha256"]}
            deps = (gi["evaluation_key"], index["evaluation_key"])
            old_plan, old_index = old_eq[key]
            logical_id = f"equivalence::{key[0]}::{request['dataset_index']}::s{key[2]}::{condition}"
            if logical_id not in forced_eq and can_reuse(old_plan, old_index, context, deps, "equivalence"):
                retained[("equivalence", condition)].append(old_plan)
                retained_index[("equivalence", condition)].append(old_index)
            else:
                jobs.append(("equivalence", condition, logical_id, context, deps, (gp, gi), (plan, index), key[2], repo_root, str(output)))
        groups = sorted({(k[0], k[1]) for k in pred})
        if len(groups) != 750:
            raise RefreshError(f"{condition}算法任务组合不是750个")
        for algorithm, dataset in groups:
            for a, b in PAIRS:
                ka, kb = (algorithm, dataset, a, condition), (algorithm, dataset, b, condition)
                pa, ia = pred[ka]
                pb, ib = pred[kb]
                ra = pa["request"]
                context = {"dataset_id": dataset, "dataset_index": ra["dataset_index"],
                           "algorithm": ra["algorithm"], "algorithm_slug": algorithm, "noise_tag": condition,
                           "variables": ra["variables"], "seed_a": a, "seed_b": b,
                           "allowed_functions": sorted(set(ra.get("allowed_functions", [])) | set(pb["request"].get("allowed_functions", []))),
                           **_binding("prediction_a", pa, ia), **_binding("prediction_b", pb, ib)}
                for label, key in (("a", ka), ("b", kb)):
                    context[f"prediction_{label}_task_id"] = numeric[key]["task_id"]
                    context[f"prediction_{label}_valid_output"] = _bool(numeric[key]["valid_output"])
                    context[f"prediction_{label}_result_sha256"] = numeric[key]["result_sha256"]
                deps = (ia["evaluation_key"], ib["evaluation_key"])
                old_plan, old_index = old_struct[(algorithm, dataset, a, b, condition)]
                logical_id = builder._structure_logical_id(algorithm, ra["dataset_index"], a, b, condition=condition)
                if can_reuse(old_plan, old_index, context, deps, "structure"):
                    retained[("structure", condition)].append(old_plan)
                    retained_index[("structure", condition)].append(old_index)
                else:
                    jobs.append(("structure", condition, logical_id, context, deps, (pa, ia), (pb, ib), a * 1000 + b, repo_root, str(output)))
        inputs[condition] = {name: {"path": value, "sha256": _sha(Path(value))} for name, value in spec.items()}
    print(json.dumps({"evidence_jobs": len(jobs), "reused": {f"{a}/{b}": len(v) for (a, b), v in retained.items()}}, ensure_ascii=False), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, result in enumerate(pool.map(_new_job, jobs, chunksize=1), 1):
            key = result["phase"], result["condition"]
            if result["kind"] == "refresh":
                results[key].append(result["record"])
            else:
                no_call[key].append(result["record"])
            if n % 50 == 0:
                print(json.dumps({"evidence_done": n, "total": len(jobs)}), flush=True)
    output_summary = {}
    all_refresh = []
    for (phase, condition), refreshed in results.items():
        prefix = output / f"{condition}_{phase}"
        write_jsonl(prefix.with_name(prefix.name + "_refresh_plan.jsonl"), refreshed)
        write_jsonl(prefix.with_name(prefix.name + "_reused_index.jsonl"), retained_index[(phase, condition)])
        write_jsonl(prefix.with_name(prefix.name + "_non_applicable.jsonl"), no_call[(phase, condition)])
        full = retained[(phase, condition)] + refreshed + no_call[(phase, condition)]
        if len(full) != 2250:
            raise RefreshError(f"{condition}/{phase}全计划不完整")
        write_jsonl(prefix.with_name(prefix.name + "_full_plan.jsonl"), full)
        all_refresh.extend(refreshed)
        output_summary[f"{condition}/{phase}"] = {"reused": len(retained[(phase, condition)]), "refresh": len(refreshed), "non_applicable": len(no_call[(phase, condition)])}
    write_jsonl(output / "downstream_refresh_plan.jsonl", all_refresh)
    manifest = {"status": "ready_for_routify_api", "new_task_count": len(all_refresh), "groups": output_summary,
                "inputs": inputs, "api_invoked": False, "local_workers": workers}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(build_refresh(json.loads(args.config.read_text()), args.output, workers=args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
