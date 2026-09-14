"""按冻结来源合并最终交付，不把旧裁决绑定到新公式。"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


CONDITIONS = ("clean", "noise001", "noise005")
SEEDS = (520, 521, 522)
FORBIDDEN_SOURCE = "all_15alg_fullcpu_v1"
SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b|\bBearer\s+[A-Za-z0-9._-]{16,}")
Key = tuple[str, str, int, str]


class ReleaseError(ValueError):
    """交付输入缺失、漂移或存在禁止来源。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ReleaseError(f"拒绝写出空结果表: {path}")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def identity(row: Mapping[str, Any]) -> Key:
    source = row.get("source", row)
    condition = str(source.get("condition") or source.get("noise_tag") or "")
    key = (
        str(source["algorithm"]).casefold(),
        str(source["dataset_id"]),
        int(source["seed"]),
        condition,
    )
    if not key[0] or not key[1] or key[2] not in SEEDS or key[3] not in CONDITIONS:
        raise ReleaseError(f"非法运行身份: {key}")
    return key


def unique_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[Key, dict[str, Any]]:
    output: dict[Key, dict[str, Any]] = {}
    for row in rows:
        key = identity(row)
        if key in output:
            raise ReleaseError(f"{label} 重复运行: {key}")
        output[key] = row
    return output


def check_raw_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record["source"]
    raw = record["result"]
    text = raw["raw_text"]
    if FORBIDDEN_SOURCE in json.dumps(source, ensure_ascii=False) or FORBIDDEN_SOURCE in text:
        raise ReleaseError("最终来源命中已废弃 fullcpu 批次")
    if SECRET_PATTERN.search(text):
        raise ReleaseError(f"原始结果包含疑似凭据，拒绝打包: {identity(record)}")
    observed = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if raw.get("sha256") != observed:
        raise ReleaseError(f"原始结果 SHA256 不匹配: {identity(record)}")
    payload = json.loads(text)
    if int(payload["seed"]) != int(source["seed"]):
        raise ReleaseError("原始结果 seed 与来源身份不一致")
    if str(payload["tool"]).casefold() != str(source["algorithm"]).casefold():
        raise ReleaseError("原始结果 algorithm 与来源身份不一致")
    return payload


def merge_final_records(
    base: Iterable[dict[str, Any]],
    updates: Iterable[dict[str, Any]],
    allowed_keys: set[Key],
) -> tuple[dict[Key, dict[str, Any]], list[dict[str, Any]]]:
    selected = unique_rows(base, "原始最终来源")
    replacements = unique_rows(updates, "新最终来源")
    if set(replacements) != allowed_keys:
        raise ReleaseError("新结果必须精确匹配声明的 replacement scope")
    if not allowed_keys <= set(selected):
        raise ReleaseError("replacement scope 含原网格之外的运行")
    provenance: list[dict[str, Any]] = []
    for key in sorted(selected):
        original = selected[key]
        current = replacements.get(key, original)
        check_raw_record(current)
        selected[key] = current
        provenance.append({
            "algorithm": key[0], "dataset_id": key[1], "seed": key[2], "condition": key[3],
            "selected_source_path": current["source"].get("path") or current["result"].get("path", ""),
            "selected_source_batch": current["source"].get("batch", ""),
            "selected_result_sha256": current["result"]["sha256"],
            "replaced_in_this_release": key in replacements,
            "previous_result_sha256": original["result"]["sha256"] if key in replacements else "",
        })
    return selected, provenance


def validate_grid(records: Mapping[Key, Any], algorithms: set[str], datasets: set[str]) -> None:
    expected = {(a, d, s, c) for a in algorithms for d in datasets for s in SEEDS for c in CONDITIONS}
    if len(algorithms) != 15 or len(datasets) != 50 or set(records) != expected:
        raise ReleaseError(f"最终来源不是完整 15 x 50 x 3 x 3 网格: {len(records)}")


def prediction_source_sha(request: Mapping[str, Any]) -> Any:
    ast = request.get("ast_source_evidence") or {}
    return request.get("result_raw_sha256") or ast.get("result_raw_sha256") or request.get("result_sha256") or request.get("prediction_result_sha256")


def bind_prediction(
    record: Mapping[str, Any], plan: Mapping[str, Any], index: Mapping[str, Any]
) -> dict[str, Any]:
    request = plan["request"]
    if prediction_source_sha(request) != record["result"]["sha256"]:
        raise ReleaseError(f"Opus5 化简仍绑定旧结果: {identity(record)}")
    request_identity = {
        "algorithm": request.get("algorithm_slug") or request.get("algorithm"),
        "dataset_id": request["dataset_id"], "seed": request["seed"],
        "condition": request.get("noise_tag") or plan.get("condition"),
    }
    if identity(request_identity) != identity(record):
        raise ReleaseError("Opus5 化简身份与最终运行不一致")
    if plan.get("evaluation_key") != index.get("evaluation_key"):
        raise ReleaseError("Opus5 计划与裁决 evaluation_key 不匹配")
    if index.get("state") != "frozen":
        raise ReleaseError(f"缺少已冻结 Opus5 化简: {identity(record)}")
    output = index.get("structured_output") or {}
    effective = index.get("effective_expression")
    if not effective:
        raise ReleaseError(f"缺少有效最终化简表达式: {identity(record)}")
    return {
        "prediction_expression": request.get("expression", ""),
        "opus5_returned_expression": output.get("simplified_expression", ""),
        "effective_prediction_expression": effective,
        "simplification_outcome": output.get("outcome", ""),
        "simplification_resolution": index.get("expression_resolution", ""),
        "pred_logical_id": index["logical_id"],
        "pred_evaluation_key": index["evaluation_key"],
        "pred_result_sha256": index.get("result_sha256", ""),
    }


def copy_checked(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise ReleaseError(f"拒绝未物化文件或链接: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise ReleaseError(f"复制哈希不一致: {destination}")
    return {"source": str(source), "sha256": source_hash, "bytes": destination.stat().st_size}


def write_checksums(output: Path) -> int:
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def verify_release_data(output: Path) -> dict[str, Any]:
    rows = read_csv(output / "provenance/datasets.csv")
    if len(rows) != 50 or len({r["dataset_name"] for r in rows}) != 50:
        raise ReleaseError("数据集清单不是50个唯一任务")
    count = 0
    for row in rows:
        for field in ("metadata", "train", "valid", "id_test", "ood_test", "formula"):
            path = output / row[f"{field}_path"]
            if path.is_symlink() or not path.is_file() or sha256(path) != row[f"{field}_sha256"]:
                raise ReleaseError(f"数据集文件缺失或哈希不匹配: {path}")
            count += 1
    return {"status": "passed", "dataset_count": len(rows), "verified_file_count": count}


def audit_source_bindings(stage: Path, output: Path) -> dict[str, Any]:
    """先生成补评清单；变量集合差异是候选，不冒充完整数学等价检测。"""
    package = stage / "exports/Core50_raw_results_20260909"
    issues = []
    counts = Counter()
    for condition in CONDITIONS:
        for record in read_jsonl(package / f"run_level/{condition}/final_results.jsonl.gz"):
            key = identity(record)
            payload = json.loads(record["result"]["raw_text"])
            artifact = payload.get("canonical_artifact") or {}
            original = str(payload.get("equation") or "")
            old_expression = str(artifact.get("normalized_expression") or "")
            old_slots = {int(i) for i in re.findall(r"\bx(\d+)\b", old_expression)}
            expected_slots: set[int] | None = None
            reason = None
            if key[0] in {"qlattice", "imcts"}:
                expected_slots = {int(i) for i in re.findall(r"\bx(\d+)\b", original)}
                if key[0] == "qlattice" and expected_slots and 0 not in expected_slots and old_slots == {i - 1 for i in expected_slots}:
                    reason = "confirmed_qlattice_zero_based_slot_shift"
                elif key[0] == "imcts" and expected_slots != old_slots:
                    reason = "candidate_imcts_variable_binding_difference"
            elif key[0] in {"drsr", "llmsr"}:
                try:
                    function = next(n for n in ast.walk(ast.parse(original)) if isinstance(n, ast.FunctionDef))
                    arguments = [a.arg for a in function.args.args if a.arg != "params"]
                    used = {n.id for node in function.body for n in ast.walk(node) if isinstance(n, ast.Name)}
                    expected_slots = {i for i, name in enumerate(arguments) if name in used}
                    if expected_slots != old_slots:
                        reason = f"candidate_{key[0]}_variable_binding_difference"
                except (SyntaxError, StopIteration):
                    reason = f"unresolved_{key[0]}_function_ast"
            common = {"algorithm": key[0], "dataset_id": key[1], "seed": key[2], "condition": condition,
                      "task_id": record["source"].get("task_id"), "result_sha256": record["result"]["sha256"]}
            if reason:
                issues.append({**common, "reason": reason, "raw_slots": sorted(expected_slots or []),
                               "old_artifact_slots": sorted(old_slots), "feature_names": payload.get("feature_names"),
                               "original_equation": original, "old_canonical_expression": old_expression})
                counts[f"{reason}/{condition}"] += 1
            if key[1] == "feynman-bonus.20":
                issues.append({**common, "reason": "equivalence_bound_to_pre_audit_ground_truth"})
                counts[f"stale_ground_truth_equivalence/{condition}"] += 1
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "binding_issues.jsonl", issues)
    report = {"status": "requires_symbolic_rebinding", "issue_count": len(issues),
              "counts": dict(sorted(counts.items())), "new_training_required": False,
              "paid_api_invoked": False,
              "scope_note": "candidate 标记需逐表达式复核；confirmed 计数不等于全仓完整错误率。"}
    (output / "binding_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def merge_release_numeric(stage: Path, inputs: Path, symbolfit_csv: Path, output: Path) -> dict[str, Any]:
    """按原始结果及当前canonical双哈希合并数值，不用原生误差覆盖统一重放。"""
    package = stage / "exports/Core50_raw_results_20260909/run_level"
    semantics = unique_rows(read_jsonl(inputs / "semantic_runs.jsonl"), "最新表达式")
    replacements = unique_rows(read_csv(symbolfit_csv), "最新SymbolFit数值")
    if len(replacements) != 300 or any(k[0] != "symbolfit" or k[3] == "clean" for k in replacements):
        raise ReleaseError("SymbolFit数值替换必须是两个噪声条件的300条")
    reports = {}
    for condition in CONDITIONS:
        rows = unique_rows(read_csv(package / condition / "numeric_run_metrics.csv"), condition)
        replaced = 0
        for key, old in rows.items():
            if key in replacements:
                current = replacements[key]
                old.update({name: current[name] for name in (
                    "task_id", "host", "result_sha256", "evaluation_status", "valid_output",
                    "invalid_reason", "evaluation_path", "artifact_rebuilt", "native_id_nmse",
                    "native_ood_nmse", "id_nmse", "ood_nmse", "id_quality", "ood_quality",
                )})
                old["canonical_artifact_sha256"] = current["replay_canonical_artifact_sha256"]
                old["formula_source"] = "canonical_artifact"
                old["replay_error"] = ""
                for axis in ("id", "ood"):
                    old[f"{axis}_quality_delta_from_native"] = ""
                replaced += 1
            semantic = semantics[key]
            if old["result_sha256"] != semantic["source_result_sha256"]:
                raise ReleaseError(f"数值绑定非当前最终结果: {key}")
            if old["canonical_artifact_sha256"] != semantic["canonical_artifact_sha256"]:
                raise ReleaseError(f"数值与当前公式的canonical artifact不一致: {key}")
            old["condition"] = condition
            old["semantic_expression_sha256"] = semantic["effective_raw_semantic_expression_sha256"]
        if len(rows) != 2250:
            raise ReleaseError(f"{condition}数值网格不是2250条")
        path = output / f"{condition}_numeric.csv"
        write_csv(path, [rows[k] for k in sorted(rows)])
        reports[condition] = {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows),
                              "replacements": replaced, "invalid_outputs": sum(str(r["valid_output"]).lower() != "true" for r in rows.values())}
    report = {"status": "passed", "row_count": 6750, "replacement_count": 300, "conditions": reports,
              "source_semantics_sha256": sha256(inputs / "semantic_runs.jsonl"), "source_symbolfit_numeric_sha256": sha256(symbolfit_csv)}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def plan_bundles(plan_path: Path, index_path: Path) -> dict[Key, tuple[dict[str, Any], dict[str, Any]]]:
    indexes = {row["logical_id"]: row for row in read_jsonl(index_path)}
    output = {}
    for plan in read_jsonl(plan_path):
        request = plan["request"]
        seed = request.get("seed")
        if seed is None:
            match = re.search(r"_s(520|521|522)_", str(request.get("task_id", "")))
            if match is None:
                raise ReleaseError(f"计划缺少 seed: {plan['logical_id']}")
            seed = int(match.group(1))
        row = {
            "algorithm": request.get("algorithm_slug") or request.get("algorithm"),
            "dataset_id": request["dataset_id"], "seed": seed,
            "condition": request.get("noise_tag") or plan["condition"],
        }
        key = identity(row)
        if key in output:
            raise ReleaseError(f"计划存在重复身份: {key}")
        output[key] = (plan, indexes[plan["logical_id"]])
    return output


def compact_judgment(plan: Mapping[str, Any], index: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    path = Path(str(index.get("result_path") or ""))
    container = {}
    if str(path) != ".":
        if not path.is_absolute():
            path = repo / path
        if not path.is_file() or sha256(path) != index["result_sha256"]:
            raise ReleaseError(f"Opus5 冻结响应缺失或哈希漂移: {path}")
        container = json.loads(path.read_text(encoding="utf-8"))
    metadata = container.get("metadata") or {}
    request = plan.get("request") or {}
    return {
        "logical_id": index["logical_id"], "evaluation_key": index["evaluation_key"],
        "task_type": index["task_type"], "condition": index["condition"],
        "state": index["state"], "attempt_id": index.get("attempt_id"),
        "source_result_sha256": prediction_source_sha(request),
        "request_sha256": metadata.get("request_sha256"),
        "frozen_response_sha256": index.get("result_sha256"),
        "requested_model": metadata.get("requested_model"),
        "response_model": metadata.get("response_model") or (container.get("envelope") or {}).get("model"),
        "requested_effort": metadata.get("requested_effort"),
        "prompt_version": plan.get("prompt_version"), "schema_version": plan.get("schema_version"),
        "dependencies": plan.get("dependencies", []),
        "input_expression": request.get("expression"),
        "effective_expression": index.get("effective_expression"),
        "expression_resolution": index.get("expression_resolution"),
        "structured_output": index.get("structured_output"),
        "audit_overlay_action": index.get("overlay_action"),
        "audit_logical_id": index.get("source_audit_logical_id"),
        "audit_record_sha256": index.get("source_final_record_sha256"),
    }


def clean_components(stage: Path) -> tuple[dict[Key, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import minimality_score, symbolic_fidelity_score
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import build_symbolic_artifact, operator_f1, tree_similarity, variable_f1

    root = stage / "work/clean_final_replacement_v1"
    downstream = root / "downstream_refresh_v2"
    components = unique_rows(read_csv(stage / "results/clean_run_metrics.csv"), "clean components")
    tasks = {(r["algorithm"].casefold(), r["dataset_id"]): r for r in read_csv(stage / "results/task_stability.csv")}
    binding = json.loads((root / "binding_manifest.json").read_text())
    replacements = set(binding["replacement_contract"]["replacement_keys"])
    pred = plan_bundles(root / "clean_pred_hybrid_active_v7_identity.jsonl", downstream / "clean_pred_frozen_active_v7.jsonl")
    equivalence = plan_bundles(downstream / "clean_equivalence_hybrid_active_v3.jsonl", downstream / "clean_equivalence_frozen_active_v3.jsonl")
    evidence = {r["logical_key"]: r for r in read_jsonl(downstream / "clean_pred_vs_gt_evidence_active_v3.jsonl")}
    gt_views = read_jsonl(stage / "audits/formula_quality_1000_0903_v2/corrections_v2/views/gt_effective.jsonl")
    gt_overrides = {r["logical_id"].split("::")[1]: build_symbolic_artifact(r["effective_expression"]) for r in gt_views if r.get("overlay_action") == "replace_expression"}
    for key in replacements:
        row = next(r for r in components.values() if r["logical_key"] == key)
        _, p = pred[identity(row)]
        artifact = build_symbolic_artifact(p["effective_expression"])
        valid = p["state"] == "frozen"
        c_ref = int(row["reference_complexity"])
        c_pred = int(artifact["node_count"]) if valid else 0
        t = v = o = 0.0
        decision = "non_applicable"
        if valid:
            decision = str(equivalence[identity(row)][1]["structured_output"]["decision"])
            if row["dataset_id"] in gt_overrides:
                gt = gt_overrides[row["dataset_id"]]
                c_ref = int(gt["node_count"])
                t, v, o = tree_similarity(gt, artifact), variable_f1(gt, artifact), operator_f1(gt, artifact)
            else:
                e = evidence[key]
                t, v, o = e["tree"]["tree_similarity"], e["variable"]["f1"], e["operator"]["f1"]
        row.update({"equivalence_decision": decision, "equivalent": decision == "equivalent",
                    "tree_similarity": t, "variable_f1": v, "operator_f1": o,
                    "reference_complexity": c_ref, "predicted_complexity": c_pred,
                    "m_sym": symbolic_fidelity_score(equivalent=decision == "equivalent", tree_similarity=t, variable_f1=v, operator_f1=o, valid=valid),
                    "m_min": minimality_score(c_ref, max(c_pred, 1), valid=valid)})
    affected = {(identity(r)[0], identity(r)[1]) for r in components.values() if r["logical_key"] in replacements}
    structure_indexes = {r["logical_id"]: r for r in read_jsonl(downstream / "clean_structure_frozen_active_v2.jsonl")}
    for plan in read_jsonl(downstream / "clean_structure_hybrid_active_v2.jsonl"):
        request = plan["request"]
        task_key = str(request["algorithm"]).casefold(), str(request["dataset_id"])
        if task_key in affected:
            pair = int(request["seed_a"]), int(request["seed_b"])
            index = structure_indexes[plan["logical_id"]]
            if index["evaluation_key"] != plan["evaluation_key"]:
                raise ReleaseError("clean 结构裁决 evaluation_key 漂移")
            tasks[task_key][f"pair_{pair[0]}_{pair[1]}"] = index["structured_output"]["decision"]
    return components, tasks


def aggregate_rows(runs: list[dict[str, Any]], decisions: Mapping[tuple[str, str], Mapping[str, Any]], condition: str):
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import RunQuality, stability_score

    groups = defaultdict(list)
    for row in runs:
        groups[(row["algorithm"].casefold(), row["dataset_id"])].append(row)
    task_rows = []
    pairs = ((520, 521), (520, 522), (521, 522))
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: int(r["seed"]))
        if tuple(int(r["seed"]) for r in group) != SEEDS:
            raise ReleaseError(f"STAB seed 不完整: {key}")
        qualities = [RunQuality(float(r["id_quality"]), float(r["ood_quality"]), str(r["valid_output"]).lower() == "true") for r in group]
        labels = [decisions[key][f"pair_{a}_{b}"] for a, b in pairs]
        score = stability_score(qualities, structural_pair_results=[v in {"mathematically_equivalent", "same_canonical_structure"} for v in labels])
        task_rows.append({"algorithm": group[0]["algorithm"], "dataset_id": key[1], "condition": condition,
                          **{f"pair_{a}_{b}": v for (a, b), v in zip(pairs, labels)},
                          "numerical_consistency": score.numerical_consistency, "validity": score.validity,
                          "structural_consistency": score.structural_consistency, "m_stab": score.score})
    algorithm_rows = []
    for algorithm in sorted({r["algorithm"] for r in runs}):
        group = [r for r in runs if r["algorithm"] == algorithm]
        task_group = [r for r in task_rows if r["algorithm"] == algorithm]
        if len(group) != 150 or len(task_group) != 50:
            raise ReleaseError(f"算法汇总网格不完整: {algorithm}/{condition}")
        values = {axis: 100 * sum(float(r[col]) for r in group) / 150 for axis, col in (("ID", "id_quality"), ("OOD", "ood_quality"), ("SYM", "m_sym"), ("MIN", "m_min"), ("EFF", "m_eff"))}
        values["STAB"] = 100 * sum(r["m_stab"] for r in task_group) / 50
        algorithm_rows.append({"algorithm": algorithm, "condition": condition, "run_count": 150, "task_count": 50,
                               **values, "formal_ready": False})
    return algorithm_rows, task_rows


def prepare_base(stage: Path, output: Path) -> dict[str, Any]:
    """只生成独立工作底稿；最新增量验收前不称为最终发布。"""
    if output.exists():
        raise ReleaseError(f"输出目录已经存在，拒绝覆盖: {output}")
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import efficiency_from_qualities

    repo = stage.parents[1]
    package = stage / "exports/Core50_raw_results_20260909"
    data_package = stage / "exports/Core50_20260908_complete"
    clean_work = stage / "work/clean_final_replacement_v1"
    downstream = clean_work / "downstream_refresh_v2"
    noise_work = stage / "work/noise_full_six_axis_20260908"
    audit_root = stage / "audits/formula_quality_1000_0903_v2/corrections_v2"
    raw = [r for condition in CONDITIONS for r in read_jsonl(package / f"run_level/{condition}/final_results.jsonl.gz")]
    records, provenance = merge_final_records(raw, [], set())
    algorithms = {k[0] for k in records}
    datasets = {k[1] for k in records}
    validate_grid(records, algorithms, datasets)
    audited_pred = {r["evaluation_key"]: r for r in read_jsonl(audit_root / "views/pred_effective.jsonl") if r.get("overlay_action") == "replace_expression"}
    gt_rows = read_csv(data_package / "formulas/core50_ground_truth_formulas.csv")
    gt = {r["dataset_name"]: r for r in gt_rows}
    if set(gt) != datasets:
        raise ReleaseError("GT 与实验数据集集合不一致")
    print("校验原始网格通过，准备 clean 最终组件", flush=True)
    clean_run_components, clean_task_decisions = clean_components(stage)
    output.mkdir(parents=True)
    copy_checked(data_package / "formulas/core50_ground_truth_formulas.csv", output / "ground_truth/formulas.csv")
    copy_checked(data_package / "formulas/current_simplified_ground_truth.jsonl", output / "ground_truth/opus5_effective_references.jsonl")
    for dataset in read_csv(data_package / "manifests/core50_manifest.csv"):
        for field in ("metadata", "train", "valid", "id_test", "ood_test", "formula"):
            relative = dataset[f"{field}_path"]
            copy_checked(data_package / relative, output / relative)
    copy_checked(data_package / "manifests/core50_manifest.csv", output / "provenance/datasets.csv")
    unresolved = []
    for condition in CONDITIONS:
        print(f"组装 {condition} 最终来源与裁决", flush=True)
        condition_records = {k: v for k, v in records.items() if k[3] == condition}
        numeric = unique_rows(read_csv(package / f"run_level/{condition}/numeric_run_metrics.csv"), condition)
        trajectories = unique_rows(read_csv(package / f"run_level/trajectories/{condition}_run_trajectories_180min.csv.gz"), condition)
        if condition == "clean":
            pred = plan_bundles(clean_work / "clean_pred_hybrid_active_v7_identity.jsonl", downstream / "clean_pred_frozen_active_v7.jsonl")
            eq = plan_bundles(downstream / "clean_equivalence_hybrid_active_v3.jsonl", downstream / "clean_equivalence_frozen_active_v3.jsonl")
            components, decisions = clean_run_components, clean_task_decisions
        else:
            pred = plan_bundles(noise_work / f"plans/{condition}_pred_hybrid_active.jsonl", noise_work / f"indexes/{condition}_pred_hybrid_frozen_index_v2.jsonl")
            eq = plan_bundles(noise_work / f"plans/{condition}_equivalence_hybrid_active.jsonl", noise_work / f"indexes/{condition}_equivalence_hybrid_frozen_index_v2.jsonl")
            components = unique_rows(read_csv(package / f"run_level/{condition}/run_formulas.csv"), condition)
            decisions = {(r["algorithm"].casefold(), r["dataset_id"]): r for r in read_csv(package / f"run_level/{condition}/task_stability.csv")}
        if any(set(value) != set(condition_records) for value in (numeric, trajectories, pred, eq, components)):
            raise ReleaseError(f"{condition} 来源、公式、数值或轨迹身份不一致")
        runs, pred_evidence, eq_evidence = [], [], []
        for key in sorted(condition_records):
            record = condition_records[key]
            number, component = numeric[key], components[key]
            if number["result_sha256"] != record["result"]["sha256"]:
                raise ReleaseError(f"数值结果绑定旧来源: {key}")
            plan, index = pred[key]
            if condition == "clean" and index["evaluation_key"] in audited_pred:
                index = audited_pred[index["evaluation_key"]]
            prediction = bind_prediction(record, plan, index)
            eq_plan, eq_index = eq[key]
            if eq_plan["evaluation_key"] != eq_index["evaluation_key"]:
                raise ReleaseError(f"等价裁决与计划不一致: {key}")
            trajectory = trajectories[key]
            q = [float(trajectory[f"q_{minute:04d}"]) for minute in range(1, 181)]
            m_eff = efficiency_from_qualities(q)
            reference = gt[key[1]]
            used_reference = component.get("effective_ground_truth_expression", reference["simplified_reference_expression"])
            reference_current = used_reference == reference["simplified_reference_expression"]
            if not reference_current:
                unresolved.append({"algorithm": key[0], "dataset_id": key[1], "seed": key[2], "condition": condition,
                                   "reason": "symbolic_metrics_bound_to_pre_audit_ground_truth",
                                   "used_reference": used_reference, "current_reference": reference["simplified_reference_expression"]})
            payload = json.loads(record["result"]["raw_text"])
            row = {
                "algorithm": number["algorithm"], "dataset_id": key[1], "seed": key[2], "condition": condition,
                "task_id": number["task_id"], "result_sha256": record["result"]["sha256"],
                "status": payload.get("status"), "valid_output": number["valid_output"],
                "id_nmse": number["id_nmse"], "ood_nmse": number["ood_nmse"],
                "id_quality": number["id_quality"], "ood_quality": number["ood_quality"],
                "numeric_evaluation_path": number.get("evaluation_path"),
                **prediction,
                "ground_truth_expression": reference["original_expression"],
                "effective_ground_truth_expression": used_reference,
                "ground_truth_reference_current": reference_current,
                "equivalence_decision": component["equivalence_decision"],
                "m_sym": component["m_sym"], "m_min": component["m_min"], "m_eff": m_eff,
                "C_ref": component.get("C_ref", component.get("reference_complexity")),
                "C_pred": component.get("C_pred", component.get("predicted_complexity")),
                "trajectory_basis": trajectory.get("trajectory_basis", ""),
                "trajectory_source_tier": trajectory.get("source_tier", ""),
                "formal_ready": False,
            }
            runs.append(row)
            pred_evidence.append(compact_judgment(plan, index, repo))
            judgment = compact_judgment(eq_plan, eq_index, repo)
            judgment["effective_decision"] = component["equivalence_decision"]
            judgment["audit_overlay_applied"] = judgment["effective_decision"] != (eq_index.get("structured_output") or {}).get("decision")
            eq_evidence.append(judgment)
        algorithms_out, tasks_out = aggregate_rows(runs, decisions, condition)
        target = output / "results" / condition
        write_csv(target / "runs.csv", runs)
        write_csv(target / "six_axis_metrics.csv", algorithms_out)
        write_csv(target / "task_stability.csv", tasks_out)
        write_jsonl(target / "final_results.jsonl.gz", [condition_records[k] for k in sorted(condition_records)])
        write_jsonl(target / "opus5_simplification.jsonl", pred_evidence)
        write_jsonl(target / "opus5_equivalence.jsonl", eq_evidence)
        copy_checked(package / f"run_level/trajectories/{condition}_run_trajectories_180min.csv.gz", target / "run_trajectories_180min.csv.gz")
        copy_checked(package / f"aggregate_results/{condition}/eff_180min.csv", target / "eff_180min.csv")
        print(json.dumps({"condition": condition, "runs": len(runs), "algorithms": len(algorithms_out)}, ensure_ascii=False), flush=True)
    write_csv(output / "provenance/final_runs.csv", provenance)
    write_jsonl(output / "provenance/unresolved.jsonl", unresolved)
    manifest = {"status": "base_prepared_not_published", "run_count": len(records), "dataset_count": 50,
                "algorithm_count": 15, "conditions": list(CONDITIONS),
                "pending": ["验收并合入最新 SymbolFit noise300 的获准 replacement scope", "复核数值重放表达式与Opus5输入语义一致性"],
                "stale_ground_truth_bindings": len(unresolved),
                "limitations": ["clean EFF 含33条显式legacy fallback", "历史noise为observed numeric best-so-far", "逐分钟SYM/MIN/STAB尚未补齐"]}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_checksums(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-raw", type=Path)
    parser.add_argument("--prepare-base", action="store_true")
    parser.add_argument("--audit-bindings", action="store_true")
    parser.add_argument("--verify-release-data", action="store_true")
    parser.add_argument("--merge-release-numeric", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--symbolfit-numeric", type=Path)
    parser.add_argument("--stage-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_raw:
        records = read_jsonl(args.verify_raw)
        for record in records:
            check_raw_record(record)
        print(json.dumps({"rows": len(records), "conditions": dict(Counter(identity(r)[3] for r in records))}))
    elif args.merge_release_numeric and args.output and args.inputs and args.symbolfit_numeric:
        print(json.dumps(merge_release_numeric(args.stage_root.resolve(), args.inputs.resolve(), args.symbolfit_numeric.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
    elif args.verify_release_data and args.output:
        print(json.dumps(verify_release_data(args.output.resolve()), ensure_ascii=False, indent=2))
    elif args.audit_bindings and args.output:
        print(json.dumps(audit_source_bindings(args.stage_root.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
    elif args.prepare_base and args.output:
        print(json.dumps(prepare_base(args.stage_root.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
    else:
        parser.error("指定 --verify-raw 或 --prepare-base --output 工作目录")


if __name__ == "__main__":
    main()
