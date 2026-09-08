from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_noise_equivalence_hybrid_plan import (
    NoiseEquivalenceHybridPlanError,
    build_noise_equivalence_hybrid_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    logical_id: str,
    *,
    dependencies: tuple[str, str],
    marker: str,
    condition: str,
) -> dict[str, object]:
    prompt_path = STAGE_ROOT / "config/prompts/equivalence.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/equivalence.v1.json"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    request = {
        "marker": marker,
        "evidence_hash": _sha(f"{logical_id}:{marker}:{dependencies}"),
    }
    normalized = {
        "request": request,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    input_hash = _sha(canonical_json(normalized))
    key = evaluation_key(
        task_type="equivalence",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        prompt_sha256=prompt_sha,
        schema_sha256=schema_sha,
        normalized_input=normalized,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type="equivalence",
        condition=condition,
        priority=30,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=dependencies,
    )
    return {
        "condition": condition,
        "dependencies": list(dependencies),
        "evaluation_key": key,
        "input_hash": input_hash,
        "logical_id": logical_id,
        "normalized_input": normalized,
        "priority": 30,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_template": prompt,
        "prompt_version": prompt_path.stem,
        "rendered_prompt": render_prompt(prompt, request, schema),
        "request": request,
        "schema_content": schema,
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha,
        "schema_version": schema_path.stem,
        "task_kind": "equivalence",
        "task_spec": json.loads(spec.canonical_json()),
        "task_type": "equivalence",
    }


def _write_plan(path: Path, rows: list[dict[str, object]]) -> list[str]:
    lines = [canonical_json(row) + "\n" for row in rows]
    path.write_text("".join(lines), encoding="utf-8")
    return lines


def _fixture(tmp_path: Path, *, condition: str = "noise001") -> dict[str, object]:
    old_rows: list[dict[str, object]] = []
    fresh_rows: list[dict[str, object]] = []
    bindings: list[dict[str, str]] = []
    changed = {("alg00", "g0001", 520), ("alg00", "g0002", 521)}
    for dataset in ("g0001", "g0002"):
        gt_key = _sha(f"gt:{dataset}")
        for seed in (520, 521, 522):
            pred_base = f"pred_simplify::alg00::{dataset}::s{seed}::{condition}"
            old_pred = _sha(f"old:{pred_base}")
            new_pred = _sha(f"new:{pred_base}") if ("alg00", dataset, seed) in changed else old_pred
            eq_base = f"equivalence::alg00::{dataset}::s{seed}::{condition}"
            old_logical_id = f"{eq_base}::v2" if dataset == "g0002" else eq_base
            old_rows.append(
                _row(
                    old_logical_id,
                    dependencies=(gt_key, old_pred),
                    marker="old",
                    condition=condition,
                )
            )
            fresh_rows.append(
                _row(
                    eq_base,
                    dependencies=(gt_key, new_pred),
                    marker="fresh",
                    condition=condition,
                )
            )
            if new_pred != old_pred:
                bindings.append(
                    {
                        "base_logical_id": pred_base,
                        "predecessor_evaluation_key": old_pred,
                        "predecessor_logical_id": pred_base,
                        "successor_evaluation_key": new_pred,
                        "successor_logical_id": f"{pred_base}::v2",
                    }
                )
    old_path = tmp_path / "old.jsonl"
    fresh_path = tmp_path / "fresh.jsonl"
    old_lines = _write_plan(old_path, old_rows)
    _write_plan(fresh_path, fresh_rows)
    manifest_path = tmp_path / "pred_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "noise_pred_hybrid_manifest.v1",
                "status": "ok",
                "conditions": {condition: {"bindings": bindings}},
            }
        ),
        encoding="utf-8",
    )
    nonapp_path = tmp_path / "nonapp.jsonl"
    nonapp_path.write_text("", encoding="utf-8")
    return {
        "old_path": old_path,
        "fresh_path": fresh_path,
        "manifest_path": manifest_path,
        "nonapp_path": nonapp_path,
        "old_rows": old_rows,
        "old_lines": old_lines,
        "bindings": bindings,
        "condition": condition,
    }


def _build(
    fixture: dict[str, object],
    tmp_path: Path,
    *,
    apply_state_db: Path | None = None,
) -> dict[str, object]:
    return build_noise_equivalence_hybrid_plan(
        condition=str(fixture["condition"]),
        predecessor_equivalence_plan_jsonl=fixture["old_path"],
        fresh_equivalence_plan_jsonl=fixture["fresh_path"],
        pred_hybrid_manifest_json=fixture["manifest_path"],
        fresh_non_applicable_jsonl=fixture["nonapp_path"],
        output_full_plan_jsonl=tmp_path / "hybrid_full.jsonl",
        output_active_plan_jsonl=tmp_path / "hybrid_active.jsonl",
        output_refresh_plan_jsonl=tmp_path / "refresh.jsonl",
        supersession_manifest_json=tmp_path / "supersessions.json",
        report_json=tmp_path / "report.json",
        expected_total_count=6,
        expected_refresh_count=2,
        apply_state_db=apply_state_db,
    )


def test_builds_hybrid_and_preserves_untouched_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = _build(fixture, tmp_path)

    assert report["counts"] == {
        "total": 6,
        "changed": 2,
        "preserved": 4,
        "refresh_callable": 2,
        "fresh_non_applicable": 0,
    }
    full_lines = (tmp_path / "hybrid_full.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    active_lines = (tmp_path / "hybrid_active.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    assert active_lines == full_lines
    refresh_rows = [json.loads(line) for line in (tmp_path / "refresh.jsonl").read_text().splitlines()]
    assert len(refresh_rows) == 2
    mapping = {
        row["predecessor_evaluation_key"]: row["successor_evaluation_key"]
        for row in fixture["bindings"]
    }
    for old, old_line, output_line in zip(
        fixture["old_rows"], fixture["old_lines"], full_lines, strict=True
    ):
        output = json.loads(output_line)
        if old["dependencies"][1] not in mapping:
            assert output_line == old_line
            continue
        assert output["dependencies"][0] == old["dependencies"][0]
        assert output["dependencies"][1] == mapping[old["dependencies"][1]]
        expected_suffix = "::v3" if old["logical_id"].endswith("::v2") else "::v2"
        assert output["logical_id"].endswith(expected_suffix)
    assert len(load_plan_jsonl(tmp_path / "hybrid_active.jsonl").entries) == 6
    assert len(load_plan_jsonl(tmp_path / "refresh.jsonl").entries) == 2


def test_rejects_changed_gt_dependency(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in Path(fixture["fresh_path"]).read_text().splitlines()]
    rows[0]["dependencies"][0] = _sha("wrong-gt")
    rows[0]["task_spec"]["dependencies"][0] = rows[0]["dependencies"][0]
    _write_plan(Path(fixture["fresh_path"]), rows)

    with pytest.raises(NoiseEquivalenceHybridPlanError, match="GT dependency"):
        _build(fixture, tmp_path)


def test_rejects_nonempty_fresh_non_applicable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    Path(fixture["nonapp_path"]).write_text('{"status":"planned_non_applicable"}\n')

    with pytest.raises(NoiseEquivalenceHybridPlanError, match="non-applicable"):
        _build(fixture, tmp_path)


def test_rejects_forbidden_fullcpu_provenance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(Path(fixture["manifest_path"]).read_text())
    manifest["source"] = "all_15alg_fullcpu_v1"
    Path(fixture["manifest_path"]).write_text(json.dumps(manifest))

    with pytest.raises(NoiseEquivalenceHybridPlanError, match="禁止来源"):
        _build(fixture, tmp_path)


def test_explicit_state_application_passes_successor_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
        build_noise_equivalence_hybrid_plan as module,
    )

    fixture = _fixture(tmp_path)
    state_db = tmp_path / "state.sqlite3"
    state_db.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "_validate_apply_state_db_path", lambda path: path)

    def fake_register(path, *, bindings, **kwargs):
        captured["path"] = path
        captured["bindings"] = bindings
        captured["kwargs"] = kwargs
        return {"requested": True, "mutated": True}

    monkeypatch.setattr(module, "_register_state", fake_register)

    report = _build(fixture, tmp_path, apply_state_db=state_db)

    assert report["state_registration"] == {"requested": True, "mutated": True}
    assert captured["path"] == state_db.resolve()
    bindings = captured["bindings"]
    assert isinstance(bindings, list)
    assert len(bindings) == 2
    assert all(binding["task_type"] == "equivalence" for binding in bindings)
    assert all(binding["successor_spec"].condition == "noise001" for binding in bindings)
