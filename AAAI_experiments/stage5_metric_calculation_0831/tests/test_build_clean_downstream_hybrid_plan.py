from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_clean_downstream_hybrid_plan import (
    CleanDownstreamHybridPlanError,
    build_clean_downstream_hybrid_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
    TaskSupersession,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
VERSION_SUFFIX = re.compile(r"::v[1-9]\d*$")
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_key(label: str) -> str:
    return _sha_text(label)


def _plan_row(
    logical_id: str,
    *,
    task_type: str,
    dependencies: tuple[str, ...],
    marker: str,
    condition: str = "clean",
) -> dict[str, object]:
    phase = "equivalence" if task_type == "equivalence" else "structure"
    prompt_path = STAGE_ROOT / f"config/prompts/{phase}.v1.txt"
    schema_path = STAGE_ROOT / f"config/schemas/{phase}.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha = _sha_file(prompt_path)
    schema_sha = _sha_file(schema_path)
    request = {
        "marker": marker,
        "evidence_hash": _sha_text(f"{logical_id}:{marker}:{dependencies}"),
    }
    normalized = {
        "request": request,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    input_hash = _sha_text(canonical_json(normalized))
    priority = 30 if task_type == "equivalence" else 50
    key = evaluation_key(
        task_type=task_type,
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
        task_type=task_type,
        condition=condition,
        priority=priority,
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
        "priority": priority,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_template": prompt_template,
        "prompt_version": prompt_path.stem,
        "rendered_prompt": render_prompt(prompt_template, request, schema),
        "request": request,
        "schema_content": schema,
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha,
        "schema_version": schema_path.stem,
        "task_kind": phase,
        "task_spec": json.loads(spec.canonical_json()),
        "task_type": task_type,
    }


def _write_plan(path: Path, rows: list[dict[str, object]]) -> list[str]:
    lines = [canonical_json(row) + "\n" for row in rows]
    path.write_text("".join(lines), encoding="utf-8")
    return lines


def _build_fixture(
    tmp_path: Path,
    *,
    formal_size: bool,
    condition: str = "clean",
    downstream_predecessor_version: int | None = None,
) -> dict[str, object]:
    algorithms = [f"alg{index:02d}" for index in range(15 if formal_size else 1)]
    datasets = range(1, 51 if formal_size else 3)
    groups = [(algorithm, f"g{dataset:04d}") for algorithm in algorithms for dataset in datasets]
    changed_by_group: dict[tuple[str, str], set[int]] = {}
    if formal_size:
        for index, group in enumerate(groups[:30]):
            changed_by_group[group] = (
                {520, 521, 522}
                if index < 18
                else ({520, 521} if index < 27 else {520})
            )
    else:
        changed_by_group[groups[0]] = {520, 521}

    pred_mappings: list[dict[str, str]] = []
    mapping_by_old: dict[str, str] = {}
    required_pred: list[str] = []
    required_eq: list[str] = []
    required_structure: list[str] = []
    old_eq: list[dict[str, object]] = []
    fresh_eq: list[dict[str, object]] = []
    old_structure: list[dict[str, object]] = []
    fresh_structure: list[dict[str, object]] = []

    for algorithm, dataset in groups:
        changed_seeds = changed_by_group.get((algorithm, dataset), set())
        old_pred_by_seed = {
            seed: _dependency_key(f"old-pred:{algorithm}:{dataset}:{seed}")
            for seed in (520, 521, 522)
        }
        new_pred_by_seed = dict(old_pred_by_seed)
        for seed in sorted(changed_seeds):
            base = f"pred_simplify::{algorithm}::{dataset}::s{seed}::{condition}"
            new_key = _dependency_key(f"new-pred:{algorithm}:{dataset}:{seed}")
            new_pred_by_seed[seed] = new_key
            required_pred.append(base)
            mapping_by_old[old_pred_by_seed[seed]] = new_key
            pred_mappings.append(
                {
                    "base_logical_id": base,
                    "predecessor_evaluation_key": old_pred_by_seed[seed],
                    "predecessor_logical_id": base,
                    "successor_evaluation_key": new_key,
                    "successor_logical_id": f"{base}::v2",
                    "identity": f"clean_pred_final_replacement::{base}",
                    "reason": "replacement",
                }
            )

        gt_key = _dependency_key(f"gt:{dataset}")
        for seed in (520, 521, 522):
            logical_id = f"equivalence::{algorithm}::{dataset}::s{seed}::{condition}"
            predecessor_logical_id = (
                f"{logical_id}::v{downstream_predecessor_version}"
                if downstream_predecessor_version is not None
                else logical_id
            )
            old_deps = (gt_key, old_pred_by_seed[seed])
            fresh_deps = (gt_key, new_pred_by_seed[seed])
            old_eq.append(
                _plan_row(
                    predecessor_logical_id,
                    task_type="equivalence",
                    dependencies=old_deps,
                    marker="old",
                    condition=condition,
                )
            )
            fresh_eq.append(
                _plan_row(
                    logical_id,
                    task_type="equivalence",
                    dependencies=fresh_deps,
                    marker="fresh",
                    condition=condition,
                )
            )
            if seed in changed_seeds:
                required_eq.append(logical_id)

        for seed_a, seed_b in SEED_PAIRS:
            condition_suffix = "" if condition == "clean" else f"::{condition}"
            logical_id = (
                f"stab_structure::{algorithm}::{dataset}::s{seed_a}-s{seed_b}"
                f"{condition_suffix}"
            )
            predecessor_logical_id = (
                f"{logical_id}::v{downstream_predecessor_version}"
                if downstream_predecessor_version is not None
                else logical_id
            )
            old_deps = (old_pred_by_seed[seed_a], old_pred_by_seed[seed_b])
            fresh_deps = (new_pred_by_seed[seed_a], new_pred_by_seed[seed_b])
            old_structure.append(
                _plan_row(
                    predecessor_logical_id,
                    task_type="stab_structure",
                    dependencies=old_deps,
                    marker="old",
                    condition=condition,
                )
            )
            fresh_structure.append(
                _plan_row(
                    logical_id,
                    task_type="stab_structure",
                    dependencies=fresh_deps,
                    marker="fresh",
                    condition=condition,
                )
            )
            if changed_seeds.intersection({seed_a, seed_b}):
                required_structure.append(logical_id)

    old_eq_path = tmp_path / "old_eq.jsonl"
    fresh_eq_path = tmp_path / "fresh_eq.jsonl"
    old_structure_path = tmp_path / "old_structure.jsonl"
    fresh_structure_path = tmp_path / "fresh_structure.jsonl"
    old_eq_lines = _write_plan(old_eq_path, old_eq)
    _write_plan(fresh_eq_path, fresh_eq)
    old_structure_lines = _write_plan(old_structure_path, old_structure)
    _write_plan(fresh_structure_path, fresh_structure)

    binding_path = tmp_path / "binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "aggregation_readiness": {
                    "required_pred_simplify_logical_ids": required_pred,
                    "required_equivalence_logical_ids": required_eq,
                    "required_structure_logical_ids": required_structure,
                    "required_structure_count": len(required_structure),
                }
            }
        ),
        encoding="utf-8",
    )
    pred_manifest_path = tmp_path / "pred_supersessions.json"
    pred_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "clean_pred_hybrid_active_plan.v1",
                "status": "ok",
                "hybrid_plan_sha256": "a" * 64,
                "counts": {"supersessions": len(pred_mappings)},
                "supersessions": pred_mappings,
            }
        ),
        encoding="utf-8",
    )
    return {
        "old_eq": old_eq,
        "fresh_eq": fresh_eq,
        "old_structure": old_structure,
        "fresh_structure": fresh_structure,
        "old_eq_path": old_eq_path,
        "fresh_eq_path": fresh_eq_path,
        "old_structure_path": old_structure_path,
        "fresh_structure_path": fresh_structure_path,
        "old_eq_lines": old_eq_lines,
        "old_structure_lines": old_structure_lines,
        "binding_path": binding_path,
        "pred_manifest_path": pred_manifest_path,
        "pred_mappings": pred_mappings,
        "mapping_by_old": mapping_by_old,
        "required_eq": set(required_eq),
        "required_structure": set(required_structure),
        "total": len(old_eq),
    }


def _build(
    fixture: dict[str, object],
    tmp_path: Path,
    *,
    apply_state_db: Path | None = None,
    pred_retirement_manifest_jsonl: Path | None = None,
    expected_eq: int | None = None,
    expected_structure: int | None = None,
    expected_single: int | None = None,
    expected_double: int | None = None,
) -> dict[str, object]:
    return build_clean_downstream_hybrid_plan(
        predecessor_equivalence_plan_jsonl=fixture["old_eq_path"],
        fresh_equivalence_plan_jsonl=fixture["fresh_eq_path"],
        predecessor_structure_plan_jsonl=fixture["old_structure_path"],
        fresh_structure_plan_jsonl=fixture["fresh_structure_path"],
        binding_manifest_json=fixture["binding_path"],
        pred_supersession_manifest_json=fixture["pred_manifest_path"],
        pred_retirement_manifest_jsonl=pred_retirement_manifest_jsonl,
        output_equivalence_plan_jsonl=tmp_path / "hybrid_eq.jsonl",
        output_structure_plan_jsonl=tmp_path / "hybrid_structure.jsonl",
        supersession_manifest_json=tmp_path / "downstream_supersessions.json",
        report_json=tmp_path / "report.json",
        expected_total_count=fixture["total"],
        expected_equivalence_count=expected_eq,
        expected_structure_count=expected_structure,
        expected_structure_single_dependency_count=expected_single,
        expected_structure_double_dependency_count=expected_double,
        apply_state_db=apply_state_db,
    )


def test_builds_exact_formal_hybrids_and_preserves_non_targets(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, formal_size=True)
    report = _build(
        fixture,
        tmp_path,
        expected_eq=75,
        expected_structure=87,
        expected_single=24,
        expected_double=63,
    )

    assert report["counts"] == {
        "equivalence_total": 2250,
        "equivalence_changed": 75,
        "equivalence_preserved": 2175,
        "structure_total": 2250,
        "structure_changed": 87,
        "structure_preserved": 2163,
        "structure_single_dependency_changed": 24,
        "structure_double_dependency_changed": 63,
        "dependency_positions_changed": 225,
        "supersessions": 162,
    }
    assert report["state_registration"] == {"requested": False, "mutated": False}
    output_eq_lines = (tmp_path / "hybrid_eq.jsonl").read_text().splitlines(keepends=True)
    output_structure_lines = (tmp_path / "hybrid_structure.jsonl").read_text().splitlines(
        keepends=True
    )
    mapping = fixture["mapping_by_old"]
    for old_row, old_line, output_line in zip(
        fixture["old_eq"], fixture["old_eq_lines"], output_eq_lines, strict=True
    ):
        base = old_row["logical_id"]
        output = json.loads(output_line)
        if base not in fixture["required_eq"]:
            assert output_line == old_line
            continue
        assert output["logical_id"] == f"{base}::v2"
        assert output["dependencies"][0] == old_row["dependencies"][0]
        assert output["dependencies"][1] == mapping[old_row["dependencies"][1]]

    changed_profiles = {1: 0, 2: 0}
    for old_row, old_line, output_line in zip(
        fixture["old_structure"],
        fixture["old_structure_lines"],
        output_structure_lines,
        strict=True,
    ):
        base = old_row["logical_id"]
        output = json.loads(output_line)
        if base not in fixture["required_structure"]:
            assert output_line == old_line
            continue
        assert output["logical_id"] == f"{base}::v2"
        changed = 0
        for old_dependency, new_dependency in zip(
            old_row["dependencies"], output["dependencies"], strict=True
        ):
            if old_dependency != new_dependency:
                changed += 1
                assert mapping[old_dependency] == new_dependency
        changed_profiles[changed] += 1
    assert changed_profiles == {1: 24, 2: 63}
    assert len(load_plan_jsonl(tmp_path / "hybrid_eq.jsonl").entries) == 2250
    assert len(load_plan_jsonl(tmp_path / "hybrid_structure.jsonl").entries) == 2250


@pytest.mark.parametrize("condition", ["noise001", "noise005"])
def test_builds_noise_condition_hybrids(tmp_path: Path, condition: str) -> None:
    fixture = _build_fixture(tmp_path, formal_size=False, condition=condition)

    report = _build(
        fixture,
        tmp_path,
        expected_eq=2,
        expected_structure=3,
        expected_single=2,
        expected_double=1,
    )

    assert report["condition"] == condition
    for path in (tmp_path / "hybrid_eq.jsonl", tmp_path / "hybrid_structure.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert {row["condition"] for row in rows} == {condition}
        assert all(f"::{condition}" in _base for _base in map(lambda row: VERSION_SUFFIX.sub("", row["logical_id"]), rows))


def test_increments_versioned_downstream_predecessors(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path,
        formal_size=False,
        downstream_predecessor_version=2,
    )

    _build(
        fixture,
        tmp_path,
        expected_eq=2,
        expected_structure=3,
        expected_single=2,
        expected_double=1,
    )

    changed_bases = fixture["required_eq"] | fixture["required_structure"]
    output_rows = [
        json.loads(line)
        for path in (tmp_path / "hybrid_eq.jsonl", tmp_path / "hybrid_structure.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    changed = [
        row for row in output_rows if VERSION_SUFFIX.sub("", row["logical_id"]) in changed_bases
    ]
    assert len(changed) == 5
    assert all(row["logical_id"].endswith("::v3") for row in changed)


@pytest.mark.parametrize("phase", ["equivalence", "structure"])
def test_rejects_wrong_or_reordered_dependency_binding(tmp_path: Path, phase: str) -> None:
    fixture = _build_fixture(tmp_path, formal_size=False)
    rows = fixture["fresh_eq"] if phase == "equivalence" else fixture["fresh_structure"]
    required = fixture["required_eq"] if phase == "equivalence" else fixture["required_structure"]
    target = next(row for row in rows if row["logical_id"] in required)
    if phase == "equivalence":
        target["dependencies"][1] = fixture["old_eq"][0]["dependencies"][1]
        target["task_spec"]["dependencies"][1] = target["dependencies"][1]
        _write_plan(fixture["fresh_eq_path"], fixture["fresh_eq"])
    else:
        target["dependencies"].reverse()
        target["task_spec"]["dependencies"] = list(target["dependencies"])
        _write_plan(fixture["fresh_structure_path"], fixture["fresh_structure"])
    with pytest.raises(CleanDownstreamHybridPlanError, match="dependency"):
        _build(fixture, tmp_path)
    assert not (tmp_path / "hybrid_eq.jsonl").exists()


def test_composes_pred_retirement_into_active_dependency_binding(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, formal_size=False)
    retired = fixture["pred_mappings"][0]
    old_successor = retired["successor_evaluation_key"]
    active_successor = _dependency_key("identity-recovery-v3")
    base = retired["base_logical_id"]
    for collection, path_key in (
        (fixture["fresh_eq"], "fresh_eq_path"),
        (fixture["fresh_structure"], "fresh_structure_path"),
    ):
        for row in collection:
            dependencies = [
                active_successor if item == old_successor else item
                for item in row["dependencies"]
            ]
            row["dependencies"] = dependencies
            row["task_spec"]["dependencies"] = dependencies
        _write_plan(fixture[path_key], collection)
    retirement_manifest = tmp_path / "pred_retirement.jsonl"
    retirement_manifest.write_text(
        canonical_json(
            {
                "schema_version": "stale_exhausted_retirement.v1",
                "revision_base": base,
                "exhausted_evaluation_key": old_successor,
                "replacement_evaluation_key": active_successor,
                "replacement_logical_id": f"{base}::v3",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = _build(
        fixture,
        tmp_path,
        pred_retirement_manifest_jsonl=retirement_manifest,
    )

    assert report["inputs"]["pred_retirement_manifest"]["row_count"] == 1
    fresh = load_plan_jsonl(tmp_path / "hybrid_eq.jsonl")
    changed = next(
        entry
        for entry in fresh.entries
        if entry.logical_id.startswith(f"equivalence::{base.split('::')[1]}::{base.split('::')[2]}::s520")
    )
    assert active_successor in changed.definition.task_spec.dependencies


def _freeze(store: TaskStateStore, spec: TaskSpec, *, now: float) -> None:
    store.register_task(spec, now=now)
    lease = store.reserve_attempt(spec.evaluation_key, now=now + 1)
    assert lease is not None
    store.freeze_result(
        lease.attempt_id,
        result_path=f"frozen/{spec.evaluation_key}.json",
        result_sha256=_sha_text(spec.evaluation_key),
        now=now + 2,
    )


def _task_spec(row: dict[str, object]) -> TaskSpec:
    spec = row["task_spec"]
    assert isinstance(spec, dict)
    return TaskSpec(
        evaluation_key=str(spec["evaluation_key"]),
        logical_id=str(spec["logical_id"]),
        task_type=str(spec["task_type"]),
        condition=str(spec["condition"]),
        priority=int(spec["priority"]),
        input_hash=str(spec["input_hash"]),
        prompt_version=str(spec["prompt_version"]),
        schema_version=str(spec["schema_version"]),
        dependencies=tuple(str(value) for value in spec["dependencies"]),
    )


def _prepare_state(
    fixture: dict[str, object],
    state_db: Path,
    *,
    leave_last_replacement_pending: bool = False,
) -> None:
    store = TaskStateStore(state_db, attempt_cap=200, logical_task_cap=200)
    mappings = fixture["pred_mappings"]
    old_pred_specs: list[TaskSpec] = []
    new_pred_specs: list[TaskSpec] = []
    for index, mapping in enumerate(mappings):
        base = str(mapping["base_logical_id"])
        old = TaskSpec(
            evaluation_key=str(mapping["predecessor_evaluation_key"]),
            logical_id=base,
            task_type="pred_simplify",
            condition="clean",
            priority=20,
            input_hash=_sha_text(f"old:{base}"),
            prompt_version="simplify.v1",
            schema_version="simplify.v1",
            dependencies=(),
        )
        new = TaskSpec(
            evaluation_key=str(mapping["successor_evaluation_key"]),
            logical_id=f"{base}::v2",
            task_type="pred_simplify",
            condition="clean",
            priority=20,
            input_hash=_sha_text(f"new:{base}"),
            prompt_version="simplify.v1",
            schema_version="simplify.v1",
            dependencies=(),
        )
        _freeze(store, old, now=float(10 + index * 10))
        old_pred_specs.append(old)
        new_pred_specs.append(new)

    downstream_rows = [
        row
        for row in [*fixture["old_eq"], *fixture["old_structure"]]
        if row["logical_id"] in fixture["required_eq"]
        or row["logical_id"] in fixture["required_structure"]
    ]
    registered_upstream = {spec.evaluation_key for spec in old_pred_specs}
    remaining_dependencies = sorted(
        {
            str(dependency)
            for row in downstream_rows
            for dependency in row["dependencies"]
        }
        - registered_upstream
    )
    for index, dependency in enumerate(remaining_dependencies):
        supporting = TaskSpec(
            evaluation_key=dependency,
            logical_id=f"supporting::{index}",
            task_type="supporting",
            condition="clean",
            priority=10,
            input_hash=_sha_text(f"supporting:{dependency}"),
            prompt_version="supporting.v1",
            schema_version="supporting.v1",
            dependencies=(),
        )
        _freeze(store, supporting, now=float(500 + index * 10))
    for index, row in enumerate(downstream_rows):
        _freeze(store, _task_spec(row), now=float(1000 + index * 10))

    store.register_supersession_batch(
        tuple(
            TaskSupersession(
                predecessor_evaluation_key=old.evaluation_key,
                successor=new,
                identity=f"pred::{old.logical_id}",
                reason="pred replacement",
                predecessor_plan_sha256="b" * 64,
                successor_plan_sha256="a" * 64,
            )
            for old, new in zip(old_pred_specs, new_pred_specs, strict=True)
        )
    )
    for index, new in enumerate(new_pred_specs):
        if leave_last_replacement_pending and index == len(new_pred_specs) - 1:
            continue
        lease = store.reserve_attempt(new.evaluation_key, now=float(2000 + index * 10))
        assert lease is not None
        store.freeze_result(
            lease.attempt_id,
            result_path=f"frozen/{new.evaluation_key}.json",
            result_sha256=_sha_text(new.evaluation_key),
            now=float(2002 + index * 10),
        )


def test_explicit_apply_is_idempotent(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, formal_size=False)
    state_db = tmp_path / "state-copy.sqlite3"
    _prepare_state(fixture, state_db)

    first = _build(fixture, tmp_path, apply_state_db=state_db)
    second = _build(fixture, tmp_path, apply_state_db=state_db)

    assert first["state_registration"]["mutated"] is True
    assert second["state_registration"]["mutated"] is False
    with sqlite3.connect(state_db) as connection:
        downstream_mappings = connection.execute(
            "SELECT COUNT(*) FROM task_supersessions WHERE identity LIKE 'clean_downstream_final_replacement::%'"
        ).fetchone()[0]
    assert downstream_mappings == len(fixture["required_eq"]) + len(
        fixture["required_structure"]
    )


def test_apply_rolls_back_when_one_replacement_dependency_is_not_frozen(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, formal_size=False)
    state_db = tmp_path / "state-copy.sqlite3"
    _prepare_state(fixture, state_db, leave_last_replacement_pending=True)
    with sqlite3.connect(state_db) as connection:
        before = list(connection.execute("SELECT evaluation_key, state FROM tasks ORDER BY 1"))

    with pytest.raises(CleanDownstreamHybridPlanError, match="frozen/non_applicable"):
        _build(fixture, tmp_path, apply_state_db=state_db)

    with sqlite3.connect(state_db) as connection:
        after = list(connection.execute("SELECT evaluation_key, state FROM tasks ORDER BY 1"))
        downstream_mappings = connection.execute(
            "SELECT COUNT(*) FROM task_supersessions WHERE identity LIKE 'clean_downstream_final_replacement::%'"
        ).fetchone()[0]
    assert after == before
    assert downstream_mappings == 0


def test_rejects_known_wrong_production_state_path(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, formal_size=False)
    wrong_path = STAGE_ROOT / "state_v2.sqlite3"
    with pytest.raises(CleanDownstreamHybridPlanError, match="错误的生产 state DB 路径"):
        _build(fixture, tmp_path, apply_state_db=wrong_path)
