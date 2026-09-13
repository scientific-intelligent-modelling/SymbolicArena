import hashlib
import json

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.scripts.build_core50_final_release import (
    ReleaseError,
    aggregate_rows,
    bind_prediction,
    check_raw_record,
    identity,
    merge_final_records,
    validate_grid,
)


def raw(expression="x0", batch="original"):
    text = json.dumps({"tool": "symbolfit", "seed": 520, "equation": expression})
    return {
        "source": {"algorithm": "symbolfit", "dataset_id": "task", "seed": 520,
                   "noise_tag": "noise001", "batch": batch},
        "result": {"raw_text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()},
    }


def test_latest_replaces_old_without_retaining_old_payload():
    previous, latest = raw(), raw("x0**2", "latest")
    rows, manifest = merge_final_records([previous], [latest], {identity(latest)})
    assert len(rows) == 1
    assert rows[identity(latest)] == latest
    assert manifest[0]["previous_result_sha256"] == previous["result"]["sha256"]


def test_replacement_scope_is_exact():
    with pytest.raises(ReleaseError, match="replacement scope"):
        merge_final_records([raw()], [raw("x0**2")], set())


def test_duplicate_run_is_rejected():
    with pytest.raises(ReleaseError, match="重复"):
        merge_final_records([raw(), raw()], [], set())


def test_aborted_fullcpu_is_rejected():
    with pytest.raises(ReleaseError, match="fullcpu"):
        check_raw_record(raw(batch="all_15alg_fullcpu_v1_formal"))


def test_raw_sha_mismatch_is_rejected():
    record = raw()
    record["result"]["raw_text"] += " "
    with pytest.raises(ReleaseError, match="SHA256"):
        check_raw_record(record)


def test_old_opus_judgment_cannot_be_reused_after_result_replacement():
    previous, latest = raw(), raw("x0**2")
    plan = {"request": {"ast_source_evidence": {"result_raw_sha256": previous["result"]["sha256"]}}}
    with pytest.raises(ReleaseError, match="仍绑定旧结果"):
        bind_prediction(latest, plan, {})


def test_credentials_are_not_published():
    with pytest.raises(ReleaseError, match="凭据"):
        check_raw_record(raw(expression="sk-" + "x" * 24))


def test_complete_grid_cannot_hide_a_missing_run():
    algorithms = {f"a{i}" for i in range(15)}
    datasets = {f"d{i}" for i in range(50)}
    records = {(a, d, s, c): None for a in algorithms for d in datasets
               for s in (520, 521, 522) for c in ("clean", "noise001", "noise005")}
    validate_grid(records, algorithms, datasets)
    records.pop(next(iter(records)))
    with pytest.raises(ReleaseError, match="完整"):
        validate_grid(records, algorithms, datasets)


def test_final_scores_are_reconstructed_from_run_and_task_components():
    rows = [{"algorithm": "symbolfit", "dataset_id": f"d{d}", "seed": seed,
             "id_quality": 0.5, "ood_quality": 0.75, "valid_output": True,
             "m_sym": 0.25, "m_min": 0.5, "m_eff": 0.8}
            for d in range(50) for seed in (520, 521, 522)]
    decisions = {("symbolfit", f"d{d}"): {
        "pair_520_521": "mathematically_equivalent",
        "pair_520_522": "same_canonical_structure",
        "pair_521_522": "mathematically_equivalent"} for d in range(50)}
    algorithms, tasks = aggregate_rows(rows, decisions, "noise001")
    assert len(tasks) == 50
    assert {k: algorithms[0][k] for k in ("ID", "OOD", "SYM", "MIN", "EFF", "STAB")} == pytest.approx(
        {"ID": 50, "OOD": 75, "SYM": 25, "MIN": 50, "EFF": 80, "STAB": 100})
    assert algorithms[0]["formal_ready"] is False
