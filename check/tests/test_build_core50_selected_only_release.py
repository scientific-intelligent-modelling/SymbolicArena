"""Selected-only indices must not carry superseded run identifiers."""

from check.build_core50_selected_only_release import (
    current_response_reference,
    current_selection_row,
)


def test_selection_row_drops_old_run_fields() -> None:
    row = {"condition": "clean", "algorithm": "gplearn", "algorithm_slug": "gplearn",
           "dataset_id": "BPG3", "seed": "520", "task_id": "current",
           "logical_key": "gplearn::BPG3::s520::clean", "selection_status": "formal_raw_retained",
           "selected_result_sha256": "a" * 64, "old_formal_task_id": "old",
           "old_formal_result_sha256": "b" * 64, "superseded": "true"}
    terminal = {"terminal_expression_sha256": "c" * 64,
                "terminal_source_sha256": "d" * 64, "terminal_archive_member": "runs/current"}
    selected = current_selection_row(row, terminal)
    assert selected["task_id"] == "current"
    assert selected["terminal_source_sha256"] == "d" * 64
    assert "old_formal_task_id" not in selected
    assert "old_formal_result_sha256" not in selected
    assert "superseded" not in selected


def test_response_index_points_inside_new_package() -> None:
    row = {"kind": "prediction", "evaluation_key": "key", "response_sha256": "a" * 64,
           "packaged_path": "symbolic/response_blobs/aa/original.json"}
    current = current_response_reference(row)
    assert current["packaged_path"] == "symbolic_current/response_blobs/aa/" + "a" * 64 + ".json"
    assert current["source_package_path"] == row["packaged_path"]
