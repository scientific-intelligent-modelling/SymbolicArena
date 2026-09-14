from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check import run_gpt_full_opus_simplification_review as full_review  # noqa: E402


def _record(
    logical_id: str,
    original: str,
    effective: str,
    *,
    outcome: str,
) -> dict[str, Any]:
    return {
        "logical_id": logical_id,
        "task_type": "prediction_simplification",
        "evaluation_key": f"eval::{logical_id}",
        "input_expression": original,
        "effective_expression": effective,
        "structured_output": {
            "outcome": outcome,
            "simplified_expression": effective,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _release_fixture(tmp_path: Path, *, prediction_count: int = 3) -> Path:
    root = tmp_path / "release"
    _write_jsonl(
        root / "ground_truth/opus5_ground_truth.jsonl",
        [_record("gt_simplify::g0001::v2", "x + 0", "x", outcome="simplified")],
    )
    for condition in full_review.CONDITIONS:
        rows = [
            _record(
                f"pred_simplify::algo_{index % 2}::g{index:04d}::s520::{condition}",
                f"x + {index} - {index}",
                "x",
                outcome=("unable" if index == prediction_count - 1 else "simplified"),
            )
            for index in range(prediction_count)
        ]
        _write_jsonl(root / f"results/{condition}/opus5_prediction.jsonl", rows)
        run_final_path = root / f"results/{condition}/run_final.csv"
        with run_final_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pred_evaluation_key", "algorithm", "condition", "dataset_id", "seed",
                    "raw_original_equation", "raw_original_equation_sha256",
                    "canonical_named_expression", "canonical_named_expression_sha256",
                ],
            )
            writer.writeheader()
            for row in rows:
                parts = row["logical_id"].split("::")
                writer.writerow(
                    {
                        "pred_evaluation_key": row["evaluation_key"],
                        "algorithm": parts[1],
                        "condition": condition,
                        "dataset_id": parts[2],
                        "seed": parts[3].removeprefix("s"),
                    }
                )
        with gzip.open(
            root / f"results/{condition}/raw_results.jsonl.gz", "wt", encoding="utf-8"
        ) as handle:
            handle.write("")
    return root


def _completed_record(task: dict[str, Any], verdict: str = "pass") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prompt_version": full_review.PROMPT_VERSION,
        "review_policy": full_review.REVIEW_POLICY,
        "logical_task_id": f"{full_review.REVIEW_POLICY}::gpt-5.6-sol::{task['sample_id']}",
        "sample_id": task["sample_id"],
        "review_key": task["review_key"],
        "pair_sha256": task["pair_sha256"],
        "model": "gpt-5.6-sol",
        "status": "completed",
        "attempts": [],
        "review": {
            "equivalence": verdict,
            "domain_preserved": True,
            "variable_mapping_preserved": verdict == "pass",
            "operator_semantics_preserved": True,
            "simplicity": "same",
            "verdict": verdict,
            "confidence": 0.9,
            "brief_reason": "fixture",
            "counterexample": None,
        },
    }


def test_full_manifest_keeps_all_rows_including_unable(tmp_path: Path) -> None:
    root = _release_fixture(tmp_path, prediction_count=3)

    tasks = full_review.build_full_manifest(
        root,
        expected_ground_truth=1,
        expected_per_condition=3,
    )

    assert len(tasks) == 10
    assert [task["sample_id"] for task in tasks] == [
        f"review_{index:06d}" for index in range(1, 11)
    ]
    assert sum(task["outcome"] == "unable" for task in tasks) == 3
    assert sum(task["review_applicable"] for task in tasks) == 7
    assert all(
        task["review_applicable"] is (task["outcome"] != "unable") for task in tasks
    )
    assert len({task["review_key"] for task in tasks}) == len(tasks)
    assert {task["occurrences"][0]["condition"] for task in tasks} == {
        "clean",
        "noise001",
        "noise005",
    }


def test_frozen_production_contract_and_one_model_request_cap() -> None:
    assert full_review.EXPECTED_GROUND_TRUTH == 50
    assert full_review.EXPECTED_PER_CONDITION == 2250
    assert full_review.EXPECTED_TOTAL == 6800
    assert full_review.EXPECTED_REVIEWABLE == 6700
    assert full_review.physical_request_cap(6700) == 10050
    assert full_review.CONCURRENCY == 32
    assert full_review.MAX_ATTEMPTS == 2


def test_progress_summary_groups_and_writes_atomic_hundred_checkpoint(
    tmp_path: Path,
) -> None:
    tasks = full_review.build_full_manifest(
        _release_fixture(tmp_path, prediction_count=3),
        expected_ground_truth=1,
        expected_per_condition=3,
    )
    output_root = tmp_path / "output"
    applicable = [task for task in tasks if task["review_applicable"]]
    for index, task in enumerate(applicable[:3]):
        path = full_review.task_path(output_root, task["sample_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_completed_record(task, "pass" if index < 2 else "fail")),
            encoding="utf-8",
        )

    summary = full_review.build_progress_summary(tasks, output_root, trigger=100)
    checkpoint = full_review.write_progress_checkpoint(
        output_root, summary, checkpoint_number=100
    )

    assert summary["overall"] == {
        "expected": 10,
        "applicable": 7,
        "non_applicable": 3,
        "reviewed": 3,
        "completed": 3,
        "failed": 0,
        "pending": 4,
        "pass": 2,
        "fail": 1,
        "undetermined": 0,
        "unavailable": 0,
        "terminal": 3,
        "attempted_all": False,
    }
    assert summary["groups"]["condition"]["clean"]["expected"] == 4
    assert summary["groups"]["algorithm"]["ground_truth"]["completed"] == 1
    assert summary["groups"]["outcome"]["simplified"]["fail"] == 1
    assert summary["groups"]["verdict"] == {
        "fail": 1,
        "pass": 2,
        "undetermined": 0,
        "unavailable": 0,
    }
    assert checkpoint.name == "checkpoint_000100.json"
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["trigger"] == 100
    assert json.loads((output_root / "checkpoint_latest.json").read_text())["overall"][
        "reviewed"
    ] == 3


def test_pending_tasks_excludes_completed_and_terminal_failed(tmp_path: Path) -> None:
    tasks = full_review.build_full_manifest(
        _release_fixture(tmp_path, prediction_count=2),
        expected_ground_truth=1,
        expected_per_condition=2,
    )
    output_root = tmp_path / "output"
    applicable = [task for task in tasks if task["review_applicable"]]
    completed = applicable[0]
    failed = applicable[1]
    for task, record in (
        (completed, _completed_record(completed)),
        (
            failed,
            {
                **_completed_record(failed),
                "status": "failed",
                "review": None,
                "attempts": [{"status": "failed"}, {"status": "failed"}],
            },
        ),
    ):
        path = full_review.task_path(output_root, task["sample_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    pending = full_review.pending_tasks(tasks, output_root)

    assert {task["sample_id"] for task in pending} == {
        task["sample_id"] for task in applicable[2:]
    }


def test_gpt_request_is_non_streaming_and_never_stored(tmp_path: Path) -> None:
    profile_path = tmp_path / "openai.toml"
    profile_path.write_text(
        "\n".join(
            [
                "[model_providers.custom]",
                f'base_url = "{full_review.base_review.OPENAI_BASE_URL}"',
                'wire_api = "responses"',
                'http_headers = {Authorization = "Bearer sk-test-only"}',
            ]
        ),
        encoding="utf-8",
    )
    profile = full_review.base_review.load_openai_profile(profile_path)
    request = full_review.base_review.build_http_request(
        full_review.MODEL,
        profile,
        "fixture",
        review_policy=full_review.REVIEW_POLICY,
    )
    payload = json.loads(request.body)

    assert request.url.endswith("/responses")
    assert payload["stream"] is False
    assert payload["store"] is False
    assert "sk-test-only" not in json.dumps(request.safe_metadata)


def test_completion_requires_no_pending_failed_or_unavailable() -> None:
    assert full_review.completion_status(
        {"pending": 0, "failed": 0, "unavailable": 0}
    ) == "completed"
    assert full_review.completion_status(
        {"pending": 0, "failed": 1, "unavailable": 1}
    ) == "partial"
    assert full_review.completion_status(
        {"pending": 1, "failed": 0, "unavailable": 0}
    ) == "partial"


def test_explicit_terminal_retry_appends_third_attempt_without_erasing_history(
    tmp_path: Path,
) -> None:
    tasks = full_review.build_full_manifest(
        _release_fixture(tmp_path, prediction_count=2),
        expected_ground_truth=1,
        expected_per_condition=2,
    )
    task = next(item for item in tasks if item["review_applicable"])
    output_root = tmp_path / "output"
    path = full_review.task_path(output_root, task["sample_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = {
        **_completed_record(task),
        "status": "failed",
        "review": None,
        "attempts": [
            {"attempt_number": 1, "status": "failed", "marker": "keep-one"},
            {"attempt_number": 2, "status": "failed", "marker": "keep-two"},
        ],
    }
    path.write_text(json.dumps(failed), encoding="utf-8")
    profile = full_review.base_review.ApiProfile(
        protocol="openai_responses",
        base_url=full_review.base_review.OPENAI_BASE_URL,
        headers={"Authorization": "Bearer sk-never-persist", "content-type": "application/json"},
        safe_metadata={"api_host": "routify-pub.alibaba-inc.com"},
    )

    def transport(
        request: full_review.base_review.HttpRequest, timeout: float
    ) -> full_review.base_review.HttpResponse:
        return full_review.base_review.HttpResponse(
            status=200,
            payload={
                "id": "resp-recovery",
                "model": "gpt-5.6-sol",
                "output_text": json.dumps(_completed_record(task)["review"]),
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            },
        )

    result = full_review.review_one_task(
        task=task,
        profile=profile,
        output_root=output_root,
        budget=full_review.base_review.RequestBudget(
            output_root / "request_budget.json", cap=10050
        ),
        transport=transport,
        allow_terminal_retry=True,
        retry_delay_seconds=0,
    )

    assert result["status"] == "completed"
    assert [attempt["attempt_number"] for attempt in result["attempts"]] == [1, 2, 3]
    assert result["attempts"][0]["marker"] == "keep-one"
    assert result["attempts"][1]["marker"] == "keep-two"
    assert result["attempts"][2]["status"] == "accepted"
    assert "sk-never-persist" not in path.read_text(encoding="utf-8")


def test_explicit_retry_requires_exactly_two_terminal_failures(tmp_path: Path) -> None:
    tasks = full_review.build_full_manifest(
        _release_fixture(tmp_path, prediction_count=2),
        expected_ground_truth=1,
        expected_per_condition=2,
    )
    task = next(item for item in tasks if item["review_applicable"])
    output_root = tmp_path / "output"
    profile = full_review.base_review.ApiProfile(
        protocol="openai_responses",
        base_url=full_review.base_review.OPENAI_BASE_URL,
        headers={"Authorization": "Bearer sk-test", "content-type": "application/json"},
        safe_metadata={},
    )

    with pytest.raises(full_review.FullReviewError, match="两次终态失败"):
        full_review.review_one_task(
            task=task,
            profile=profile,
            output_root=output_root,
            budget=full_review.base_review.RequestBudget(
                output_root / "request_budget.json", cap=10050
            ),
            transport=lambda request, timeout: pytest.fail("不应发请求"),
            allow_terminal_retry=True,
            retry_delay_seconds=0,
        )


def test_retry_cli_requires_one_explicit_task_identity() -> None:
    parser = full_review.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["retry-one"])
    by_sample = parser.parse_args(["retry-one", "--sample-id", "review_005745"])
    by_logical = parser.parse_args(
        ["retry-one", "--logical-id", "pred_simplify::gplearn::g0049::s521::noise001"]
    )

    assert by_sample.sample_id == "review_005745"
    assert by_sample.logical_id is None
    assert by_logical.logical_id.endswith("noise001")
    assert by_logical.sample_id is None
