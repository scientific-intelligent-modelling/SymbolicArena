from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.scripts import (  # noqa: E402
    run_core50_release_routify as release,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.revise_exhausted_symbolic_plan import (  # noqa: E402
    _successor_logical_id,
)


def _write_profile(path: Path, *, base_url: str, token: str = "sk-fixture") -> None:
    path.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": base_url,
                    "ANTHROPIC_AUTH_TOKEN": token,
                    "ANTHROPIC_MODEL": "ignored-by-contract",
                },
                "effortLevel": "low",
            }
        ),
        encoding="utf-8",
    )


def test_attempt_cap_is_exactly_ceiling_one_and_a_half() -> None:
    assert release.physical_attempt_cap(1) == 2
    assert release.physical_attempt_cap(2) == 3
    assert release.physical_attempt_cap(135) == 203
    assert release.state_attempt_cap(135, cache_import_total=6) == 209


def test_load_release_channel_accepts_only_routify_and_never_persists_secret(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "routify.json"
    _write_profile(
        profile,
        base_url="https://routify-pub.alibaba-inc.com/protocol/anthropic",
        token="sk-must-not-leak",
    )

    channel, safe = release.load_release_channel(profile)

    assert channel.name == "routify"
    assert channel.auth_token == "sk-must-not-leak"
    assert safe == {
        "api_channel": "routify",
        "api_host": "routify-pub.alibaba-inc.com",
        "api_path": "/protocol/anthropic",
        "credential_loaded": True,
    }
    assert "sk-must-not-leak" not in json.dumps(safe)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://yapi.click",
        "https://routify-pub.alibaba-inc.com.evil.invalid/protocol/anthropic",
        "http://routify-pub.alibaba-inc.com/protocol/anthropic",
        "https://routify-pub.alibaba-inc.com/other",
    ],
)
def test_load_release_channel_rejects_every_noncanonical_endpoint(
    tmp_path: Path,
    base_url: str,
) -> None:
    profile = tmp_path / "invalid.json"
    _write_profile(profile, base_url=base_url)

    with pytest.raises(release.ReleaseControlError, match="Routify"):
        release.load_release_channel(profile)


def test_release_paths_are_shared_and_confined_to_llm_workspace(tmp_path: Path) -> None:
    paths = release.ReleasePaths.from_root(tmp_path / "release_v2" / "llm", "pred")

    assert paths.state_db == tmp_path / "release_v2" / "llm" / "release_state.sqlite3"
    assert paths.attempts_dir == tmp_path / "release_v2" / "llm" / "attempts"
    assert paths.frozen_dir == tmp_path / "release_v2" / "llm" / "frozen"
    assert paths.report_json == tmp_path / "release_v2" / "llm" / "reports" / "pred.json"
    with pytest.raises(release.ReleaseControlError, match="phase"):
        release.ReleasePaths.from_root(tmp_path / "release_v2" / "llm", "../escape")


def test_run_phase_uses_one_shared_budget_and_single_routify_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "routify.json"
    _write_profile(
        profile,
        base_url="https://routify-pub.alibaba-inc.com/protocol/anthropic",
    )
    plan = tmp_path / "release_v2" / "llm" / "plans" / "pred.jsonl"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        release,
        "load_plan_jsonl",
        lambda _: SimpleNamespace(
            entries=tuple(
                SimpleNamespace(
                    logical_id=f"task::{index}",
                    definition=SimpleNamespace(task_spec=SimpleNamespace(condition="clean")),
                )
                for index in range(10)
            )
        ),
    )
    captured: dict[str, Any] = {}

    def fake_execute_plan(**kwargs: Any) -> int:
        captured.update(kwargs)
        store = SimpleNamespace()
        runner = kwargs["runner_factory"](
            store,
            attempts_dir=kwargs["attempts_dir"],
            frozen_dir=kwargs["frozen_dir"],
        )
        assert len(runner.channels) == 1
        assert runner.channels[0].name == "routify"
        assert runner._api_payload("fixture") == {
            "model": "claude-opus-5",
            "stream": False,
            "max_tokens": 4096,
            "system": release.STRICT_EVALUATOR_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": "fixture"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        }
        assert runner.backoff_schedule_seconds == (2.0, 8.0)
        return 0

    monkeypatch.setattr(release, "execute_plan", fake_execute_plan)

    rc = release.run_phase(
        work_root=tmp_path / "release_v2" / "llm",
        phase="pred",
        plan_jsonl=plan,
        profile=profile,
        logical_total=20,
        workers=32,
        limit=1,
    )

    assert rc == 0
    assert captured["attempt_cap"] == 30
    assert captured["logical_task_cap"] == 20
    assert captured["max_attempts_per_task"] == 3
    assert captured["workers"] == 32
    assert captured["state_db"] == tmp_path / "release_v2" / "llm" / "release_state.sqlite3"
    assert captured["limit"] == 1


def test_run_phase_allows_small_canary_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "routify.json"
    _write_profile(
        profile,
        base_url="https://routify-pub.alibaba-inc.com/protocol/anthropic",
    )
    work_root = tmp_path / "release_v2" / "llm" / "identity_canary"
    plan = tmp_path / "release_v2" / "inputs" / "identity_canary.jsonl"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        release,
        "load_plan_jsonl",
        lambda _: SimpleNamespace(
            entries=(
                SimpleNamespace(
                    logical_id="task::canary",
                    definition=SimpleNamespace(task_spec=SimpleNamespace(condition="clean")),
                ),
            )
        ),
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        release,
        "execute_plan",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    assert release.run_phase(
        work_root=work_root,
        phase="identity",
        plan_jsonl=plan,
        profile=profile,
        logical_total=6,
        workers=6,
    ) == 0
    assert captured["workers"] == 6
    assert captured["attempt_cap"] == 9


def test_plan_must_live_under_release_llm_workspace(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "routify.json"
    _write_profile(
        profile,
        base_url="https://routify-pub.alibaba-inc.com/protocol/anthropic",
    )
    outside_plan = tmp_path / "outside.jsonl"
    outside_plan.write_text("{}\n", encoding="utf-8")

    with pytest.raises(release.ReleaseControlError, match="plan"):
        release.run_phase(
            work_root=tmp_path / "release_v2" / "llm",
            phase="pred",
            plan_jsonl=outside_plan,
            profile=profile,
            logical_total=1,
            workers=32,
            limit=1,
        )


def test_exports_safe_frozen_index_and_predecessor_manifest(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite3"
    attempts_dir = tmp_path / "attempts"
    frozen_dir = tmp_path / "frozen"
    attempts_dir.mkdir()
    frozen_dir.mkdir()
    store = TaskStateStore(
        state_db,
        attempt_cap=2,
        logical_task_cap=1,
        max_attempts_per_task=2,
    )
    key = "a" * 64
    spec = TaskSpec(
        evaluation_key=key,
        logical_id="gt_simplify::fixture",
        task_type="gt_simplify",
        condition="clean",
        priority=10,
        input_hash="input",
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    store.register_task(spec)
    lease = store.reserve_attempt(key)
    attempt_path = attempts_dir / f"{lease.attempt_id}.json"
    attempt_path.write_text(json.dumps({"attempt_id": lease.attempt_id}), encoding="utf-8")
    result_path = frozen_dir / f"{key}.json"
    result_path.write_text(
        json.dumps(
            {
                "attempt_id": lease.attempt_id,
                "metadata": {
                    "requested_model": "claude-opus-5",
                    "requested_effort": "xhigh",
                    "api_channel": "routify",
                    "stream": False,
                    "estimated_cost_cny": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    result_sha = release._sha256_file(result_path)
    store.freeze_result(
        lease.attempt_id,
        result_path=str(result_path.resolve()),
        result_sha256=result_sha,
    )

    index_report = release.export_frozen_index(
        state_db=state_db,
        output_jsonl=tmp_path / "frozen_index.jsonl",
    )
    predecessor_report = release.export_predecessor_manifest(
        state_db=state_db,
        attempts_dir=attempts_dir,
        output_json=tmp_path / "predecessor.json",
    )

    assert index_report["frozen_count"] == 1
    index_row = json.loads((tmp_path / "frozen_index.jsonl").read_text())
    assert index_row["logical_id"] == "gt_simplify::fixture"
    assert index_row["api_channel"] == "routify"
    assert index_row["attempt_id"] == lease.attempt_id
    assert index_row["state_binding_attempt_id"] == lease.attempt_id
    assert "structured_output" not in index_row
    assert predecessor_report["attempt_count"] == 1
    predecessor = json.loads((tmp_path / "predecessor.json").read_text())
    assert predecessor["attempt_ids"] == [lease.attempt_id]
    assert predecessor["attempt_file_sha256"][lease.attempt_id] == release._sha256_file(
        attempt_path
    )


def test_prepare_release_state_reserves_cache_import_rows_outside_api_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "release_v2" / "llm" / "main"
    plan = tmp_path / "release_v2" / "inputs" / "bootstrap.jsonl"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}\n", encoding="utf-8")
    spec = TaskSpec(
        evaluation_key="b" * 64,
        logical_id="gt_simplify::bootstrap",
        task_type="gt_simplify",
        condition="clean",
        priority=10,
        input_hash="input",
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    monkeypatch.setattr(
        release,
        "load_plan_jsonl",
        lambda _: SimpleNamespace(
            entries=(
                SimpleNamespace(evaluation_key=spec.evaluation_key, definition=SimpleNamespace(task_spec=spec)),
            )
        ),
    )

    store = release.prepare_release_state(
        work_root=work_root,
        plan_jsonls=(plan,),
        logical_total=20,
        cache_import_total=6,
    )

    assert store.attempt_cap == 36
    assert store.logical_task_cap == 20
    assert store.task_state(spec.evaluation_key) == "pending"


def test_export_api_attempt_ledger_keeps_only_real_routify_calls(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for attempt_id, error_class in (("a" * 64 + ".a01", None), ("b" * 64 + ".a01", "api_timeout")):
        (source / f"{attempt_id}.json").write_text(
            json.dumps(
                {
                    "attempt_id": attempt_id,
                    "metadata": {
                        "api_channel": "routify",
                        "requested_model": "claude-opus-5",
                        "requested_effort": "xhigh",
                        "stream": False,
                        "error_class": error_class,
                        "estimated_cost_cny": 0.1,
                    },
                }
            ),
            encoding="utf-8",
        )
    (source / "cache.json").write_text(
        json.dumps({"provider": "cache_import", "network_request": False}),
        encoding="utf-8",
    )

    report = release.export_api_attempt_ledger(
        source_attempt_dirs=(source,),
        output_root=tmp_path / "ledger",
    )

    assert report["attempt_count"] == 2
    assert report["accepted_count"] == 1
    assert report["failed_count"] == 1
    assert report["estimated_cost_cny"] == pytest.approx(0.2)
    manifest = json.loads((tmp_path / "ledger" / "predecessor_attempts.json").read_text())
    assert manifest["attempt_count"] == 2
    assert set(manifest["attempt_ids"]) == {"a" * 64 + ".a01", "b" * 64 + ".a01"}


def test_structure_successor_accepts_noise_condition_suffix() -> None:
    assert _successor_logical_id(
        "stab_structure::symbolfit::g0012::s520-s522::noise005",
        task_type="stab_structure",
        logical_id_suffix="v2",
    ) == "stab_structure::symbolfit::g0012::s520-s522::noise005::v2"
