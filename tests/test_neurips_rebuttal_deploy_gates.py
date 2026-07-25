import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = (
    ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "01_new3algs_full664_3seeds_clean_1h"
)


def test_collect_gate_requires_every_task_to_be_done() -> None:
    script = (
        BATCH_DIR / "deploy/04_collect_audit_from_iaaccn22.sh"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"jq -e '\n(?P<filter>.*?)\n' \"\$STATE_SUMMARY\"",
        script,
        flags=re.DOTALL,
    )
    assert match is not None
    jq_filter = match.group("filter")
    assert "(.task_states.done // 0) == 5976" in script
    assert "([.task_states[]] | add == 5976)" in script
    completed = subprocess.run(
        ["jq", "-e", jq_filter],
        input=json.dumps(
            {"task_states": {"done": 5976, "pending": 0, "running": 0}}
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    failed = subprocess.run(
        ["jq", "-e", jq_filter],
        input=json.dumps(
            {
                "task_states": {
                    "done": 5975,
                    "failed": 1,
                    "pending": 0,
                    "running": 0,
                }
            }
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert failed.returncode != 0


def test_full_dispatch_allows_one_post_hotfix_retry() -> None:
    script = (
        BATCH_DIR / "deploy/03_full_dispatch_from_iaaccn22.sh"
    ).read_text(encoding="utf-8")

    assert "--retry-limit 2" in script


def test_symf_gate_requires_complete_new3_and_stage3_grids() -> None:
    script = (
        BATCH_DIR / "deploy/05_generate_symf_from_iaaccn22.sh"
    ).read_text(encoding="utf-8")

    assert "(.final_ready == true)" in script
    assert "(.new3.expected_tasks == 5976)" in script
    assert "(.new3.present_results == 5976)" in script
    assert "(.stage3.rows == 7968)" in script
    assert "--expected-runs 13944" in script
    assert "--expected-algorithms 7" in script
    assert "(.runs == 13944)" in script
    assert "(.datasets == 664)" in script
    assert "(.algorithms == 7)" in script
    assert "symbolic_metrics_formal_summary.json" in script
    assert (
        '["dso", "fepysr", "imcts", "jaxsr", '
        '"pyoperon", "symbolfit", "udsr"]'
    ) in script
    assert "full664_7alg_leaderboard_with_symf.summary.json" in script
