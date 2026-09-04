from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("generate_all_15alg_fullcpu_v1.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("all_15alg_fullcpu_v1_generator", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generator: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    module = _load_module()
    root = tmp_path_factory.mktemp("all_15alg_fullcpu_v1")
    report = module.generate_assets(asset_root=root, run_dry_run=True)
    return module, root, report


def test_composite_ledger_proves_exact_partition(generated) -> None:
    module, root, report = generated
    ledger = report["composite_ledger"]
    assert ledger["mother_count"] == 6750
    assert ledger["excluded_count"] == 143
    assert ledger["new_queue_count"] == 6607
    assert ledger["identity"] == "143 + 6607 = 6750"
    assert ledger["excluded_and_queue_disjoint"] is True
    assert ledger["excluded_union_queue_equals_mother"] is True
    assert ledger["mother_algorithm_counts"] == {tool: 450 for tool in sorted(module.TOOLS)}
    assert ledger["queue_algorithm_counts"]["jaxsr"] == 387
    assert ledger["queue_algorithm_counts"]["imcts"] == 371
    assert ledger["queue_algorithm_counts"]["drsr"] == 449
    assert ledger["queue_condition_counts"] == {
        "clean": 2250,
        "noise001": 2171,
        "noise005": 2186,
    }
    assert ledger["mother_cpu_weight_counts"] == {"1": 3600, "4": 3150}
    assert ledger["queue_cpu_weight_counts"] == {"1": 3457, "4": 3150}
    assert ledger["queue_cpu_weight_total"] == 16057

    with (root / "manifests/composite_ledger_6750.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        mother = list(csv.DictReader(handle))
    with (root / "manifests/new_queue_6607.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        queue = list(csv.DictReader(handle))
    mother_ids = {row["task_id"] for row in mother}
    queue_ids = {row["task_id"] for row in queue}
    excluded_ids = {
        row["task_id"]
        for row in mother
        if row["excluded_by_all_conditions_cpu_v2"] == "true"
    }
    assert len(mother_ids) == len(mother) == 6750
    assert len(queue_ids) == len(queue) == 6607
    assert len(excluded_ids) == 143
    assert not queue_ids.intersection(excluded_ids)
    assert queue_ids.union(excluded_ids) == mother_ids


def test_sources_params_and_local_dry_run_are_strict(generated) -> None:
    _, root, report = generated
    sources = report["source_validation"]
    assert sources["all_three_sha_equal"] is True
    assert len({entry["sha256"] for entry in sources["sources"].values()}) == 1
    assert report["parameter_validation"]["count"] == 45
    assert report["parameter_validation"]["by_source_family"] == {
        "drsr3h": 3,
        "formal3h_13alg": 39,
        "llmsr3h": 3,
    }
    for entry in report["parameter_validation"]["files"].values():
        assert entry["timeout_in_seconds"] == 10800
        assert entry["progress_snapshot_interval_seconds"] == 60

    dry_run = report["dry_run"]
    assert dry_run["status"] == "passed"
    assert dry_run["remote_connection_attempted"] is False
    assert dry_run["task_count"] == 6607
    assert dry_run["task_ids_unique"] is True
    assert dry_run["condition_counts"] == {
        "clean": 2250,
        "noise001": 2171,
        "noise005": 2186,
    }
    assert dry_run["cpu_weight_counts"] == {"1": 3457, "4": 3150}
    assert "--dry-run" in dry_run["command"]
    assert "--preflight-only" not in dry_run["command"]
    assert (root / "queues/dry_run_validation/state").is_dir()


def test_commands_encode_full_cpu_guard_and_shared_old_sessions(generated) -> None:
    module, root, report = generated
    expected_hosts = "iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29"
    expected_tools = " ".join(module.TOOLS)
    for mode in ("preflight", "dry_run", "formal"):
        text = (root / f"commands/run_all_15alg_{mode}.sh").read_text(encoding="utf-8")
        assert "\n+" not in text
        assert f"--hosts {expected_hosts}" in text
        assert f"--tools {expected_tools}" in text
        assert "--noise-sigmas 0 0.01 0.05" in text
        assert "--max-jobs-per-host 230" in text
        assert "--max-cpu-used-ratio" not in text
        assert "--max-new-jobs-per-host-per-poll 64" in text
        assert '"0.50:64,0.70:32,0.85:8,0.90:2"' in text
        assert "--max-load-ratio 0.90" in text
        assert "--max-memory-used-ratio 0.90" in text
        assert "--min-free-mem-gb 32" in text
        assert "--poll-seconds 30" in text
        assert "--session-prefix all_conditions_cpu_v2_" in text
        assert "--host-session-count-prefix all_conditions_cpu_v2_" in text
        assert "--llm-model-bucket-limits base:0,turbo:30" in text
    formal = (root / "commands/run_all_15alg_formal.sh").read_text(encoding="utf-8")
    assert "--force-rerun-existing" in formal
    assert "SIM_RESUME_EXISTING_QUEUE" in formal
    assert "RESUME_ARGS+=(--skip-support-sync)" in formal
    assert '"${RESUME_ARGS[@]}"' in formal
    assert report["resources"]["cpu_weight_budget_per_host"] is None
    assert report["resources"]["max_cpu_used_ratio"] is None
    assert report["resources"]["load_driven_dispatch"] is True
    assert report["resources"]["logical_cpus_per_host"] == 256
    assert report["resources"]["physical_cpus_per_host"] == 128

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "取消固定 CPU weight 硬封顶" in readme
    assert "真实 load 逼近但不超过 0.90" in readme
    assert "143 + 6607 = 6750" in readme

    environment = dict(os.environ)
    environment["SIM_REPO_ROOT"] = str(module.REPO_ROOT)
    executed = subprocess.run(
        ["bash", str(root / "commands/run_all_15alg_dry_run.sh")],
        cwd=module.REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr[-2000:]
    assert '"tasks": 6607' in executed.stdout


def test_runtime_fingerprints_deployment_and_manifest(generated) -> None:
    module, root, report = generated
    fingerprint_path = root / "manifests/runtime_code_fingerprints.json"
    fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert fingerprints["file_count"] == 23
    assert fingerprints["coverage"]["wrappers"] == 15
    assert len({entry["path"] for entry in fingerprints["files"]}) == 23
    for entry in fingerprints["files"]:
        path = report["asset_root"]
        del path  # 仅强调指纹针对仓库运行时代码，不是资产副本。
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0

    deploy = (root / "commands/deploy_via_iaaccn22.sh").read_text(encoding="utf-8")
    fanout = (root / "commands/fanout_from_iaaccn22.sh").read_text(encoding="utf-8")
    inferred_repo_root = (
        module.DEFAULT_ASSET_ROOT / "commands" / "../../../../.."
    ).resolve()
    assert inferred_repo_root == module.REPO_ROOT.resolve()
    assert '/../../../../.." && pwd)' in deploy
    assert "--delete" not in deploy
    assert "--delete" not in fanout
    assert "iaaccn22:$REMOTE_ROOT/" in deploy
    assert "fanout_from_iaaccn22.sh" in deploy
    for number in range(23, 30):
        assert f"10.10.100.{number}" in fanout
    assert "llm_config" not in deploy.lower()
    assert report["deployment"]["executed"] is False
    assert report["deployment"]["copies_llm_config_or_secrets"] is False

    environment = dict(os.environ)
    environment["SIM_REPO_ROOT"] = str(module.REPO_ROOT)
    local_verify = subprocess.run(
        ["bash", str(root / "commands/deploy_via_iaaccn22.sh"), "--verify-local-only"],
        cwd=module.REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert local_verify.returncode == 0, local_verify.stderr
    assert "local_deployment_inputs_ok" in local_verify.stdout
    assert "files=23" in local_verify.stdout

    manifest_path = root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, metadata in manifest["files"].items():
        assert _sha(root / relative) == metadata["sha256"]
    self_hash = (root / "asset_manifest.sha256").read_text(encoding="utf-8").split()[0]
    assert self_hash == _sha(manifest_path)
    assert not any("llm_config" in relative.lower() for relative in manifest["files"])
    assert Counter(path.suffix for path in (root / "params").iterdir())[".json"] == 45
