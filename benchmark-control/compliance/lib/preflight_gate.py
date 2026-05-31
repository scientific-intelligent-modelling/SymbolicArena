from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def check_preflight_report(
    *,
    report_path: Path,
    expected_hosts: list[str],
    batch_dir: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    host_reports = report.get("hosts", [])
    if not isinstance(host_reports, list):
        host_reports = []
    by_host = {
        str(item.get("host")): item
        for item in host_reports
        if isinstance(item, dict) and item.get("host")
    }
    requested_tools = [str(tool) for tool in report.get("requested_tools", [])]
    requested_envs = [str(env) for env in report.get("requested_envs", [])]
    requested_params = [str(name) for name in report.get("requested_params", [])] or requested_tools

    issues: list[str] = []
    for host in expected_hosts:
        item = by_host.get(host)
        if item is None:
            issues.append(f"missing host in preflight report: {host}")
            continue
        _check_host(
            host=host,
            item=item,
            requested_tools=requested_tools,
            requested_envs=requested_envs,
            requested_params=requested_params,
            issues=issues,
        )

    summary = {
        "ready_for_smoke": not issues,
        "report_path": str(report_path),
        "expected_hosts": expected_hosts,
        "reported_hosts": sorted(by_host),
        "issues": issues,
    }
    out_dir = batch_dir / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _check_host(
    *,
    host: str,
    item: dict[str, Any],
    requested_tools: list[str],
    requested_envs: list[str],
    requested_params: list[str],
    issues: list[str],
) -> None:
    if item.get("ok") is False:
        issues.append(f"{host} preflight transport failed: {item.get('stage') or item.get('error') or 'unknown'}")
        return
    _check_files(host, item.get("files"), requested_tools, requested_params, issues)
    _check_dataset_sync(host, item.get("dataset_sync"), issues)
    _check_params(host, item.get("params"), requested_params, issues)
    _check_envs(host, item.get("envs"), requested_envs, issues)


def _check_files(
    host: str,
    files: object,
    requested_tools: list[str],
    requested_params: list[str],
    issues: list[str],
) -> None:
    if not isinstance(files, dict):
        issues.append(f"{host} files report missing")
        return
    required_labels = [
        "scheduler",
        "launcher",
        "runner",
        "toolbox_config",
        "source_csv",
    ]
    for tool in requested_tools:
        required_labels.append(f"{tool}_wrapper")
    for params_name in requested_params:
        required_labels.append(f"{params_name}_params")
    for label in required_labels:
        if label not in files:
            issues.append(f"{host} file {label} missing")
    for label, payload in files.items():
        if not isinstance(payload, dict):
            issues.append(f"{host} file {label} report invalid")
            continue
        if payload.get("matches_expected") is not True:
            issues.append(f"{host} file {label} mismatch")


def _check_dataset_sync(host: str, dataset_sync: object, issues: list[str]) -> None:
    if not isinstance(dataset_sync, dict):
        issues.append(f"{host} dataset_sync report missing")
        return
    if dataset_sync.get("all_match") is not True:
        missing = dataset_sync.get("missing_files")
        size = dataset_sync.get("size_mismatch_files")
        hashes = dataset_sync.get("hash_mismatch_files")
        issues.append(f"{host} dataset sync mismatch missing={missing} size={size} hash={hashes}")


def _check_params(host: str, params: object, requested_params: list[str], issues: list[str]) -> None:
    if not isinstance(params, dict):
        issues.append(f"{host} params report missing")
        return
    for params_name in requested_params:
        payload = params.get(params_name)
        if not isinstance(payload, dict):
            issues.append(f"{host} params {params_name} missing")
            continue
        if payload.get("exists") is not True:
            issues.append(f"{host} params {params_name} file missing")
        if payload.get("json_ok") is not True:
            issues.append(f"{host} params {params_name} json invalid")
        tool = params_name.split("__", 1)[0].split("_", 1)[0]
        if tool in {"llmsr", "drsr"} and payload.get("llm_config_exists") is not True:
            issues.append(f"{host} params {params_name} llm config missing")


def _check_envs(host: str, env_report: object, requested_envs: list[str], issues: list[str]) -> None:
    if not isinstance(env_report, dict):
        issues.append(f"{host} env report missing")
        return
    envs = env_report.get("envs")
    if not isinstance(envs, dict):
        issues.append(f"{host} envs report missing")
        return
    for env in requested_envs:
        payload = envs.get(env)
        if not isinstance(payload, dict):
            issues.append(f"{host} env {env} missing")
            continue
        if payload.get("ok") is not True:
            issues.append(f"{host} env {env} import failed")
