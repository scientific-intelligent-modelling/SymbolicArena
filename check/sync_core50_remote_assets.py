#!/usr/bin/env python3
"""同步并预检 Core50 正式评测远端资产。

本脚本服务于 `12 algorithms x Core50` 正式评测准备阶段，分层处理：

1. `--preflight-only`：只读检查远端 Core50 数据、代码文件 hash、conda 环境导入。
2. `--sync-code`：同步 12 个算法 wrapper、benchmark runner、配置、参数和 Core50 实验文件。
3. `--sync-data`：仅同步 Core50 清单中的 50 个数据集目录。
4. `--sync-missing-envs`：只补远端不存在的 conda env，不覆盖已有环境。

注意：环境替换/覆盖不在本脚本默认行为内，避免误伤已有可用环境。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import socket
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE50_ROOT = REPO_ROOT / "exp-planning/04.Core50正式全量评测"
CORE50_CSV = CORE50_ROOT / "core50_datasets.csv"
REMOTE_ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
REMOTE_DATA_ROOT = Path("/home/zhangziwen/sim-datasets-data")
REMOTE_CONDA_ENVS = Path("/home/zhangziwen/anaconda3/envs")
GENERATED_ROOT = CORE50_ROOT / "generated/remote_sync"

DEFAULT_HOSTS = (
    "iaaccn22",
    "iaaccn23",
    "iaaccn24",
    "iaaccn25",
    "iaaccn26",
    "iaaccn27",
    "iaaccn28",
    "iaaccn29",
)

TOOL_ENVS = {
    "gplearn": "sim_base",
    "llmsr": "sim_llm",
    "pyoperon": "sim_base",
    "drsr": "sim_llm",
    "pysr": "sim_base",
    "dso": "sim_dso",
    "tpsr": "sim_tpsr",
    "e2esr": "sim_e2esr",
    "qlattice": "sim_qLattice",
    "imcts": "sim_iMCTS",
    "udsr": "sim_dso",
    "ragsr": "sim_ragsr",
}

ENV_IMPORTS = {
    "sim_base": [
        "scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper",
    ],
    "sim_llm": [
        "scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper",
    ],
    "sim_dso": [
        "scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.udsr_wrapper.wrapper",
    ],
    "sim_tpsr": ["scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper"],
    "sim_e2esr": ["scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper"],
    "sim_qLattice": ["scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper"],
    "sim_iMCTS": ["scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper"],
    "sim_ragsr": ["scientific_intelligent_modelling.algorithms.ragsr_wrapper.wrapper"],
}

ENV_SOURCES = {
    "sim_base": "iaaccn23",
    "sim_llm": "iaaccn23",
    "sim_dso": "iaaccn23",
    "sim_iMCTS": "iaaccn23",
    "sim_e2esr": "iaaccn24",
    "sim_qLattice": "iaaccn26",
    "sim_tpsr": "iaaccn25",
    "sim_ragsr": "iaaccn27",
}

SYNC_CODE_DIRS = (
    "scientific_intelligent_modelling/algorithms",
    "scientific_intelligent_modelling/srkit",
    "scientific_intelligent_modelling/benchmarks",
    "scientific_intelligent_modelling/config",
    "exp-planning/02.E1选择验证/generated/params",
    "exp-planning/02.E1选择验证/llm_configs",
    "exp-planning/04.Core50正式全量评测",
)

SYNC_CODE_FILES = (
    "check/launch_e1_benchmark.py",
    "check/run_e1_candidate200_12alg_load_queue.py",
    "check/sync_core50_remote_assets.py",
)

REQUIRED_DATA_FILES = ("metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv", "formula.py")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, safe_text(exc.stdout), safe_text(exc.stderr) or "timeout")


def host_number(host: str) -> str | None:
    suffix = host.removeprefix("iaaccn")
    return suffix if suffix.isdigit() else None


def internal_target(host: str) -> str:
    number = host_number(host)
    return f"10.10.100.{number}" if number else host


def is_local_host(host: str) -> bool:
    local_names = {socket.gethostname(), socket.getfqdn(), "localhost", "127.0.0.1"}
    short_names = {name.split(".")[0] for name in local_names}
    return host in local_names or host in short_names


def ssh(host: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    if is_local_host(host):
        return run(["bash", "-lc", command], timeout=timeout)
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            host,
            command,
        ],
        timeout=timeout,
    )


def scp(local_path: Path, host: str, remote_path: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    if is_local_host(host):
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        return run(["cp", str(local_path), str(remote_path)], timeout=timeout)
    return run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            str(local_path),
            f"{host}:{remote_path}",
        ],
        timeout=timeout,
    )


def rsync_path(local_path: Path, host: str, remote_path: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    local = f"{local_path}/" if local_path.is_dir() else str(local_path)
    remote = f"{host}:{remote_path}/" if local_path.is_dir() else f"{host}:{remote_path}"
    return run(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            local,
            remote,
        ],
        timeout=timeout,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_core50_rows() -> list[dict[str, str]]:
    with CORE50_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 50:
        raise SystemExit(f"Core50 清单行数不是 50: {CORE50_CSV} -> {len(rows)}")
    return rows


def local_dataset_dir(row: dict[str, str]) -> Path:
    dataset_dir = Path(row["dataset_dir"])
    if dataset_dir.is_absolute():
        try:
            rel = dataset_dir.relative_to("/home/zhangziwen/sim-datasets-data")
            return REPO_ROOT / "sim-datasets-data" / rel
        except ValueError:
            return dataset_dir
    return REPO_ROOT / dataset_dir


def remote_dataset_dir(row: dict[str, str]) -> Path:
    dataset_dir = Path(row["dataset_dir"])
    if dataset_dir.is_absolute():
        try:
            return REMOTE_DATA_ROOT / dataset_dir.relative_to("/home/zhangziwen/sim-datasets-data")
        except ValueError:
            return dataset_dir
    if dataset_dir.parts and dataset_dir.parts[0] == "sim-datasets-data":
        return REMOTE_DATA_ROOT.joinpath(*dataset_dir.parts[1:])
    return REMOTE_DATA_ROOT / dataset_dir


def local_file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size": None, "sha256": None}
    return {"exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}


def core50_fingerprints(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        dataset_dir = local_dataset_dir(row)
        files = {name: local_file_fingerprint(dataset_dir / name) for name in REQUIRED_DATA_FILES}
        out.append(
            {
                "dataset_name": row["dataset_name"],
                "dataset_dir": row["dataset_dir"],
                "files": files,
            }
        )
    return out


def local_code_hashes() -> dict[str, str | None]:
    rels = list(SYNC_CODE_FILES)
    for rel_dir in SYNC_CODE_DIRS:
        base = REPO_ROOT / rel_dir
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts:
                continue
            if "generated" in path.parts and "remote_sync" in path.parts:
                continue
            rels.append(str(path.relative_to(REPO_ROOT)))
    out = {}
    for rel in sorted(set(rels)):
        path = REPO_ROOT / rel
        out[rel] = sha256_file(path) if path.exists() else None
    return out


def write_remote_preflight_script() -> Path:
    path = GENERATED_ROOT / "remote_core50_preflight.py"
    content = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


REMOTE_ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
REMOTE_DATA_ROOT = Path("/home/zhangziwen/sim-datasets-data")
REMOTE_CONDA_ENVS = Path("/home/zhangziwen/anaconda3/envs")
REQUIRED_DATA_FILES = ("metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv", "formula.py")


def run(cmd: str, timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": 124, "stdout": str(exc.stdout or "")[-4000:], "stderr": str(exc.stderr or "timeout")[-4000:]}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "size": None, "sha256": None}
    return {"exists": True, "size": path.stat().st_size, "sha256": sha256(path)}


def remote_dataset_dir(dataset_dir: str) -> Path:
    path = Path(dataset_dir)
    if path.is_absolute():
        try:
            return REMOTE_DATA_ROOT / path.relative_to("/home/zhangziwen/sim-datasets-data")
        except ValueError:
            return path
    if path.parts and path.parts[0] == "sim-datasets-data":
        return REMOTE_DATA_ROOT.joinpath(*path.parts[1:])
    return REMOTE_DATA_ROOT / path


def check_data(expected: list[dict]) -> dict:
    missing = 0
    size_mismatch = 0
    hash_mismatch = 0
    examples = []
    compared = 0
    for item in expected:
        dataset_dir = remote_dataset_dir(item["dataset_dir"])
        for name, expected_fp in item["files"].items():
            compared += 1
            actual = fingerprint(dataset_dir / name)
            problems = []
            if not actual["exists"]:
                missing += 1
                problems.append("missing")
            else:
                if actual["size"] != expected_fp.get("size"):
                    size_mismatch += 1
                    problems.append("size_mismatch")
                if actual["sha256"] != expected_fp.get("sha256"):
                    hash_mismatch += 1
                    problems.append("sha256_mismatch")
            if problems and len(examples) < 20:
                examples.append({
                    "dataset_name": item["dataset_name"],
                    "file": name,
                    "remote_path": str(dataset_dir / name),
                    "problems": problems,
                    "expected": expected_fp,
                    "actual": actual,
                })
    return {
        "datasets": len(expected),
        "compared_files": compared,
        "missing_files": missing,
        "size_mismatch_files": size_mismatch,
        "hash_mismatch_files": hash_mismatch,
        "all_match": missing == 0 and size_mismatch == 0 and hash_mismatch == 0,
        "examples": examples,
    }


def check_code(expected_hashes: dict[str, str | None]) -> dict:
    bad = []
    checked = 0
    for rel, expected in expected_hashes.items():
        checked += 1
        path = REMOTE_ROOT / rel
        actual = sha256(path)
        if expected != actual and len(bad) < 50:
            bad.append({"rel": rel, "expected": expected, "actual": actual, "exists": path.exists()})
    return {"checked_files": checked, "mismatch_count": sum(1 for rel, expected in expected_hashes.items() if sha256(REMOTE_ROOT / rel) != expected), "examples": bad}


def check_envs(env_imports: dict[str, list[str]]) -> dict:
    out = {}
    for env, modules in env_imports.items():
        env_dir = REMOTE_CONDA_ENVS / env
        code = "import importlib\n" + "\n".join(f"importlib.import_module({m!r})" for m in modules) + "\nprint('import_ok')\n"
        tmp = Path(tempfile.gettempdir()) / f"core50_import_check_{env}.py"
        tmp.write_text(code, encoding="utf-8")
        cmd = f"cd {shlex.quote(str(REMOTE_ROOT))} && PYTHONPATH=. conda run -n {shlex.quote(env)} python {shlex.quote(str(tmp))}"
        try:
            result = run(cmd, timeout=120)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        out[env] = {
            "env_dir_exists": env_dir.exists(),
            "required_modules": modules,
            "returncode": result["returncode"],
            "ok": result["returncode"] == 0 and "import_ok" in result["stdout"],
            "stdout_tail": result["stdout"][-1000:],
            "stderr_tail": result["stderr"][-1500:],
        }
    return out


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    report = {
        "host": payload["host"],
        "data": check_data(payload["core50_fingerprints"]),
        "code": check_code(payload["code_hashes"]),
        "envs": check_envs(payload["env_imports"]),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def sync_code_to_host(host: str) -> dict[str, Any]:
    result = ssh(host, f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}", timeout=30)
    if result.returncode != 0:
        return {"ok": False, "stage": "mkdir_root", "error": result.stderr or result.stdout}
    failures = []
    for rel_dir in SYNC_CODE_DIRS:
        local = REPO_ROOT / rel_dir
        remote = REMOTE_ROOT / rel_dir
        if not local.exists():
            failures.append({"path": rel_dir, "error": "local_missing"})
            continue
        mkdir = ssh(host, f"mkdir -p {shlex.quote(str(remote.parent))}", timeout=30)
        if mkdir.returncode != 0:
            failures.append({"path": rel_dir, "error": mkdir.stderr or mkdir.stdout})
            continue
        res = rsync_path(local, host, remote, timeout=900)
        if res.returncode != 0:
            failures.append({"path": rel_dir, "error": res.stderr or res.stdout})
    for rel_file in SYNC_CODE_FILES:
        local = REPO_ROOT / rel_file
        remote = REMOTE_ROOT / rel_file
        mkdir = ssh(host, f"mkdir -p {shlex.quote(str(remote.parent))}", timeout=30)
        if mkdir.returncode != 0:
            failures.append({"path": rel_file, "error": mkdir.stderr or mkdir.stdout})
            continue
        res = rsync_path(local, host, remote, timeout=120)
        if res.returncode != 0:
            failures.append({"path": rel_file, "error": res.stderr or res.stdout})
    return {"ok": not failures, "failures": failures[:20], "failure_count": len(failures)}


def sync_data_to_host(host: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        local = local_dataset_dir(row)
        remote = remote_dataset_dir(row)
        mkdir = ssh(host, f"mkdir -p {shlex.quote(str(remote.parent))}", timeout=30)
        if mkdir.returncode != 0:
            failures.append({"dataset": row["dataset_name"], "error": mkdir.stderr or mkdir.stdout})
            continue
        res = rsync_path(local, host, remote, timeout=900)
        if res.returncode != 0:
            failures.append({"dataset": row["dataset_name"], "error": res.stderr or res.stdout})
    return {"ok": not failures, "failures": failures[:20], "failure_count": len(failures)}


def sync_missing_env(source_host: str, target_host: str, env: str) -> dict[str, Any]:
    if source_host == target_host:
        return {"env": env, "source": source_host, "target": target_host, "ok": True, "skipped": "source_is_target"}
    target = internal_target(target_host)
    source_dir = REMOTE_CONDA_ENVS / env
    target_dir = REMOTE_CONDA_ENVS / env
    command = (
        f"test -d {shlex.quote(str(source_dir))} && "
        f"ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"{shlex.quote(target)} 'mkdir -p {shlex.quote(str(target_dir.parent))}' && "
        f"rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' "
        f"{shlex.quote(str(source_dir))}/ {shlex.quote(target)}:{shlex.quote(str(target_dir))}/"
    )
    result = ssh(source_host, command, timeout=3600)
    return {
        "env": env,
        "source": source_host,
        "target": target_host,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1500:],
    }


def sync_missing_envs(host_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return launch_missing_env_installers(host_reports)


def load_env_configs() -> dict[str, Any]:
    path = REPO_ROOT / "scientific_intelligent_modelling/config/envs_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("env_list", {})


def write_remote_env_installer() -> Path:
    path = GENERATED_ROOT / "remote_install_envs_from_config.py"
    content = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REMOTE_ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
REMOTE_CONDA_ENVS = Path("/home/zhangziwen/anaconda3/envs")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run(command: str, log_file, *, cwd: Path | None = None) -> int:
    log_file.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] $ {command}\n")
    log_file.flush()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        executable="/bin/bash",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
    proc.wait()
    log_file.write(f"[returncode] {proc.returncode}\n")
    log_file.flush()
    return int(proc.returncode)


def conda_env_exists(env_name: str) -> bool:
    return (REMOTE_CONDA_ENVS / env_name).exists()


def install_env(env_name: str, env_config: dict, log_file) -> dict:
    env_dir = REMOTE_CONDA_ENVS / env_name
    if env_dir.exists():
        log_file.write(f"[skip] {env_name} already exists at {env_dir}; not overwriting.\n")
        return {"env": env_name, "ok": True, "skipped": "exists"}

    python_version = str(env_config.get("python_version") or "3.10")
    conda_packages = [str(pkg) for pkg in env_config.get("conda_packages", [])]
    pip_packages = [str(pkg) for pkg in env_config.get("pip_packages", [])]
    channels = [str(channel) for channel in env_config.get("channels", [])]
    post_commands = [str(command) for command in env_config.get("post_install_commands", [])]

    create_cmd = ["conda", "create", "-y", "-n", env_name, f"python={python_version}", *conda_packages]
    for channel in channels:
        create_cmd.extend(["-c", channel])
    rc = run(shell_join(create_cmd), log_file, cwd=REMOTE_ROOT)
    if rc != 0:
        return {"env": env_name, "ok": False, "stage": "conda_create", "returncode": rc}

    for package in pip_packages:
        cmd = shell_join(["conda", "run", "-n", env_name, "python", "-m", "pip", "install", package])
        rc = run(cmd, log_file, cwd=REMOTE_ROOT)
        if rc != 0:
            return {"env": env_name, "ok": False, "stage": "pip_install", "package": package, "returncode": rc}

    for command in post_commands:
        cmd = f"cd {shlex.quote(str(REMOTE_ROOT))} && conda run -n {shlex.quote(env_name)} bash -lc {shlex.quote(command)}"
        rc = run(cmd, log_file, cwd=REMOTE_ROOT)
        if rc != 0:
            return {"env": env_name, "ok": False, "stage": "post_install", "command": command, "returncode": rc}

    return {"env": env_name, "ok": True, "skipped": None}


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    env_names = payload["envs"]
    env_configs = payload["env_configs"]
    log_path = Path(payload["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[start] env installer host={payload.get('host')} envs={env_names}\n")
        for env_name in env_names:
            config = env_configs.get(env_name)
            if not config:
                result = {"env": env_name, "ok": False, "stage": "missing_config"}
            else:
                result = install_env(env_name, config, log_file)
            results.append(result)
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        log_file.write("[done]\n")

    result_path = log_path.with_suffix(".result.json")
    result_path.write_text(json.dumps({"host": payload.get("host"), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result_path": str(result_path), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def launch_missing_env_installers(host_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    env_configs = load_env_configs()
    remote_script = Path("/tmp/core50_install_envs_from_config.py")
    local_script = write_remote_env_installer()
    install_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    actions = []
    for report in host_reports:
        host = report["host"]
        if not report.get("ok"):
            actions.append({"host": host, "ok": False, "stage": "preflight_failed"})
            continue
        missing_envs = [
            env
            for env, item in sorted(report.get("envs", {}).items())
            if not item.get("ok") and not item.get("env_dir_exists")
        ]
        if not missing_envs:
            actions.append({"host": host, "ok": True, "skipped": "no_missing_envs"})
            continue
        copy = scp(local_script, host, remote_script, timeout=60)
        if copy.returncode != 0:
            actions.append({"host": host, "ok": False, "stage": "scp_installer", "error": copy.stderr or copy.stdout})
            continue
        payload = {
            "host": host,
            "envs": missing_envs,
            "env_configs": {env: env_configs.get(env) for env in missing_envs},
            "log_path": str(REMOTE_ROOT / "experiments" / f"core50_env_install_{install_id}" / f"{host}.log"),
        }
        local_payload = GENERATED_ROOT / f"env_install_request_{host}_{install_id}.json"
        local_payload.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        remote_payload = Path(f"/tmp/core50_env_install_request_{host}.json")
        copy_payload = scp(local_payload, host, remote_payload, timeout=60)
        if copy_payload.returncode != 0:
            actions.append({"host": host, "ok": False, "stage": "scp_payload", "error": copy_payload.stderr or copy_payload.stdout})
            continue
        session = f"core50_env_install_{install_id}"
        command = (
            f"tmux new-session -d -s {shlex.quote(session)} "
            f"bash -lc {shlex.quote(f'python {remote_script} {remote_payload}')}"
        )
        started = ssh(host, command, timeout=30)
        actions.append(
            {
                "host": host,
                "ok": started.returncode == 0,
                "envs": missing_envs,
                "session": session,
                "log_path": payload["log_path"],
                "error": None if started.returncode == 0 else (started.stderr or started.stdout),
            }
        )
    return actions


def run_preflight(hosts: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    remote_script = Path("/tmp/core50_remote_preflight.py")
    local_script = write_remote_preflight_script()
    payload_base = {
        "core50_fingerprints": core50_fingerprints(rows),
        "code_hashes": local_code_hashes(),
        "env_imports": ENV_IMPORTS,
    }
    reports = []
    for host in hosts:
        copied = scp(local_script, host, remote_script, timeout=60)
        if copied.returncode != 0:
            reports.append({"host": host, "ok": False, "stage": "scp_script", "error": copied.stderr or copied.stdout})
            continue
        payload = {**payload_base, "host": host}
        local_payload = GENERATED_ROOT / f"preflight_request_{host}.json"
        local_payload.parent.mkdir(parents=True, exist_ok=True)
        local_payload.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        remote_payload = Path(f"/tmp/core50_preflight_request_{host}.json")
        copied_payload = scp(local_payload, host, remote_payload, timeout=60)
        if copied_payload.returncode != 0:
            reports.append({"host": host, "ok": False, "stage": "scp_payload", "error": copied_payload.stderr or copied_payload.stdout})
            continue
        result = ssh(host, f"python {shlex.quote(str(remote_script))} {shlex.quote(str(remote_payload))}", timeout=900)
        if result.returncode != 0:
            reports.append({"host": host, "ok": False, "stage": "run_script", "error": result.stderr or result.stdout})
            continue
        try:
            report = json.loads(result.stdout.strip().splitlines()[-1])
            report["ok"] = True
            reports.append(report)
        except Exception as exc:
            reports.append({"host": host, "ok": False, "stage": "parse_report", "error": repr(exc), "stdout_tail": result.stdout[-2000:]})
    summary = {"time": now(), "hosts": reports}
    report_path = GENERATED_ROOT / f"core50_preflight_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def summarize_preflight(summary: dict[str, Any]) -> str:
    lines = []
    for report in summary["hosts"]:
        host = report["host"]
        if not report.get("ok"):
            lines.append(f"{host}: HOST_FAIL {report.get('stage')} {str(report.get('error'))[:120]}")
            continue
        data = report["data"]
        code = report["code"]
        env_ok = sorted(env for env, item in report["envs"].items() if item.get("ok"))
        env_bad = sorted(env for env, item in report["envs"].items() if not item.get("ok"))
        lines.append(
            f"{host}: data={'ok' if data['all_match'] else 'bad'} "
            f"code_mismatch={code['mismatch_count']} "
            f"env_ok={','.join(env_ok) or '-'} env_bad={','.join(env_bad) or '-'}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步并预检 Core50 / 12算法远端资产")
    parser.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sync-code", action="store_true")
    parser.add_argument("--sync-data", action="store_true")
    parser.add_argument("--sync-missing-envs", action="store_true", help="兼容旧参数；现在按 envs_config.json 远端安装缺失环境，不复制 env 目录。")
    parser.add_argument("--install-missing-envs", action="store_true", help="按 envs_config.json 在远端下载并安装缺失环境。")
    parser.add_argument("--post-preflight", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_core50_rows()
    print(json.dumps({"event": "start", "time": now(), "hosts": args.hosts, "core50_rows": len(rows)}, ensure_ascii=False), flush=True)

    if args.sync_code:
        code_results = {host: sync_code_to_host(host) for host in args.hosts}
        print(json.dumps({"event": "sync_code_done", "results": code_results}, ensure_ascii=False), flush=True)

    if args.sync_data:
        data_results = {host: sync_data_to_host(host, rows) for host in args.hosts}
        print(json.dumps({"event": "sync_data_done", "results": data_results}, ensure_ascii=False), flush=True)

    needs_env_install = args.sync_missing_envs or args.install_missing_envs
    if args.preflight_only or args.post_preflight or needs_env_install:
        summary = run_preflight(args.hosts, rows)
        print(json.dumps({"event": "preflight_done", "report_path": summary["report_path"]}, ensure_ascii=False), flush=True)
        print(summarize_preflight(summary), flush=True)
    else:
        summary = {"hosts": []}

    if needs_env_install:
        env_actions = sync_missing_envs(summary["hosts"])
        env_path = GENERATED_ROOT / f"missing_env_sync_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        env_path.write_text(json.dumps({"time": now(), "actions": env_actions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "sync_missing_envs_done", "report_path": str(env_path), "actions": env_actions}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
