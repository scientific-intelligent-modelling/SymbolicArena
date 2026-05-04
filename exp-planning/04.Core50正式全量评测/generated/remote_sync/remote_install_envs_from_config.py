#!/usr/bin/env python3
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
    python_version = str(env_config.get("python_version") or "3.10")
    conda_packages = [str(pkg) for pkg in env_config.get("conda_packages", [])]
    pip_packages = [str(pkg) for pkg in env_config.get("pip_packages", [])]
    channels = [str(channel) for channel in env_config.get("channels", [])]
    post_commands = [str(command) for command in env_config.get("post_install_commands", [])]

    if env_dir.exists():
        log_file.write(f"[repair] {env_name} already exists at {env_dir}; installing configured packages in-place.\n")
    else:
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
        stripped = command.strip()
        if stripped.startswith("pip "):
            cmd = f"cd {shlex.quote(str(REMOTE_ROOT))} && conda run -n {shlex.quote(env_name)} python -m pip {stripped[4:]}"
        elif stripped.startswith("python "):
            cmd = f"cd {shlex.quote(str(REMOTE_ROOT))} && conda run -n {shlex.quote(env_name)} python {stripped[7:]}"
        else:
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
