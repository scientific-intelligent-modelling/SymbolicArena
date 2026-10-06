from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from scientific_intelligent_modelling.srkit.config_manager import config_manager
from scientific_intelligent_modelling.srkit.conda_env_manager import env_manager

from .manifest import load_manifest, repository_root, write_json
from .source import inspect_source


def setup_environment(manifest_path, output_root):
    manifest = load_manifest(manifest_path)
    name = manifest["environment"]["name"]
    config = config_manager.get_config("envs_config")["env_list"][name]
    registrations = config_manager.get_config("toolbox_config")["tool_mapping"]
    if any(tool != manifest["tool_name"] and value["env"] == name for tool, value in registrations.items()):
        raise ValueError("Automated setup requires an environment dedicated to this algorithm")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(output)
    python = env_manager.get_env_python(name)
    if python is None:
        command = ["conda", "create", "-y", "-n", name, "python=" + config["python_version"], *config.get("conda_packages", [])]
        for channel in config.get("channels", []):
            command.extend(["-c", channel])
        subprocess.run(command, check=True, env=environment)
        python = env_manager.get_env_python(name)
    if python is None:
        raise RuntimeError("The new algorithm environment could not be located")
    requirements = config.get("pip_packages", [])
    if requirements:
        subprocess.run([python, "-m", "pip", "install", *requirements], check=True, env=environment)
    for command_text in config.get("post_install_commands", []):
        command = shlex.split(command_text)
        if command[0] == "pip":
            command = [python, "-m", "pip", *command[1:]]
        elif command[0] == "python":
            command[0] = python
        subprocess.run(command, check=True, cwd=repository_root(), env=environment)
    subprocess.run([python, "-m", "pip", "check"], check=True, env=environment)
    installed = subprocess.check_output([python, "-m", "pip", "freeze"], text=True, env=environment)
    (output / "packages.txt").write_text(installed, encoding="utf-8")
    write_json(output / "environment.json", {"name": name, "python": python, "requirements": requirements})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and verify an agent-driven symbolic regression integration")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Pin and inspect an upstream source checkout")
    prepare.add_argument("--source-url", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--output-root", required=True)
    for command in ("validate", "setup", "accept", "_worker"):
        child = commands.add_parser(command)
        child.add_argument("--manifest", required=True)
        child.add_argument("--output-root", required=True)
        if command in ("accept", "_worker"):
            child.add_argument("--stage", choices=("smoke", "budget", "core50", "all"), default="smoke")
            child.add_argument("--dataset-dir")
            child.add_argument("--datasets-root")
            child.add_argument("--workers", type=int, default=1)
            child.add_argument("--tmux", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = inspect_source(args.source_url, args.revision, args.output_root)
        task = Path(args.output_root) / "agent_task.md"
        task.write_text(
            "使用 $sr-tool-onboarder 接入以下真实符号回归源码。\n\n"
            + "来源：" + report["url"] + "\n版本：" + report["revision"]
            + "\n本地目录：" + report["checkout"]
            + "\n源码清单：" + str(Path(args.output_root) / "source-review.json")
            + "\n\n阅读真实训练入口，使用文件编辑工具实现包装器、注册配置和 integration manifest。"
            + "运行 sim-onboard validate、setup、accept；记录原生预测、序列化、预算与逐分钟证据。"
            + "验收失败时根据真实日志修复；完成报告必须引用实际验收输出。\n",
            encoding="utf-8",
        )
        print(json.dumps({"source_review": str(Path(args.output_root) / "source-review.json"), "agent_task": str(task)}, ensure_ascii=False))
        return
    manifest = load_manifest(args.manifest)
    if args.command == "setup":
        setup_environment(args.manifest, args.output_root)
        return
    if args.command == "validate":
        from .acceptance import validate_integration

        report = validate_integration(args.manifest)
        write_json(Path(args.output_root) / "structure.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    if args.command == "accept":
        python = env_manager.get_env_python(manifest["environment"]["name"])
        if python is None:
            raise RuntimeError("Run sim-onboard setup to create the algorithm environment")
        output = Path(args.output_root).resolve()
        output.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["TMPDIR"] = str(output)
        environment["PYTHONPATH"] = str(repository_root())
        command = [python, "-m", "scientific_intelligent_modelling.onboarding.cli", "_worker",
                   "--manifest", str(Path(args.manifest).resolve()), "--output-root", str(output),
                   "--stage", args.stage, "--workers", str(args.workers)]
        if args.dataset_dir:
            command.extend(["--dataset-dir", str(Path(args.dataset_dir).resolve())])
        if args.datasets_root:
            command.extend(["--datasets-root", str(Path(args.datasets_root).resolve())])
        if args.tmux:
            command.append("--tmux")
        subprocess.run(command, check=True, cwd=repository_root(), env=environment)
        return
    from .acceptance import run_acceptance

    report = run_acceptance(args.manifest, args.output_root, args.stage, args.dataset_dir, args.datasets_root, args.workers, args.tmux)
    print(json.dumps({"passed": report["passed"], "report": str(Path(args.output_root) / "acceptance.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
