import argparse
import shlex
import subprocess
from pathlib import Path

from scientific_intelligent_modelling.onboarding.manifest import load_manifest, repository_root, sha256_file, write_json
from scientific_intelligent_modelling.onboarding.source import verify_source


def main():
    parser = argparse.ArgumentParser(description="Deploy a pinned integration branch into a new remote checkout")
    parser.add_argument("--host", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository-url", default="https://github.com/scientific-intelligent-modelling/SymbolicArena.git")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-conda-prefix", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--environment-archive", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    manifest = load_manifest(args.manifest)
    source = verify_source(manifest, args.source)
    remote_root = args.remote_root.rstrip("/")
    checkout = remote_root + "/checkout"
    remote_source = remote_root + "/upstream"
    environment = args.remote_conda_prefix.rstrip("/") + "/envs/" + manifest["environment"]["name"]
    python = environment + "/bin/python"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host]

    def remote(command, capture=False):
        result = subprocess.run(ssh + [shlex.join(command)], check=True, text=True, stdout=subprocess.PIPE if capture else None, timeout=180)
        return result.stdout.strip() if capture else None

    if subprocess.run(ssh + [shlex.join(["test", "-e", remote_root])], timeout=30).returncode == 0:
        raise ValueError("Remote task directory already exists; select a new isolated task directory")
    remote(["mkdir", "-p", remote_root, remote_root + "/temporary"])
    remote(["git", "clone", "--depth", "1", "--branch", args.branch, args.repository_url, checkout])
    revision = remote(["git", "-C", checkout, "rev-parse", "HEAD"], capture=True)
    if revision != args.revision:
        raise ValueError("Remote branch revision differs from the requested commit")
    subprocess.run(["rsync", "-a", "--protect-args", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10", str(source) + "/", args.host + ":" + remote_source + "/"], check=True, timeout=180)
    existing_environment = subprocess.run(ssh + [shlex.join(["test", "-x", python])], timeout=30).returncode == 0
    if not existing_environment:
        archive = Path(args.environment_archive).resolve()
        remote_archive = remote_root + "/environment.tar.gz"
        subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(archive), args.host + ":" + remote_archive], check=True, timeout=900)
        remote(["mkdir", "-p", environment])
        remote(["tar", "-xzf", remote_archive, "-C", environment])
        remote([python, environment + "/bin/conda-unpack"])
    env = [
        "env", manifest["source"]["environment_variable"] + "=" + remote_source,
        "TMPDIR=" + remote_root + "/temporary", "PYTHONPATH=" + checkout,
        "SIM_ONBOARD_TMUX_SOCKET=" + remote_root + "/t.sock",
    ]
    remote(env + [python, "-m", "pip", "install", "--no-build-isolation", "-e", checkout])
    remote(env + [python, "-m", "pip", "check"])
    relative_manifest = Path(args.manifest).resolve().relative_to(repository_root())
    remote_manifest = checkout + "/" + str(relative_manifest)
    remote(env + [
        python, "-m", "scientific_intelligent_modelling.onboarding.cli", "validate",
        "--manifest", remote_manifest, "--output-root", remote_root + "/structure",
    ])
    session = "onboard_" + manifest["tool_name"] + "_" + args.revision[:8]
    result_root = remote_root + "/core50"
    launcher = env + [
        "bash", checkout + "/tools/sr_onboarder/run_core50.sh", python, remote_manifest,
        args.datasets_root, result_root, str(args.workers), remote_root + "/core50.log",
    ]
    remote(["tmux", "-S", remote_root + "/t.sock", "new-session", "-d", "-s", session, shlex.join(launcher)])
    report = {
        "host": args.host,
        "remote_root": remote_root,
        "checkout": checkout,
        "revision": revision,
        "source": remote_source,
        "environment": environment,
        "environment_archive_sha256": sha256_file(Path(args.environment_archive)),
        "environment_already_present": existing_environment,
        "source_revision": manifest["source"]["revision"],
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "datasets_root": args.datasets_root,
        "workers": args.workers,
        "session": session,
        "acceptance_report": result_root + "/acceptance.json",
        "log": remote_root + "/core50.log",
        "completed": False,
    }
    write_json(Path(args.output_root) / "deployment.json", report)
    print("Started remote Core50 acceptance in " + session)


if __name__ == "__main__":
    main()
