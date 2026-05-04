#!/usr/bin/env python3
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
