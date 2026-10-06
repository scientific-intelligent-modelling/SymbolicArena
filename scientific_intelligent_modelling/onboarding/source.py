from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

from .manifest import sha256_file, write_json


def inspect_source(url: str, revision: str, output_root: str | Path) -> dict:
    root = Path(output_root).resolve()
    checkout = root / "upstream"
    if not checkout.exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", url, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "switch", "--detach", revision], check=True)
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError("Source checkout revision does not match the requested commit")
    files = subprocess.check_output(["git", "-C", str(checkout), "ls-files"], text=True).splitlines()
    imported = set()
    for relative in files:
        path = checkout / relative
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(item.name.split(".")[0] for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
    report = {
        "schema": "symbolicarena-source-review-v1",
        "url": url,
        "revision": head,
        "checkout": str(checkout),
        "files": {relative: sha256_file(checkout / relative) for relative in files if (checkout / relative).is_file()},
        "imports": sorted(imported),
        "licenses": [name for name in files if Path(name).name.upper().startswith(("LICENSE", "COPYING"))],
        "code_edits": "agent_apply_patch",
    }
    write_json(root / "source-review.json", report)
    return report


def verify_source(manifest: dict, source_path: str | Path | None = None) -> Path:
    variable = manifest["source"]["environment_variable"]
    configured = source_path or os.environ.get(variable)
    if not configured:
        raise ValueError("Set " + variable + " to the pinned upstream checkout")
    source = Path(configured).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["source"]["revision"]:
        raise ValueError("Upstream commit differs from the integration manifest")
    for name, expected in manifest["source"]["files"].items():
        path = source / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("Source checksum mismatch: " + str(path))
    return source
