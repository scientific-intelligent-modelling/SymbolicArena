from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "tool_name", "wrapper_class", "source", "environment", "parameters", "core50"],
    "properties": {
        "schema": {"const": "symbolicarena-integration-v1"},
        "tool_name": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "wrapper_class": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$"},
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["url", "revision", "mode", "environment_variable", "files", "license"],
            "properties": {
                "url": {"type": "string", "pattern": "^https://"},
                "revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "mode": {"enum": ["external"]},
                "environment_variable": {"type": "string", "pattern": "^SIM_[A-Z0-9_]+_SOURCE$"},
                "files": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                "license": {"type": "string", "minLength": 1},
                "patch": {"type": "string"},
            },
        },
        "environment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "python_version"],
            "properties": {
                "name": {"type": "string", "pattern": "^sim_[a-z0-9_]+$"},
                "python_version": {"type": "string"},
            },
        },
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["defaults", "smoke", "budget", "core50"],
            "properties": {key: {"type": "object"} for key in ("defaults", "smoke", "budget", "core50")},
        },
        "core50": {
            "type": "object",
            "additionalProperties": False,
            "required": ["selection", "sha256", "count", "seeds"],
            "properties": {
                "selection": {"type": "string"},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "count": {"const": 50},
                "seeds": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer"}},
            },
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    Draft202012Validator(SCHEMA).validate(value)
    for relative in value["source"]["files"]:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Source file must be a relative path: " + relative)
    return value


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(output)
