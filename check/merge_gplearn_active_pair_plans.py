"""Select one frozen Opus pair plan per current gplearn run or seed pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(row: dict) -> tuple:
    req = row["request"]
    prefix = (row["condition"], req["dataset_id"])
    return prefix + ((int(req["seed"]),) if row["task_type"] == "equivalence" else
                     (int(req["seed_left"]), int(req["seed_right"])))


def _index(items: list[dict], label: str) -> dict[tuple, dict]:
    result = {}
    for row in items:
        key = identity(row)
        if key in result:
            raise ValueError(f"duplicate {label} identity: {key}")
        result[key] = row
    return result


def _frozen_by_source(directory: Path) -> dict[str, tuple[dict, str]]:
    result = {}
    for path in directory.glob("*.json"):
        raw = path.read_bytes()
        row = json.loads(raw)
        key = row.get("source_evaluation_key")
        if not isinstance(key, str) or key in result:
            raise ValueError(f"duplicate or missing frozen source key: {path}")
        result[key] = (row, hashlib.sha256(raw).hexdigest())
    return result


def select_active(*, base_plan: Path, retry_plan: Path, base_frozen_dir: Path,
                  retry_frozen_dir: Path, output: Path, expected_rows: int = 450) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    base_rows, retry_rows = rows(base_plan), rows(retry_plan)
    base, retry = _index(base_rows, "base"), _index(retry_rows, "retry")
    if len(base) != expected_rows or not set(retry) <= set(base):
        raise ValueError(f"pair plans do not cover the frozen {expected_rows} identities")
    base_frozen = _frozen_by_source(base_frozen_dir)
    retry_frozen = _frozen_by_source(retry_frozen_dir)
    selected = []
    provenance = []
    for key, original in sorted(base.items()):
        extra = retry.get(key)
        if extra is not None:
            if original["evaluation_key"] in base_frozen:
                raise ValueError(f"retry would supersede an already frozen base judgment: {key}")
            old_req, new_req = original["request"], extra["request"]
            stripped = {name: value for name, value in new_req.items()
                        if name not in {"prior_exhausted_source_evaluation_key",
                                        "prior_exhausted_model_evaluation_key"}}
            if (stripped != old_req or extra["dependencies"] != original["dependencies"] or
                    new_req.get("prior_exhausted_source_evaluation_key") != original["evaluation_key"]):
                raise ValueError(f"retry dependency/input drift: {key}")
            active, source, version = extra, retry_frozen, "json_retry"
        else:
            active, source, version = original, base_frozen, "base"
        item = source.get(active["evaluation_key"])
        if item is None:
            raise ValueError(f"active judgment not frozen: {key}")
        frozen, response_sha = item
        if (frozen.get("status") != "frozen" or frozen.get("model") != "claude-opus-4-8"
            or frozen.get("source_input_hash") != active["input_hash"]
            or frozen.get("request") != active["request"]
            or frozen.get("plan_dependencies") != active["dependencies"]
            or frozen.get("semantic_validation", {}).get("status") != "promotable"):
            raise ValueError(f"active frozen response binding invalid: {key}")
        selected.append(active)
        provenance.append({"identity": list(key), "version": version,
                           "source_evaluation_key": active["evaluation_key"],
                           "model_evaluation_key": frozen["evaluation_key"],
                           "response_sha256": response_sha,
                           "terminal_evidence_sha256": active["request"]["evidence_hash"],
                           "base_source_evaluation_key": original["evaluation_key"]})
    output.mkdir(parents=True)
    phase = selected[0]["task_type"]
    path = output / f"{phase}_active_plan.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    provenance_path = output / f"{phase}_selection.jsonl"
    with provenance_path.open("w", encoding="utf-8") as handle:
        for row in provenance:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    report = {"schema_version": "gplearn_active_pair_plan.v1", "phase": phase,
              "active_rows": len(selected), "base_rows": len(base)-len(retry),
              "retry_rows": len(retry), "base_plan_sha256": sha(base_plan),
              "retry_plan_sha256": sha(retry_plan),
              "active_plan_sha256": sha(path), "selection_sha256": sha(provenance_path)}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--retry-plan", type=Path, required=True)
    parser.add_argument("--base-frozen-dir", type=Path, required=True)
    parser.add_argument("--retry-frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_active(base_plan=args.base_plan, retry_plan=args.retry_plan,
                                   base_frozen_dir=args.base_frozen_dir,
                                   retry_frozen_dir=args.retry_frozen_dir,
                                   output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
