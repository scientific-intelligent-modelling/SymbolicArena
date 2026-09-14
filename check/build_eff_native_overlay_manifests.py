#!/usr/bin/env python3
"""验证 host raw bundles 并生成 clean/noise EFF-only overlay manifests。"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "AAAI_experiments/stage5_metric_calculation_0831/work/final_release_20260913/release_v2"
BUNDLES = RELEASE / "eff_native_raw_bundles_20260914/host_bundles"
OUTPUTS = {
    "clean": RELEASE / "eff_native_missing165_20260913/validation",
    "noise": RELEASE / "eff_native_noise_missing309_20260914/validation",
}
EXPECTED = {"clean": 165, "noise": 309}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    grouped = defaultdict(list)
    bundle_hashes = {}
    record_count = 0
    for bundle in sorted(BUNDLES.glob("iaaccn*.jsonl.gz")):
        bundle_hashes[bundle.name] = sha_bytes(bundle.read_bytes())
        with gzip.open(bundle, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if sha_bytes(row["raw_text"].encode("utf-8")) != row["sha256"]:
                    raise ValueError(f"{bundle}:{line_number} raw SHA 漂移")
                row["bundle_path"] = str(bundle.relative_to(REPO))
                row["bundle_sha256"] = bundle_hashes[bundle.name]
                grouped[row["task_id"]].append(row)
                record_count += 1
    if len(grouped) != 474 or record_count != 474 * 181:
        raise ValueError(f"bundle scope 漂移: tasks={len(grouped)}, records={record_count}")

    manifests = {"clean": [], "noise": []}
    for task_id, rows in grouped.items():
        results = [row for row in rows if row["record_type"] == "result"]
        snapshots = sorted(
            (row for row in rows if row["record_type"] == "snapshot"),
            key=lambda row: row["minute"],
        )
        if len(results) != 1 or [row["minute"] for row in snapshots] != list(range(1, 181)):
            raise ValueError(f"{task_id}: result/snapshot grid 漂移")
        result = results[0]
        result_payload = json.loads(result["raw_text"])
        candidate_count = 0
        for snapshot in snapshots:
            payload = json.loads(snapshot["raw_text"])
            candidate_count += int(bool(payload.get("candidate_available")))
            if payload.get("condition") != result["condition"]:
                raise ValueError(f"{task_id}: snapshot condition 漂移")
        scope = "clean" if result["condition"] == "clean" else "noise"
        manifest = {
            "schema_version": "eff_native_overlay_run.v1",
            "replacement_scope": "eff_only",
            "base_final_preserved": True,
            "terminal_result_for_eff_audit_only": True,
            "task_id": task_id,
            "algorithm": result["tool"],
            "dataset_id": result_payload.get("dataset"),
            "condition": result["condition"],
            "seed": result["seed"],
            "source_host": result["source_host"],
            "source_batch": (
                "eff_native_missing165_20260913"
                if scope == "clean"
                else "eff_native_noise_missing309_20260914"
            ),
            "result_path": result["source_path"],
            "result_sha256": result["sha256"],
            "result_status": result_payload.get("status"),
            "bundle_path": result["bundle_path"],
            "bundle_sha256": result["bundle_sha256"],
            "snapshot_count": len(snapshots),
            "snapshot_bindings": [
                {
                    "minute": row["minute"],
                    "source_path": row["source_path"],
                    "sha256": row["sha256"],
                }
                for row in snapshots
            ],
            "candidate_count": candidate_count,
            "eff_resolution": (
                "native_incumbent_trajectory"
                if candidate_count
                else "auditable_no_valid_incumbent"
            ),
            "q_star": None if candidate_count else 0.0,
            "m_eff": None if candidate_count else 0.0,
            "candidate_selection_uses_id_ood": False,
        }
        manifests[scope].append(manifest)

    for scope, rows in manifests.items():
        rows.sort(key=lambda row: row["task_id"])
        if len(rows) != EXPECTED[scope]:
            raise ValueError(f"{scope}: {len(rows)} != {EXPECTED[scope]}")
        output = OUTPUTS[scope]
        path = output / "overlay_manifest.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "schema_version": "eff_native_overlay_manifest.v1",
            "status": "passed",
            "replacement_scope": "eff_only",
            "base_final_preserved": True,
            "terminal_result_for_eff_audit_only": True,
            "run_count": len(rows),
            "condition_counts": dict(Counter(row["condition"] for row in rows)),
            "algorithm_counts": dict(Counter(row["algorithm"] for row in rows)),
            "auditable_no_valid_incumbent_count": sum(
                row["eff_resolution"] == "auditable_no_valid_incumbent" for row in rows
            ),
            "overlay_jsonl": str(path.relative_to(REPO)),
            "overlay_sha256": sha_bytes(path.read_bytes()),
            "bundle_sha256": bundle_hashes,
        }
        summary_path = output / "overlay_manifest.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksums = output / "SHA256SUMS"
        artifacts = [path, summary_path, output / "source_evidence.jsonl", output / "source_manifest.csv", output / "validation_report.json"]
        checksums.write_text(
            "".join(f"{sha_bytes(item.read_bytes())}  {item.name}\n" for item in artifacts),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
