#!/usr/bin/env python3
"""在 controller 上审计分布式 native-EFF 重跑的 result 与 180 分钟证据。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REMOTE_CODE = r'''
import glob, hashlib, json, math, os, sys
root=sys.argv[1]
tasks=json.loads(sys.argv[2])
out=[]
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
 return h.hexdigest()
for task in tasks:
 tid=task["task_id"]
 base_pattern=root+"/**/tasks/"+tid+"/**"
 minute_candidates=glob.glob(base_pattern+"/progress/minute_*.json",recursive=True)
 by_name={}
 for path in minute_candidates:
  suffix=path.split("/tasks/"+tid+"/",1)[-1]
  if "/experiments/" in suffix: continue
  name=os.path.basename(path)
  if name not in by_name or len(path)<len(by_name[name]): by_name[name]=path
 result_candidates=glob.glob(base_pattern+"/result.json",recursive=True)
 result_candidates=[p for p in result_candidates if "/experiments/" not in p.split("/tasks/"+tid+"/",1)[-1]]
 result_path=min(result_candidates,key=len) if result_candidates else None
 minutes=[]; contract_errors=[]; candidate_count=0; heartbeat_count=0; excluded_post_budget=[]
 for name,path in sorted(by_name.items()):
  try: minute_number=int(name.removeprefix("minute_").removesuffix(".json"))
  except Exception: minute_number=-1
  if minute_number < 1 or minute_number > 180:
   excluded_post_budget.append({"minute":name,"path":path,"sha256":sha(path)})
   continue
  try: x=json.load(open(path))
  except Exception as e:
   contract_errors.append(name+":parse:"+repr(e)); continue
  available=bool(x.get("candidate_available"))
  if available: candidate_count+=1
  else: heartbeat_count+=1
  tool=task["tool"]
  condition=task["noise_tag"]
  if x.get("condition")!=condition: contract_errors.append(name+":condition")
  if available and tool=="e2esr":
   ok=(x.get("internal_objective")=="decoder_length_normalized_log_likelihood" and x.get("internal_objective_direction")=="max" and x.get("internal_objective_value") is not None and bool(x.get("candidate_sha256")))
   if not ok: contract_errors.append(name+":e2esr_native")
  elif available and tool=="imcts":
   ok=(x.get("internal_objective")=="native_reward" and x.get("internal_objective_direction")=="max" and x.get("internal_objective_value") is not None and bool(x.get("expression_vector")) and bool(x.get("candidate_sha256")))
   if not ok: contract_errors.append(name+":imcts_native")
  elif available and tool=="pysr":
   ok=(x.get("internal_objective")=="hof_loss" and x.get("internal_objective_direction")=="min" and x.get("internal_objective_value") is not None and bool(x.get("candidate_original_equation")) and bool(x.get("candidate_sha256")))
   if not ok: contract_errors.append(name+":pysr_native")
  elif available and tool in {"gplearn","pyoperon"}:
   try: loss=float(x.get("source_loss")); ok=math.isfinite(loss)
   except Exception: ok=False
   if not ok: contract_errors.append(name+":"+tool+"_native_loss")
  if not available and tool in {"e2esr","imcts","pysr"}:
   expected={"e2esr":("decoder_length_normalized_log_likelihood","max"),"imcts":("native_reward","max"),"pysr":("hof_loss","min")}[tool]
   if x.get("internal_objective")!=expected[0] or x.get("internal_objective_direction")!=expected[1] or not x.get("native_objective_unavailable"):
    contract_errors.append(name+":heartbeat_contract")
  minutes.append({"minute":name,"path":path,"sha256":sha(path),"record_type":x.get("record_type"),"candidate_available":available})
 result=None
 if result_path:
  try: result=json.load(open(result_path))
  except Exception as e: contract_errors.append("result_parse:"+repr(e))
 evidence_paths=[]
 if task["tool"]=="llmsr":
  evidence_paths=glob.glob(base_pattern+"/best_history/*.json",recursive=True)+glob.glob(base_pattern+"/samples/top*.json",recursive=True)
  evidence_paths=sorted(set(evidence_paths))
 out.append({
  "task_id":tid,"tool":task["tool"],"condition":task["noise_tag"],"seed":task["seed"],
  "result_path":result_path,"result_sha256":sha(result_path) if result_path else None,
  "result_status":result.get("status") if isinstance(result,dict) else None,
  "result_has_equation":bool(result.get("equation")) if isinstance(result,dict) else False,
  "minute_count":len(minutes),"minute_evidence":minutes,"candidate_count":candidate_count,
  "heartbeat_count":heartbeat_count,"contract_errors":contract_errors,
  "excluded_post_budget":excluded_post_budget,
  "llmsr_evidence_count":len(evidence_paths),
  "llmsr_evidence_sha256":hashlib.sha256(json.dumps([(p,sha(p)) for p in evidence_paths],sort_keys=True,separators=(",",":")).encode()).hexdigest() if evidence_paths else None})
print(json.dumps(out))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for task in state["tasks"].values():
        grouped[task["assigned_host"]].append(task)
    rows = []
    host_errors = []
    for host, tasks in sorted(grouped.items()):
        command = ["python", "-c", REMOTE_CODE, args.experiment_root, json.dumps(tasks)]
        if host != "iaaccn22":
            command = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "10.10.100." + host.replace("iaaccn", ""), shlex.join(command),
            ]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=300)
        if proc.returncode:
            host_errors.append({"host": host, "stderr": proc.stderr[-2000:]})
            continue
        rows.extend(json.loads(proc.stdout))
    if len(rows) != args.expected:
        raise SystemExit(f"audit rows {len(rows)} != {args.expected}; hosts={host_errors}")
    unresolved = []
    auditable_zero = []
    for row in rows:
        expected_minutes = {f"minute_{minute:04d}.json" for minute in range(1, 181)}
        observed = {item["minute"] for item in row["minute_evidence"]}
        if observed != expected_minutes:
            row["contract_errors"].append(
                f"minute_grid_missing={len(expected_minutes-observed)},extra={len(observed-expected_minutes)}"
            )
        if row["result_status"] not in {"ok", "timed_out"}:
            row["contract_errors"].append(f"result_status={row['result_status']}")
        if row["tool"] == "llmsr" and row["llmsr_evidence_count"] == 0:
            row["contract_errors"].append("llmsr_native_evidence_missing")
        if row["candidate_count"] == 0:
            row["eff_resolution"] = "auditable_no_valid_incumbent"
            row["m_eff"] = 0.0
            row["q_star"] = 0.0
            auditable_zero.append(
                {
                    "task_id": row["task_id"],
                    "reason": "no_finite_native_incumbent",
                    "m_eff": 0.0,
                    "q_star": 0.0,
                }
            )
        else:
            row["eff_resolution"] = "native_incumbent_trajectory"
    rows.sort(key=lambda row: row["task_id"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_root / "source_evidence.jsonl"
    with evidence_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.output_root / "source_manifest.csv"
    fields = [
        "task_id", "tool", "condition", "seed", "result_path", "result_sha256",
        "result_status", "result_has_equation", "minute_count", "candidate_count",
        "heartbeat_count", "llmsr_evidence_count", "llmsr_evidence_sha256",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    error_rows = [row for row in rows if row["contract_errors"]]
    report = {
        "status": "passed" if not error_rows and not unresolved and not host_errors else "requires_review",
        "run_count": len(rows),
        "state_counts": dict(Counter(task["state"] for task in state["tasks"].values())),
        "tool_counts": dict(Counter(row["tool"] for row in rows)),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "all_have_180_minutes": all(row["minute_count"] == 180 for row in rows),
        "excluded_post_budget_snapshot_count": sum(
            len(row.get("excluded_post_budget", [])) for row in rows
        ),
        "contract_error_count": len(error_rows),
        "contract_error_preview": [
            {"task_id": row["task_id"], "errors": row["contract_errors"][:5]}
            for row in error_rows[:10]
        ],
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "auditable_no_valid_incumbent_count": len(auditable_zero),
        "auditable_no_valid_incumbent": auditable_zero,
        "host_errors": host_errors,
        "source_evidence": str(evidence_path),
        "source_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    report_path = args.output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
