from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOLS = "qlattice drsr dso e2esr fepysr gplearn imcts jaxsr llmsr pyoperon pysr ragsr symbolfit tpsr udsr"
HOSTS = "iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29"
HARVEST_ROOTS = " \\\n+  ".join(
    f"--experiment-root benchmark-runs/compliance/latest/smoke/remote-experiments/{host}"
    for host in HOSTS.split()
)


def write_stage1_queue_commands(*, batch_dir: Path) -> list[Path]:
    deploy_dir = batch_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    scripts = [
        (
            "01_preflight_from_iaaccn22.sh",
            _preflight_script(),
        ),
        (
            "02_smoke_dispatch_from_iaaccn22.sh",
            _smoke_dispatch_script(),
        ),
        (
            "03_full_dispatch_from_iaaccn22.sh",
            _full_dispatch_script(),
        ),
    ]
    paths: list[Path] = []
    for filename, content in scripts:
        path = deploy_dir / filename
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
        paths.append(path)
    return paths


def _common_header() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/home/zhangziwen/workplace/scientific-intelligent-modelling
cd "$REMOTE_ROOT"
BATCH_ID="$(basename "$(readlink -f benchmark-runs/compliance/latest)")"
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1
"""


def _preflight_script() -> str:
    return (
        _common_header()
        + f"""
python check/run_e1_candidate200_12alg_load_queue.py \\
  --batch-name "${{BATCH_ID}}_preflight" \\
  --source-csv benchmark-runs/compliance/latest/queues/smoke_2datasets_source.csv \\
  --expected-rows 2 \\
  --queue-root benchmark-runs/compliance/latest/queues/load_queue_smoke \\
  --params-root exp-planning/02.E1选择验证/generated/params \\
  --hosts {HOSTS} \\
  --tools {TOOLS} \\
  --seeds 520 \\
  --controller-host iaaccn22 \\
  --use-internal-ips \\
  --preflight-only \\
  --preflight-host-timeout 240 \\
  --preflight-report benchmark-runs/compliance/latest/deploy/preflight_15alg_smoke_from_iaaccn22.json \\
  2>&1 | tee benchmark-runs/compliance/latest/deploy/preflight_15alg_smoke_from_iaaccn22.log

python benchmark-control/compliance/launchers/check_preflight_report.py \\
  --batch-dir benchmark-runs/compliance/latest \\
  --report benchmark-runs/compliance/latest/deploy/preflight_15alg_smoke_from_iaaccn22.json \\
  --expected-hosts {HOSTS}
"""
    )


def _smoke_dispatch_script() -> str:
    return (
        _common_header()
        + f"""
python check/run_e1_candidate200_12alg_load_queue.py \\
  --batch-name "${{BATCH_ID}}_smoke" \\
  --source-csv benchmark-runs/compliance/latest/queues/smoke_2datasets_source.csv \\
  --expected-rows 2 \\
  --queue-root benchmark-runs/compliance/latest/queues/load_queue_smoke \\
  --params-root exp-planning/02.E1选择验证/generated/params \\
  --hosts {HOSTS} \\
  --tools {TOOLS} \\
  --seeds 520 \\
  --controller-host iaaccn22 \\
  --use-internal-ips \\
  --session-prefix compliance_smoke_ \\
  --host-session-count-prefix compliance_smoke_ \\
  --poll-seconds 60 \\
  2>&1 | tee benchmark-runs/compliance/latest/deploy/smoke_dispatch_from_iaaccn22.log

python benchmark-control/compliance/launchers/prepare_smoke_batch.py \\
  --batch-dir benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/collect_remote_batch.py \\
  --batch-dir benchmark-runs/compliance/latest/smoke \\
  --batch-id "${{BATCH_ID}}_smoke" \\
  --hosts {HOSTS} \\
  --controller-host iaaccn22 \\
  --use-internal-ips

python benchmark-control/compliance/launchers/harvest_batch.py \\
  --batch-dir benchmark-runs/compliance/latest/smoke \\
  {HARVEST_ROOTS}

python benchmark-control/compliance/launchers/audit_batch.py \\
  --batch-dir benchmark-runs/compliance/latest/smoke \\
  --write-heartbeat \\
  --write-rerun \\
  --round-id 1

python benchmark-control/compliance/launchers/check_audit_success.py \\
  --batch-dir benchmark-runs/compliance/latest/smoke \\
  --expected-total-tasks 30
"""
    )


def _full_dispatch_script() -> str:
    return (
        _common_header()
        + f"""
python check/run_e1_candidate200_12alg_load_queue.py \\
  --batch-name "${{BATCH_ID}}" \\
  --source-csv benchmark-runs/compliance/latest/queues/ssr50_source.csv \\
  --expected-rows 50 \\
  --queue-root benchmark-runs/compliance/latest/queues/load_queue_full \\
  --params-root exp-planning/02.E1选择验证/generated/params \\
  --hosts {HOSTS} \\
  --tools {TOOLS} \\
  --seeds 520 \\
  --controller-host iaaccn22 \\
  --use-internal-ips \\
  --session-prefix compliance_1h_ \\
  --host-session-count-prefix compliance_1h_ \\
  --poll-seconds 60 \\
  2>&1 | tee benchmark-runs/compliance/latest/deploy/full_dispatch_from_iaaccn22.log
"""
    )


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default="benchmark-runs/compliance/latest")
    args = parser.parse_args()
    paths = write_stage1_queue_commands(batch_dir=_resolve_repo_path(args.batch_dir))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
