from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOLS = "gplearn pyoperon pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit"
HOSTS = "iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29"
SEEDS = "520 521 522"
NOISE_SIGMAS = "0 0.01 0.05"
EXPECTED_SMOKE_TASKS = 234
EXPECTED_FULL_TASKS = 5850
REMOTE_ROOT = "/home/zhangziwen/workplace/scientific-intelligent-modelling"
HUB = "iaaccn22"
INTERNAL_TARGETS = "10.10.100.23 10.10.100.24 10.10.100.25 10.10.100.26 10.10.100.27 10.10.100.28 10.10.100.29"


def _harvest_roots(base: str) -> str:
    return " \\\n  ".join(f"--experiment-root {base}/{host}" for host in HOSTS.split())


SMOKE_HARVEST_ROOTS = _harvest_roots("benchmark-runs/formal24h/latest/smoke/remote-experiments")
FULL_HARVEST_ROOTS = _harvest_roots("benchmark-runs/formal24h/latest/remote-experiments")


def write_full24h_queue_commands(*, batch_dir: Path) -> list[Path]:
    deploy_dir = batch_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    scripts = [
        ("00_sync_code_and_batch_to_iaaccn22.sh", _sync_script(batch_dir=batch_dir)),
        ("01_preflight_from_iaaccn22.sh", _preflight_script()),
        ("02_smoke_dispatch_from_iaaccn22.sh", _smoke_dispatch_script()),
        ("03_full_dispatch_from_iaaccn22.sh", _full_dispatch_script()),
    ]
    paths: list[Path] = []
    for filename, content in scripts:
        path = deploy_dir / filename
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
        paths.append(path)
    return paths


def _common_header() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT={REMOTE_ROOT}
cd "$REMOTE_ROOT"
BATCH_ID="$(basename "$(readlink -f benchmark-runs/formal24h/latest)")"
export SIM_QUEUE_CONTROLLER_IS_LOCAL=1
"""


def _sync_script(*, batch_dir: Path) -> str:
    batch_id = batch_dir.resolve().name
    batch_rel = _repo_relative(batch_dir)
    sync_items = [
        "check/run_e1_candidate200_12alg_load_queue.py",
        "check/launch_e1_benchmark.py",
        "scientific_intelligent_modelling/",
        "benchmark-control/compliance/",
        "exp-planning/02.E1选择验证/generated/params/",
        f"{batch_rel}/",
    ]
    sync_items_array = "\n".join(f'  "{item}"' for item in sync_items)
    rsync_filters_array = "\n".join(
        f'  "{item}"'
        for item in (
            "--exclude=.git/",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            "--exclude=*.pyo",
        )
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
BATCH_ID="{batch_id}"

RSYNC_FILTERS=(
{rsync_filters_array}
)

SYNC_ITEMS=(
{sync_items_array}
)

echo "[sync] local -> {HUB}"
rsync -aR "${{RSYNC_FILTERS[@]}}" "${{SYNC_ITEMS[@]}}" {HUB}:{REMOTE_ROOT}/
ssh -o BatchMode=yes -o ConnectTimeout=10 {HUB} \\
  "cd {REMOTE_ROOT} && ln -sfn \\"$BATCH_ID\\" benchmark-runs/formal24h/latest"

echo "[sync] {HUB} -> iaaccn23~29 over internal network"
ssh -o BatchMode=yes -o ConnectTimeout=10 {HUB} 'bash -s' <<'REMOTE_SYNC'
set -euo pipefail
cd {REMOTE_ROOT}
BATCH_ID="{batch_id}"
RSYNC_FILTERS=(
{rsync_filters_array}
)
SYNC_ITEMS=(
{sync_items_array}
)
failures=0
for target in {INTERNAL_TARGETS}; do
  echo "[sync] iaaccn22 -> $target"
  if ! rsync -aR "${{RSYNC_FILTERS[@]}}" "${{SYNC_ITEMS[@]}}" "$target":{REMOTE_ROOT}/; then
    echo "SYNC_FAIL $target"
    failures=$((failures + 1))
    continue
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" \\
    "cd {REMOTE_ROOT} && ln -sfn \\"$BATCH_ID\\" benchmark-runs/formal24h/latest"; then
    echo "LINK_FAIL $target"
    failures=$((failures + 1))
    continue
  fi
done
if [[ "$failures" -gt 0 ]]; then
  echo "[sync] $failures internal target(s) failed"
  exit 1
fi
REMOTE_SYNC

echo "[sync] done"
"""


def _preflight_script() -> str:
    return (
        _common_header()
        + f"""
python check/run_e1_candidate200_12alg_load_queue.py \\
  --batch-name "${{BATCH_ID}}_preflight" \\
  --source-csv benchmark-runs/formal24h/latest/queues/smoke_2datasets_source.csv \\
  --expected-rows 2 \\
  --queue-root benchmark-runs/formal24h/latest/queues/load_queue_smoke \\
  --params-root benchmark-runs/formal24h/latest/params_smoke \\
  --hosts {HOSTS} \\
  --tools {TOOLS} \\
  --seeds {SEEDS} \\
  --noise-sigmas {NOISE_SIGMAS} \\
  --controller-host iaaccn22 \\
  --use-internal-ips \\
  --preflight-only \\
  --preflight-host-timeout 240 \\
  --preflight-report benchmark-runs/formal24h/latest/deploy/preflight_13alg_3seed_3noise_from_iaaccn22.json \\
  2>&1 | tee benchmark-runs/formal24h/latest/deploy/preflight_13alg_3seed_3noise_from_iaaccn22.log

python benchmark-control/compliance/launchers/check_preflight_report.py \\
  --batch-dir benchmark-runs/formal24h/latest \\
  --report benchmark-runs/formal24h/latest/deploy/preflight_13alg_3seed_3noise_from_iaaccn22.json \\
  --expected-hosts {HOSTS}
"""
    )


def _preflight_gate_check() -> str:
    return """
python - <<'PY'
import json
from pathlib import Path

path = Path("benchmark-runs/formal24h/latest/preflight/preflight_gate_summary.json")
if not path.exists():
    raise SystemExit(f"preflight gate summary missing: {path}; run deploy/01_preflight_from_iaaccn22.sh first")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("ready_for_smoke") is not True:
    raise SystemExit(f"preflight gate not ready_for_smoke: {payload.get('issues')}")
print(f"preflight gate passed: {path}")
PY
"""


def _smoke_gate_check() -> str:
    return """
python - <<'PY'
import json
from pathlib import Path

path = Path("benchmark-runs/formal24h/latest/smoke/audit/audit_gate_summary.json")
if not path.exists():
    raise SystemExit(f"smoke audit gate summary missing: {path}; run deploy/02_smoke_dispatch_from_iaaccn22.sh first")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("audit_passed") is not True:
    raise SystemExit(f"smoke audit gate failed: {payload.get('issues')}")
print(f"smoke audit gate passed: {path}")
PY
"""


def _smoke_dispatch_script() -> str:
    return (
        _common_header()
        + _preflight_gate_check()
        + f"""
python check/run_e1_candidate200_12alg_load_queue.py \\
  --batch-name "${{BATCH_ID}}_smoke" \\
  --source-csv benchmark-runs/formal24h/latest/queues/smoke_2datasets_source.csv \\
  --expected-rows 2 \\
  --queue-root benchmark-runs/formal24h/latest/queues/load_queue_smoke \\
  --params-root benchmark-runs/formal24h/latest/params_smoke \\
  --hosts {HOSTS} \\
  --tools {TOOLS} \\
  --seeds {SEEDS} \\
  --noise-sigmas {NOISE_SIGMAS} \\
  --controller-host iaaccn22 \\
  --use-internal-ips \\
  --session-prefix formal24h_smoke_ \\
  --host-session-count-prefix formal24h_smoke_ \\
  --poll-seconds 60 \\
  2>&1 | tee benchmark-runs/formal24h/latest/deploy/smoke_dispatch_from_iaaccn22.log

python benchmark-control/compliance/launchers/prepare_smoke_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest

python benchmark-control/compliance/launchers/collect_remote_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest/smoke \\
  --batch-id "${{BATCH_ID}}_smoke" \\
  --hosts {HOSTS} \\
  --controller-host iaaccn22 \\
  --use-internal-ips

python benchmark-control/compliance/launchers/harvest_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest/smoke \\
  {SMOKE_HARVEST_ROOTS}

python benchmark-control/compliance/launchers/audit_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest/smoke \\
  --write-heartbeat \\
  --write-rerun \\
  --round-id 1

python benchmark-control/compliance/launchers/check_audit_success.py \\
  --batch-dir benchmark-runs/formal24h/latest/smoke \\
  --expected-total-tasks {EXPECTED_SMOKE_TASKS}
"""
    )


def _full_dispatch_script() -> str:
    return (
        _common_header()
        + _smoke_gate_check()
        + f"""
python check/run_e1_candidate200_12alg_load_queue.py \\
  --batch-name "${{BATCH_ID}}" \\
  --source-csv benchmark-runs/formal24h/latest/queues/ssr50_source.csv \\
  --expected-rows 50 \\
  --queue-root benchmark-runs/formal24h/latest/queues/load_queue_full \\
  --params-root benchmark-runs/formal24h/latest/params \\
  --hosts {HOSTS} \\
  --tools {TOOLS} \\
  --seeds {SEEDS} \\
  --noise-sigmas {NOISE_SIGMAS} \\
  --controller-host iaaccn22 \\
  --use-internal-ips \\
  --session-prefix formal24h_full_ \\
  --host-session-count-prefix formal24h_full_ \\
  --poll-seconds 60 \\
  2>&1 | tee benchmark-runs/formal24h/latest/deploy/full_dispatch_from_iaaccn22.log

python benchmark-control/compliance/launchers/collect_remote_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest \\
  --batch-id "${{BATCH_ID}}" \\
  --hosts {HOSTS} \\
  --controller-host iaaccn22 \\
  --use-internal-ips

python benchmark-control/compliance/launchers/harvest_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest \\
  {FULL_HARVEST_ROOTS}

python benchmark-control/compliance/launchers/audit_batch.py \\
  --batch-dir benchmark-runs/formal24h/latest \\
  --write-heartbeat \\
  --write-rerun \\
  --round-id 1

python benchmark-control/compliance/launchers/check_audit_success.py \\
  --batch-dir benchmark-runs/formal24h/latest \\
  --expected-total-tasks {EXPECTED_FULL_TASKS}
"""
    )


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix().rstrip("/")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default="benchmark-runs/formal24h/latest")
    args = parser.parse_args()
    paths = write_full24h_queue_commands(batch_dir=_resolve_repo_path(args.batch_dir))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
