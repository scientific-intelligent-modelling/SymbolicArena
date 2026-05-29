from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REMOTE_ROOT = "/home/zhangziwen/workplace/scientific-intelligent-modelling"
HUB = "iaaccn22"
INTERNAL_TARGETS = [
    "10.10.100.23",
    "10.10.100.24",
    "10.10.100.25",
    "10.10.100.26",
    "10.10.100.27",
    "10.10.100.28",
    "10.10.100.29",
]
SYNC_PATHS = [
    "check/run_e1_candidate200_12alg_load_queue.py",
    "check/launch_e1_benchmark.py",
    "benchmark-control/compliance/",
    "exp-planning/02.E1选择验证/generated/params/",
]
RSYNC_FILTERS = [
    "--exclude=__pycache__/",
    "--exclude=*.pyc",
    "--exclude=*.pyo",
]


def write_remote_sync_commands(*, batch_dir: Path) -> Path:
    deploy_dir = batch_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    path = deploy_dir / "00_sync_code_and_batch_to_iaaccn22.sh"
    path.write_text(_script_content(batch_dir=batch_dir), encoding="utf-8")
    path.chmod(0o755)
    return path


def _script_content(*, batch_dir: Path) -> str:
    batch_id = batch_dir.resolve().name
    batch_rel = _repo_relative(batch_dir)
    sync_items_array = "\n".join(f'  "{item}"' for item in [*SYNC_PATHS, f"{batch_rel}/"])
    rsync_filters_array = "\n".join(f'  "{item}"' for item in RSYNC_FILTERS)
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
  "cd {REMOTE_ROOT} && ln -sfn \\"$BATCH_ID\\" benchmark-runs/compliance/latest"

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
for target in {' '.join(INTERNAL_TARGETS)}; do
  echo "[sync] iaaccn22 -> $target"
  if ! rsync -aR "${{RSYNC_FILTERS[@]}}" "${{SYNC_ITEMS[@]}}" "$target":{REMOTE_ROOT}/; then
    echo "SYNC_FAIL $target"
    failures=$((failures + 1))
    continue
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" \\
    "cd {REMOTE_ROOT} && ln -sfn \\"$BATCH_ID\\" benchmark-runs/compliance/latest"; then
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
    parser.add_argument("--batch-dir", default="benchmark-runs/compliance/latest")
    args = parser.parse_args()
    print(write_remote_sync_commands(batch_dir=_resolve_repo_path(args.batch_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
