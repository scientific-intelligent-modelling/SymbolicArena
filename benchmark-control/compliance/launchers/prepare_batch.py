from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"
sys.path.insert(0, str(LIB))

from manifest import generate_manifest


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolbox-config", default="scientific_intelligent_modelling/config/toolbox_config.json")
    parser.add_argument("--ssr50-root", default="sim-datasets-data/ssr50")
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    summary = generate_manifest(
        toolbox_config_path=Path(args.toolbox_config),
        ssr50_root=Path(args.ssr50_root),
        batch_dir=Path(args.batch_dir),
        git_revision=_git_revision(),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
