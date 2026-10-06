import argparse
from pathlib import Path

from scientific_intelligent_modelling.onboarding.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "budget", "core50", "all"), default="smoke")
    parser.add_argument("--dataset-dir", default="examples/BPG0")
    parser.add_argument("--datasets-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [
        "accept", "--manifest", str(root / "tools/sr_onboarder/manifests/dgp.json"),
        "--output-root", args.output_root, "--stage", args.stage,
        "--dataset-dir", args.dataset_dir, "--workers", str(args.workers),
    ]
    if args.datasets_root:
        command.extend(["--datasets-root", args.datasets_root])
    main(command)
