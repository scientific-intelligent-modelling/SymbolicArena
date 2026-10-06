import argparse
import json
from pathlib import Path

from scientific_intelligent_modelling.benchmarks.runner import run_benchmark_task
from scientific_intelligent_modelling.onboarding.manifest import load_manifest, sha256_file, write_json
from scientific_intelligent_modelling.onboarding.source import verify_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    manifest = load_manifest(request["manifest"])
    verify_source(manifest)
    result_path = run_benchmark_task(
        tool_name=manifest["tool_name"], dataset_dir=request["dataset_dir"],
        output_root=request["output_root"], seed=request["seed"],
        params_override=request["parameters"],
    )
    write_json(request_path.parent / "completion.json", {"result_path": str(result_path), "result_sha256": sha256_file(result_path)})


if __name__ == "__main__":
    main()
