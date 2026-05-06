import argparse
import os
import sys
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np

from .benchmarks.runner import run_benchmark_task
from .srkit.regressor import SymbolicRegressor
from .pipelines.iterative_experiment import IterativeExperimentPipeline


def _resolve_dataset_path(path: str) -> str:
    """
    note 
    - noteenvironment note SIM_DATASETS_PATH note 
      note note sim-datasets-data note 
    - note 
    note note 
    """
    env_var_name = "SIM_DATASETS_PATH"
    base = os.environ.get(env_var_name)
    if base:
        base = os.path.expanduser(base)
        # note note
        if not os.path.isabs(path):
            path = os.path.join(base, path)
    return os.path.abspath(path)


def _convert_scalar(value: str) -> Any:
    """note bool/int/float note """
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_key_value_pairs(pairs: List[str]) -> Dict[str, Any]:
    """
    note key=value note 
    note int/float/bool note note 
    """
    out: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"note: '{item}' note key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"note: '{item}'")
        out[key] = _convert_scalar(value)
    return out


def _parse_unknown_to_kwargs(unknown: List[str]) -> Dict[str, Any]:
    """
    note kwargs note 
    - note: --key value
    - note: --key=value
    - note --flag note note True
    - key note '-' note '-' note '_'
    note -k note note argparse note 
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("-"):
            # current note
            i += 1
            continue

        # note --xxx
        if token.startswith("--"):
            # note --key=value note
            if "=" in token:
                key, value = token[2:].split("=", 1)
                key = key.replace("-", "_")
                kwargs[key] = _convert_scalar(value)
                i += 1
                continue

            # note --key value note --flag
            key = token[2:].replace("-", "_")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
                kwargs[key] = _convert_scalar(unknown[i + 1])
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            # note note argparse note
            i += 1
    return kwargs


def _load_dataset(
    path: str,
    target_column: Optional[str] = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    note 

    note 
    - CSV note default note note --no-header note
    - Numpy note .npy / .npz default note arr_0 note 

    note 
    - note target_column note y note X
    - note default note y note X
    """
    # noteenvironment note
    path = _resolve_dataset_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"note: {path}")

    ext = os.path.splitext(path)[1].lower()

    # Numpy note
    if ext in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "arr_0" not in data.files:
                raise ValueError(f"npz note 'arr_0' note: {data.files}")
            arr = data["arr_0"]
        else:
            arr = data
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("note 2 note (note + note)")
        X = arr[:, :-1]
        y = arr[:, -1]
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    # note/CSV
    if has_header:
        # note genfromtxt note note
        arr = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float)
        if arr.size == 0:
            raise ValueError(f"note: {path}")
        field_names = list(arr.dtype.names or [])
        if len(field_names) < 2:
            raise ValueError("note 2 note (note + note)")

        if target_column:
            if target_column not in field_names:
                raise ValueError(
                    f"note '{target_column}' note: {', '.join(field_names)}"
                )
            y = arr[target_column]
            feature_names = [f for f in field_names if f != target_column]
        else:
            # default note y
            y = arr[field_names[-1]]
            feature_names = field_names[:-1]

        X_cols = [arr[name] for name in feature_names]
        X = np.vstack(X_cols).T
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    # note note
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        raise ValueError("note 2 note (note + note)")
    X = arr[:, :-1]
    y = arr[:, -1]
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _is_canonical_dataset_dir(path: str) -> bool:
    """notecurrent note """
    p = Path(path)
    return (
        p.is_dir()
        and (p / "metadata.yaml").is_file()
        and (p / "train.csv").is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    """note default note """
    parser = argparse.ArgumentParser(
        prog="sim-cli",
        description="Scientific Intelligent Modelling note",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        required=True,
        help="note symbolic regressionnote note toolbox_config.json note tool_name note: gplearn, pysr, drsr, llmsr note",
    )
    parser.add_argument(
        "-t",
        "--train-path",
        required=True,
        help="note note: CSV note .npy .npz",
    )

    # note note WandB note 
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        type=str,
        help="note note WandB note config.dataset.name note ",
    )

    # note note note LLMSR  note
    parser.add_argument(
        "--seed",
        type=int,
        default=1314,
        help="note default 1314 note seed note ",
    )
    parser.add_argument(
        "--timeout-in-seconds",
        "--timeout_in_seconds",
        dest="timeout_in_seconds",
        type=int,
        default=None,
        help="note note  notecurrent note",
    )

    # prompts note/note notecurrent note 
    parser.add_argument(
        "--prompts-type",
        "--prompts_type",
        dest="prompts_type",
        type=str,
        help="current note prompts note/note note WandB note config.prompts_type note ",
    )

    # WandB note
    parser.add_argument("--use_wandb", action="store_true", help="note WandB note")
    parser.add_argument("--wandb_project", type=str, default="my-awesome-project", help="WandB note")
    parser.add_argument("--wandb_entity", type=str, default="my-awesome-entity", help="WandB note/note")
    parser.add_argument("--wandb_name", type=str, default="sim-run", help="WandB note")
    parser.add_argument("--wandb_group", type=str, default="sim-group", help="WandB note")
    parser.add_argument("--wandb_tags", type=str, default="sim,llmsr", help="WandB note note")

    # note
    parser.add_argument(
        "--redirect_io",
        action="store_true",
        help="note note sim-cli note std.out/std.err",
    )

    return parser

def build_pipeline_parser() -> argparse.ArgumentParser:
    """note """
    parser = argparse.ArgumentParser(
        prog="sim-cli run-pipeline",
        description="note symbolic regressionnote",
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset_dir",
        dest="dataset_dir",
        type=str,
        required=True,
        help="note (note train.csv, valid.csv, metadata.yaml note)"
    )
    parser.add_argument(
        "--tool-name",
        "--tool_name",
        dest="tool_name",
        type=str,
        required=True,
        help="symbolic regressionnote (note: gplearn, pysr, llmsr)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="note/note"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1314,
        help="note"
    )
    parser.add_argument(
        "--params-json",
        "--params_json",
        dest="params_json",
        type=str,
        default="{}",
        help='JSON note (note: \'{"population_size": 100}\')'
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        type=str,
        default="iterative_experiment_report.json",
        help="note (JSON)"
    )
    return parser

def run_pipeline_command(argv: List[str]) -> None:
    """note run-pipeline note """
    parser = build_pipeline_parser()
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir
    tool_name = args.tool_name
    iterations = args.iterations
    seed = args.seed
    output_path = args.output_path
    
    try:
        params = json.loads(args.params_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON parameters: {e}")
        sys.exit(1)

    print(f"[sim-cli] Running Iterative Pipeline")
    print(f"Dataset: {dataset_dir}")
    print(f"Algorithm: {tool_name}")
    print(f"Iterations: {iterations}")

    try:
        pipeline = IterativeExperimentPipeline(
            dataset_dir=dataset_dir,
            algorithm=tool_name,
            params=params,
            seed=seed
        )
        report = pipeline.run(num_iterations=iterations)

        # note
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"\n[sim-cli] Experiment report saved to {output_path}")

        # note
        print("\n--- Experiment Summary ---")
        print(f"Total Fit Time: {report['total_fit_time']:.4f}s")
        print(f"Final ID Test RMSE: {report['final_id_test_metrics']['rmse']:.4f}")
        print(f"Final OOD Test RMSE: {report['final_ood_test_metrics']['rmse']:.4f}")

    except Exception as e:
        print(f"\n[sim-cli] An error occurred during the pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    """note """
    if argv is None:
        argv = sys.argv[1:]

    # note run-pipeline note
    if len(argv) > 0 and argv[0] == "run-pipeline":
        # note "run-pipeline" note
        run_pipeline_command(argv[1:])
        return

    # default note note (Backward Compatibility)
    parser = build_parser()
    # note parse_known_args note note wrapper note
    args, unknown = parser.parse_known_args(argv)

    algorithm = args.algorithm
    train_path = args.train_path

    # note CLI note note wrapper
    extra_params: Dict[str, Any] = {}
    # note note --llm_config_path xxx  note wrapper note
    extra_kwargs = _parse_unknown_to_kwargs(unknown)
    extra_params.update(extra_kwargs)
    # note -a/-t note note meta.json note CLI note
    extra_params.setdefault("algorithm", algorithm)
    extra_params.setdefault("train_path", train_path)
    # note note WandB
    if getattr(args, "dataset_name", None):
        extra_params.setdefault("dataset_name", args.dataset_name)
    # note seed note LLMSRRegressor note manifest note
    if getattr(args, "seed", None) is not None:
        extra_params.setdefault("seed", args.seed)
    if getattr(args, "timeout_in_seconds", None) is not None:
        extra_params.setdefault("timeout_in_seconds", args.timeout_in_seconds)
    # note prompts_type note WandB notecurrent prompts note/note
    if getattr(args, "prompts_type", None):
        extra_params.setdefault("prompts_type", args.prompts_type)

    # WandB note note
    if args.use_wandb:
        wandb_tags = None
        if args.wandb_tags:
            wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        extra_params.update(
            {
                "use_wandb": True,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_name": args.wandb_name,
                "wandb_group": args.wandb_group,
                "wandb_tags": wandb_tags,
            }
        )

    print(f"[sim-cli] note: {algorithm}")
    resolved_train_path = _resolve_dataset_path(train_path)
    print(f"[sim-cli] note: {resolved_train_path}")

    if _is_canonical_dataset_dir(resolved_train_path):
        # note CLI note note --train-path note 
        # note benchmark runner 
        output_root = extra_params.pop(
            "output_root",
            os.path.join(os.getcwd(), "bench_results", "sim_cli"),
        )
        print(f"[sim-cli] note note runner")
        print(f"[sim-cli] note: {os.path.abspath(str(output_root))}")
        result_path = run_benchmark_task(
            tool_name=algorithm,
            dataset_dir=resolved_train_path,
            output_root=output_root,
            seed=args.seed,
            params_override=extra_params,
        )
        print(f"[sim-cli] note note: {result_path}")
        return

    # note note
    reg = SymbolicRegressor(
        tool_name=algorithm,
        **extra_params,
    )

    if args.redirect_io:
        # note stdout/stderr note std.out / std.err
        exp_dir = getattr(reg, "experiment_dir", os.getcwd())
        os.makedirs(exp_dir, exist_ok=True)
        stdout_path = os.path.join(exp_dir, "std.out")
        stderr_path = os.path.join(exp_dir, "std.err")
        with open(stdout_path, "a", encoding="utf-8") as f_out, open(
            stderr_path, "a", encoding="utf-8"
        ) as f_err, redirect_stdout(f_out), redirect_stderr(f_err):
            print(f"[sim-cli] note: {algorithm}")
            print(f"[sim-cli] note: {resolved_train_path}")

            # note
            X, y = _load_dataset(
                resolved_train_path,
                target_column=None,   # note note CSV default note
                delimiter=",",
                has_header=True,
            )
            print(f"[sim-cli] note: X note={X.shape}, y note={y.shape}")

            print("[sim-cli] note...")
            reg.fit(X, y)
            print("[sim-cli] note ")
    else:
        # note
        X, y = _load_dataset(
            resolved_train_path,
            target_column=None,   # note note CSV default note
            delimiter=",",
            has_header=True,
        )
        print(f"[sim-cli] note: X note={X.shape}, y note={y.shape}")

        print("[sim-cli] note...")
        reg.fit(X, y)
        print("[sim-cli] note ")

    # WandB note
    if args.use_wandb:
        pass

if __name__ == "__main__":
    main()
