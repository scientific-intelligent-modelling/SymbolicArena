"""SymbolFit numerical and serialization checks with a 180-second worker limit."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

from three_algorithm_quality_probe import metrics, run_bounded, write_json


ROOT = Path(__file__).resolve().parents[1]


def worker(job_path):
    job = json.loads(job_path.read_text())
    out = job_path.parent
    started = time.monotonic()
    result = {'status': 'starting', 'versions': {}}
    predictions = {}
    try:
        from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor
        from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact
        import sympy as sp

        for name in ('symbolfit', 'pysr', 'lmfit', 'numpy', 'sympy'):
            result['versions'][name] = importlib.metadata.version(name)
        arrays = np.load(out / 'inputs.npz')
        reg = SymbolFitRegressor(**job['params'])
        reg.fit(arrays['train_X'], arrays['train_y'])
        result.update(equation=reg.get_optimal_equation(), artifact=reg.export_canonical_symbolic_program(),
                      coordinate_transform=reg._coordinate_transform,
                      reported_refit_rmse=float(reg._best_candidate['RMSE']))
        reg.model.func_candidates.to_csv(out / 'refit_candidates.csv', index=False)
        row = reg._best_candidate
        result['parameterized_equation'] = row['Parameterized equation, unscaled']
        params = row.get('Parameters: (best-fit, +1, -1)', {})
        result['best_fit_parameters'] = {str(k): float(v[0]) for k, v in params.items()}
        x_names = ['x' + str(i) for i in range(arrays['train_X'].shape[1])]
        p_names = sorted(result['best_fit_parameters'])
        # Bind native parameters as function arguments, independently of wrapper substitution.
        reference = sp.lambdify(x_names + p_names, sp.sympify(result['parameterized_equation']), 'numpy')
        serialized = reg.serialize()
        restored = SymbolFitRegressor.deserialize(serialized)
        result['serialized_bytes'] = len(serialized.encode())
        result['metrics'] = {}
        for split in ('train', 'id', 'ood'):
            X, y = arrays[split + '_X'], arrays[split + '_y']
            modes = {
                'native': reg.predict(X),
                'restored': restored.predict(X),
                'exported': _predict_from_canonical_artifact(result['artifact'], X),
                'refit_reference': reference(*[X[:, i] for i in range(X.shape[1])],
                    *[result['best_fit_parameters'][p] for p in p_names]),
            }
            result['metrics'][split] = {}
            for mode, values in modes.items():
                values = np.asarray(values, dtype=float)
                if values.ndim == 0:
                    values = np.full(len(y), float(values))
                values = values.reshape(-1)
                predictions[split + '_' + mode] = values
                result['metrics'][split][mode] = metrics(y, values)
            for mode in ('restored', 'exported', 'refit_reference'):
                result['metrics'][split][mode + '_matches_native'] = bool(np.allclose(
                    predictions[split + '_' + mode], predictions[split + '_native'], rtol=1e-7, atol=1e-10))
        result['status'] = 'ok'
    except Exception:
        result.update(status='error', traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.monotonic() - started
    if predictions:
        np.savez(out / 'predictions.npz', **predictions)
    write_json(out / 'worker.json', result)
    return 0 if result['status'] == 'ok' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--case', choices=['shifted', 'zero_mean', 'constant_column'], default='shifted')
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--fit-seconds', type=int, default=45)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if not args.output or not 0 < args.fit_seconds < args.seconds <= 180:
        parser.error('Require output and 0 < fit-seconds < seconds <= 180')
    from scientific_intelligent_modelling.benchmarks.runner import load_canonical_dataset

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    arrays = {}
    data_hashes = {}
    if args.dataset:
        dataset = load_canonical_dataset(args.dataset)
        for name, split in [('train', dataset.train), ('id', dataset.id_test), ('ood', dataset.ood_test)]:
            arrays[name + '_X'], arrays[name + '_y'] = split.X, split.y
        names, target = dataset.feature_names, dataset.target_name
        data_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in args.dataset.iterdir() if p.suffix in ('.csv', '.yaml')}
    else:
        rng = np.random.RandomState(520)
        for split, n in [('train', 80), ('id', 32), ('ood', 32)]:
            if args.case == 'zero_mean':
                x = np.arange(-n // 2, n // 2 + 1, dtype=float)
                x = x[x != 0] / 10
                if split == 'ood':
                    x *= 5
                X, y = x.reshape(-1, 1), 2 * x
            else:
                X = rng.uniform(-1, 1, (n, 2)) * [2.0, .5] + [10., -5.]
                if split == 'ood':
                    X += [5., 1.]
                if args.case == 'constant_column':
                    X[:, 0] = 50.
                    y = 3.25 * X[:, 1] - .75
                else:
                    y = 2 * X[:, 0] - 3 * X[:, 1] + .7
            arrays[split + '_X'], arrays[split + '_y'] = X, y
        names = ['x' + str(i) for i in range(arrays['train_X'].shape[1])]
        target = 'y'
    np.savez(out / 'inputs.npz', **arrays)
    params = {'seed': 520, 'n_features': len(names), 'feature_names': names, 'target_name': target,
              'exp_path': str(out), 'exp_name': 'native', 'timeout_in_seconds': args.fit_seconds,
              'fill_timeout_budget': False, 'niterations': 3, 'maxsize': 8, 'max_complexity': 8,
              'unary_operators': ['sin', 'cos'], 'binary_operators': ['+', '-', '*', '/']}
    source_files = ['scientific_intelligent_modelling/algorithms/symbolfit_wrapper/wrapper.py',
                    'scientific_intelligent_modelling/benchmarks/normalizers.py']
    job = {'algorithm': 'symbolfit', 'case': str(args.dataset or args.case), 'params': params,
           'diagnostic_only': True, 'data_sha256': data_hashes,
           'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files}}
    write_json(out / 'job.json', job)
    stop = threading.Event()

    def collect_progress():
        from scientific_intelligent_modelling.benchmarks.runner import (
            _extract_periodic_candidate, _predict_from_canonical_artifact,
        )
        from scientific_intelligent_modelling.benchmarks.normalizers import normalize_external_infix_artifact

        minute = 0
        while not stop.wait(60):
            minute += 1
            record = {'algorithm': 'symbolfit', 'condition': 'clean_as_stored', 'seed': 520,
                      'dataset_id': job['case'], 'minute': minute,
                      'logical_key': 'symbolfit/{}/520/{}'.format(job['case'], minute),
                      'source_path': str(out / 'native'), 'judge_version': None,
                      'formal_ready': False, 'unresolved_axes': ['SYM', 'MIN', 'EFF', 'STAB'],
                      'metric_contract': 'diagnostic_mse_nmse_mean_y_squared_r2_v1'}
            try:
                candidate = _extract_periodic_candidate('symbolfit', out / 'native',
                    snapshot_minute=minute, snapshot_elapsed_seconds=minute * 60)
                record['candidate'] = candidate
                if candidate:
                    record['candidate_sha256'] = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
                    artifact = normalize_external_infix_artifact(candidate['equation'], tool_name='symbolfit',
                                                                 expected_n_features=len(names), shift_one_based=False)
                    record['metrics'] = {split: metrics(arrays[split + '_y'],
                        _predict_from_canonical_artifact(artifact, arrays[split + '_X']))
                        for split in ('train', 'id', 'ood')}
                else:
                    record['unresolved_reason'] = 'no_candidate_at_checkpoint'
            except Exception as exc:
                record['unresolved_reason'] = repr(exc)
            with (out / 'minute_evidence.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')

    sampler = threading.Thread(target=collect_progress, daemon=True)
    sampler.start()
    try:
        execution = run_bounded([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(out / 'job.json')],
                                args.seconds, out / 'worker.log', dict(os.environ, OPENBLAS_NUM_THREADS='1',
                                OMP_NUM_THREADS='1', JULIA_NUM_THREADS='1'))
    finally:
        stop.set()
        sampler.join()
    result = json.loads((out / 'worker.json').read_text()) if (out / 'worker.json').exists() else {}
    issues = []
    if execution['hard_timeout'] or result.get('status') != 'ok':
        issues.append('worker_did_not_complete')
    for split, value in result.get('metrics', {}).items():
        for mode in ('native', 'exported', 'restored', 'refit_reference'):
            if not value.get(mode, {}).get('finite'):
                issues.append(split + ':' + mode + '_not_finite')
        for mode in ('exported', 'restored', 'refit_reference'):
            if not value.get(mode + '_matches_native'):
                issues.append(split + ':' + mode + '_mismatch')
    report = dict(job, execution=execution, worker=result, validation={'ok': not issues, 'issues': issues})
    write_json(out / 'report.json', report)
    print(json.dumps({'case': job['case'], 'execution': execution, 'validation': report['validation'],
                      'metrics': result.get('metrics'), 'traceback': result.get('traceback')}, indent=2))
    return 124 if execution['hard_timeout'] else (1 if issues else 0)


if __name__ == '__main__':
    sys.exit(main())
