"""离线比较原生预测、导出公式与进度快照，每个 worker 最长运行 180 秒。"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'e2esr': 'E2ESRRegressor', 'tpsr': 'TPSRRegressor', 'ragsr': 'RAGSRRegressor'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run_bounded(command, seconds, log_path, env=None):
    if not 0 < seconds <= 180:
        raise ValueError('Worker wall-time limit must be in (0, 180] seconds')
    started = time.monotonic()
    timed_out = False
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env, cwd=str(ROOT))
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # 即使主进程已退出，也回收同组子进程，防止超时后遗留计算任务。
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {'returncode': process.returncode, 'hard_timeout': timed_out,
            'hard_limit_seconds': seconds, 'wall_seconds': time.monotonic() - started}


def worker(job_path):
    import numpy as np

    job = json.loads(job_path.read_text())
    out = job_path.parent
    started = time.monotonic()
    result = {'status': 'starting', 'python': sys.version}
    write_json(out / 'worker.json', result)
    try:
        arrays = np.load(str(out / 'inputs.npz'))
        module = importlib.import_module('scientific_intelligent_modelling.algorithms.' +
                                         job['algorithm'] + '_wrapper.wrapper')
        cls = getattr(module, CLASSES[job['algorithm']])
        reg = cls(**job['params'])
        if hasattr(reg, '_write_progress_state'):
            original = reg._write_progress_state

            def capture(*args, **kwargs):
                payload = dict(args[0]) if args else dict(kwargs)
                if job['algorithm'] == 'e2esr' and 'training_mse' in payload:
                    payload.update(score=payload['training_mse'], internal_loss=payload['training_mse'],
                                   internal_objective='native_training_mse', objective_direction='min',
                                   selection_policy=reg._SELECTION_POLICY)
                payload['probe_elapsed_seconds'] = time.monotonic() - started
                with (out / 'emitted_candidates.jsonl').open('a') as handle:
                    handle.write(json.dumps(payload, default=str) + '\n')
                return original(*args, **kwargs)

            reg._write_progress_state = capture
        result['status'] = 'fitting'
        write_json(out / 'worker.json', result)
        reg.fit(arrays['train_X'], arrays['train_y'])
        if job['algorithm'] == 'e2esr':
            result['selection_policy'] = reg._SELECTION_POLICY
            result['completed_rounds'] = reg._budget_chunks_run
            result['refinement_type'] = reg.best_tree.get('refinement_type')
        result['equation'] = reg.get_optimal_equation()
        result['artifact'] = reg.export_canonical_symbolic_program()
        predictions = {}
        for split in ('train', 'id', 'ood'):
            if split + '_X' in arrays:
                predictions[split + '_native'] = np.asarray(reg.predict(arrays[split + '_X'])).reshape(-1)
        np.savez(str(out / 'predictions.npz'), **predictions)
        result['status'] = 'native_evaluated'
        write_json(out / 'worker.json', result)
        if job['algorithm'] == 'ragsr':
            result['backend_expression'] = str(reg.model.model())
            result['scalers'] = {}
            for name in ('x_scaler', 'y_scaler'):
                scaler = getattr(reg.model, name, None)
                result['scalers'][name] = {'class': type(scaler).__name__, 'params': {
                    key: np.asarray(getattr(scaler, key)).tolist()
                    for key in ('scale_', 'min_', 'mean_', 'data_min_', 'data_max_')
                    if hasattr(scaler, key)}}
            restored = cls.deserialize(reg.serialize())
            for split in ('train', 'id', 'ood'):
                if split + '_X' in arrays:
                    predictions[split + '_restored'] = np.asarray(restored.predict(arrays[split + '_X'])).reshape(-1)
            np.savez(str(out / 'predictions.npz'), **predictions)
        result['status'] = 'ok'
    except Exception:
        result['status'] = 'error'
        result['traceback'] = traceback.format_exc()
    result['elapsed_seconds'] = time.monotonic() - started
    write_json(out / 'worker.json', result)
    return 0 if result['status'] == 'ok' else 1


def metrics(target, prediction):
    import numpy as np

    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    invalid = {'finite': False, 'mse': None, 'nmse_mean_y_squared': None, 'r2': None}
    if (target.size == 0 or target.shape != prediction.shape
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        return invalid
    with np.errstate(over='ignore', invalid='ignore'):
        mse = float(np.mean((target - prediction) ** 2))
        denominator = float(np.mean(target ** 2))
        variance = float(np.var(target))
    if not all(np.isfinite(v) for v in (mse, denominator, variance)):
        return invalid
    return {'finite': True, 'mse': mse, 'nmse_mean_y_squared': mse / denominator if denominator else None,
            'r2': 1 - mse / variance if variance else None}


def validation_status(report):
    issues = []
    execution = report['execution']
    if execution.get('hard_timeout'):
        issues.append('hard_timeout')
    if execution.get('returncode') != 0 or report['worker'].get('status') != 'ok':
        issues.append('worker_did_not_complete')
    if not report['metrics']:
        issues.append('missing_evaluation')
    for split, values in report['metrics'].items():
        for mode in ('native', 'exported'):
            if not values.get(mode, {}).get('finite', False):
                issues.append(split + ':' + mode + '_unresolved')
        if values.get('export_matches_native') is not True:
            issues.append(split + ':export_mismatch')
        if 'restored' in values and (
                not values['restored'].get('finite') or values.get('restored_matches_native') is not True):
            issues.append(split + ':restored_mismatch')
        if (report.get('algorithm') == 'e2esr'
                and report['worker'].get('selection_policy') == 'e2esr_training_mse_v1'
                and (not values.get('snapshot', {}).get('finite')
                     or values.get('snapshot_matches_native') is not True)):
            issues.append(split + ':snapshot_mismatch')
    return {'ok': not issues, 'issues': issues,
            'exit_code': 124 if execution.get('hard_timeout') else (1 if issues else 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--algorithm', choices=sorted(CLASSES))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--case', choices=['centered', 'shifted'], default='shifted')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--fit-seconds', type=float, default=45)
    parser.add_argument('--hard-stop-seconds', type=float,
                        help='Optional earlier worker cutoff for timeout recovery checks')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.worker:
        return worker(args.worker.resolve())
    if not args.algorithm or not args.output:
        parser.error('--algorithm and --output are required')
    if not 0 < args.seconds <= 180 or not 0 < args.fit_seconds < args.seconds:
        parser.error('Require 0 < fit-seconds < seconds <= 180')
    if args.hard_stop_seconds is not None and not 0 < args.hard_stop_seconds <= args.seconds:
        parser.error('Require 0 < hard-stop-seconds <= seconds')

    import numpy as np
    from scientific_intelligent_modelling.benchmarks import normalizers
    from scientific_intelligent_modelling.benchmarks.runner import (
        load_canonical_dataset, _predict_from_canonical_artifact,
        _extract_e2esr_periodic_candidate, _recover_timeout_payload_from_candidate,
    )

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    arrays = {}
    if args.dataset:
        data = load_canonical_dataset(args.dataset)
        for name, split in [('train', data.train), ('id', data.id_test), ('ood', data.ood_test)]:
            if split is not None and split.rows:
                arrays[name + '_X'], arrays[name + '_y'] = split.X, split.y
        names, target = data.feature_names, data.target_name
    else:
        rng = np.random.RandomState(520)
        for split, size in [('train', 80), ('id', 32), ('ood', 32)]:
            x = rng.uniform(-1, 1, (size, 2)) if split != 'ood' else rng.uniform(1, 2, (size, 2))
            if args.case == 'shifted':
                x = x * [2.0, 0.5] + [10.0, -5.0]
            arrays[split + '_X'], arrays[split + '_y'] = x, 2 * x[:, 0] - 3 * x[:, 1] + 0.7
        names, target = ['x0', 'x1'], 'y'
    np.savez(str(out / 'inputs.npz'), **arrays)
    params = {'seed': 520, 'n_features': len(names), 'feature_names': names, 'target_name': target,
              'exp_path': str(out), 'exp_name': 'native'}
    if args.algorithm == 'e2esr':
        params.update(timeout_in_seconds=args.fit_seconds, max_number_bags=1, n_trees_to_refine=4)
    elif args.algorithm == 'tpsr':
        params.update(timeout_in_seconds=args.fit_seconds, max_number_bags=1, n_trees_to_refine=4,
                      cpu_num_threads=1, horizon=40, rollout=1, width=2)
    else:
        params.update(n_pop=32, n_gen=3, gene_num=3, time_limit=args.fit_seconds, cpu_num_threads=1)
    source_paths = [ROOT / 'scientific_intelligent_modelling' / 'algorithms' / (args.algorithm + '_wrapper') / 'wrapper.py',
                    ROOT / 'scientific_intelligent_modelling/benchmarks/normalizers.py']
    if args.algorithm == 'e2esr':
        source_paths.append(ROOT / 'scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/sklearn_wrapper.py')
        source_paths.append(ROOT / 'scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/utils_wrapper.py')
    job = {'algorithm': args.algorithm, 'case': str(args.dataset or args.case), 'params': params,
           'diagnostic_only': True, 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in source_paths}}
    write_json(out / 'job.json', job)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    execution = run_bounded([args.python, '-u', str(Path(__file__).resolve()), '--worker', str(out / 'job.json')],
                            args.hard_stop_seconds or args.seconds, out / 'worker.log', env)
    report = dict(job, execution=execution, metrics={})
    result_path = out / 'worker.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    report['worker'] = result
    selected_snapshot = _extract_e2esr_periodic_candidate(out / 'native') if args.algorithm == 'e2esr' else None
    if args.algorithm == 'e2esr':
        report['selected_snapshot'] = selected_snapshot
    predictions = np.load(str(out / 'predictions.npz')) if (out / 'predictions.npz').exists() else {}
    for split in ('train', 'id', 'ood'):
        if split + '_y' not in arrays:
            continue
        report['metrics'][split] = {}
        for kind in ('native', 'restored'):
            if split + '_' + kind in predictions:
                report['metrics'][split][kind] = metrics(arrays[split + '_y'], predictions[split + '_' + kind])
        if split + '_restored' in predictions and split + '_native' in predictions:
            report['metrics'][split]['restored_matches_native'] = bool(np.allclose(
                predictions[split + '_restored'], predictions[split + '_native'], rtol=1e-7, atol=1e-10))
        if result.get('artifact'):
            try:
                replay = _predict_from_canonical_artifact(result['artifact'], arrays[split + '_X'])
                report['metrics'][split]['exported'] = metrics(arrays[split + '_y'], replay)
                if split + '_native' in predictions:
                    delta = np.abs(replay - predictions[split + '_native'])
                    report['metrics'][split]['export_max_abs_delta'] = float(np.max(delta)) if np.isfinite(delta).all() else None
                    report['metrics'][split]['export_matches_native'] = bool(np.allclose(
                        replay, predictions[split + '_native'], rtol=1e-7, atol=1e-10))
            except Exception as exc:
                report['metrics'][split]['export_error'] = repr(exc)
        if selected_snapshot is not None:
            try:
                artifact = normalizers.normalize_e2esr_artifact(selected_snapshot['equation'], expected_n_features=len(names))
                replay = _predict_from_canonical_artifact(artifact, arrays[split + '_X'])
                report['metrics'][split]['snapshot'] = metrics(arrays[split + '_y'], replay)
                if split + '_native' in predictions:
                    report['metrics'][split]['snapshot_matches_native'] = bool(np.allclose(
                        replay, predictions[split + '_native'], rtol=1e-7, atol=1e-10))
            except Exception as exc:
                report['metrics'][split]['snapshot_error'] = repr(exc)
    if args.algorithm == 'e2esr' and args.dataset:
        report['timeout_recovery'] = _recover_timeout_payload_from_candidate(
            tool_name='e2esr', dataset=data, experiment_dir=out / 'native')
    events_path = out / 'emitted_candidates.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    report['snapshots'] = []
    for event in events:
        entry = {k: event.get(k) for k in ('source', 'score', 'training_mse', 'stage', 'refinement_type',
                                          'native_model_score', 'probe_elapsed_seconds', 'equation')}
        try:
            artifact = getattr(normalizers, 'normalize_' + args.algorithm + '_artifact')(event['equation'], expected_n_features=len(names))
            entry['id'] = metrics(arrays['id_y'], _predict_from_canonical_artifact(artifact, arrays['id_X']))
        except Exception as exc:
            entry['error'] = repr(exc)
        report['snapshots'].append(entry)
    report['validation'] = validation_status(report)
    write_json(out / 'report.json', report)
    print(json.dumps({'output': str(out), 'execution': execution, 'metrics': report['metrics'],
                      'worker_status': result.get('status'), 'snapshots': len(events),
                      'validation': report['validation']}, indent=2))
    return report['validation']['exit_code']


if __name__ == '__main__':
    sys.exit(main())
