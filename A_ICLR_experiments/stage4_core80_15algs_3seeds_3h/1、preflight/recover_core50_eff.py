import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import shutil

import numpy as np

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import extract_expression_body
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import efficiency_from_qualities, phi_nmse
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import load_canonical_dataset, regression_metrics
from build_core50_minute_evidence import completed_snapshot_horizon
from run_core50_comparisons import canonical, sha, write_json


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / 'A_ICLR_experiments/stage4_core80_15algs_3seeds_3h'
METRICS = STAGE / '3、metrics'
OUTPUT = METRICS / 'eff_recovery'
SNAPSHOT = OUTPUT / 'input_snapshot'
DATASETS = {}


def native_parameter_quality(snapshot, final, candidate_path):
    candidate = json.loads(candidate_path.read_text())
    if candidate['sample_order'] != snapshot['source_sample_order']:
        raise ValueError('原生候选编号不一致')
    body = extract_expression_body(snapshot['equation'])
    if ast.dump(ast.parse(body, mode='eval')) != ast.dump(ast.parse(extract_expression_body(candidate['function']), mode='eval')):
        raise ValueError('原生候选公式不一致')
    objective = candidate['score'] if snapshot['tool'].lower() == 'drsr' else candidate['nmse']
    if not math.isclose(objective, snapshot['internal_objective_value'], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('原生训练目标与参数记录不一致')
    tree = ast.parse(body, mode='eval')
    functions = {'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'any', 'sum', 'tanh', 'power'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr in functions):
            raise ValueError('候选包含未允许的属性访问')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in {'sum', 'len', 'range'}:
            raise ValueError('候选包含未允许的函数调用')
    code = compile(tree, '<frozen-native-parameters>', 'eval')
    path = ROOT / final['expected_dataset_rel']
    if path not in DATASETS:
        DATASETS[path] = load_canonical_dataset(path)
    dataset = DATASETS[path]
    params = np.asarray(candidate['params'], dtype=float)
    environment = {'__builtins__': {}, 'np': np, 'params': params, 'MAX_NPARAMS': len(params),
                   'sum': sum, 'len': len, 'range': range}
    metrics = {}
    valid = True
    for name in ('id_test', 'ood_test'):
        split = getattr(dataset, name)
        variables = {f'x{i}': split.X[:, i] for i in range(split.X.shape[1])}
        with np.errstate(all='ignore'):
            prediction = np.asarray(eval(code, {**environment, **variables}), dtype=float)
        if prediction.ndim == 0:
            prediction = np.full(split.rows, float(prediction))
        prediction = prediction.reshape(-1)
        if prediction.shape != np.asarray(split.y).reshape(-1).shape:
            raise ValueError('原生预测长度不符')
        if not np.all(np.isfinite(prediction)):
            valid = False
            metrics[name] = None
        else:
            metrics[name] = regression_metrics(split.y, prediction, acc_threshold=0.1)
    iq = phi_nmse(metrics['id_test']['nmse']) if valid else 0.0
    oq = phi_nmse(metrics['ood_test']['nmse']) if valid else 0.0
    return {'q': (iq + oq) / 2, 'id_quality': iq, 'ood_quality': oq, 'valid_output': valid,
        'id_nmse': metrics['id_test']['nmse'] if metrics['id_test'] else None,
        'ood_nmse': metrics['ood_test']['nmse'] if metrics['ood_test'] else None,
        'expression': snapshot['equation'], 'frozen_parameter_values': candidate['params'],
        'status': 'resolved' if valid else 'invalid', 'reason': None if valid else 'native_prediction_nonfinite',
        'evaluation_path': 'frozen_native_parameters.v1',
        'recovery_evidence': {'path': str(candidate_path), 'sha256': sha(candidate_path),
                              'sample_order': candidate['sample_order'], 'native_objective': objective}}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'verification.json').exists():
        raise FileExistsError('EFF恢复结果已经存在')
    SNAPSHOT.mkdir(exist_ok=True)
    names = ('algorithm_six_axis.csv', 'noise_supplement.csv', 'terminal_run_metrics.csv',
             'terminal_run_metrics.jsonl.gz', 'unresolved.json', 'verification.json')
    for name in names:
        target = SNAPSHOT / name
        if not target.exists():
            shutil.copy2(METRICS / name, target)
    with gzip.open(SNAPSHOT / 'terminal_run_metrics.jsonl.gz', 'rt') as handle:
        runs = {row['logical_key']: row for row in map(json.loads, handle)}
    keys = {key for key, row in runs.items() if row['eff_score'] is None}
    points = defaultdict(list)
    manifests = {}
    for path in sorted((METRICS / 'inputs/minute_evidence_full').glob('*/*/run_minutes.jsonl.gz')):
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if sha(path) != manifest['numeric_sha256']:
            raise ValueError('数值轨迹哈希不符')
        manifests[str(path)] = manifest['numeric_sha256']
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                if row['logical_key'] in keys:
                    points[row['logical_key']].append(row)
    audit, cache = [], {}
    delta = OUTPUT / 'recovered_run_minutes.jsonl.gz'
    with gzip.open(delta, 'wt') as handle:
        for key in sorted(keys):
            row, trace = runs[key], points[key]
            if [point['minute'] for point in trace] != list(range(1, 181)):
                raise ValueError('分钟网格不完整')
            algorithm_roots = {p.name.lower(): p for p in (STAGE / '2.2 core50 experiments' / row['condition']).iterdir() if p.is_dir()}
            directory = algorithm_roots[row['algorithm']] / row['dataset_id'] / str(row['seed'])
            result_path = directory / 'result.json'
            if sha(result_path) != row['source_sha256']:
                raise ValueError('最终运行来源发生变化')
            final = json.loads(result_path.read_text())
            paths = {int(path.stem.removeprefix('minute_')): path for path in (directory / 'progress').glob('minute_*.json')}
            payloads = {minute: json.loads(path.read_text()) for minute, path in paths.items() if minute <= 180}
            horizon = completed_snapshot_horizon(final, payloads)
            changed, methods = [], Counter()
            for point in trace:
                minute = point['minute']
                if point['q'] is not None:
                    continue
                if minute > horizon and trace[horizon - 1]['q'] is not None:
                    origin = trace[horizon - 1]
                    if final['seconds'] > minute * 60:
                        raise ValueError('结束结果不能用于更早时刻')
                    preserved = {field: point[field] for field in ('logical_key', 'run_minute_key', 'minute', 'condition', 'algorithm', 'dataset_id', 'dataset_index', 'seed', 'result_sha256')}
                    point.update(origin)
                    point.update(preserved)
                    point['selection_source'] = f'finished_run_carry_forward:{horizon}'
                    point['recovery_evidence'] = {'path': str(paths[horizon]), 'sha256': sha(paths[horizon]),
                        'result_sha256': row['source_sha256'], 'termination_seconds': final['seconds'],
                        'source_minute': horizon}
                    methods['finished_run_carry_forward'] += 1
                elif row['algorithm'] in ('drsr', 'llmsr') and minute in payloads:
                    snapshot = payloads[minute]
                    order = snapshot.get('source_sample_order')
                    if order is None or not snapshot.get('equation'):
                        continue
                    if row['algorithm'] == 'drsr' and row['condition'] == 'clean' and row['dataset_index'] == 'g0625':
                        history = OUTPUT / 'inputs/drsr_clean_g0625_s521'
                    else:
                        history = directory / 'experiments' / Path(final['experiment_dir']).name / 'best_history'
                    candidate = history / f'best_sample_{order}.json'
                    if not candidate.exists():
                        continue
                    cache_key = str(candidate)
                    if cache_key not in cache:
                        cache[cache_key] = native_parameter_quality(snapshot, final, candidate)
                    point.update(cache[cache_key])
                    point['source_path'], point['source_sha256'] = str(paths[minute]), sha(paths[minute])
                    point['selection_source'] = 'frozen_native_parameters'
                    methods['frozen_native_parameters'] += 1
                else:
                    continue
                changed.append(minute)
            qualities = [point['q'] for point in trace]
            missing = [point['minute'] for point in trace if point['q'] is None]
            if not missing:
                value = efficiency_from_qualities(qualities)
                maximum, accumulated = max(qualities), 0.0
                for point in trace:
                    relative = point['q'] / maximum if maximum else 0.0
                    accumulated += relative
                    point.update(q_star=maximum, relative_progress=relative, cumulative_eff=accumulated / point['minute'])
                if not math.isclose(value, trace[-1]['cumulative_eff'], abs_tol=1e-12):
                    raise ValueError('EFF复算不一致')
                row['eff_score'], row['eff_missing_minutes'] = value, []
                row['eff_recovery'] = {'methods': dict(methods), 'changed_minutes': changed, 'evidence': str(delta)}
            for point in trace:
                handle.write(canonical(point) + '\n')
            audit.append({'logical_key': key, 'algorithm': row['algorithm'], 'condition': row['condition'],
                'recovered': not missing, 'methods': dict(methods), 'changed_minutes': changed, 'remaining_minutes': missing})
    by_group = defaultdict(list)
    for row in runs.values():
        by_group[row['condition'], row['algorithm']].append(row['eff_score'])
    summary = []
    for (condition, algorithm), values in sorted(by_group.items()):
        available = [value for value in values if value is not None]
        if len(values) != 150:
            raise ValueError('算法运行分母不符')
        summary.append({'condition': condition, 'algorithm': algorithm,
            'EFF': 100 * sum(available) / len(available) if available else None,
            'EFF_available': len(available), 'EFF_expected': 150,
            'EFF_aggregation': 'mean_available_runs' if len(available) < 150 else 'mean_all_runs'})
    with (OUTPUT / 'eff_means.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with gzip.open(OUTPUT / 'terminal_run_metrics.jsonl.gz', 'wt') as handle:
        for row in runs.values():
            handle.write(canonical(row) + '\n')
    report = {'runs_checked': len(keys), 'runs_recovered': sum(row['recovered'] for row in audit),
        'remaining': [row for row in audit if not row['recovered']], 'audit': audit,
        'source_minute_sha256': manifests, 'new_training_runs': 0, 'new_api_requests': 0,
        'mean_policy': 'available_runs_with_explicit_denominator',
        'script_path': str(Path(__file__)), 'script_sha256': sha(Path(__file__)),
        'files': {path.name: sha(path) for path in OUTPUT.iterdir() if path.is_file()}}
    write_json(OUTPUT / 'verification.json', report)
    print(json.dumps({'recovered': report['runs_recovered'], 'remaining': len(report['remaining']),
        'means': [row for row in summary if row['algorithm'] in ('drsr', 'llmsr', 'e2esr', 'jaxsr', 'qlattice')]}, ensure_ascii=False), flush=True)


def publish():
    recovery = json.loads((OUTPUT / 'verification.json').read_text())
    for name, expected in recovery['files'].items():
        if sha(OUTPUT / name) != expected:
            raise ValueError('恢复产物哈希不一致')
    recovery['script_path'], recovery['script_sha256'] = str(Path(__file__)), sha(Path(__file__))
    write_json(OUTPUT / 'verification.json', recovery)
    with gzip.open(OUTPUT / 'terminal_run_metrics.jsonl.gz', 'rt') as handle:
        updated = {row['logical_key']: row for row in map(json.loads, handle)}
    with gzip.open(SNAPSHOT / 'terminal_run_metrics.jsonl.gz', 'rt') as handle:
        original = {row['logical_key']: row for row in map(json.loads, handle)}
    if set(updated) != set(original) or len(updated) != 6750:
        raise ValueError('恢复前后运行身份不一致')
    for key, row in updated.items():
        excluded = {'eff_score', 'eff_missing_minutes', 'eff_recovery'}
        if {k: v for k, v in row.items() if k not in excluded} != {k: v for k, v in original[key].items() if k not in excluded}:
            raise ValueError('EFF以外的指标发生变化')
    by_group = defaultdict(list)
    for row in updated.values():
        by_group[row['condition'], row['algorithm']].append(row['eff_score'])
    with (OUTPUT / 'eff_means.csv').open(newline='') as handle:
        means = {(row['condition'], row['algorithm']): row for row in csv.DictReader(handle)}
    for key, values in by_group.items():
        valid = [value for value in values if value is not None]
        row = means[key]
        if len(values) != 150 or len(valid) != int(row['EFF_available']):
            raise ValueError('EFF统计数量不符')
        if not math.isclose(float(row['EFF']), 100 * sum(valid) / len(valid), abs_tol=1e-12):
            raise ValueError('EFF均值复算不符')
    for name in ('algorithm_six_axis.csv', 'noise_supplement.csv'):
        with (SNAPSHOT / name).open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = [*reader.fieldnames, 'EFF_aggregation']
            rows = list(reader)
        for row in rows:
            mean = means[row['condition'], row['algorithm']]
            row.update({key: mean[key] for key in ('EFF', 'EFF_available', 'EFF_expected', 'EFF_aggregation')})
        with (METRICS / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    csv.field_size_limit(100000000)
    with (SNAPSHOT / 'terminal_run_metrics.csv').open(newline='') as source, \
            (METRICS / 'terminal_run_metrics.csv').open('w', newline='') as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            value = updated[row['logical_key']]
            row.update(EFF=value['eff_score'], eff_missing_minutes=canonical(value['eff_missing_minutes']))
            writer.writerow(row)
    shutil.copy2(OUTPUT / 'terminal_run_metrics.jsonl.gz', METRICS / 'terminal_run_metrics.jsonl.gz')
    issues = json.loads((SNAPSHOT / 'unresolved.json').read_text())
    retained = []
    for issue in issues['runs']:
        row = updated[issue['logical_key']]
        if row['eff_score'] is not None:
            issue['missing_axes'] = [axis for axis in issue['missing_axes'] if axis != 'EFF']
            issue['eff_missing_minutes'] = []
        if issue['missing_axes'] or issue['status'] != 'resolved':
            retained.append(issue)
    issues['runs'] = retained
    issues['axis_counts']['EFF'] = sum(row['eff_score'] is None for row in updated.values())
    write_json(METRICS / 'unresolved.json', issues)
    report = json.loads((SNAPSHOT / 'verification.json').read_text())
    report['terminal_unresolved'] = issues['axis_counts']
    report['eff_aggregation'] = 'available_runs_with_explicit_denominator'
    report['eff_recovery'] = {'path': str(OUTPUT / 'verification.json'),
                            'sha256': sha(OUTPUT / 'verification.json'), 'recovered_runs': recovery['runs_recovered']}
    report['formal_ready'] = not any(issues['axis_counts'].values())
    for name in report['files']:
        report['files'][name] = {'path': str(METRICS / name), 'sha256': sha(METRICS / name)}
    for name in ('eff_recovery/verification.json', 'eff_recovery/recovered_run_minutes.jsonl.gz', 'eff_recovery/eff_means.csv'):
        report['files'][name] = {'path': str(METRICS / name), 'sha256': sha(METRICS / name)}
    write_json(METRICS / 'verification.json', report)
    print(json.dumps({'published': str(METRICS), 'runs_recovered': recovery['runs_recovered'],
                      'remaining': issues['axis_counts']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish()
    else:
        main()
