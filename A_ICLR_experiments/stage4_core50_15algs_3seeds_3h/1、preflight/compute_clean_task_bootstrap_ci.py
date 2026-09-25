import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = Path(__file__).resolve().parents[1] / '3、metrics'
AXES = ('ID', 'OOD', 'SYM', 'MIN', 'EFF', 'STAB')
FIELDS = ('id_quality', 'ood_quality', 'sym_score', 'min_score', 'eff_score')
SEEDS = (520, 521, 522)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    source = METRICS / 'terminal_run_metrics.jsonl.gz'
    point_path = METRICS / 'algorithm_six_axis.csv'
    verification = json.loads((METRICS / 'verification.json').read_text())
    for path in (source, point_path):
        if sha(path) != verification['files'][path.name]['sha256']:
            raise ValueError(f'输入版本与当前汇总不符: {path.name}')
    with point_path.open(newline='') as handle:
        point_rows = list(csv.DictReader(handle))
    algorithms = [row['algorithm'] for row in point_rows]
    with gzip.open(source, 'rt') as handle:
        records = [row for row in map(json.loads, handle) if row['condition'] == 'clean']
    tasks = sorted({row['dataset_index'] for row in records})
    if len(records) != 2250 or len(set(algorithms)) != 15 or len(tasks) != 50:
        raise ValueError('clean输入必须覆盖15算法、50任务、3个seed')
    algorithm_index = {name: i for i, name in enumerate(algorithms)}
    task_index = {name: i for i, name in enumerate(tasks)}
    seed_index = {seed: i for i, seed in enumerate(SEEDS)}
    run_values = np.full((15, 50, 3, 5), np.nan)
    stab_values = np.full((15, 50, 3), np.nan)
    seen, dataset_names, input_rows = set(), {}, []
    for row in records:
        identity = (row['algorithm'], row['dataset_index'], row['seed'])
        if identity in seen:
            raise ValueError(f'重复运行: {identity}')
        seen.add(identity)
        a, t, s = algorithm_index[identity[0]], task_index[identity[1]], seed_index[identity[2]]
        if dataset_names.setdefault(identity[1], row['dataset_id']) != row['dataset_id']:
            raise ValueError('任务编号对应的数据集名称不一致')
        values = [row[field] for field in FIELDS]
        if any(value is None for value in values) or row['stab_score'] is None:
            raise ValueError(f'clean存在缺失指标: {identity}')
        run_values[a, t, s] = values
        stab_values[a, t, s] = row['stab_score']
        input_rows.append({'algorithm': identity[0], 'task': identity[1], 'dataset': row['dataset_id'],
                           'seed': identity[2], **dict(zip(AXES[:5], values))})
    for values in (run_values, stab_values):
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError('输入指标必须完整且位于0至1')
    if not np.all(stab_values == stab_values[:, :, :1]):
        raise ValueError('同一algorithm-task的STAB在三个seed记录中不一致')
    task_stab = stab_values[:, :, 0]
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, 50, size=(1000, 50))
    # 同一组task索引用于所有算法和指标，每个task的三个seed整体保留。
    run_bootstrap = run_values[:, draws, :, :].mean(axis=(2, 3))
    stab_bootstrap = task_stab[:, draws].mean(axis=2)
    bootstrap = np.concatenate((run_bootstrap, stab_bootstrap[:, :, None]), axis=2)
    means = np.concatenate((run_values.mean(axis=(1, 2)), task_stab.mean(axis=1)[:, None]), axis=1)
    counts = np.array([np.bincount(draw, minlength=50) for draw in draws])
    task_values = np.concatenate((run_values.mean(axis=2), task_stab[:, :, None]), axis=2)
    weighted = np.einsum('bt,atc->abc', counts, task_values) / 50
    np.testing.assert_allclose(bootstrap, weighted, rtol=1e-12, atol=1e-12)
    reference = np.array([[float(row[axis]) for axis in AXES] for row in point_rows])
    np.testing.assert_allclose(means * 100, reference, rtol=1e-12, atol=1e-12)
    lower, upper = np.quantile(bootstrap * 100, [0.025, 0.975], axis=1, method='linear')
    if lower.shape != (15, 6) or np.any(lower > upper):
        raise ValueError('置信区间维度或上下界异常')
    output = METRICS / 'clean_six_axis_task_bootstrap_ci.csv'
    evidence = METRICS / 'clean_task_bootstrap'
    evidence.mkdir(exist_ok=True)
    output_rows = [{'algorithm': algorithm, 'axis': axis, 'mean': float(means[a, j] * 100),
                    'ci_lower': float(lower[a, j]), 'ci_upper': float(upper[a, j])}
                   for a, algorithm in enumerate(algorithms) for j, axis in enumerate(AXES)]
    if len(output_rows) != 90:
        raise ValueError('输出必须为90行')
    write_csv(output, ('algorithm', 'axis', 'mean', 'ci_lower', 'ci_upper'), output_rows)
    write_csv(evidence / 'clean_run_scores.csv', ('algorithm', 'task', 'dataset', 'seed', *AXES[:5]),
              sorted(input_rows, key=lambda row: (row['algorithm'], row['task'], row['seed'])))
    write_csv(evidence / 'clean_task_stab.csv', ('algorithm', 'task', 'dataset', 'STAB'),
              [{'algorithm': algorithm, 'task': task, 'dataset': dataset_names[task], 'STAB': float(task_stab[a, t])}
               for a, algorithm in enumerate(algorithms) for t, task in enumerate(tasks)])
    np.save(evidence / 'task_draws.npy', draws, allow_pickle=False)
    report = {'condition': 'clean', 'runs': 2250, 'algorithm_tasks': 750, 'output_rows': 90,
        'task_count_per_replicate': 50, 'replicates': 1000, 'seed': args.seed,
        'seeds_per_task': list(SEEDS), 'shared_draws_across_algorithms_and_axes': True,
        'sampling': 'tasks_with_replacement_preserving_three_seeds', 'ci_method': 'percentile',
        'quantiles': [0.025, 0.975], 'quantile_interpolation': 'linear',
        'mean_definition': 'original_sample_mean', 'input_scale': '0-1', 'output_scale': '0-100',
        'numpy_version': np.__version__, 'rng': type(rng.bit_generator).__name__,
        'algorithms': algorithms, 'task_order': tasks, 'source_sha256': {path.name: sha(path) for path in (source, point_path)},
        'script_sha256': sha(Path(__file__)), 'mean_matches_current_summary': True,
        'run_resampling_matches_task_multiplicities': True,
        'files': {str(path.relative_to(METRICS)): sha(path) for path in
                  (output, evidence / 'clean_run_scores.csv', evidence / 'clean_task_stab.csv', evidence / 'task_draws.npy')}}
    (evidence / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'rows': len(output_rows), 'replicates': 1000,
                      'scale': '0-100', 'seed': args.seed, 'mean_matches_summary': True}))


if __name__ == '__main__':
    main()
