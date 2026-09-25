import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sys
import time

from aggregate_core50_minute_metrics import terminal_metrics
from run_core50_comparisons import canonical, sha, write_json
from run_core50_minute_opus import read_minutes


def aggregate(root, output):
    terminal = root.parent / 'core50_comparisons'
    status = json.loads((terminal / 'terminal_continuation.json').read_text())
    if not status['complete']:
        raise ValueError('最终公式比较尚未结束')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'verification.json').exists():
        raise FileExistsError('目标已有汇总验收文件')
    print(json.dumps({'phase': 'reading_numeric_trajectories', 'time': time.time()}), flush=True)
    efficiencies, counts, qualities, missing = {}, Counter(), defaultdict(list), defaultdict(list)
    for row in read_minutes(root):
        key = row['logical_key']
        counts[key] += 1
        if row['minute'] != counts[key]:
            raise ValueError(f'数值轨迹分钟顺序异常: {key}')
        if row['q'] is None:
            missing[key].append(row['minute'])
        else:
            qualities[key].append(row['q'])
        if row['minute'] == 180:
            efficiencies[key] = row
            row['missing_minutes'] = missing[key]
    if len(counts) != 6750 or set(counts.values()) != {180} or len(efficiencies) != 6750:
        raise ValueError('数值轨迹覆盖不足')
    for key, row in efficiencies.items():
        if missing[key]:
            if row['cumulative_eff'] is not None:
                raise ValueError('轨迹缺失时EFF应保持空缺')
            continue
        maximum = max(qualities[key])
        expected = sum(qualities[key]) / (180 * maximum) if maximum else 0.0
        if row['cumulative_eff'] is None or not math.isclose(expected, row['cumulative_eff'], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f'EFF复算不一致: {key}')
    print(json.dumps({'phase': 'aggregating_terminal_metrics', 'time': time.time()}), flush=True)
    unresolved = terminal_metrics(root, output, efficiencies, export_sources=True)
    fields = ['condition', 'algorithm', 'dataset_index', 'dataset_id', 'seed', 'logical_key',
              'status', 'reason', 'ID', 'OOD', 'SYM', 'MIN', 'EFF', 'STAB',
              'original_expression', 'effective_expression', 'simplification_outcome',
              'source_path', 'source_sha256', 'eff_missing_minutes']
    axes = {'ID': 'id_quality', 'OOD': 'ood_quality', 'SYM': 'sym_score', 'MIN': 'min_score',
            'EFF': 'eff_score', 'STAB': 'stab_score'}
    identities, issues = set(), []
    with gzip.open(output / 'terminal_run_metrics.jsonl.gz', 'rt') as source, \
            (output / 'terminal_run_metrics.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for line in source:
            row = json.loads(line)
            identity = (row['condition'], row['algorithm'], row['dataset_id'], row['seed'])
            if identity in identities:
                raise ValueError('最终运行重复')
            identities.add(identity)
            formula = row.get('simplification') or {}
            result = {name: row.get(name) for name in fields}
            result.update({axis: row[field] for axis, field in axes.items()})
            result.update(original_expression=formula.get('original_expression'),
                          effective_expression=formula.get('effective_expression'),
                          simplification_outcome=formula.get('outcome'),
                          eff_missing_minutes=canonical(row['eff_missing_minutes']))
            writer.writerow(result)
            missing_axes = [axis for axis, field in axes.items() if row[field] is None]
            if missing_axes or row['status'] != 'resolved':
                issues.append({'logical_key': row['logical_key'], 'condition': row['condition'],
                    'algorithm': row['algorithm'], 'dataset_index': row['dataset_index'], 'seed': row['seed'],
                    'status': row['status'], 'reason': row.get('reason'), 'missing_axes': missing_axes,
                    'eff_missing_minutes': row['eff_missing_minutes'], 'source_sha256': row['source_sha256']})
    if len(identities) != 6750:
        raise ValueError('最终运行数量不符')
    input_manifests = {str(path): sha(path) for path in sorted((root / 'inputs/minutes').glob('*/*/manifest.json'))}
    input_manifests.update({str(path): sha(path) for path in sorted((root / 'inputs/terminal_numeric').glob('*.manifest.json'))})
    write_json(output / 'input_manifest.json', {'scope': 'terminal_formulas_only',
        'manifests': input_manifests, 'terminal_state_sha256': sha(terminal / 'terminal_continuation.json'),
        'dependencies_sha256': sha(terminal / 'dependencies.json')})
    write_json(output / 'unresolved.json', {'axis_counts': unresolved, 'runs': issues})
    report = {'scope': 'terminal_formulas_only', 'run_count': 6750, 'task_count': 2250,
        'numeric_run_minute_count': sum(counts.values()), 'terminal_unresolved': unresolved,
        'formal_ready': not unresolved, 'new_api_requests': 0,
        'eff_aggregation': 'available_runs_with_explicit_denominator',
        'score_scale': {'run': '0..1', 'algorithm': '0..100'},
        'files': {path.name: {'sha256': sha(path), 'path': str(path)} for path in sorted(output.iterdir()) if path.is_file()}}
    write_json(output / 'verification.json', report)
    return report


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output)), flush=True)
