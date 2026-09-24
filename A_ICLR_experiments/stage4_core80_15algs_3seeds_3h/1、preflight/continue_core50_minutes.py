import argparse
import fcntl
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import time

from aggregate_core50_minute_metrics import aggregate
from run_core50_comparisons import run, sha, write_json
from run_core50_minute_opus import prepare_pairs, prepare_simplifications, read_minutes


def main():
    sys.setrecursionlimit(20000)
    runtime = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime')
    root = runtime / 'core50_minutes'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args = argparse.Namespace(runtime=runtime, root=root, workers=300, prepare_workers=50,
        logical_task_cap=1000000, channel_settings='routify=/home/zhangziwen/.config/core50-opus/routify.json')
    status_path = root / 'pipeline_status.json'
    receipt = root / 'inputs.complete.json'
    while not receipt.exists():
        time.sleep(10)
    inputs = json.loads(receipt.read_text())
    if inputs['run_count'] != 6750 or inputs['generator_version'] != 5 or len(inputs['shards']) != 45:
        raise ValueError('逐分钟输入版本或覆盖不完整')
    for relative, expected_sha in inputs['shards'].items():
        path = root / 'inputs/minutes' / relative / 'manifest.json'
        if sha(path) != expected_sha:
            raise ValueError(f'逐分钟输入清单变化: {relative}')
    write_json(status_path, {'phase': 'preparing_simplifications', 'time': time.time(), 'complete': False})
    report = prepare_simplifications(args)
    groups = {}
    for row in read_minutes(root):
        group = (row['condition'], row['algorithm'], row['dataset_id'], row['minute'])
        groups.setdefault(group, {})[row['seed']] = row['expression_key'] if row['valid_output'] else None
    pairs = set()
    for (_, _, dataset, _), seeds in groups.items():
        if set(seeds) != {520, 521, 522}:
            raise ValueError('逐分钟预算清单缺少seed')
        for a, b in combinations(seeds.values(), 2):
            if a and b and a in report['aliases'] and b in report['aliases']:
                pairs.add((dataset, *sorted((a, b))))
    requests = report['new_requests'] + len(report['aliases']) + len(pairs)
    budget = {'prediction_requests': report['new_requests'], 'reused_predictions': report['reused'],
        'equivalence_requests_upper_bound': len(report['aliases']), 'structure_requests_upper_bound': len(pairs),
        'total_requests_upper_bound': requests, 'max_retries_per_task': 5, 'retry_cap': requests * 5,
        'total_attempts_upper_bound': requests * 6, 'max_tokens': 65536, 'workers': 300,
        'first_pass_estimate_cny': [requests * 0.02, requests * 0.12],
        'unresolved_expressions': len(report['unresolved'])}
    write_json(root / 'budget.json', budget)
    write_json(status_path, {'phase': 'awaiting_budget_notification', 'time': time.time(), 'complete': False})
    notified = root / 'budget_notified.json'
    while not notified.exists():
        time.sleep(10)
    if json.loads(notified.read_text())['budget_sha256'] != sha(root / 'budget.json'):
        raise ValueError('预算通知版本不一致')
    terminal_status = runtime / 'core50_comparisons/terminal_continuation.json'
    write_json(status_path, {'phase': 'minute_simplification_api', 'time': time.time(), 'complete': False})
    runner = runtime / 'core50_comparisons/bin/run_core50_minute_opus.py'
    subprocess.run([sys.executable, '-u', str(runner), 'simplify'], check=True)
    write_json(status_path, {'phase': 'preparing_minute_comparisons', 'time': time.time(), 'complete': False})
    subprocess.run([sys.executable, '-u', str(runner), 'pairs'], check=True)
    write_json(status_path, {'phase': 'minute_comparison_api', 'time': time.time(), 'complete': False})
    subprocess.run([sys.executable, '-u', str(runner), 'compare'], check=True)
    write_json(status_path, {'phase': 'waiting_terminal_comparisons', 'time': time.time(), 'complete': False})
    while not json.loads(terminal_status.read_text()).get('complete'):
        time.sleep(20)
    write_json(status_path, {'phase': 'aggregating_metrics', 'time': time.time(), 'complete': False})
    subprocess.run([sys.executable, '-u', str(runtime / 'core50_comparisons/bin/aggregate_core50_minute_metrics.py'),
                    '--root', str(root), '--output', str(root / 'results')], check=True)
    result = json.loads((root / 'results/verification.json').read_text())
    write_json(status_path, {'phase': 'verification_finished', 'time': time.time(), 'complete': True,
        'formal_ready': result['formal_ready'], 'verification': str(root / 'results/verification.json')})


if __name__ == '__main__':
    main()
