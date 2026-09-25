import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import clean_task_builder as builder
from build_core50_formula_recovery import build_entry
from run_core50_comparisons import canonical, sha, write_json


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.agent/work/GOAL-CORE50'
REMOTE = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime')
CONDITIONS = ('clean', 'noise001', 'noise005')


def read_freeze(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_one(arguments):
    sys.setrecursionlimit(20000)
    row, gt, probes, output = arguments
    source = row['source']
    saved = output / 'records' / f"{source['task_id']}.json"
    if saved.exists():
        existing = json.loads(saved.read_text())
        if existing['result_sha256'] != row['result']['sha256']:
            raise ValueError(f"{source['task_id']}: 输入版本改变")
        return existing
    payload = json.loads(row['result']['raw_text'])
    expression, _ = builder.select_formula_with_source(payload)
    if expression and any(isinstance(node, ast.Call) and ast.unparse(node.func) in ('np.linalg.norm', 'linalg.norm')
                          for node in ast.walk(ast.parse(expression))):
        return {'task_id': source['task_id'], 'result_sha256': row['result']['sha256'],
                'task': None, 'no_call': None, 'unresolved': 'batch_dependent_numpy_norm',
                'expression': expression}
    contract = builder._load_prompt_schema(ROOT, prompt_path=ROOT /
        'AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt')
    recovery, issue = build_entry(row, output / 'recovery')
    if issue:
        return {'task_id': source['task_id'], 'result_sha256': row['result']['sha256'],
                'task': None, 'no_call': None, 'unresolved': issue['reason'], 'evidence': issue}
    task, no_call = builder._build_pred_task(row, contract=contract,
        ground_truth_variables={x['dataset_id']: x['ordered_variables'] for x in gt},
        ground_truth_targets={x['dataset_id']: x['target'] for x in gt},
        recovery_entries={source['task_id']: recovery} if recovery else {},
        dataset_probes=probes, condition=source['noise_tag'])
    record = task.to_json_record() if task else None
    if record:
        for field in ('prompt_path', 'schema_path'):
            record[field] = str(REMOTE / Path(record[field]).relative_to(ROOT))
    return {'task_id': source['task_id'], 'result_sha256': row['result']['sha256'],
            'task': record, 'no_call': no_call}


def main():
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=WORK / 'remaining_terminal')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = ROOT / 'AAAI_experiments/stage5_metric_calculation_0831/reports'
    gt = builder._read_jsonl(reports / 'ground_truth_extract.jsonl')
    probes, probe_sha = builder.load_dataset_probes(reports / 'dataset_probes.jsonl')
    pending = []
    manifests = []
    for condition in CONDITIONS:
        old_path = WORK / 'source_freezes_collected_v5' / f'{condition}_runs_available.jsonl.gz'
        full_path = WORK / 'source_freezes_full' / f'{condition}_runs.jsonl.gz'
        old = {row['source']['task_id']: row['result']['sha256'] for row in read_freeze(old_path)}
        full = read_freeze(full_path)
        if len(full) != 2250:
            raise ValueError(f'{condition}: 训练输入数量异常')
        for row in full:
            key = row['source']['task_id']
            if key in old:
                if old[key] != row['result']['sha256']:
                    raise ValueError(f'{key}: 训练来源改变')
            else:
                pending.append((row, gt, probes, args.output))
        manifests.append({'path': str(full_path), 'sha256': sha(full_path), 'previous_sha256': sha(old_path)})
    if len(pending) != 1441:
        raise ValueError(f'待处理数量变化: {len(pending)}')
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build_one, pending, chunksize=1):
            path = args.output / 'records' / f"{result['task_id']}.json"
            write_json(path, result)
            if result['task']:
                plan_path = args.output / 'plans' / f"missing_{result['task_id']}.jsonl"
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                content = canonical(result['task']) + '\n'
                if plan_path.exists() and plan_path.read_text() != content:
                    raise ValueError(f'计划内容变化: {plan_path}')
                plan_path.write_text(content)
                counts['callable'] += 1
            else:
                counts['unresolved' if result.get('unresolved') else 'no_call'] += 1
            if sum(counts.values()) % 50 == 0:
                print(dict(counts), flush=True)
    write_json(args.output / 'manifest.json', {'counts': dict(counts), 'inputs': manifests,
        'dataset_probes_sha256': probe_sha, 'max_attempts_per_task': 6,
        'max_calls': counts['callable'] * 6, 'retry_cap': counts['callable'] * 5})
    print(dict(counts), flush=True)


if __name__ == '__main__':
    main()
