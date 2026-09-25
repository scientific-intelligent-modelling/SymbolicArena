from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import select_formula_with_source
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import phi_nmse
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import PerformanceReplayCache, PerformanceReplayError, replay_payload_performance
from run_core50_comparisons import canonical, sha, write_json


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.agent/work/GOAL-CORE50'
CACHE = PerformanceReplayCache()


def evaluate(row):
    source = row['source']
    raw = row['result']['raw_text']
    result_sha = hashlib.sha256(raw.encode()).hexdigest()
    if result_sha != row['result']['sha256']:
        raise ValueError('最终数值输入哈希不一致')
    payload = json.loads(raw)
    record = {'condition': source['noise_tag'], 'algorithm': source['algorithm'],
        'dataset_id': source['dataset_id'], 'dataset_index': source['dataset_index'], 'seed': int(source['seed']),
        'logical_key': f"{source['algorithm']}::{source['dataset_id']}::s{source['seed']}::{source['noise_tag']}",
        'source_path': source['path'], 'source_sha256': result_sha,
        'id_quality': None, 'ood_quality': None, 'valid_output': None, 'status': 'unresolved',
        'native_id_nmse': (payload.get('id_test') or {}).get('nmse'),
        'native_ood_nmse': (payload.get('ood_test') or {}).get('nmse')}
    expression, _ = select_formula_with_source(payload)
    if not expression:
        return {**record, 'id_quality': 0.0, 'ood_quality': 0.0, 'valid_output': False,
            'status': 'invalid', 'reason': 'missing_budget_expression'}
    recovery = WORK / 'remaining_terminal/recovered_parameters' / f"recovered_params_s{source['seed']}.jsonl"
    if source['algorithm'] == 'llmsr' and source['dataset_index'] == 'g0641' and source['noise_tag'] == 'noise005' and recovery.exists():
        request = json.loads(recovery.read_text())['request']
        if request['ast_source_evidence']['result_raw_sha256'] != result_sha:
            raise ValueError('参数恢复来源不一致')
        payload = dict(payload)
        payload['equation'] = f"def equation({', '.join(payload['feature_names'])}):\n    return {request['expression']}"
        payload['canonical_artifact'] = None
        record['formula_recovery'] = {'plan_path': str(recovery), 'plan_sha256': sha(recovery),
                                      'evidence': request['ast_source_evidence']}
    try:
        replay = replay_payload_performance(payload, algorithm=source['algorithm'], repo_root=ROOT,
            cache=CACHE, task_id=source['task_id'], condition=source['noise_tag'], result_sha256=result_sha)
    except PerformanceReplayError as exc:
        return {**record, 'reason': str(exc)}
    return {**record, **replay, 'status': 'resolved' if replay['valid_output'] else 'invalid',
            'reason': replay['invalid_reason']}


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    output = WORK / 'terminal_numeric_full'
    output.mkdir(parents=True, exist_ok=True)
    for condition in ('clean', 'noise001', 'noise005'):
        source = WORK / 'source_freezes_full' / f'{condition}_runs.jsonl.gz'
        destination = output / f'{condition}.jsonl.gz'
        complete = output / f'{condition}.manifest.json'
        if complete.exists():
            report = json.loads(complete.read_text())
            if report['source_sha256'] != sha(source) or report['output_sha256'] != sha(destination):
                raise ValueError('已计算的最终数值证据变化')
            continue
        with gzip.open(source, 'rt') as handle:
            inputs = [json.loads(line) for line in handle]
        counts = Counter()
        with ProcessPoolExecutor(max_workers=2) as pool, gzip.open(destination.with_suffix('.pending'), 'wt') as handle:
            for result in pool.map(evaluate, inputs, chunksize=5):
                handle.write(canonical(result) + '\n')
                counts[result['status']] += 1
        destination.with_suffix('.pending').replace(destination)
        write_json(complete, {'source_sha256': sha(source), 'output_sha256': sha(destination),
            'runs': len(inputs), 'states': dict(counts), 'evaluation_path': 'canonical_replay.v1'})
        print({'condition': condition, 'states': dict(counts)}, flush=True)
