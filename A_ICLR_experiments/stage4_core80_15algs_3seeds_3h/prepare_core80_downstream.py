from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import resource
import signal
import sqlite3
import sys
import time

from sympy.core.cache import clear_cache

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as builder
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index import _resolve_simplify_effective_expression


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / '.agent/work/EXP-001/opus_postprocess'
WORK = ROOT / '.agent/work/EXP-001/followup'
OUTPUT = WORK / 'downstream_plans'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def initialize_worker():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))


def context_binding(prefix, plan, frozen):
    return {f'{prefix}_logical_id': plan.logical_id,
            f'{prefix}_plan_evaluation_key': plan.evaluation_key,
            f'{prefix}_frozen_plan_sha256': frozen.plan_sha256,
            f'{prefix}_frozen_evaluation_key': frozen.evaluation_key,
            f'{prefix}_simplify_status': frozen.simplified_status,
            f'{prefix}_expression_resolution': frozen.expression_resolution}


def pair_evidence(arguments):
    def expired(signum, frame):
        raise TimeoutError('pair evidence exceeded 180 seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 180)
    try:
        return {'evidence': builder._build_full_pair_evidence(**arguments)}
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}'}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        clear_cache()
        gc.collect()


def prepare_pass():
    sys.setrecursionlimit(100000)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    emitted = {row['logical_id'] for path in OUTPUT.glob('batch_*.jsonl') for row in rows(path)}
    base = json.loads((WORK / 'base/selected.json').read_text())
    prior = json.loads((ROOT / '.agent/work/EXP-001/oversample/opus/selected.json').read_text())
    bindings = {}
    old_db = OLD / 'state/opus_pred_simplify.sqlite3'
    with sqlite3.connect(f'file:{old_db}?mode=ro', uri=True) as connection:
        for key, path, digest in connection.execute('SELECT evaluation_key,result_path,result_sha256 FROM frozen_results'):
            bindings[key] = {'result_path': path, 'result_sha256': digest, 'source_db': str(old_db)}
    for key, record in {**prior, **base}.items():
        bindings[key] = {**record, 'source_db': record.get('source_db') or str(Path(record['result_path']).parent.parent / 'state.sqlite3')}
    plans = {}
    for path in [OLD / 'plans/simplify_core80.jsonl', WORK / 'base_plan.jsonl']:
        plan_sha = sha(path)
        for row in rows(path):
            plans[row['logical_id']] = (row, plan_sha)
    predictions = {}
    numeric = {}
    ground_truth = {}
    dependencies = {}
    unresolved = []
    for logical_id, (row, plan_sha) in plans.items():
        key = row['evaluation_key']
        if key not in bindings:
            unresolved.append({'logical_id': logical_id, 'reason': 'missing_frozen_dependency'})
            continue
        binding = bindings[key]
        path = Path(binding['result_path'])
        assert sha(path) == binding['result_sha256']
        frozen = json.loads(path.read_text())
        assert frozen['evaluation_key'] == key and frozen['request'] == row['request']
        assert frozen['validation']['ok'] is True
        effective, resolution = _resolve_simplify_effective_expression(request=row['request'],
                                        structured_output=frozen['structured_output'], context=logical_id)
        plan_record = builder.SimplifyPlanRecord(logical_id, row['task_type'], key, row['priority'], row['request'])
        frozen_record = builder.FrozenSimplifyRecord(key, logical_id, row['task_type'], plan_sha, 'frozen',
                            frozen['structured_output']['outcome'], frozen['structured_output'].get('simplified_expression'),
                            effective, resolution, frozen['structured_output'], None, binding['result_sha256'])
        dependencies[key] = binding
        item = (plan_record, frozen_record)
        request = row['request']
        if row['task_type'] == 'gt_simplify':
            ground_truth[request['dataset_id']] = item
        else:
            result_path = Path(request['ast_source_evidence']['result_path'])
            if sha(result_path) != request['ast_source_evidence']['result_raw_sha256']:
                unresolved.append({'logical_id': logical_id, 'reason': 'source_result_changed'})
                continue
            payload = json.loads(result_path.read_text())
            valid = all(isinstance(payload.get(split), dict) and isinstance(payload[split].get('nmse'), (int, float))
                        and math.isfinite(payload[split]['nmse']) for split in ['id_test', 'ood_test'])
            numeric[key] = {'valid': valid, 'task_id': request['task_id'],
                            'result_sha256': request['ast_source_evidence']['result_raw_sha256']}
            predictions[(row['condition'], request['algorithm_slug'], request['dataset_index'], request['seed'])] = item
    assert len(ground_truth) == 80
    temporary = WORK / 'dependencies.pending'
    temporary.write_text(json.dumps(dependencies, ensure_ascii=False) + '\n')
    temporary.replace(WORK / 'dependencies.json')
    contracts = {kind: builder._load_prompt_schema(ROOT, task_kind=kind) for kind in ['equivalence', 'structure']}
    jobs = []
    for (condition, algorithm, dataset_index, seed), prediction in predictions.items():
        pred, pred_frozen = prediction
        gt, gt_frozen = ground_truth[pred.request['dataset_id']]
        logical_id = f'equivalence::{algorithm}::{dataset_index}::s{seed}::{condition}'
        request = {'dataset_id': pred.request['dataset_id'], 'dataset_index': dataset_index,
                   'algorithm': pred.request['algorithm'], 'algorithm_slug': algorithm,
                   'seed': seed, 'noise_tag': condition, 'variables': pred.request['variables'],
                   'effective_ground_truth_expression': gt_frozen.effective_expression,
                   'simplified_ground_truth_expression': gt_frozen.simplified_expression,
                   'effective_prediction_expression': pred_frozen.effective_expression,
                   'simplified_prediction_expression': pred_frozen.simplified_expression,
                   **context_binding('ground_truth', gt, gt_frozen),
                   **context_binding('prediction', pred, pred_frozen),
                   'prediction_task_id': numeric[pred.evaluation_key]['task_id'],
                   'prediction_valid_output': numeric[pred.evaluation_key]['valid'],
                   'prediction_result_sha256': numeric[pred.evaluation_key]['result_sha256']}
        if logical_id not in emitted:
            jobs.append((condition, logical_id, 'equivalence', gt, gt_frozen, pred, pred_frozen, seed, request))
    grouped = defaultdict(dict)
    for (condition, algorithm, dataset_index, seed), item in predictions.items():
        grouped[condition, algorithm, dataset_index][seed] = item
    for (condition, algorithm, dataset_index), seeds in grouped.items():
        for seed_a, seed_b in combinations([520, 521, 522], 2):
            logical_id = builder._structure_logical_id(algorithm, dataset_index, seed_a, seed_b, condition=condition)
            if seed_a not in seeds or seed_b not in seeds:
                unresolved.append({'logical_id': logical_id, 'reason': 'missing_seed_expression'})
                continue
            left, left_frozen = seeds[seed_a]
            right, right_frozen = seeds[seed_b]
            if not (numeric[left.evaluation_key]['valid'] and numeric[right.evaluation_key]['valid']):
                unresolved.append({'logical_id': logical_id, 'reason': 'numeric_metrics_unavailable'})
                continue
            request = {'dataset_id': left.request['dataset_id'], 'dataset_index': dataset_index,
                       'algorithm': left.request['algorithm'], 'algorithm_slug': algorithm, 'noise_tag': condition,
                       'seed_a': seed_a, 'seed_b': seed_b, 'variables': left.request['variables'],
                       'effective_prediction_a_expression': left_frozen.effective_expression,
                       'effective_prediction_b_expression': right_frozen.effective_expression,
                       'simplified_prediction_a_expression': left_frozen.simplified_expression,
                       'simplified_prediction_b_expression': right_frozen.simplified_expression,
                       **context_binding('prediction_a', left, left_frozen),
                       **context_binding('prediction_b', right, right_frozen),
                       'prediction_a_task_id': numeric[left.evaluation_key]['task_id'],
                       'prediction_b_task_id': numeric[right.evaluation_key]['task_id'],
                       'prediction_a_valid_output': True, 'prediction_b_valid_output': True,
                       'prediction_a_result_sha256': numeric[left.evaluation_key]['result_sha256'],
                       'prediction_b_result_sha256': numeric[right.evaluation_key]['result_sha256']}
            if logical_id not in emitted:
                jobs.append((condition, logical_id, 'structure', left, left_frozen, right, right_frozen, seed_a*1000+seed_b, request))
    chunks = []
    count = max((int(path.stem.split('_')[1]) for path in OUTPUT.glob('batch_*.jsonl')), default=0)
    def flush():
        nonlocal count
        if not chunks:
            return
        count += 1
        path = OUTPUT / f'batch_{count:05d}.jsonl'
        assert not path.exists()
        temporary = path.with_suffix('.pending')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in chunks))
        temporary.replace(path)
        emitted.update(row['logical_id'] for row in chunks)
        chunks.clear()
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker) as pool:
        futures = {}
        for job in jobs:
            condition, logical_id, kind, left, left_frozen, right, right_frozen, seed, context = job
            arguments = dict(logical_id=logical_id, phase=kind, left_plan=left, left_frozen=left_frozen,
                             right_plan=right, right_frozen=right_frozen, pair_seed=seed)
            futures[pool.submit(pair_evidence, arguments)] = job
        for future in as_completed(futures):
            condition, logical_id, kind, left, left_frozen, right, right_frozen, seed, context = futures[future]
            response = future.result()
            if 'error' in response:
                unresolved.append({'logical_id': logical_id, 'reason': response['error']})
                continue
            evidence = response['evidence']
            request = {**context, 'allowed_functions': evidence['allowed_functions'],
                       'deterministic_evidence': evidence, 'evidence_hash': evidence['evidence_sha256']}
            if kind == 'structure':
                request['deterministic_pair_evidence'] = evidence
            task = builder._task_from_request(logical_id=logical_id,
                        task_type='equivalence' if kind == 'equivalence' else 'stab_structure',
                        priority=builder.EQUIVALENCE_PRIORITY if kind == 'equivalence' else builder.STRUCTURE_PRIORITY,
                        request=request, evidence_hash=evidence['evidence_sha256'], contract=contracts[kind],
                        dependencies=(left_frozen.evaluation_key, right_frozen.evaluation_key), condition=condition)
            chunks.append(builder._task_json_record(task))
            if len(chunks) >= 100:
                flush()
                print(json.dumps({'batches': count, 'unresolved': len(unresolved)}), flush=True)
    flush()
    (OUTPUT / 'unresolved.json').write_text(json.dumps(unresolved, ensure_ascii=False, indent=2)+'\n')
    missing = sum(item['reason'] == 'missing_frozen_dependency' for item in unresolved)
    if missing == 0:
        (OUTPUT / 'complete.json').write_text(json.dumps({'batches': count, 'prepared_tasks': len(emitted), 'unresolved': len(unresolved)})+'\n')
    return missing == 0


def main():
    while not prepare_pass():
        time.sleep(10)


if __name__ == '__main__':
    main()
