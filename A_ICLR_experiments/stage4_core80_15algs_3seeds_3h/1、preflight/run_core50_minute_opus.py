import argparse
from collections import defaultdict
from contextlib import closing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
import inspect
from itertools import combinations
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import clean_task_builder as clean
from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as symbolic
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index import _resolve_simplify_effective_expression
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import SymbolicEvidenceError
from run_core50_comparisons import (
    canonical, context_binding, dependencies, evidence_job, import_dependency,
    read_frozen_rows, rows, run, sha, source_index, write_json,
)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def expression_key(dataset, variables, expression):
    return digest({'dataset_id': dataset, 'variables': variables, 'expression': expression, 'semantics': 'canonical_replay.v1'})


def read_minutes(root):
    for path in sorted((root / 'inputs/minutes').glob('*/*/run_minutes.jsonl.gz')):
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if sha(path) != manifest['numeric_sha256']:
            raise ValueError(f'逐分钟证据哈希不一致: {path}')
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                yield json.loads(line)


def frozen_records(root):
    plan_paths = {row['evaluation_key']: path for path in (root / 'plans').glob('*.jsonl') for row in rows(path)}
    records = json.loads((root / 'dependencies.json').read_text())
    for db in (root / 'execution').glob('*/state.sqlite3'):
        with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as connection:
            keys = [row[0] for row in connection.execute('SELECT evaluation_key FROM frozen_results')]
        source_plans = {key: plan_paths[key] if key in plan_paths else Path(records[key]['source_plan']) for key in keys}
        for record in read_frozen_rows(db, plan_by_key=source_plans):
            records[record['task']['evaluation_key']] = record
    return records


def materialize(record):
    frozen = record['frozen']
    path = Path(frozen['result_path'])
    if sha(path) != frozen['result_sha256']:
        raise ValueError(f'裁决文件哈希不一致: {path}')
    artifact = json.loads(path.read_text())
    key = record['task']['evaluation_key']
    if artifact['evaluation_key'] != key or not artifact['validation']['ok']:
        raise ValueError(f'裁决绑定无效: {key}')
    return artifact


def simplify_pair(record):
    artifact = materialize(record)
    task, result = record['task'], artifact['structured_output']
    request = artifact['request']
    effective, resolution = _resolve_simplify_effective_expression(request=request,
        structured_output=result, context=task['logical_id'])
    plan = symbolic.SimplifyPlanRecord(task['logical_id'], task['task_type'], task['evaluation_key'], task['priority'], request)
    frozen = symbolic.FrozenSimplifyRecord(task['evaluation_key'], task['logical_id'], task['task_type'],
        record['source_plan_sha256'], 'frozen', result['outcome'], result.get('simplified_expression'),
        effective, resolution, result, None, record['frozen']['result_sha256'])
    return plan, frozen


def publish(root, filename, records):
    path = root / 'plans' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(canonical(row) + '\n' for row in records)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'已发布计划变化: {path}')
        return
    temporary = path.with_suffix('.pending')
    temporary.write_text(text)
    temporary.replace(path)


def build_simplification(arguments):
    row, probe, runtime = arguments
    sys.setrecursionlimit(20000)
    contract = clean._load_prompt_schema(runtime, prompt_path=runtime /
        'AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt')
    try:
        functions, assumptions, evidence = clean._build_symbolic_request_evidence(
            expression=row['expression'], variables=row['variables'], probe=probe)
    except (clean.CleanTaskBuilderError, SymbolicEvidenceError) as exc:
        return row['key'], None, str(exc)
    request = {'dataset_id': row['dataset_id'], 'dataset_index': row['dataset_index'],
        'variables': row['variables'], 'allowed_functions': functions,
        'expression': row['expression'], 'original_expression': row['expression'],
        'domain_assumptions': assumptions, 'probe_points': probe['points'],
        'probe_source': probe['schema_version'], 'probe_sample_sha256': probe['sample_sha256'],
        'dataset_probe_evidence': probe, 'deterministic_evidence': evidence,
        'source_evidence': row, 'scope': 'shared_minute_expression'}
    request['evidence_hash'] = digest(request)
    task = clean._build_task_definition(logical_id=f"minute_pred::{row['key']}", task_type='pred_simplify',
        priority=clean.PRED_PRIORITY, request=request, evidence_hash=request['evidence_hash'],
        contract=contract, condition='clean')
    return row['key'], task.to_json_record(), None


def prepare_simplifications(args):
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    completed = root / 'simplification_plan.json'
    if completed.exists():
        return json.loads(completed.read_text())
    comparison_root = args.runtime / 'core50_comparisons'
    numeric = source_index(comparison_root)
    dependencies(comparison_root, args.runtime / 'core50_new15_opus_v5', numeric)
    upstream = json.loads((comparison_root / 'dependencies.json').read_text())
    write_json(root / 'dependencies.json', upstream)
    cached = {}
    gt = {}
    for key, record in upstream.items():
        artifact = materialize(record)
        request = artifact['request']
        if record['task']['task_type'] == 'gt_simplify':
            gt[request['dataset_id']] = key
        elif record['task']['task_type'] == 'pred_simplify':
            identity = expression_key(request['dataset_id'], request['variables'], request['expression'])
            cached.setdefault(identity, key)
    probes, _ = clean.load_dataset_probes(comparison_root / 'inputs/dataset_probes.jsonl')
    descriptions = {}
    for path in sorted((root / 'inputs/minutes').glob('*/*/expressions.jsonl')):
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if sha(path) != manifest['expressions_sha256']:
            raise ValueError(f'表达式输入哈希不一致: {path}')
        for row in rows(path):
            if expression_key(row['dataset_id'], row['variables'], row['expression']) != row['key']:
                raise ValueError('表达式去重键不一致')
            descriptions.setdefault(row['key'], row)
    aliases, unresolved, tasks = {}, {}, []
    jobs = []
    for key, row in sorted(descriptions.items()):
        if key in cached:
            aliases[key] = cached[key]
            continue
        jobs.append((row, probes[row['dataset_id']], args.runtime))
    cache = sqlite3.connect(root / 'preparation_cache.sqlite3')
    cache.execute('PRAGMA journal_mode=WAL')
    cache.execute('CREATE TABLE IF NOT EXISTS prepared (semantic_key TEXT PRIMARY KEY, input_hash TEXT NOT NULL, version TEXT NOT NULL, result TEXT NOT NULL)')
    version = digest({'builder': inspect.getsource(build_simplification),
        'symbolic_evidence': sha(args.runtime / 'AAAI_experiments/stage5_metric_calculation_0831/pipeline/symbolic_evidence.py'),
        'prompt': sha(args.runtime / 'AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt')})
    waiting = []
    def accept(result):
        key, task, error = result
        if error is not None:
            unresolved[key] = error
        else:
            tasks.append(task)
            aliases[key] = task['evaluation_key']
    for job in jobs:
        cached_row = cache.execute('SELECT input_hash,version,result FROM prepared WHERE semantic_key=?', (job[0]['key'],)).fetchone()
        input_hash = digest([job[0], job[1]])
        if cached_row is not None and cached_row[:2] == (input_hash, version):
            accept(json.loads(cached_row[2]))
        else:
            waiting.append((job, input_hash))
    with ProcessPoolExecutor(max_workers=args.prepare_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator = iter(waiting)
        futures = {}
        number = 0
        while True:
            while len(futures) < args.prepare_workers * 2:
                item = next(iterator, None)
                if item is None:
                    break
                job, input_hash = item
                futures[pool.submit(build_simplification, job)] = (job[0]['key'], input_hash)
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                key, input_hash = futures.pop(future)
                cache.execute('INSERT OR REPLACE INTO prepared VALUES (?,?,?,?)', (key, input_hash, version, canonical(result)))
                accept(result)
                number += 1
                if number % 50 == 0:
                    cache.commit()
                    write_json(root / 'preparation_progress.json', {'completed': len(tasks) + len(unresolved),
                        'uncached_expressions': len(jobs), 'reused_terminal': len(aliases) - len(tasks),
                        'unresolved': len(unresolved)})
        cache.commit()
    cache.close()
    tasks.sort(key=lambda task: task['logical_id'])
    for index in range(0, len(tasks), 50):
        publish(root, f'minute_pred_{index // 50:05d}.jsonl', tasks[index:index + 50])
    report = {'expressions': len(descriptions), 'new_requests': len(tasks),
        'reused': len(aliases) - len(tasks), 'unresolved': unresolved, 'aliases': aliases, 'ground_truth': gt,
        'retry_cap': len(tasks) * 5, 'max_calls': len(tasks) * 6}
    write_json(completed, report)
    write_json(root / 'preparation.complete.json', {'phase': 'minute_simplifications'})
    return report


def prepare_pairs(args):
    root = args.root
    report = json.loads((root / 'simplification_plan.json').read_text())
    records = frozen_records(root)
    write_json(root / 'dependencies.json', records)
    resolved = {key: simplify_pair(records[value]) for key, value in report['aliases'].items() if value in records}
    ground_truth = {name: simplify_pair(records[key]) for name, key in report['ground_truth'].items()}
    probes, _ = clean.load_dataset_probes(args.runtime / 'core50_comparisons/inputs/dataset_probes.jsonl')
    emitted = {row['logical_id'] for p in (root / 'plans').glob('minute_pair_*.jsonl') for row in rows(p)}
    groups = defaultdict(dict)
    for row in read_minutes(root):
        groups[row['condition'], row['algorithm'], row['dataset_id'], row['minute']][row['seed']] = row.get('expression_key')
    pairs = {}
    for (_, _, dataset, _), by_seed in groups.items():
        if set(by_seed) != {520, 521, 522}:
            raise ValueError('逐分钟结构比较缺少seed')
        for a, b in combinations((520, 521, 522), 2):
            if by_seed[a] and by_seed[b]:
                keys = sorted([by_seed[a], by_seed[b]])
                pairs.setdefault(digest({'dataset': dataset, 'pair': keys}), (dataset, *keys))
    jobs = []
    for key, (pred, pf) in resolved.items():
        dataset = pred.request['dataset_id']
        gt, gf = ground_truth[dataset]
        logical_id = f'minute_equivalence::{key}'
        request = {'dataset_id': dataset, 'variables': probes[dataset]['variables'],
            'effective_ground_truth_expression': gf.effective_expression,
            'simplified_ground_truth_expression': gf.simplified_expression,
            'effective_prediction_expression': pf.effective_expression,
            'simplified_prediction_expression': pf.simplified_expression,
            'prediction_valid_output': True, **context_binding('ground_truth', gt, gf),
            **context_binding('prediction', pred, pf)}
        jobs.append({'logical_id': logical_id, 'kind': 'equivalence', 'request': request,
            'dependencies': (gt.evaluation_key, pred.evaluation_key),
            'arguments': ('equivalence', logical_id, gt, gf, pred, pf, 0, probes[dataset])})
    for key, (dataset, left, right) in pairs.items():
        if left not in resolved or right not in resolved:
            continue
        a, af = resolved[left]
        b, bf = resolved[right]
        logical_id = f'minute_structure::{key}'
        request = {'dataset_id': dataset, 'variables': probes[dataset]['variables'],
            'effective_prediction_a_expression': af.effective_expression,
            'effective_prediction_b_expression': bf.effective_expression,
            'simplified_prediction_a_expression': af.simplified_expression,
            'simplified_prediction_b_expression': bf.simplified_expression,
            'prediction_a_valid_output': True, 'prediction_b_valid_output': True,
            **context_binding('prediction_a', a, af), **context_binding('prediction_b', b, bf)}
        jobs.append({'logical_id': logical_id, 'kind': 'structure', 'request': request,
            'dependencies': (a.evaluation_key, b.evaluation_key),
            'arguments': ('structure', logical_id, a, af, b, bf, 0, probes[dataset])})
    jobs = [job for job in jobs if job['logical_id'] not in emitted]
    contracts = {kind: symbolic._load_prompt_schema(args.runtime, task_kind=kind) for kind in ('equivalence', 'structure')}
    pending, batch, failures = {}, [], {}
    iterator = iter(jobs)
    index = len(list((root / 'plans').glob('minute_pair_*.jsonl')))
    with ProcessPoolExecutor(max_workers=args.prepare_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            while len(pending) < args.prepare_workers:
                job = next(iterator, None)
                if job is None:
                    break
                pending[pool.submit(evidence_job, job)] = job
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                evidence = future.result()
                if 'error' in evidence:
                    failures[job['logical_id']] = evidence['error']
                    continue
                evidence = evidence['evidence']
                request = {**job['request'], 'allowed_functions': evidence['allowed_functions'],
                    'deterministic_evidence': evidence, 'evidence_hash': evidence['evidence_sha256']}
                if job['kind'] == 'structure':
                    request['deterministic_pair_evidence'] = evidence
                task = symbolic._task_from_request(logical_id=job['logical_id'],
                    task_type='equivalence' if job['kind'] == 'equivalence' else 'stab_structure',
                    priority=symbolic.EQUIVALENCE_PRIORITY if job['kind'] == 'equivalence' else symbolic.STRUCTURE_PRIORITY,
                    request=request, evidence_hash=evidence['evidence_sha256'], contract=contracts[job['kind']],
                    dependencies=job['dependencies'], condition='clean')
                batch.append(symbolic._task_json_record(task))
                if len(batch) == 50:
                    publish(root, f'minute_pair_{index:05d}.jsonl', batch)
                    index += 1
                    batch.clear()
    if batch:
        publish(root, f'minute_pair_{index:05d}.jsonl', batch)
    result = {'unique_structure_pairs': len(pairs), 'resolved_expressions': len(resolved),
        'new_pair_requests': len(jobs) - len(failures), 'unresolved': failures}
    write_json(root / 'pair_plan.json', result)
    return result


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'simplify', 'pairs', 'compare'))
    parser.add_argument('--runtime', type=Path, default=Path('/home/zhangziwen/sim-runtime/core50-opus-runtime'))
    args = parser.parse_args()
    args.root = args.runtime / 'core50_minutes'
    stop_path = args.root / 'stopped_by_user.json'
    if stop_path.exists() and json.loads(stop_path.read_text()).get('scope') == 'terminal_formulas_only':
        raise SystemExit('当前仅处理最终公式，逐分钟Opus执行入口已停用。')
    args.workers = 300
    args.prepare_workers = 50
    args.logical_task_cap = 1000000
    args.channel_settings = 'routify=/home/zhangziwen/.config/core50-opus/routify.json'
    if args.mode == 'prepare':
        print(json.dumps({k: v for k, v in prepare_simplifications(args).items() if k not in ('aliases', 'ground_truth', 'unresolved')}))
    elif args.mode == 'pairs':
        print(json.dumps(prepare_pairs(args)))
    else:
        run(args)
