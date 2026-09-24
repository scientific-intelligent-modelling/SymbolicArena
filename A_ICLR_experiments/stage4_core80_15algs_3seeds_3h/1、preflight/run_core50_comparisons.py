import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import fcntl
import gzip
import hashlib
from itertools import combinations
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import threading
import time

import psutil
from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as builder
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import AnthropicApiRunner, HttpxAnthropicTransport
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import load_dataset_probes
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index import _resolve_simplify_effective_expression
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_anthropic_api_plan import load_api_channels
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import load_plan_jsonl
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import build_pair_evidence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(canonical(value) + '\n')
    temporary.replace(path)


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_index(root):
    report = json.loads((root / 'inputs/sixaxis_input_audit_full.json').read_text())
    if report['available_runs'] != 6750 or report['missing_runs']:
        raise ValueError('训练输入尚未完整')
    result = {}
    for condition in ('clean', 'noise001', 'noise005'):
        path = root / 'inputs' / f'{condition}_runs.jsonl.gz'
        if sha(path) != report['source_freezes'][condition]['sha256']:
            raise ValueError(f'训练输入文件哈希不一致: {path}')
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                raw = row['result']['raw_text']
                digest = hashlib.sha256(raw.encode()).hexdigest()
                if digest != row['result']['sha256']:
                    raise ValueError('训练结果哈希不一致')
                payload = json.loads(raw)
                source = row['source']
                key = (condition, source['algorithm'], source['dataset_index'], int(source['seed']))
                if key in result:
                    raise ValueError(f'重复训练身份: {key}')
                valid = all(isinstance((payload.get(s) or {}).get('nmse'), (float, int))
                            and math.isfinite(payload[s]['nmse']) for s in ('id_test', 'ood_test'))
                result[key] = {'sha256': digest, 'valid': valid, 'task_id': source['task_id'],
                               'dataset_id': source['dataset_id'], 'source_path': source['path']}
    return result


def read_frozen_rows(db, plan_path):
    plan_sha = sha(plan_path)
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        frozen_rows = connection.execute('SELECT * FROM frozen_results').fetchall()
        for frozen in frozen_rows:
            key = frozen['evaluation_key']
            task = connection.execute('SELECT * FROM tasks WHERE evaluation_key=?', (key,)).fetchone()
            attempt = connection.execute('SELECT * FROM attempts WHERE attempt_id=?', (frozen['attempt_id'],)).fetchone()
            if task['state'] != 'frozen' or attempt['status'] != 'accepted':
                raise ValueError(f'已接受结果状态异常: {key}')
            yield {'task': dict(task), 'attempt': dict(attempt), 'frozen': dict(frozen),
                   'source_db': str(db), 'source_plan': str(plan_path), 'source_plan_sha256': plan_sha}


def dependencies(root, prior, numeric):
    historical = json.loads((root / 'dependency_pack/manifest.json').read_text())['records']
    all_records = {r['task']['evaluation_key']: r for r in historical}
    for location in (prior, prior / 'retry64k'):
        for db in sorted((location / 'state').glob('*.sqlite3')):
            condition = db.stem
            name = f'{condition}_retry64k.jsonl' if location.name == 'retry64k' else (
                'gt_pending_api_iaaccn22.jsonl' if condition == 'gt' else f'{condition}_pred_pending_api_iaaccn22.jsonl')
            for record in read_frozen_rows(db, location / 'plans' / name):
                all_records[record['task']['evaluation_key']] = record
    retry_plan = root / 'plans/retry_exact.jsonl'
    if retry_plan.exists():
        for db in (root / 'execution').glob('pred_simplify__*/state.sqlite3'):
            for record in read_frozen_rows(db, retry_plan):
                all_records[record['task']['evaluation_key']] = record
    plans = {}
    predictions = {}
    ground_truth = {}
    for key, record in all_records.items():
        frozen = record['frozen']
        path = Path(frozen['result_path'])
        if sha(path) != frozen['result_sha256']:
            raise ValueError(f'已接受响应哈希异常: {key}')
        artifact = json.loads(path.read_text())
        if artifact['evaluation_key'] != key or not artifact['validation']['ok']:
            raise ValueError(f'响应身份或验收状态异常: {key}')
        request = artifact['request']
        task = record['task']
        effective, resolution = _resolve_simplify_effective_expression(
            request=request, structured_output=artifact['structured_output'], context=task['logical_id'])
        plan = builder.SimplifyPlanRecord(task['logical_id'], task['task_type'], key, task['priority'], request)
        output = artifact['structured_output']
        frozen_record = builder.FrozenSimplifyRecord(key, task['logical_id'], task['task_type'],
            record['source_plan_sha256'], 'frozen', output['outcome'], output.get('simplified_expression'),
            effective, resolution, output, None, frozen['result_sha256'])
        if task['task_type'] == 'gt_simplify':
            identity = request['dataset_id']
            if identity in ground_truth:
                raise ValueError(f'重复GT: {identity}')
            ground_truth[identity] = (plan, frozen_record)
        else:
            identity = (task['condition_name'], request['algorithm_slug'], request['dataset_index'], int(request['seed']))
            source = numeric[identity]
            expected = request['ast_source_evidence']['result_raw_sha256']
            if expected != source['sha256']:
                raise ValueError(f'训练结果与公式绑定不一致: {identity}')
            if identity in predictions:
                raise ValueError(f'重复预测公式: {identity}')
            predictions[identity] = (plan, frozen_record)
        plans[key] = record
    if len(ground_truth) != 50:
        raise ValueError(f'GT覆盖数量异常: {len(ground_truth)}')
    write_json(root / 'dependencies.json', plans)
    return predictions, ground_truth


def context_binding(prefix, plan, frozen):
    return {f'{prefix}_logical_id': plan.logical_id, f'{prefix}_plan_evaluation_key': plan.evaluation_key,
            f'{prefix}_frozen_plan_sha256': frozen.plan_sha256, f'{prefix}_frozen_evaluation_key': frozen.evaluation_key,
            f'{prefix}_simplify_status': frozen.simplified_status, f'{prefix}_expression_resolution': frozen.expression_resolution}


def evidence_worker(channel, arguments):
    kind, logical_id, left, lf, right, rf, seed, probe = arguments
    allowed = sorted(builder._pair_allowed_functions(left, lf) | builder._pair_allowed_functions(right, rf))
    evidence = build_pair_evidence(lf.effective_expression, rf.effective_expression,
        allowed_variables=probe['variables'], allowed_functions=allowed, seed=seed,
        probe_points=probe['points'], probe_source=probe['schema_version'],
        probe_sample_sha256=probe['sample_sha256'], include_tree_distance=kind == 'equivalence')
    result = {'schema_version': 'symbolic_pair_evidence.v2', 'phase': kind, 'pair_seed': seed,
              'pair_evidence': evidence, 'dataset_probe': probe, 'allowed_variables': probe['variables'],
              'allowed_functions': allowed,
              'domain_assumptions': {'lhs': builder._request_domain_assumptions(left.request, expression=lf.effective_expression, context=left.logical_id),
                                     'rhs': builder._request_domain_assumptions(right.request, expression=rf.effective_expression, context=right.logical_id)},
              'lhs_binding': builder._upstream_binding(plan_record=left, frozen_record=lf, role='lhs'),
              'rhs_binding': builder._upstream_binding(plan_record=right, frozen_record=rf, role='rhs'),
              'comparison_probe_policy': 'current_hash_verified_core50_dataset_probes'}
    result['evidence_sha256'] = hashlib.sha256(canonical(result).encode()).hexdigest()
    channel.send(result)
    channel.close()


def evidence_job(job):
    sys.setrecursionlimit(20000)
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=evidence_worker, args=(sender, job['arguments']))
    process.start()
    sender.close()
    try:
        if not receiver.poll(180):
            return {'error': 'pair_evidence_timeout_180s'}
        if process.exitcode is not None and process.exitcode != 0:
            return {'error': f'pair_evidence_exit_{process.exitcode}'}
        try:
            return {'evidence': receiver.recv()}
        except EOFError:
            return {'error': 'pair_evidence_process_exited'}
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


def prepare(args):
    root = args.root
    (root / 'plans').mkdir(parents=True, exist_ok=True)
    lock = (root / 'prepare.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    numeric = source_index(root)
    predictions, gt = dependencies(root, args.prior, numeric)
    probes, probe_sha = load_dataset_probes(root / 'inputs/dataset_probes.jsonl')
    report = json.loads((root / 'inputs/sixaxis_input_audit_full.json').read_text())
    expected_probe = next(v for k,v in report['inputs'].items() if k.endswith('/dataset_probes.jsonl'))
    if probe_sha != expected_probe:
        raise ValueError('dataset probes输入哈希不一致')
    emitted = {r['logical_id'] for p in (root / 'plans').glob('batch_*.jsonl') for r in rows(p)}
    jobs = []
    unresolved = []
    for identity, source in numeric.items():
        condition, algorithm, dataset, seed = identity
        logical_id = f'equivalence::{algorithm}::{dataset}::s{seed}::{condition}'
        if identity not in predictions:
            unresolved.append({'logical_id': logical_id, 'reason': 'missing_accepted_simplification'})
            continue
        pred, pf = predictions[identity]
        target, tf = gt[source['dataset_id']]
        request = {'dataset_id': source['dataset_id'], 'dataset_index': dataset, 'algorithm': pred.request['algorithm'],
                   'algorithm_slug': algorithm, 'seed': seed, 'noise_tag': condition, 'variables': pred.request['variables'],
                   'effective_ground_truth_expression': tf.effective_expression, 'simplified_ground_truth_expression': tf.simplified_expression,
                   'effective_prediction_expression': pf.effective_expression, 'simplified_prediction_expression': pf.simplified_expression,
                   **context_binding('ground_truth', target, tf), **context_binding('prediction', pred, pf),
                   'prediction_task_id': source['task_id'], 'prediction_valid_output': source['valid'],
                   'prediction_result_sha256': source['sha256']}
        if logical_id not in emitted:
            jobs.append({'condition': condition, 'logical_id': logical_id, 'kind': 'equivalence', 'request': request,
                         'dependencies': (target.evaluation_key, pred.evaluation_key),
                         'arguments': ('equivalence', logical_id, target, tf, pred, pf, seed, probes[source['dataset_id']])})
    groups = defaultdict(dict)
    for identity, source in numeric.items():
        groups[identity[:3]][identity[3]] = source
    for (condition, algorithm, dataset), sources in groups.items():
        for a,b in combinations((520,521,522), 2):
            logical_id = builder._structure_logical_id(algorithm, dataset, a, b, condition=condition)
            keys = [(condition, algorithm, dataset, s) for s in (a,b)]
            if any(k not in predictions for k in keys) or not (sources[a]['valid'] and sources[b]['valid']):
                unresolved.append({'logical_id': logical_id, 'reason': 'missing_valid_seed_pair'})
                continue
            left, lf = predictions[keys[0]]
            right, rf = predictions[keys[1]]
            source = sources[a]
            request = {'dataset_id': source['dataset_id'], 'dataset_index': dataset, 'algorithm': left.request['algorithm'],
                       'algorithm_slug': algorithm, 'noise_tag': condition, 'seed_a': a, 'seed_b': b, 'variables': left.request['variables'],
                       'effective_prediction_a_expression': lf.effective_expression, 'effective_prediction_b_expression': rf.effective_expression,
                       'simplified_prediction_a_expression': lf.simplified_expression, 'simplified_prediction_b_expression': rf.simplified_expression,
                       **context_binding('prediction_a', left, lf), **context_binding('prediction_b', right, rf),
                       'prediction_a_task_id': sources[a]['task_id'], 'prediction_b_task_id': sources[b]['task_id'],
                       'prediction_a_valid_output': True, 'prediction_b_valid_output': True,
                       'prediction_a_result_sha256': sources[a]['sha256'], 'prediction_b_result_sha256': sources[b]['sha256']}
            if logical_id not in emitted:
                jobs.append({'condition': condition, 'logical_id': logical_id, 'kind': 'structure', 'request': request,
                             'dependencies': (left.evaluation_key, right.evaluation_key),
                             'arguments': ('structure', logical_id, left, lf, right, rf, a*1000+b, probes[source['dataset_id']])})
    jobs.sort(key=lambda j: len(j['arguments'][3].effective_expression)+len(j['arguments'][5].effective_expression))
    counts = dict(Counter(j['kind'] for j in jobs))
    write_json(root / 'preparation.json', {'status': 'running', 'ready_pairs': counts, 'unresolved_count': len(unresolved),
                                         'max_tokens': 65536, 'max_attempts_per_task': 6, 'new_attempt_cap': 6*len(jobs)})
    print(json.dumps({'ready_pairs': counts, 'unresolved_count': len(unresolved)}), flush=True)
    contracts = {kind: builder._load_prompt_schema(args.runtime, task_kind=kind) for kind in ('equivalence','structure')}
    pending = {}
    iterator = iter(jobs)
    batch = []
    number = len(list((root / 'plans').glob('batch_*.jsonl')))
    def flush():
        nonlocal number
        if not batch:
            return
        number += 1
        path = root / 'plans' / f'batch_{number:05d}.jsonl'
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix('.pending')
        temporary.write_text(''.join(canonical(r)+'\n' for r in batch))
        temporary.replace(path)
        batch.clear()
        print(json.dumps({'published_batches': number}), flush=True)
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
                result = future.result()
                if 'error' in result:
                    unresolved.append({'logical_id': job['logical_id'], 'reason': result['error']})
                    continue
                evidence = result['evidence']
                request = {**job['request'], 'allowed_functions': evidence['allowed_functions'],
                           'deterministic_evidence': evidence, 'evidence_hash': evidence['evidence_sha256']}
                if job['kind'] == 'structure':
                    request['deterministic_pair_evidence'] = evidence
                task = builder._task_from_request(logical_id=job['logical_id'],
                    task_type='equivalence' if job['kind'] == 'equivalence' else 'stab_structure',
                    priority=builder.EQUIVALENCE_PRIORITY if job['kind'] == 'equivalence' else builder.STRUCTURE_PRIORITY,
                    request=request, evidence_hash=evidence['evidence_sha256'], contract=contracts[job['kind']],
                    dependencies=job['dependencies'], condition=job['condition'])
                batch.append(builder._task_json_record(task))
                if len(batch) >= 10:
                    flush()
            done.clear()
    flush()
    write_json(root / 'unresolved.json', unresolved)
    write_json(root / 'preparation.complete.json', {'batches': number, 'ready_pairs': counts, 'unresolved_count': len(unresolved)})


def import_dependency(store, record):
    key = record['task']['evaluation_key']
    frozen = record['frozen']
    if sha(frozen['result_path']) != frozen['result_sha256']:
        raise ValueError(f'依赖响应哈希不一致: {key}')
    with sqlite3.connect(store.path, timeout=60) as connection:
        existing = connection.execute('SELECT state FROM tasks WHERE evaluation_key=?', (key,)).fetchone()
        if existing:
            if existing[0] != 'frozen':
                raise ValueError(f'依赖尚未接受: {key}')
            return
        for table, row in (('tasks', record['task']), ('attempts', record['attempt']), ('frozen_results', frozen)):
            columns = ','.join(row)
            placeholders = ','.join('?' for _ in row)
            connection.execute(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})', tuple(row.values()))
        connection.execute('INSERT INTO events(evaluation_key,event_type,event_at,details_json) VALUES (?,?,?,?)',
                           (key, 'verified_dependency_import', time.time(), canonical({'source_db': record['source_db'], 'result_sha256': frozen['result_sha256']})))


def run(args):
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    channels = load_api_channels([args.channel_settings], allow_single_channel=True)
    transport = HttpxAnthropicTransport(channels, max_connections=args.workers)
    semaphore = threading.BoundedSemaphore(8)
    runners = {}
    entries = {}
    loaded = set()
    completed = set()
    futures = {}
    def runner_for(spec):
        group = (spec.task_type, spec.condition)
        if group not in runners:
            directory = root / 'execution' / '__'.join(group)
            store = TaskStateStore(directory / 'state.sqlite3', attempt_cap=20300,
                                   logical_task_cap=20000, max_attempts_per_task=6)
            store.recover_expired_leases()
            runner = AnthropicApiRunner(store, attempts_dir=directory/'attempts', frozen_dir=directory/'frozen',
                channels=channels, transport=transport, timeout_seconds=1800, lease_seconds=2100,
                max_tokens=65536, per_channel_concurrency=args.workers, semantic_validation_concurrency=8,
                allow_single_channel=True)
            runner._semantic_semaphore = semaphore
            runner.semantic_validator_timeout_seconds = 180
            runners[group] = runner
        return runners[group]
    def refresh():
        new_paths = [p for p in sorted((root/'plans').glob('*.jsonl')) if p not in loaded]
        new_paths.sort(key=lambda p: (not p.name.startswith('retry_'), p.name))
        new_paths = new_paths[:10]
        if not new_paths:
            return
        dependencies = json.loads((root/'dependencies.json').read_text())
        for path in new_paths:
            plan = load_plan_jsonl(path)
            for entry in plan.entries:
                if entry.evaluation_key in entries:
                    raise ValueError(f'重复任务: {entry.logical_id}')
                spec = entry.definition.task_spec
                runner = runner_for(spec)
                for key in spec.dependencies:
                    import_dependency(runner.store, dependencies[key])
                runner.store.register_task(spec)
                entries[entry.evaluation_key] = entry
                state = runner.store.task_state(entry.evaluation_key)
                if state == 'frozen' and runner._load_existing_frozen(entry.evaluation_key) is None:
                    raise ValueError(f'已接受响应缺失: {entry.logical_id}')
                if state in ('frozen', 'exhausted'):
                    completed.add(entry.evaluation_key)
            loaded.add(path)
    def progress():
        counts = defaultdict(Counter)
        for key, entry in entries.items():
            spec = entry.definition.task_spec
            counts[spec.task_type][runners[spec.task_type,spec.condition].store.task_state(key)] += 1
        memory = psutil.virtual_memory()
        data = {'time': time.time(), 'workers': args.workers, 'max_tokens': 65536,
                'registered': len(entries), 'in_flight': len(futures), 'by_type': {k:dict(v) for k,v in counts.items()},
                'cpu_percent': psutil.cpu_percent(), 'memory_total': memory.total, 'memory_available': memory.available,
                'preparation_complete': (root/'preparation.complete.json').exists()}
        write_json(root/'progress.json', data)
        print(json.dumps(data), flush=True)
    last_report = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            refresh()
            active = {e.evaluation_key for e in futures.values()}
            candidates = defaultdict(deque)
            for key, entry in entries.items():
                if key in completed or key in active:
                    continue
                candidates[entry.definition.task_spec.task_type].append(entry)
            while len(futures) < args.workers and any(candidates.values()):
                for kind in sorted(candidates):
                    if len(futures) >= args.workers:
                        break
                    if not candidates[kind]:
                        continue
                    entry = candidates[kind].popleft()
                    spec = entry.definition.task_spec
                    runner = runners[spec.task_type,spec.condition]
                    if runner.store.task_state(entry.evaluation_key) not in ('pending','retry_wait'):
                        continue
                    futures[pool.submit(runner.execute, entry.definition)] = entry
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED) if futures else (set(),set())
            for future in done:
                entry = futures.pop(future)
                result = future.result()
                completed.add(entry.evaluation_key)
                with (root/'events.jsonl').open('a') as handle:
                    handle.write(canonical({'time':time.time(),'logical_id':entry.logical_id,'state':result.state,
                                            'error':result.error_class,'cost_cny':result.total_cost_cny})+'\n')
            if time.time()-last_report >= 60:
                progress()
                last_report = time.time()
            all_plans_loaded = all(p in loaded for p in (root/'plans').glob('*.jsonl'))
            if (root/'preparation.complete.json').exists() and all_plans_loaded and len(completed) == len(entries):
                progress()
                break
            if not futures:
                time.sleep(2)


def extend(args):
    root = args.root
    lock = (root / 'extension.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = json.loads((root / 'retry_exact_manifest.json').read_text())['count']
    while True:
        terminal = 0
        for db in (root / 'execution').glob('pred_simplify__*/state.sqlite3'):
            with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
                terminal += connection.execute("SELECT count(*) FROM tasks WHERE task_type='pred_simplify' AND state IN ('frozen','exhausted')").fetchone()[0]
        if (root / 'preparation.complete.json').exists() and terminal == expected:
            break
        time.sleep(15)
    previous_paths = set((root / 'plans').glob('batch_*.jsonl'))
    prepare(args)
    added_paths = set((root / 'plans').glob('batch_*.jsonl')) - previous_paths
    added_keys = {row['evaluation_key'] for path in added_paths for row in rows(path)}
    write_json(root / 'extension.complete.json', {'time': time.time(), 'retry_tasks_terminal': terminal,
                                                 'formal_ready': False})
    while added_keys:
        registered = set()
        for db in (root / 'execution').glob('*/state.sqlite3'):
            with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
                registered.update(r[0] for r in connection.execute('SELECT evaluation_key FROM tasks'))
        if added_keys <= registered:
            return
        controller_lock = (root / 'controller.lock').open('a+')
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            controller_lock.close()
            time.sleep(5)
            continue
        fcntl.flock(controller_lock, fcntl.LOCK_UN)
        controller_lock.close()
        run(args)
        return


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare','run','extend'))
    parser.add_argument('--runtime', type=Path, default=Path('/home/zhangziwen/sim-runtime/core50-opus-runtime'))
    parser.add_argument('--root', type=Path, default=Path('/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_comparisons'))
    parser.add_argument('--prior', type=Path, default=Path('/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_new15_opus_v5'))
    parser.add_argument('--workers', type=int, default=150)
    parser.add_argument('--prepare-workers', type=int, default=8)
    parser.add_argument('--channel-settings', default='routify=/home/zhangziwen/.config/core50-opus/routify.json')
    arguments = parser.parse_args()
    if arguments.mode == 'prepare':
        prepare(arguments)
    elif arguments.mode == 'run':
        run(arguments)
    else:
        extend(arguments)
