import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

from opus_remaining_replicas import ReplicaRunner, SYSTEM, write_json
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import HttpxAnthropicTransport, STRICT_EVALUATOR_SYSTEM_PROMPT
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_anthropic_api_plan import load_api_channels
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import load_plan_jsonl
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.agent/work/EXP-001/followup'


def import_dependency(store, key, index):
    with sqlite3.connect(store.path, timeout=60) as target:
        existing = target.execute('SELECT state FROM tasks WHERE evaluation_key=?', (key,)).fetchone()
        if existing:
            assert existing[0] == 'frozen'
            return
        binding = index[key]
        with sqlite3.connect(f"file:{binding['source_db']}?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            task = source.execute('SELECT * FROM tasks WHERE evaluation_key=?', (key,)).fetchone()
            frozen = source.execute('SELECT * FROM frozen_results WHERE evaluation_key=?', (key,)).fetchone()
            assert task and task['state'] == 'frozen' and frozen
            assert frozen['result_sha256'] == binding['result_sha256']
            assert hashlib.sha256(Path(frozen['result_path']).read_bytes()).hexdigest() == frozen['result_sha256']
            attempt = source.execute('SELECT * FROM attempts WHERE attempt_id=?', (frozen['attempt_id'],)).fetchone()
            assert attempt and attempt['status'] == 'accepted'
            for table, record in [('tasks', task), ('attempts', attempt), ('frozen_results', frozen)]:
                columns = ','.join(record.keys())
                placeholders = ','.join('?' for _ in record)
                target.execute(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})', tuple(record))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-dir', type=Path)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--recover-interrupted', action='store_true')
    args = parser.parse_args()
    if args.plan is None and args.plan_dir is None:
        parser.error('必须提供输入计划')
    sys.setrecursionlimit(100000)
    work = BASE / args.phase
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    entries = {}
    plan_bindings = {}
    loaded_paths = set()
    dependencies = {}
    selected_path = work / 'selected.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    channels = load_api_channels([f'routify={Path.home() / ".claude/settings.json"}'], allow_single_channel=True)
    transport = HttpxAnthropicTransport(channels, max_connections=args.workers)
    semantic = threading.BoundedSemaphore(1)
    runners = {}
    def runner_for(group):
        if group in runners:
            return runners[group]
        directory = work / ('__'.join(group))
        store = TaskStateStore(directory / 'state.sqlite3', attempt_cap=10**9, logical_task_cap=100000,
                               max_attempts_per_task=10**9)
        store.recover_expired_leases()
        if args.recover_interrupted:
            with sqlite3.connect(store.path) as connection:
                interrupted = connection.execute("SELECT attempt_id FROM attempts WHERE status='running'").fetchall()
            for (attempt_id,) in interrupted:
                store.finish_failure(attempt_id, error_class='controller_interrupted', retryable=True)
        runner = ReplicaRunner(store, attempts_dir=directory / 'attempts', frozen_dir=directory / 'frozen',
                               channels=channels, transport=transport, timeout_seconds=1800, lease_seconds=3600,
                               max_tokens=65536, per_channel_concurrency=args.workers, semantic_validation_concurrency=1,
                               allow_single_channel=True,
                               system_prompt=SYSTEM if group[0].endswith('simplify') else STRICT_EVALUATOR_SYSTEM_PROMPT)
        runner._semantic_semaphore = semantic
        runner.semantic_validator_timeout_seconds = 180
        runner.semantic_validator_command_builder = lambda: [sys.executable, str(Path(__file__).with_name('limited_semantic_worker.py'))]
        runners[group] = runner
        return runner
    def refresh():
        paths = ([args.plan] if args.plan else []) + (sorted(args.plan_dir.glob('batch_*.jsonl')) if args.plan_dir else [])
        for path in paths:
            if path in loaded_paths:
                continue
            plan = load_plan_jsonl(path)
            if args.plan_dir and path.parent == args.plan_dir:
                dependencies.update(json.loads((BASE / 'dependencies.json').read_text()))
            for entry in plan.entries:
                if entry.evaluation_key in entries:
                    raise RuntimeError(f'duplicate task: {entry.logical_id}')
                spec = entry.definition.task_spec
                runner = runner_for((spec.task_type, spec.condition))
                for dependency in spec.dependencies:
                    import_dependency(runner.store, dependency, dependencies)
                runner.store.register_task(spec)
                entries[entry.evaluation_key] = entry
                plan_bindings[entry.evaluation_key] = (str(path.resolve()), plan.plan_sha256)
            loaded_paths.add(path)
    refresh()
    write_json(work / 'configuration.json', {'plan': str(args.plan or args.plan_dir),
               'workers': args.workers, 'model': 'claude-opus-5',
               'system_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest()})
    def progress():
        counts = Counter(entries[key].definition.task_spec.task_type for key in selected if key in entries)
        totals = Counter(entry.definition.task_spec.task_type for entry in entries.values())
        data = {'time': time.time(), 'phase': args.phase, 'selected': len(selected), 'total': len(entries),
                'workers': args.workers, 'by_task_type': {kind: {'selected': counts[kind], 'total': total}
                                                       for kind, total in totals.items()},
                'preparation_complete': args.plan_dir is None or (args.plan_dir / 'complete.json').exists(),
                'status': 'complete' if len(selected) == len(entries) and (args.plan_dir is None or (args.plan_dir / 'complete.json').exists()) else 'running'}
        write_json(work / 'progress.json', data)
        write_json(BASE / 'progress.json', data)
        return data
    print(json.dumps(progress()), flush=True)
    retry_at = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        while True:
            if args.plan_dir and (args.plan_dir / 'failed').exists():
                raise RuntimeError('下游计划生成失败，请检查 preparation 日志')
            refresh()
            in_flight = {entry.evaluation_key for entry in futures.values()}
            for key, entry in entries.items():
                if len(futures) >= args.workers:
                    break
                if key in selected or key in in_flight or retry_at.get(key, 0) > time.time():
                    continue
                spec = entry.definition.task_spec
                runner = runners[spec.task_type, spec.condition]
                if runner.store.task_state(key) == 'exhausted':
                    raise RuntimeError(f'不可自动重试: {entry.logical_id}')
                futures[pool.submit(runner.execute_once, entry.definition)] = entry
            if not futures:
                if progress()['status'] == 'complete':
                    break
                time.sleep(1)
                continue
            completed, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                entry = futures.pop(future)
                result = future.result()
                event = {'time': time.time(), 'logical_id': entry.logical_id, 'state': result.state,
                         'error': result.error_class, 'cost_cny': result.total_cost_cny}
                if result.state == 'frozen':
                    spec = entry.definition.task_spec
                    directory = work / ('__'.join((spec.task_type, spec.condition)))
                    selected[entry.evaluation_key] = {**event, 'evaluation_key': entry.evaluation_key,
                        'result_path': result.result_path, 'result_sha256': result.result_sha256,
                        'source_db': str(directory / 'state.sqlite3'), 'plan_path': plan_bindings[entry.evaluation_key][0],
                        'plan_sha256': plan_bindings[entry.evaluation_key][1]}
                    write_json(selected_path, selected)
                else:
                    retry_at[entry.evaluation_key] = time.time() + 5
                with (work / 'events.jsonl').open('a') as handle:
                    handle.write(json.dumps(event) + '\n')
                progress()
                print(json.dumps(event), flush=True)
    print(json.dumps(progress()), flush=True)


if __name__ == '__main__':
    main()
