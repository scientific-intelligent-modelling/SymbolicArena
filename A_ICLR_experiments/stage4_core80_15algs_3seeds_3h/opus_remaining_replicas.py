import argparse
import concurrent.futures
import fcntl
import hashlib
from itertools import zip_longest
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import psutil

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import render_prompt
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import ClaudeRunnerCircuitBreaker
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    API_TRANSPORT_VERSION, AnthropicApiRunner, HttpxAnthropicTransport,
    STRICT_EVALUATOR_SYSTEM_PROMPT,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_anthropic_api_plan import load_api_channels
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import load_plan_jsonl
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / '.agent/work/EXP-001/opus_postprocess'
WORK = ROOT / '.agent/work/EXP-001/oversample/opus'
SYSTEM = STRICT_EVALUATOR_SYSTEM_PROMPT + '''
Exact-expression policy v2:
The authoritative expression is request.expression. Canonical forms and numerical probes are supporting evidence; they may contain rounded coefficients and must not replace the authoritative expression.
Treat every supplied decimal literal as an exact decimal value. Never replace a product, quotient, sum, or difference of numeric literals with a rounded decimal approximation. Keep the original numeric arithmetic unevaluated, or use an exactly equal integer/rational representation.
For example, retain (1.2345678901234567/3)*x instead of rounding its coefficient to a finite decimal. Approximate numerical agreement is insufficient for exact equivalence.
Perform only exact, domain-preserving reductions. Preserve variable names and real-domain restrictions, especially for roots, powers, logarithms, and protected operators. Do not expand expressions solely for presentation.
Return outcome unchanged only when justified by the mathematical task. Do not use unchanged or unable merely to bypass validation.
Complete the requested judgment within one response. Return exactly the supplied JSON schema, with all required keys, no additional keys or surrounding text, and a brief reason. No tool calls or simulated computations.
'''


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stop_old():
    targets = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if process.pid == os.getpid():
            continue
        if any(str(OLD / name) in arg or arg == f'.agent/work/EXP-001/opus_postprocess/{name}'
               for name in ['opus_daemon.py', 'run_opus_plan.py'] for arg in args):
            targets.append(process)
    children = {child.pid: child for parent in targets for child in parent.children(recursive=True)}
    for process in targets + list(children.values()):
        if process.is_running():
            process.terminate()
    _, alive = psutil.wait_procs(targets + list(children.values()), timeout=10)
    for process in alive:
        process.kill()
    _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError('Opus 旧进程未全部终止')
    write_json(WORK / 'stopped.json', {'time': time.time(), 'parents': [p.pid for p in targets],
                                     'children': sorted(children)})
    print(json.dumps({'stopped': [p.pid for p in targets], 'children': len(children)}), flush=True)


class ReplicaRunner(AnthropicApiRunner):
    def execute_once(self, definition):
        self.store.register_task(definition.task_spec)
        cached = self._load_existing_frozen(definition.task_spec.evaluation_key)
        if cached is not None:
            return cached
        prompt_sha, schema_sha = self._verify_task_definition(definition)
        lease = self.store.reserve_attempt(definition.task_spec.evaluation_key, lease_seconds=self.lease_seconds)
        return self._run_attempt(
            definition=definition, task_kind='simplify',
            prompt=render_prompt(definition.prompt_template, definition.request, definition.schema),
            prompt_sha256=prompt_sha, schema_sha256=schema_sha,
            command=self._build_and_validate_command(definition.schema),
            attempt_id=lease.attempt_id, attempt_number=lease.attempt_number,
            lease_expires_at=lease.lease_expires_at,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-old', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--recover-interrupted', action='store_true')
    parser.add_argument('--include-noise', action='store_true')
    parser.add_argument('--replicas', type=int)
    parser.add_argument('--max-tokens', type=int, default=65536)
    parser.add_argument('--channel-settings', default=f'routify={Path.home() / ".claude/settings.json"}')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers 必须为正整数')
    if args.max_tokens < 1 or (args.replicas is not None and args.replicas < 1):
        parser.error('--replicas 和 --max-tokens 必须为正整数')
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stop_old:
        stop_old()
        return
    lock = (WORK / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.setrecursionlimit(100000)
    conditions = ['clean', 'noise001', 'noise005'] if args.include_noise else ['clean']
    plans = {condition: load_plan_jsonl(OLD / f'plans/simplify_core80_{condition}.jsonl')
             for condition in conditions}
    plan = plans['clean']
    with sqlite3.connect(f'file:{OLD}/state/opus_pred_simplify.sqlite3?mode=ro', uri=True) as connection:
        frozen = {row[0] for row in connection.execute("SELECT evaluation_key FROM tasks WHERE state='frozen'")}
    entries_by_condition = {
        condition: [entry for entry in item.entries if entry.evaluation_key not in frozen]
        for condition, item in plans.items()
    }
    entries = [entry for group in entries_by_condition.values() for entry in group]
    replicas = {condition: args.replicas or (3 if condition == 'clean' else 1) for condition in conditions}
    selection_path = WORK / 'selected.json'
    selected = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    for record in selected.values():
        actual = hashlib.sha256(Path(record['result_path']).read_bytes()).hexdigest()
        if actual != record['result_sha256']:
            raise RuntimeError(f"已选结果校验失败: {record['result_path']}")
    configuration = {
        'plan': str(plan.plan_path), 'plan_sha256': plan.plan_sha256,
        'task_count': len(entries), 'replicas': replicas, 'workers': args.workers,
        'condition_plans': {condition: {'path': str(item.plan_path), 'sha256': item.plan_sha256}
                            for condition, item in plans.items()},
        'timeout_seconds': 1800, 'system_prompt': SYSTEM,
        'transport_version': API_TRANSPORT_VERSION, 'max_tokens': args.max_tokens,
        'system_prompt_version': 'exact_expression.v2',
        'semantic_validation_concurrency': 2, 'semantic_memory_limit_bytes': 2 * 1024**3,
        'system_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'selection': 'first_valid_completion', 'logical_ids': [e.logical_id for e in entries],
    }
    manifest_path = WORK / 'request_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['plan_sha256'] != plan.plan_sha256:
            raise RuntimeError('恢复任务的输入计划哈希发生改变')
    else:
        write_json(manifest_path, configuration)
    for condition in conditions:
        if condition == 'clean':
            continue
        condition_manifest = WORK / condition / 'request_manifest.json'
        current = configuration['condition_plans'][condition]
        if condition_manifest.exists():
            if json.loads(condition_manifest.read_text()) != current:
                raise RuntimeError(f'{condition} 输入计划哈希发生改变')
        else:
            write_json(condition_manifest, current)
    with (WORK / 'run_configurations.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.time(), **configuration}) + '\n')
    runners = {}
    channels = load_api_channels([args.channel_settings], allow_single_channel=True)
    transport = HttpxAnthropicTransport(channels, max_connections=args.workers)
    semantic_semaphore = threading.BoundedSemaphore(2)
    for condition, replica in ((condition, replica) for condition in conditions
                               for replica in range(1, replicas[condition] + 1)):
        directory = (WORK if condition == 'clean' else WORK / condition) / f'replica_{replica}'
        store = TaskStateStore(directory / 'state.sqlite3', attempt_cap=10**9,
                               logical_task_cap=40000, max_attempts_per_task=10**9)
        store.register_tasks([entry.definition.task_spec for entry in entries_by_condition[condition]
                              if entry.evaluation_key not in selected])
        store.recover_expired_leases()
        if args.recover_interrupted:
            with sqlite3.connect(store.path) as connection:
                interrupted = connection.execute("SELECT attempt_id FROM attempts WHERE status='running'").fetchall()
            for (attempt_id,) in interrupted:
                store.finish_failure(attempt_id, error_class='controller_interrupted', retryable=True)
        runner = ReplicaRunner(store, attempts_dir=directory / 'attempts',
                               frozen_dir=directory / 'frozen', channels=channels,
                               transport=transport, timeout_seconds=1800, lease_seconds=3600,
                               max_tokens=args.max_tokens, per_channel_concurrency=args.workers,
                               semantic_validation_concurrency=2, allow_single_channel=True,
                               system_prompt=SYSTEM)
        runner._semantic_semaphore = semantic_semaphore
        runner.semantic_validator_command_builder = lambda: [
            sys.executable, str(Path(__file__).with_name('limited_semantic_worker.py')),
        ]
        runners[condition, replica - 1] = runner
    def by_condition():
        return {condition: {'total': len(group),
                            'selected': sum(entry.evaluation_key in selected for entry in group)}
                for condition, group in entries_by_condition.items()}
    progress_path = WORK / 'progress.json'
    round_number = json.loads(progress_path.read_text()).get('round', 0) if progress_path.exists() else 0
    startup = {'time': time.time(), 'round': round_number, 'selected': len(selected),
               'total': len(entries), 'workers': args.workers, 'replicas': replicas,
               'by_condition': by_condition(), 'status': 'running',
               'transport_version': API_TRANSPORT_VERSION}
    write_json(progress_path, startup)
    print(json.dumps(startup), flush=True)
    while len(selected) < len(entries):
        round_number += 1
        pending_groups = [[entry for entry in group if entry.evaluation_key not in selected]
                          for group in entries_by_condition.values()]
        pending = [entry for group in zip_longest(*pending_groups) for entry in group if entry is not None]
        jobs = [(entry, replica) for entry in pending
                for replica in range(replicas[entry.definition.task_spec.condition])
                if runners[entry.definition.task_spec.condition, replica].store.task_state(entry.evaluation_key) != 'exhausted']
        if not jobs:
            raise RuntimeError('剩余任务全部发生不可自动重试的错误，请检查 attempts')
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            def execute(entry, replica):
                if entry.evaluation_key in selected:
                    return None
                try:
                    return runners[entry.definition.task_spec.condition, replica].execute_once(entry.definition)
                except ClaudeRunnerCircuitBreaker as exc:
                    return {'error': str(exc), 'replica': replica + 1}

            futures = {pool.submit(execute, entry, replica): (entry, replica) for entry, replica in jobs}
            for future in concurrent.futures.as_completed(futures):
                entry, replica = futures[future]
                result = future.result()
                event = {'time': time.time(), 'logical_id': entry.logical_id,
                         'condition': entry.definition.task_spec.condition,
                         'replica': replica + 1, 'round': round_number}
                if result is None:
                    event['status'] = 'sibling_selected'
                elif isinstance(result, dict):
                    event.update(result)
                else:
                    event.update({'status': result.state, 'error_class': result.error_class,
                                  'attempt_id': result.attempt_id, 'cost_usd': result.total_cost_usd,
                                  'estimated_cost_cny': result.total_cost_cny,
                                  'transport_version': API_TRANSPORT_VERSION})
                    if result.state == 'frozen' and entry.evaluation_key not in selected:
                        selected[entry.evaluation_key] = {
                            **event, 'evaluation_key': entry.evaluation_key,
                            'result_path': result.result_path, 'result_sha256': result.result_sha256,
                        }
                        write_json(selection_path, selected)
                with (WORK / 'events.jsonl').open('a') as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + '\n')
                write_json(WORK / 'progress.json', {'time': time.time(), 'round': round_number,
                           'selected': len(selected), 'total': len(entries), 'workers': args.workers,
                           'replicas': replicas, 'by_condition': by_condition(),
                           'transport_version': API_TRANSPORT_VERSION, 'last_event': event})
                print(json.dumps(event, ensure_ascii=False), flush=True)
        time.sleep(5)
    print(json.dumps({'status': 'complete', 'selected': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
