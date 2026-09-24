import json
import hashlib
from pathlib import Path
import sqlite3
import sys
import time
from contextlib import closing

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import AnthropicApiResponse, AnthropicApiRunner
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import canonical_json, render_prompt
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import _row_to_definition
from run_core50_comparisons import rows, sha, write_json


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    root = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_comparisons')
    audit = json.loads((root / 'terminal_state_audit.json').read_text())
    targets = {entry['task']['evaluation_key']: entry for entry in audit['active']}
    if len(targets) > 2:
        raise ValueError('待恢复运行数量发生变化')
    plans = {}
    for path in (root / 'plans').glob('*.jsonl'):
        for row in rows(path):
            if row['evaluation_key'] in targets:
                plans[row['evaluation_key']] = (row, path)
    changes = []
    for key, entry in targets.items():
        db = Path(entry['db'])
        with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as connection:
            meta = dict(connection.execute('SELECT key,value FROM meta'))
            state = connection.execute('SELECT state FROM tasks WHERE evaluation_key=?', (key,)).fetchone()[0]
        if state == 'frozen':
            continue
        if state != 'running':
            raise ValueError(f'运行状态变化: {key}: {state}')
        attempt_id = entry['attempt']['attempt_id']
        target = db.parent / 'attempts' / f'{attempt_id}.json'
        row, path = plans[key]
        store = TaskStateStore(db, attempt_cap=int(meta['attempt_cap']), logical_task_cap=int(meta['logical_task_cap']),
            max_attempts_per_task=int(meta['max_attempts_per_task']))
        if target.exists():
            source = json.loads(target.read_text())
            definition = _row_to_definition(row, line_number=1).definition
            metadata = source['metadata']
            if (source['request'] != definition.request or metadata['http_status'] != 200
                    or source['prompt'] != render_prompt(definition.prompt_template, definition.request, definition.schema)):
                raise ValueError('已保存响应的输入绑定不一致')
            for field, recorded in (('api_request', 'api_request_sha256'), ('api_response', 'persisted_api_response_sha256')):
                if hashlib.sha256(canonical_json(source[field]).encode()).hexdigest() != metadata[recorded]:
                    raise ValueError('已保存HTTP证据哈希不一致')
            structured, _, _, response_metadata = AnthropicApiRunner._extract_structured_output(
                AnthropicApiResponse(200, source['api_response'], '', {}), task_kind='structure', schema=definition.schema)
            recovered = db.parent / 'recovered_frozen' / f'{key}.json'
            frozen = {**source, 'logical_id': entry['task']['logical_id'], 'task_type': 'stab_structure', 'task_kind': 'structure',
                'structured_output': structured, 'command': ['anthropic-messages-api', '--model', metadata['requested_model']],
                'stdout': canonical_json(source['api_response']), 'stderr': '', 'envelope': source['api_response'],
                'original_validation': source['validation'], 'original_metadata': metadata,
                'metadata': {**metadata, 'error_class': None, 'retryable': False, 'response_metadata': response_metadata},
                'validation': {'ok': True, 'error_class': None, 'error_message': None, 'structured_output': structured},
                'revalidation': {'source_attempt': str(target), 'source_sha256': sha(target), 'network_request': False,
                                 'reason': '验收已保存的HTTP响应并完成中断的冻结步骤'}}
            write_json(recovered, frozen)
            error = 'controller_interrupted_after_audit_write'
            store.finish_failure(attempt_id, error_class=error, retryable=True)
            store.promote_failed_attempt(attempt_id, result_path=str(recovered), result_sha256=sha(recovered),
                allowed_error_classes=[error], audit_reason=frozen['revalidation']['reason'])
            changes.append({'evaluation_key': key, 'state': 'frozen', 'network_requests': 0})
            continue
        error = 'controller_interrupted_during_audit_serialization'
        write_json(target, {'attempt_id': attempt_id, 'evaluation_key': key, 'request': row['request'],
            'api_request': None, 'api_response': None,
            'validation': {'ok': False, 'error_class': error, 'error_message': '进程在审计JSON序列化阶段停止，完整响应未保存'},
            'metadata': {'error_class': error, 'retryable': True, 'response_unavailable': True,
                'attempt_number': entry['attempt']['attempt_number'], 'physical_attempt_accounted': True,
                'reserved_at': entry['attempt']['reserved_at'], 'recovery_time': time.time(),
                'estimated_cost_cny': None, 'source_plan': str(path), 'source_plan_sha256': sha(path),
                'thread_profile_sha256': sha(root / 'thread_profile.json')}})
        state = store.finish_failure(attempt_id, error_class=error, retryable=True)
        changes.append({'evaluation_key': key, 'state': state, 'attempt_count_preserved': entry['attempt']['attempt_number']})
    write_json(root / 'serialization_recovery.json', {'tasks': changes, 'network_requests': 0})
    print(changes)
