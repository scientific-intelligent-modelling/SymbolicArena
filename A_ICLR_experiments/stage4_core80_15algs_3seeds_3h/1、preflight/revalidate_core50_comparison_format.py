import json
import hashlib
from pathlib import Path
import sqlite3
import sys
import time

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import AnthropicApiResponse, AnthropicApiRunner
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import canonical_json, render_prompt
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import _row_to_definition
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore
from run_core50_comparisons import rows, sha, write_json


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def main():
    sys.setrecursionlimit(20000)
    root = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_comparisons')
    targets = {}
    for db in (root / 'execution').glob('equivalence__*/state.sqlite3'):
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            for row in connection.execute("SELECT * FROM tasks WHERE state='exhausted' AND last_error_class='structured_output_invalid'"):
                targets[row['evaluation_key']] = (db, dict(row))
    definitions = {}
    for path in (root / 'plans').glob('*.jsonl'):
        for number, row in enumerate(rows(path), 1):
            if row['evaluation_key'] in targets:
                definitions[row['evaluation_key']] = _row_to_definition(row, line_number=number).definition
    output = []
    for key, (db, task) in targets.items():
        definition = definitions[key]
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
            meta = dict(connection.execute('SELECT key,value FROM meta'))
            attempt_id = connection.execute('SELECT attempt_id FROM attempts WHERE evaluation_key=? ORDER BY attempt_number DESC LIMIT 1', (key,)).fetchone()[0]
        path = db.parent / 'attempts' / f'{attempt_id}.json'
        source = json.loads(path.read_text())
        metadata = source['metadata']
        if (source['request'] != definition.request
                or source['prompt'] != render_prompt(definition.prompt_template, definition.request, definition.schema)
                or digest(source['api_request']) != metadata['api_request_sha256']
                or digest(source['api_response']) != metadata['persisted_api_response_sha256']):
            raise ValueError('历史HTTP请求或响应绑定不一致')
        result, _, _, response_metadata = AnthropicApiRunner._extract_structured_output(
            AnthropicApiResponse(metadata['http_status'], source['api_response'], '', {}),
            task_kind='equivalence', schema=definition.schema)
        if response_metadata.get('schema_normalizations') != ['equivalence_null_assumptions_note.v1']:
            raise ValueError('本次复核仅允许已声明的空字段规范化')
        proof = {'source_attempt': str(path), 'source_sha256': sha(path), 'time': time.time(),
            'normalization': 'equivalence_null_assumptions_note.v1', 'network_request': False,
            'reason': '保留原始响应，规范化额外的空assumptions_note字段；数学裁决及其他字段保持不变'}
        frozen = {**source, 'logical_id': task['logical_id'], 'task_type': 'equivalence', 'task_kind': 'equivalence',
            'structured_output': result, 'command': ['anthropic-messages-api', '--model', metadata['requested_model']],
            'stdout': canonical_json(source['api_response']), 'stderr': '', 'envelope': source['api_response'],
            'original_metadata': metadata, 'original_validation': source['validation'], 'revalidation': proof,
            'metadata': {**metadata, 'error_class': None, 'retryable': False, 'response_metadata': response_metadata},
            'validation': {'ok': True, 'error_class': None, 'error_message': None, 'structured_output': result}}
        target = db.parent / 'frozen' / f'{key}.json'
        if target.exists():
            raise FileExistsError(target)
        write_json(target, frozen)
        store = TaskStateStore(db, attempt_cap=int(meta['attempt_cap']), logical_task_cap=int(meta['logical_task_cap']),
            max_attempts_per_task=int(meta['max_attempts_per_task']))
        store.promote_failed_attempt(attempt_id, result_path=str(target), result_sha256=sha(target),
            allowed_error_classes=['structured_output_invalid'], audit_reason=proof['reason'])
        output.append({'evaluation_key': key, 'logical_id': task['logical_id'], 'decision': result['decision'], 'proof': proof})
        print(output[-1], flush=True)
    write_json(root / 'comparison_format_revalidation.json', {'recovered': output})


if __name__ == '__main__':
    main()
