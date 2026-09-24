from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import sympy
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import AnthropicApiRunner, AnthropicApiResponse
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import canonical_json, render_prompt
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import _row_to_definition
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore


def sha(data):
    return hashlib.sha256(data).hexdigest()


def revalidate(row, definition):
    db = Path(row['source_db'])
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
        meta = dict(connection.execute('SELECT key,value FROM meta').fetchall())
        state = connection.execute('SELECT state FROM tasks WHERE evaluation_key=?', (row['evaluation_key'],)).fetchone()[0]
    if state == 'frozen':
        return {'logical_id': row['logical_id'], 'status': 'already_frozen'}
    path = Path(row['history'][-1]['path'])
    raw = path.read_bytes()
    source = json.loads(raw)
    metadata = source['metadata']
    if source['request'] != definition.request or source['prompt'] != render_prompt(definition.prompt_template, definition.request, definition.schema):
        raise ValueError(f"历史请求与计划不匹配: {row['logical_id']}")
    if sha(canonical_json(source['api_request']).encode()) != metadata['api_request_sha256']:
        raise ValueError('HTTP请求哈希不一致')
    if sha(canonical_json(source['api_response']).encode()) != metadata['persisted_api_response_sha256']:
        raise ValueError('HTTP响应哈希不一致')
    response = AnthropicApiResponse(metadata['http_status'], source['api_response'], '', {})
    structured, _, _, _ = AnthropicApiRunner._extract_structured_output(response, task_kind='simplify', schema=definition.schema)
    payload = {'evaluation_key': row['evaluation_key'], 'request': definition.request, 'structured_output': structured}
    command = [sys.executable, '-m', 'AAAI_experiments.stage5_metric_calculation_0831.pipeline.semantic_validation_worker']
    try:
        result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=185)
    except subprocess.TimeoutExpired:
        return {'logical_id': row['logical_id'], 'status': 'unresolved', 'reason': 'semantic_recheck_timeout_185s'}
    if result.returncode:
        return {'logical_id': row['logical_id'], 'status': 'unresolved', 'reason': result.stderr[-2000:]}
    checked = json.loads(result.stdout)
    if checked['status'] != 'ok':
        return {'logical_id': row['logical_id'], 'status': 'unresolved', 'reason': checked}
    proof = {'source_attempt': str(path), 'source_attempt_sha256': sha(raw), 'time': time.time(),
             'sympy_version': sympy.__version__, 'reason': '使用输入计划的SymPy版本复核历史HTTP响应',
             'network_request': False}
    frozen = {**source, 'logical_id': row['logical_id'], 'task_type': definition.task_spec.task_type,
              'task_kind': 'simplify', 'structured_output': structured,
              'command': ['anthropic-messages-api', '--model', metadata['requested_model']],
              'stdout': canonical_json(source['api_response']), 'stderr': '', 'envelope': source['api_response'],
              'original_metadata': metadata, 'original_validation': source['validation'], 'revalidation': proof,
              'metadata': {**metadata, 'error_class': None, 'retryable': False},
              'validation': {'ok': True, 'error_class': None, 'error_message': None,
                             'structured_output': structured, 'semantic_evidence': checked['semantic_evidence']}}
    target = db.parent.parent / 'frozen' / db.stem / f"{row['evaluation_key']}.json"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(frozen) + '\n')
    store = TaskStateStore(db, attempt_cap=int(meta['attempt_cap']), logical_task_cap=int(meta['logical_task_cap']),
                           max_attempts_per_task=int(meta['max_attempts_per_task']))
    store.promote_failed_attempt(source['attempt_id'], result_path=str(target), result_sha256=sha(target.read_bytes()),
        allowed_error_classes=['validation_failed', 'semantic_validator_error'], audit_reason=proof['reason'])
    return {'logical_id': row['logical_id'], 'status': 'recovered', 'decision': checked['semantic_evidence']['decision'],
            'result_path': str(target)}


def main():
    sys.setrecursionlimit(20000)
    base = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_new15_opus_v5')
    failures = json.loads((base/'failure_audit/failed.json').read_text())
    selected = [r for r in failures if r['last_message'] == '请求中的 symbolic artifact 与原公式不一致'
                or r['history'][-1]['error_class'] == 'semantic_validator_error']
    targets = {r['evaluation_key'] for r in selected}
    definitions = {}
    for location in (base, base/'retry64k'):
        for path in (location/'plans').glob('*iaaccn22.jsonl' if location == base else '*retry64k.jsonl'):
            with path.open() as handle:
                for number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if row['evaluation_key'] in targets:
                        definitions[row['evaluation_key']] = _row_to_definition(row, line_number=number).definition
    if set(definitions) != targets:
        raise ValueError('缺少源请求计划')
    output = base/'failure_audit/revalidation.jsonl'
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(revalidate, row, definitions[row['evaluation_key']]): row for row in selected}
        for future in as_completed(futures):
            item = future.result()
            with output.open('a') as handle:
                handle.write(json.dumps(item, ensure_ascii=False)+'\n')
            print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
