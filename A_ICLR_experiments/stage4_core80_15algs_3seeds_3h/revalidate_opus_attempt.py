import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import AnthropicApiRunner, AnthropicApiResponse
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import canonical_json, render_prompt
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import _row_to_definition
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.agent/work/EXP-001/oversample/opus'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    sys.setrecursionlimit(100000)
    source = json.loads(args.attempt.read_text())
    key = source['evaluation_key']
    selected = json.loads((WORK / 'selected.json').read_text())
    assert key not in selected
    plan = ROOT / f'.agent/work/EXP-001/opus_postprocess/plans/simplify_core80_{args.condition}.jsonl'
    with plan.open() as handle:
        rows = ((number, json.loads(line)) for number, line in enumerate(handle, 1))
        number, row = next((number, row) for number, row in rows if row['evaluation_key'] == key)
    entry = _row_to_definition(row, line_number=number)
    definition = entry.definition
    assert definition.request == source['request']
    assert source['prompt'] == render_prompt(definition.prompt_template, definition.request, definition.schema)
    metadata = source['metadata']
    assert sha(canonical_json(source['api_request']).encode()) == metadata['api_request_sha256']
    assert sha(canonical_json(source['api_response']).encode()) == metadata['persisted_api_response_sha256']
    assert sha(source['api_request']['system'].encode()) == metadata['system_prompt_sha256']
    assert source['api_request']['messages'] == [{'role': 'user', 'content': source['prompt']}]
    response = AnthropicApiResponse(metadata['http_status'], source['api_response'], '', {})
    assert response.status_code == 200
    structured, _, _, _ = AnthropicApiRunner._extract_structured_output(response, task_kind='simplify', schema=definition.schema)
    payload = {'evaluation_key': key, 'request': source['request'], 'structured_output': structured}
    replay = subprocess.run([sys.executable, str(Path(__file__).with_name('limited_semantic_worker.py'))],
                            input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=180)
    checked = json.loads(replay.stdout)
    assert checked['status'] == 'ok', checked
    directory = args.attempt.parent.parent
    store = TaskStateStore(directory / 'state.sqlite3', attempt_cap=10**9,
                           logical_task_cap=40000, max_attempts_per_task=10**9)
    frozen_path = directory / 'frozen' / f'{key}.json'
    assert not frozen_path.exists()
    provenance = {'attempt_path': str(args.attempt.resolve()), 'attempt_sha256': sha(args.attempt.read_bytes()),
                  'time': time.time(), 'reason': '精确十进制重建与高精度数值复核修复后重新验收',
                  'semantic_validator_sha256': sha((ROOT / 'AAAI_experiments/stage5_metric_calculation_0831/pipeline/symbolic_evidence.py').read_bytes())}
    result = {**source, 'logical_id': entry.logical_id, 'task_type': definition.task_spec.task_type,
              'task_kind': 'simplify', 'structured_output': structured,
              'command': ['anthropic-messages-api', '--model', metadata['requested_model'],
                          '--effort', metadata['requested_effort'], '--stream', 'false'],
              'stdout': canonical_json(source['api_response']), 'stderr': '', 'envelope': source['api_response'],
              'original_metadata': metadata,
              'metadata': {**metadata, 'error_class': None, 'retryable': False,
                           'semantic_validator_sha256': provenance['semantic_validator_sha256']},
              'original_validation': source['validation'], 'revalidation': provenance,
              'validation': {'ok': True, 'error_class': None, 'error_message': None,
                             'structured_output': structured, 'semantic_evidence': checked['semantic_evidence']}}
    write(frozen_path, result)
    result_sha = sha(frozen_path.read_bytes())
    store.promote_failed_attempt(source['attempt_id'], result_path=str(frozen_path.resolve()),
                                result_sha256=result_sha, allowed_error_classes=['validation_failed', 'semantic_validator_error'],
                                audit_reason=provenance['reason'])
    selected[key] = {'logical_id': entry.logical_id, 'condition': args.condition, 'evaluation_key': key,
                     'attempt_id': source['attempt_id'], 'result_path': str(frozen_path.resolve()),
                     'result_sha256': result_sha, 'time': time.time(), 'revalidation': provenance}
    write(WORK / 'selected.json', selected)
    progress = json.loads((WORK / 'progress.json').read_text())
    progress['selected'] = len(selected)
    progress['by_condition'][args.condition]['selected'] += 1
    progress.update(time=time.time(), status='complete' if progress['selected'] == progress['total'] else 'stopped')
    write(WORK / 'progress.json', progress)
    print(json.dumps({'logical_id': entry.logical_id, 'proof': checked['semantic_evidence']['proof_basis'],
                      'selected': len(selected), 'total': progress['total']}))


if __name__ == '__main__':
    main()
