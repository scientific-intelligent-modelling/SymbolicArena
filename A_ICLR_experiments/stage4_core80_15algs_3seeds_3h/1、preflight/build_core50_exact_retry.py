import json
from pathlib import Path
import sqlite3
import sys

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.clean_task_builder import _build_task_definition, _load_prompt_schema, PRED_PRIORITY
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import load_plan_jsonl
from run_core50_comparisons import source_index, sha, write_json, canonical


def main():
    sys.setrecursionlimit(20000)
    runtime = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime')
    root = runtime/'core50_comparisons'
    prior = runtime/'core50_new15_opus_v5'
    failures = json.loads((prior/'failure_audit/failed.json').read_text())
    numeric = source_index(root)
    prompt = runtime/'AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt'
    contract = _load_prompt_schema(runtime, prompt_path=prompt)
    path = root/'plans/retry_exact.jsonl'
    if path.exists():
        raise FileExistsError(path)
    manifest = []
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as output:
        for row in failures:
            with sqlite3.connect(f"file:{row['source_db']}?mode=ro", uri=True) as connection:
                state = connection.execute('SELECT state FROM tasks WHERE evaluation_key=?', (row['evaluation_key'],)).fetchone()[0]
            if state == 'frozen':
                continue
            if state != 'exhausted':
                raise ValueError(f"状态变化: {row['logical_id']}")
            source_path = Path(row['history'][-1]['path'])
            source = json.loads(source_path.read_text())
            request = source['request']
            key = (request['noise_tag'], request['algorithm_slug'], request['dataset_index'], int(request['seed']))
            if request['ast_source_evidence']['result_raw_sha256'] != numeric[key]['sha256']:
                raise ValueError(f'重试公式来源改变: {key}')
            task = _build_task_definition(logical_id=row['logical_id']+'::v2', task_type='pred_simplify',
                priority=PRED_PRIORITY, request=request, evidence_hash=request['evidence_hash'],
                contract=contract, condition=request['noise_tag'])
            output.write(canonical(task.to_json_record())+'\n')
            manifest.append({'previous_key': row['evaluation_key'], 'evaluation_key': task.evaluation_key,
                             'logical_id': task.logical_id, 'source_db': row['source_db'],
                             'source_attempt': str(source_path), 'source_attempt_sha256': sha(source_path),
                             'previous_reason': row['last_message']})
    temporary.replace(path)
    loaded = load_plan_jsonl(path)
    write_json(root/'retry_exact_manifest.json', {'tasks': manifest, 'count': len(manifest),
        'plan_sha256': loaded.plan_sha256, 'prompt_sha256': sha(prompt), 'max_tokens': 65536,
        'max_attempts_per_task': 6, 'new_attempt_cap': 6*len(manifest), 'retry_cap': 5*len(manifest)})
    print({'requests': len(manifest), 'max_attempts': 6*len(manifest), 'retry_cap': 5*len(manifest)})


if __name__ == '__main__':
    main()
