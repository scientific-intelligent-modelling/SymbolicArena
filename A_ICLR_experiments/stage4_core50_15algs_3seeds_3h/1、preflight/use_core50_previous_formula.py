import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import clean_task_builder as clean
from AAAI_experiments.stage5_metric_calculation_0831.pipeline import symbolic_task_builder as symbolic
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import replay_payload_performance
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import RunQuality, stability_score
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index import _resolve_simplify_effective_expression
from aggregate_core50_minute_metrics import symbolic_binding
from run_core50_comparisons import context_binding, evidence_job, read_frozen_rows, run, sha, write_json, canonical
from run_core50_minute_opus import materialize, simplify_pair


REPO = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parent.parent
METRICS = STAGE / '3、metrics'
LOCAL = METRICS / 'formula_override_g0436_s522'
RUNTIME = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime')
REMOTE = RUNTIME / 'core50_previous_g0436_s522'
KEY = 'llmsr::feynman-i.11.19::s522::noise005'


def prepare():
    LOCAL.mkdir(parents=True, exist_ok=True)
    directory = STAGE / '2.2 core50 experiments/noise005/llmsr/feynman-i.11.19/522'
    original = json.loads((directory / 'result.json').read_text())
    history = directory / 'experiments' / Path(original['experiment_dir']).name / 'best_history'
    rejected = set()
    for path in sorted((directory / 'progress').glob('minute_*.json'), reverse=True):
        snapshot = json.loads(path.read_text())
        if not snapshot.get('equation') or snapshot.get('elapsed_seconds', 10801) > 10800:
            continue
        tree = ast.parse(snapshot['equation'])
        if any(isinstance(node, ast.Call) and ast.unparse(node.func) in ('np.linalg.norm', 'linalg.norm') for node in ast.walk(tree)):
            rejected.add(snapshot.get('source_sample_order'))
            continue
        candidate_path = history / f"best_sample_{snapshot['source_sample_order']}.json"
        candidate = json.loads(candidate_path.read_text())
        same_formula = ast.dump(ast.parse(clean.extract_expression_body(candidate['function']), mode='eval')) == ast.dump(ast.parse(clean.extract_expression_body(snapshot['equation']), mode='eval'))
        if (not same_formula
                or candidate['nmse'] != snapshot['source_internal_loss']
                or candidate['params'] != snapshot['canonical_artifact']['parameter_values']):
            raise ValueError('历史原生候选、参数或目标绑定不一致')
        break
    else:
        raise ValueError('未找到上一个可处理的预算内原生候选')
    selected = dict(original)
    for field in ('equation', 'canonical_artifact', 'train', 'valid', 'source_sample_order', 'source_internal_loss'):
        selected[field] = snapshot.get(field)
    selection = {'policy': 'user_authorized_previous_usable_native_candidate',
        'original_result_path': str(directory / 'result.json'), 'original_result_sha256': sha(directory / 'result.json'),
        'rejected_sample_orders': sorted(rejected), 'selected_sample_order': candidate['sample_order'],
        'selected_snapshot_path': str(path), 'selected_snapshot_sha256': sha(path),
        'selected_elapsed_seconds': snapshot['elapsed_seconds'], 'selected_minute': snapshot['elapsed_minutes'],
        'candidate_path': str(candidate_path), 'candidate_sha256': sha(candidate_path),
        'criterion': 'latest prior native incumbent with supported scalar expression; no ID/OOD selection'}
    selected['selection_override'] = selection
    replay = replay_payload_performance(selected, algorithm='llmsr', repo_root=REPO)
    for field in ('canonical_artifact', 'id_test', 'ood_test'):
        selected[field] = replay[field]
    write_json(LOCAL / 'selected_result.json', selected)
    write_json(LOCAL / 'numeric.json', replay)
    write_json(LOCAL / 'selection.json', selection)
    shutil.copy2(candidate_path, LOCAL / 'selected_candidate.json')
    shutil.copy2(path, LOCAL / 'selected_snapshot.json')
    probes, _ = clean.load_dataset_probes(REPO / 'AAAI_experiments/stage5_metric_calculation_0831/reports/dataset_probes.jsonl')
    probe = probes[original['dataset']]
    expression, _ = clean.select_formula_with_source({'canonical_artifact': replay['canonical_artifact']})
    expression, mapping = clean.map_indexed_variables(expression, original['feature_names'])
    functions, assumptions, evidence = clean._build_symbolic_request_evidence(
        expression=expression, variables=original['feature_names'], probe=probe)
    contract = clean._load_prompt_schema(REPO, prompt_path=REPO /
        'AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt')
    request = {'dataset_id': original['dataset'], 'dataset_index': 'g0436', 'algorithm': 'llmsr',
        'algorithm_slug': 'llmsr', 'seed': 522, 'noise_tag': 'noise005', 'variables': original['feature_names'],
        'expression': expression, 'original_expression': expression, 'allowed_functions': functions,
        'domain_assumptions': assumptions, 'probe_points': probe['points'], 'probe_source': probe['schema_version'],
        'probe_sample_sha256': probe['sample_sha256'], 'dataset_probe_evidence': probe, 'deterministic_evidence': evidence,
        'ast_source_evidence': {'result_path': str(REMOTE / 'selected_result.json'),
            'result_raw_sha256': sha(LOCAL / 'selected_result.json'), 'variable_mapping': mapping,
            'selection_override': selection}}
    request['evidence_hash'] = hashlib.sha256(canonical(request).encode()).hexdigest()
    task = clean._build_task_definition(logical_id='pred_simplify::llmsr::g0436::s522::noise005',
        task_type='pred_simplify', priority=clean.PRED_PRIORITY, request=request,
        evidence_hash=request['evidence_hash'], contract=contract, condition='noise005').to_json_record()
    for field in ('prompt_path', 'schema_path'):
        task[field] = str(RUNTIME / Path(task[field]).relative_to(REPO))
    (LOCAL / 'plans').mkdir(exist_ok=True)
    (LOCAL / 'plans/prediction.jsonl').write_text(canonical(task) + '\n')
    write_json(LOCAL / 'probe.json', probe)
    write_json(LOCAL / 'preparation.complete.json', {'scope': 'one_final_formula_override', 'requests': 4})
    print(json.dumps({'selected_sample_order': candidate['sample_order'], 'last_witness_minute': snapshot['elapsed_minutes'],
        'id_nmse': replay['id_test']['nmse'], 'ood_nmse': replay['ood_test']['nmse'],
        'new_requests': 4, 'retry_limit': 20, 'attempt_limit': 24}), flush=True)


def execute():
    root = REMOTE
    dependencies = json.loads((RUNTIME / 'core50_comparisons/dependencies.json').read_text())
    required = {'gt_simplify::feynman-i.11.19', 'pred_simplify::llmsr::g0436::s520::noise005',
                'pred_simplify::llmsr::g0436::s521::noise005'}
    cached = {key: row for key, row in dependencies.items() if row['task']['logical_id'] in required}
    if len(cached) != 3:
        raise ValueError('既有GT和两个seed化简依赖不完整')
    write_json(root / 'dependencies.json', cached)
    args = argparse.Namespace(root=root, runtime=RUNTIME, workers=3, logical_task_cap=8,
        channel_settings='routify=/home/zhangziwen/.config/core50-opus/routify.json')
    run(args)
    database = root / 'execution/pred_simplify__noise005/state.sqlite3'
    prediction = list(read_frozen_rows(database, root / 'plans/prediction.jsonl'))
    if len(prediction) != 1:
        raise RuntimeError('替代公式化简尚未验收')
    pred_record = prediction[0]
    cached[pred_record['task']['evaluation_key']] = pred_record
    write_json(root / 'dependencies.json', cached)
    if not (root / 'plans/comparisons.jsonl').exists():
        gt_record = next(row for row in cached.values() if row['task']['task_type'] == 'gt_simplify')
        pred, pf = simplify_pair(pred_record)
        gt, gf = simplify_pair(gt_record)
        probe = json.loads((root / 'probe.json').read_text())
        common = {'dataset_id': 'feynman-i.11.19', 'dataset_index': 'g0436', 'algorithm': 'llmsr',
                  'algorithm_slug': 'llmsr', 'noise_tag': 'noise005', 'variables': pred.request['variables']}
        request = {**common, 'seed': 522, 'effective_ground_truth_expression': gf.effective_expression,
            'simplified_ground_truth_expression': gf.simplified_expression,
            'effective_prediction_expression': pf.effective_expression, 'simplified_prediction_expression': pf.simplified_expression,
            **context_binding('ground_truth', gt, gf), **context_binding('prediction', pred, pf),
            'prediction_task_id': 'llmsr_s522_noise005_g0436', 'prediction_valid_output': True,
            'prediction_result_sha256': sha(root / 'selected_result.json')}
        jobs = [('equivalence', 'equivalence::llmsr::g0436::s522::noise005', gt, gf, pred, pf, 522, request)]
        for seed in (520, 521):
            old_record = next(row for row in cached.values() if row['task']['logical_id'] == f'pred_simplify::llmsr::g0436::s{seed}::noise005')
            old, of = simplify_pair(old_record)
            request = {**common, 'seed_a': seed, 'seed_b': 522,
                'effective_prediction_a_expression': of.effective_expression, 'effective_prediction_b_expression': pf.effective_expression,
                'simplified_prediction_a_expression': of.simplified_expression, 'simplified_prediction_b_expression': pf.simplified_expression,
                **context_binding('prediction_a', old, of), **context_binding('prediction_b', pred, pf),
                'prediction_a_task_id': f'llmsr_s{seed}_noise005_g0436', 'prediction_b_task_id': 'llmsr_s522_noise005_g0436',
                'prediction_a_valid_output': True, 'prediction_b_valid_output': True,
                'prediction_a_result_sha256': old.request['ast_source_evidence']['result_raw_sha256'],
                'prediction_b_result_sha256': sha(root / 'selected_result.json')}
            jobs.append(('structure', symbolic._structure_logical_id('llmsr', 'g0436', seed, 522, condition='noise005'), old, of, pred, pf, seed*1000+522, request))
        planned = []
        for kind, logical_id, left, lf, right, rf, seed, request in jobs:
            result = evidence_job({'arguments': (kind, logical_id, left, lf, right, rf, seed, probe), 'timeout_seconds': 180})
            if 'error' in result:
                raise RuntimeError(result['error'])
            evidence = result['evidence']
            request.update(allowed_functions=evidence['allowed_functions'], deterministic_evidence=evidence,
                           evidence_hash=evidence['evidence_sha256'])
            if kind == 'structure':
                request['deterministic_pair_evidence'] = evidence
            task = symbolic._task_from_request(logical_id=logical_id,
                task_type='equivalence' if kind == 'equivalence' else 'stab_structure',
                priority=symbolic.EQUIVALENCE_PRIORITY if kind == 'equivalence' else symbolic.STRUCTURE_PRIORITY,
                request=request, evidence_hash=evidence['evidence_sha256'],
                contract=symbolic._load_prompt_schema(RUNTIME, task_kind=kind),
                dependencies=(left.evaluation_key, right.evaluation_key), condition='noise005')
            planned.append(symbolic._task_json_record(task))
        (root / 'plans/comparisons.jsonl').write_text(''.join(canonical(row) + '\n' for row in planned))
    run(args)
    progress = json.loads((root / 'progress.json').read_text())
    if progress['registered'] != 4 or any(set(counts) != {'frozen'} for counts in progress['by_type'].values()):
        raise RuntimeError('替代公式的四项后处理未全部验收')
    write_json(root / 'complete.json', {'completed_tasks': 4, 'progress': progress})


def publish():
    api = LOCAL / 'api'
    if json.loads((api / 'complete.json').read_text())['completed_tasks'] != 4:
        raise ValueError('替代公式的API验收未完成')
    plans = {row['evaluation_key']: path for path in (api / 'plans').glob('*.jsonl')
             for row in (json.loads(line) for line in path.read_text().splitlines() if line)}
    records = {}
    for db in (api / 'execution').glob('*/state.sqlite3'):
        for record in read_frozen_rows(db, plan_by_key=plans, evaluation_keys=set(plans)):
            path = api / Path(record['frozen']['result_path']).relative_to(REMOTE)
            if sha(path) != record['frozen']['result_sha256']:
                raise ValueError('新裁决哈希不符')
            records[record['task']['evaluation_key']] = (record, json.loads(path.read_text()), path)
    if len(records) != 4:
        raise ValueError('本地新裁决数量不符')
    backup = LOCAL / 'input_snapshot'
    backup.mkdir(exist_ok=True)
    for name in ('algorithm_six_axis.csv', 'noise_supplement.csv', 'terminal_run_metrics.csv',
                 'terminal_run_metrics.jsonl.gz', 'unresolved.json', 'verification.json', 'opus_sources.json'):
        if not (backup / name).exists():
            shutil.copy2(METRICS / name, backup / name)
    with gzip.open(backup / 'terminal_run_metrics.jsonl.gz', 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    target = next(row for row in rows if row['logical_key'] == KEY)
    selection = json.loads((LOCAL / 'selection.json').read_text())
    if target['source_sha256'] != selection['original_result_sha256']:
        raise ValueError('被替代的结果版本不一致')
    numeric = json.loads((LOCAL / 'numeric.json').read_text())
    snapshot = json.loads((LOCAL / 'selected_snapshot.json').read_text())
    target['original_source'] = {'path': target['source_path'], 'sha256': target['source_sha256']}
    target['eff_source_result_sha256'] = target['source_sha256']
    target['eff_selection_policy'] = 'original_native_training_trajectory'
    target['source_path'], target['source_sha256'] = str(LOCAL / 'selected_result.json'), sha(LOCAL / 'selected_result.json')
    target['selection_override'] = selection
    for field in ('canonical_artifact', 'canonical_artifact_sha256', 'id_test', 'ood_test', 'id_quality',
                  'ood_quality', 'valid_output', 'evaluation_path', 'invalid_reason'):
        target[field] = numeric[field]
    target['native_id_nmse'] = (snapshot.get('id_test') or {}).get('nmse')
    target['native_ood_nmse'] = (snapshot.get('ood_test') or {}).get('nmse')
    target['status'], target['reason'] = ('resolved', None) if numeric['valid_output'] else ('invalid', numeric['invalid_reason'])
    pred_record, pred_artifact, pred_path = next(value for value in records.values() if value[0]['task']['task_type'] == 'pred_simplify')
    effective, resolution = _resolve_simplify_effective_expression(request=pred_artifact['request'],
        structured_output=pred_artifact['structured_output'], context=pred_record['task']['logical_id'])
    target['simplification'] = {'original_expression': pred_artifact['request']['expression'],
        'effective_expression': effective,
        'outcome': pred_artifact['structured_output']['outcome'], 'resolution': resolution,
        'evaluation_key': pred_record['task']['evaluation_key'], 'result_path': str(pred_path),
        'result_sha256': pred_record['frozen']['result_sha256'], 'source_result_sha256': target['source_sha256']}
    eq_record, eq_artifact, _ = next(value for value in records.values() if value[0]['task']['task_type'] == 'equivalence')
    if eq_artifact['request']['prediction_result_sha256'] != target['source_sha256']:
        raise ValueError('新数值结果与符号裁决未绑定')
    target['symbolic_binding'] = symbolic_binding(eq_record, artifact=eq_artifact, recompute=True)
    target.update(sym_score=target['symbolic_binding']['sym_score'], min_score=target['symbolic_binding']['min_score'])
    group = sorted([row for row in rows if row['condition'] == 'noise005' and row['algorithm'] == 'llmsr'
                    and row['dataset_index'] == 'g0436'], key=lambda row: row['seed'])
    old_binding = group[0]['structure_bindings'][0]
    sources = json.loads((backup / 'opus_sources.json').read_text())
    old_source = sources[old_binding['evaluation_key']]
    old_path = METRICS / 'opus' / Path(old_source['result_path']).relative_to(RUNTIME)
    if sha(old_path) != old_binding['sha256']:
        raise ValueError('既有seed520/521结构裁决哈希不符')
    old = json.loads(old_path.read_text())
    decisions = [old['structured_output']['decision'] in ('mathematically_equivalent', 'same_canonical_structure')]
    bindings = [old_binding]
    for seed in (520, 521):
        record, artifact, path = next(value for value in records.values()
            if value[0]['task']['task_type'] == 'stab_structure' and value[1]['request']['seed_a'] == seed)
        if artifact['request']['prediction_b_result_sha256'] != target['source_sha256']:
            raise ValueError('新结构比较引用的公式版本不符')
        decisions.append(artifact['structured_output']['decision'] in ('mathematically_equivalent', 'same_canonical_structure'))
        bindings.append({'evaluation_key': record['task']['evaluation_key'], 'sha256': record['frozen']['result_sha256'],
                         'result_path': str(path)})
    stab = stability_score([RunQuality(row['id_quality'], row['ood_quality'], row['valid_output']) for row in group],
                           structural_pair_results=decisions).score
    for row in group:
        row['stab_score'], row['structure_bindings'] = stab, bindings
    with gzip.open(METRICS / 'terminal_run_metrics.jsonl.gz', 'wt') as handle:
        for row in rows:
            handle.write(canonical(row) + '\n')
    axes = {'ID': 'id_quality', 'OOD': 'ood_quality', 'SYM': 'sym_score', 'MIN': 'min_score', 'EFF': 'eff_score', 'STAB': 'stab_score'}
    affected = {row['logical_key']: row for row in group}
    csv.field_size_limit(100000000)
    with (backup / 'terminal_run_metrics.csv').open(newline='') as source, (METRICS / 'terminal_run_metrics.csv').open('w', newline='') as dest:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['logical_key'] in affected:
                record = affected[row['logical_key']]
                row.update({axis: record[field] for axis, field in axes.items()})
                if row['logical_key'] == KEY:
                    row.update(original_expression=target['simplification']['original_expression'],
                        effective_expression=target['simplification']['effective_expression'],
                        simplification_outcome=target['simplification']['outcome'],
                        source_path=target['source_path'], source_sha256=target['source_sha256'])
            writer.writerow(row)
    with (backup / 'noise_supplement.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, summaries = reader.fieldnames, list(reader)
    summary = next(row for row in summaries if row['condition'] == 'noise005' and row['algorithm'] == 'llmsr')
    same_algorithm = [row for row in rows if row['condition'] == 'noise005' and row['algorithm'] == 'llmsr']
    for axis, field in axes.items():
        if axis == 'EFF':
            continue
        values = {row['dataset_id']: row[field] for row in same_algorithm}.values() if axis == 'STAB' else [row[field] for row in same_algorithm]
        values = list(values)
        expected = 50 if axis == 'STAB' else 150
        if len(values) != expected or any(value is None for value in values):
            raise ValueError('LLM-SR噪声指标覆盖不完整')
        summary[axis], summary[f'{axis}_available'] = 100 * sum(values) / expected, expected
    with (METRICS / 'noise_supplement.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    unresolved = json.loads((backup / 'unresolved.json').read_text())
    unresolved['runs'] = [row for row in unresolved['runs'] if row['logical_key'] not in affected]
    for axis in ('SYM', 'MIN', 'STAB'):
        unresolved['axis_counts'][axis] = 0
    write_json(METRICS / 'unresolved.json', unresolved)
    for key, (record, _, path) in records.items():
        sources[key] = {'task_type': record['task']['task_type'], 'logical_id': record['task']['logical_id'],
            'result_path': str(path), 'result_sha256': record['frozen']['result_sha256'],
            'source_plan': record['source_plan'], 'source_plan_sha256': record['source_plan_sha256']}
    write_json(METRICS / 'opus_sources.json', sources)
    receipt = {'logical_key': KEY, 'selected_sample_order': selection['selected_sample_order'],
        'source_sha256': target['source_sha256'], 'summary': summary, 'new_frozen_tasks': 4,
        'EFF_policy': 'original_native_training_trajectory_unchanged', 'original_result_preserved': True,
        'files': {name: sha(METRICS / name) for name in ('noise_supplement.csv', 'terminal_run_metrics.csv', 'terminal_run_metrics.jsonl.gz')}}
    write_json(LOCAL / 'verification.json', receipt)
    report = json.loads((backup / 'verification.json').read_text())
    report['terminal_unresolved'] = unresolved['axis_counts']
    report['formula_selection_override'] = {'path': str(LOCAL / 'verification.json'), 'sha256': sha(LOCAL / 'verification.json')}
    for name in report['files']:
        report['files'][name] = {'path': str(METRICS / name), 'sha256': sha(METRICS / name)}
    report['files']['formula_override_g0436_s522/verification.json'] = {'path': str(LOCAL / 'verification.json'), 'sha256': sha(LOCAL / 'verification.json')}
    write_json(METRICS / 'verification.json', report)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', 'publish'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        execute()
    else:
        publish()
