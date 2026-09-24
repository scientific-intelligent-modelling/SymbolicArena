import argparse
from collections import Counter, defaultdict
import csv
import gzip
from itertools import combinations
import json
import math
from pathlib import Path
import sys

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    RunQuality, minimality_score, stability_score, symbolic_fidelity_score,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    build_symbolic_artifact, operator_f1, tree_similarity, variable_f1,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_task_builder import _pair_allowed_functions
from run_core50_comparisons import canonical, sha, write_json
from run_core50_minute_opus import digest, frozen_records, materialize, read_minutes, simplify_pair


def symbolic_binding(record):
    artifact = materialize(record)
    pair = artifact['request']['deterministic_evidence']['pair_evidence']
    lhs, rhs = pair['lhs_artifact'], pair['rhs_artifact']
    tree, variables, operators = tree_similarity(lhs, rhs), variable_f1(lhs, rhs), operator_f1(lhs, rhs)
    for actual, expected in ((tree, pair['tree']['tree_similarity']),
                             (variables, pair['variable']['f1']), (operators, pair['operator']['f1'])):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"符号指标复算不一致: {record['task']['logical_id']}")
    decision = artifact['structured_output']['decision']
    return {'evaluation_key': record['task']['evaluation_key'], 'result_path': record['frozen']['result_path'],
        'result_sha256': record['frozen']['result_sha256'], 'decision': decision,
        'sym_score': symbolic_fidelity_score(equivalent=decision == 'equivalent', tree_similarity=tree,
                                            variable_f1=variables, operator_f1=operators),
        'min_score': minimality_score(lhs['node_count'], rhs['node_count']),
        'tree_similarity': tree, 'variable_f1': variables, 'operator_f1': operators,
        'reference_complexity': lhs['node_count'], 'predicted_complexity': rhs['node_count']}


def minimality_binding(prediction, ground_truth):
    pred, pf = simplify_pair(prediction)
    gt, gf = simplify_pair(ground_truth)
    functions = _pair_allowed_functions(pred, pf) | _pair_allowed_functions(gt, gf)
    left = build_symbolic_artifact(gf.effective_expression, allowed_variables=gt.request['variables'], allowed_functions=functions)
    right = build_symbolic_artifact(pf.effective_expression, allowed_variables=pred.request['variables'], allowed_functions=functions)
    return {'min_score': minimality_score(left['node_count'], right['node_count']),
        'prediction_key': pred.evaluation_key, 'ground_truth_key': gt.evaluation_key,
        'prediction_sha256': prediction['frozen']['result_sha256'],
        'ground_truth_sha256': ground_truth['frozen']['result_sha256'],
        'reference_complexity': left['node_count'], 'predicted_complexity': right['node_count']}


def terminal_metrics(root, output, efficiencies):
    comparison_root = root.parent / 'core50_comparisons'
    records = frozen_records(comparison_root)
    equivalences, structures = {}, {}
    predictions, ground_truth = {}, {}
    for record in records.values():
        kind = record['task']['task_type']
        if kind in ('pred_simplify', 'gt_simplify'):
            request = materialize(record)['request']
            if kind == 'gt_simplify':
                ground_truth[request['dataset_id']] = record
            else:
                predictions[request['noise_tag'], request['algorithm_slug'], request['dataset_id'], request['seed']] = record
            continue
        if kind not in ('equivalence', 'stab_structure'):
            continue
        artifact = materialize(record)
        request = artifact['request']
        identity = (request['noise_tag'], request['algorithm_slug'], request['dataset_id'])
        if kind == 'equivalence':
            equivalences[(*identity, request['seed'])] = (record, artifact)
        else:
            structures[(*identity, request['seed_a'], request['seed_b'])] = (record, artifact)
    groups = defaultdict(dict)
    for condition in ('clean', 'noise001', 'noise005'):
        path = root / 'inputs/terminal_numeric' / f'{condition}.jsonl.gz'
        manifest = json.loads((path.parent / f'{condition}.manifest.json').read_text())
        if sha(path) != manifest['output_sha256'] or manifest['runs'] != 2250:
            raise ValueError('最终数值覆盖或哈希不一致')
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                identity = (condition, row['algorithm'], row['dataset_id'])
                key = (*identity, row['seed'])
                if row['seed'] in groups[identity]:
                    raise ValueError('最终运行身份重复')
                row['sym_score'], row['min_score'] = None, None
                eff = efficiencies[row['logical_key']]
                if eff['result_sha256'] != row['source_sha256']:
                    raise ValueError('最终结果与EFF输入版本不一致')
                row['eff_score'] = eff['cumulative_eff']
                if row.get('reason') == 'missing_budget_expression':
                    row['sym_score'], row['min_score'] = 0.0, 0.0
                elif key in equivalences:
                    record, artifact = equivalences[key]
                    if artifact['request']['prediction_result_sha256'] != row['source_sha256']:
                        raise ValueError('最终符号裁决来源不一致')
                    row['symbolic_binding'] = symbolic_binding(record)
                    row.update(sym_score=row['symbolic_binding']['sym_score'], min_score=row['symbolic_binding']['min_score'])
                if row['min_score'] is None and key in predictions:
                    row['minimality_binding'] = minimality_binding(predictions[key], ground_truth[row['dataset_id']])
                    row['min_score'] = row['minimality_binding']['min_score']
                groups[identity][row['seed']] = row
    if len(groups) != 2250:
        raise ValueError('最终算法与数据集覆盖不完整')
    aggregate_rows = defaultdict(lambda: defaultdict(list))
    unresolved = Counter()
    with gzip.open(output / 'terminal_run_metrics.jsonl.gz', 'wt') as handle:
        for identity, seeds in sorted(groups.items()):
            if set(seeds) != {520, 521, 522}:
                raise ValueError('最终结果三个seed覆盖不完整')
            decisions, bindings = [], []
            for a, b in combinations((520, 521, 522), 2):
                if seeds[a]['valid_output'] is False or seeds[b]['valid_output'] is False:
                    decisions.append(False)
                    bindings.append({'status': 'not_applicable_invalid_output'})
                elif (*identity, a, b) in structures:
                    record, artifact = structures[(*identity, a, b)]
                    request = artifact['request']
                    if (request['prediction_a_result_sha256'] != seeds[a]['source_sha256']
                            or request['prediction_b_result_sha256'] != seeds[b]['source_sha256']):
                        raise ValueError('最终结构裁决来源不一致')
                    decisions.append(artifact['structured_output']['decision'] in ('mathematically_equivalent', 'same_canonical_structure'))
                    bindings.append({'evaluation_key': record['task']['evaluation_key'], 'sha256': record['frozen']['result_sha256']})
                else:
                    decisions.append(None)
                    bindings.append({'status': 'missing_accepted_structure'})
            stab = None
            if all(d is not None for d in decisions) and all(r['valid_output'] is not None for r in seeds.values()):
                stab = stability_score([RunQuality(r['id_quality'], r['ood_quality'], r['valid_output'])
                                        for r in seeds.values()], structural_pair_results=decisions).score
            group = aggregate_rows[identity[:2]]
            group['STAB'].append(stab)
            for row in seeds.values():
                row['stab_score'], row['structure_bindings'] = stab, bindings
                for axis, field in (('ID', 'id_quality'), ('OOD', 'ood_quality'), ('SYM', 'sym_score'),
                                    ('MIN', 'min_score'), ('EFF', 'eff_score')):
                    group[axis].append(row[field])
                    if row[field] is None:
                        unresolved[axis] += 1
                handle.write(canonical(row) + '\n')
            if stab is None:
                unresolved['STAB'] += 1
    summaries = []
    for (condition, algorithm), axes in sorted(aggregate_rows.items()):
        row = {'condition': condition, 'algorithm': algorithm}
        for axis in ('ID', 'OOD', 'SYM', 'MIN', 'EFF', 'STAB'):
            expected = 50 if axis == 'STAB' else 150
            values = axes[axis]
            if len(values) != expected:
                raise ValueError('最终指标分母错误')
            valid = [v for v in values if v is not None]
            row[axis] = 100 * sum(valid) / expected if len(valid) == expected else None
            row[f'{axis}_available'] = len(valid)
            row[f'{axis}_expected'] = expected
        summaries.append(row)
    for name, selected in (('algorithm_six_axis.csv', [r for r in summaries if r['condition'] == 'clean']),
                           ('noise_supplement.csv', [r for r in summaries if r['condition'] != 'clean'])):
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    return dict(unresolved)


def aggregate(root, output):
    records = frozen_records(root)
    plan = json.loads((root / 'simplification_plan.json').read_text())
    evidence = {}
    structures = {}
    for key, record in records.items():
        logical_id = record['task']['logical_id']
        if not logical_id.startswith(('minute_equivalence::', 'minute_structure::')):
            continue
        artifact = materialize(record)
        identity = logical_id.split('::', 1)[1]
        binding = {'evaluation_key': key, 'result_path': record['frozen']['result_path'],
                   'result_sha256': record['frozen']['result_sha256'], 'decision': artifact['structured_output']['decision']}
        if logical_id.startswith('minute_structure::'):
            structures[identity] = binding
            continue
        evidence[identity] = symbolic_binding(record)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(dict)
    summary = defaultdict(lambda: defaultdict(list))
    unresolved = Counter()
    runs = set()
    efficiencies = {}
    for row in read_minutes(root):
        runs.add(row['logical_key'])
        if row['minute'] == 180:
            efficiencies[row['logical_key']] = row
        key = (row['condition'], row['algorithm'], row['dataset_id'], row['minute'])
        if row['seed'] in groups[key]:
            raise ValueError(f'逐分钟记录重复: {row["run_minute_key"]}')
        if row['status'] == 'invalid':
            row.update(sym_score=0.0, min_score=0.0, symbolic_status='not_applicable_invalid_output')
        elif row.get('expression_key') in evidence:
            row['symbolic_binding'] = evidence[row['expression_key']]
            row.update(sym_score=row['symbolic_binding']['sym_score'], min_score=row['symbolic_binding']['min_score'],
                       symbolic_status='accepted')
        else:
            row.update(sym_score=None, min_score=None, symbolic_status='unresolved')
            unresolved[row.get('reason') or 'missing_accepted_symbolic_comparison'] += 1
            prediction_key = plan['aliases'].get(row.get('expression_key'))
            if prediction_key in records:
                row['minimality_binding'] = minimality_binding(records[prediction_key], records[plan['ground_truth'][row['dataset_id']]])
                row['min_score'] = row['minimality_binding']['min_score']
        groups[key][row['seed']] = row
    if len(runs) != 6750 or len(groups) != 405000:
        raise ValueError(f'六维覆盖不足: runs={len(runs)}, task_minutes={len(groups)}')
    run_path = output / 'run_minute_metrics.jsonl.gz'
    task_path = output / 'task_minute_stability.jsonl.gz'
    with gzip.open(run_path, 'wt') as run_out, gzip.open(task_path, 'wt') as task_out:
        for (condition, algorithm, dataset, minute), by_seed in sorted(groups.items()):
            if set(by_seed) != {520, 521, 522}:
                raise ValueError('STAB三个seed覆盖不完整')
            run_rows = [by_seed[seed] for seed in (520, 521, 522)]
            decisions, bindings = [], []
            for a, b in combinations(run_rows, 2):
                if a['valid_output'] is False or b['valid_output'] is False:
                    decisions.append(False)
                    bindings.append({'status': 'not_applicable_invalid_output'})
                    continue
                if a['valid_output'] is None or b['valid_output'] is None or not a['expression_key'] or not b['expression_key']:
                    decisions.append(None)
                    bindings.append({'status': 'unresolved_source'})
                    continue
                key = digest({'dataset': dataset, 'pair': sorted([a['expression_key'], b['expression_key']])})
                binding = structures.get(key)
                if binding is None:
                    decisions.append(None)
                    bindings.append({'status': 'missing_accepted_structure', 'comparison_key': key})
                else:
                    decisions.append(binding['decision'] in ('mathematically_equivalent', 'same_canonical_structure'))
                    bindings.append(binding)
            score = None
            components = {}
            if all(d is not None for d in decisions) and all(
                r['id_quality'] is not None and r['ood_quality'] is not None and r['valid_output'] is not None for r in run_rows):
                value = stability_score([RunQuality(r['id_quality'], r['ood_quality'], r['valid_output']) for r in run_rows],
                                        structural_pair_results=decisions)
                score = value.score
                components = {'numerical_consistency': value.numerical_consistency, 'validity': value.validity,
                              'structural_consistency': value.structural_consistency}
            else:
                unresolved['incomplete_stability_evidence'] += 1
            task_out.write(canonical({'condition': condition, 'algorithm': algorithm, 'dataset_id': dataset,
                'minute': minute, 'stab_score': score, 'structure_bindings': bindings, **components}) + '\n')
            key = (condition, algorithm, minute)
            summary[key]['STAB'].append(score)
            for row in run_rows:
                row['stab_score'] = score
                run_out.write(canonical(row) + '\n')
                for axis, field in (('ID', 'id_quality'), ('OOD', 'ood_quality'), ('SYM', 'sym_score'),
                                    ('MIN', 'min_score'), ('EFF', 'cumulative_eff')):
                    summary[key][axis].append(row[field])
    curves = []
    for (condition, algorithm, minute), axes in sorted(summary.items()):
        row = {'condition': condition, 'algorithm': algorithm, 'minute': minute}
        for axis in ('ID', 'OOD', 'SYM', 'MIN', 'EFF', 'STAB'):
            expected = 50 if axis == 'STAB' else 150
            values = axes[axis]
            if len(values) != expected:
                raise ValueError(f'{axis}: 指标分母不一致')
            available = [v for v in values if v is not None]
            row[axis] = 100 * sum(available) / expected if len(available) == expected else None
            row[f'{axis}_available'] = len(available)
            row[f'{axis}_expected'] = expected
        curves.append(row)
    for name, data in (('algorithm_minute_metrics.csv', curves),
                       ('algorithm_minute180_metrics.csv', [r for r in curves if r['minute'] == 180])):
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    terminal_unresolved = terminal_metrics(root, output, efficiencies)
    report = {'run_count': len(runs), 'run_minute_count': 1215000, 'task_minute_count': len(groups),
        'unresolved': dict(unresolved), 'terminal_unresolved': terminal_unresolved,
        'formal_ready': not unresolved and not terminal_unresolved,
        'files': {p.name: {'sha256': sha(p), 'path': str(p)} for p in
                  (run_path, task_path, output / 'algorithm_minute_metrics.csv', output / 'algorithm_six_axis.csv',
                   output / 'noise_supplement.csv', output / 'terminal_run_metrics.jsonl.gz')}}
    write_json(output / 'verification.json', report)
    return report


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output)))
