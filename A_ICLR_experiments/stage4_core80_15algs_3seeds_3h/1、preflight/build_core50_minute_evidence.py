import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import clean_task_builder as builder
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import efficiency_from_qualities
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import (
    PerformanceReplayCache, PerformanceReplayError, replay_payload_performance,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff import (
    EffPreparationContractError, NATIVE_OBJECTIVE_ADAPTERS, _finite_number,
    _normalize_checkpoint_drift, canonical_expression, reconstruct_native_incumbent_trajectory,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.trajectory_repairs import _parse_frozen_snapshot
from run_core50_comparisons import canonical, sha, write_json


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.agent/work/GOAL-CORE50'


def expression_key(dataset, variables, expression):
    return hashlib.sha256(canonical({'dataset_id': dataset, 'variables': variables,
        'expression': expression, 'semantics': 'canonical_replay.v1'}).encode()).hexdigest()


def process_shard(arguments):
    sys.setrecursionlimit(20000)
    source_path, output, expected_runs = arguments
    condition, host = source_path.parent.name, source_path.name.removesuffix('.jsonl.gz')
    destination = output / condition / host
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'manifest.json').exists():
        report = json.loads((destination / 'manifest.json').read_text())
        if report['source_sha256'] != sha(source_path):
            raise ValueError(f'逐分钟输入变化: {source_path}')
        if report.get('generator_version') == 5 and report.get('generator_sha256') == sha(Path(__file__)):
            return report
    numeric = destination / 'run_minutes.jsonl.gz'
    expressions = {}
    counts = Counter()
    with gzip.open(source_path, 'rt') as source, gzip.open(numeric.with_suffix('.pending'), 'wt') as target:
        for line in source:
            record = json.loads(line)
            identity = record['source']
            logical_key = f"{identity['algorithm']}::{identity['dataset_id']}::s{identity['seed']}::{condition}"
            snapshots = {int(s['minute']): s for s in record['snapshots']}
            raw = record['result']['raw_text']
            if hashlib.sha256(raw.encode()).hexdigest() != record['result']['sha256']:
                raise ValueError(f'{logical_key}: 最终结果哈希不一致')
            final = json.loads(raw)
            variables = final['feature_names']
            payloads = {m: _parse_frozen_snapshot(s, logical_key=logical_key, minute=m)
                        for m, s in snapshots.items() if s['status'] == 'ok'}
            endpoint_audit = []
            for minute, payload in payloads.items():
                elapsed = payload.get('elapsed_seconds')
                if (payload.get('record_type') == 'budget_end_internal_best'
                        and isinstance(elapsed, (int, float)) and math.isfinite(elapsed)
                        and math.ceil(elapsed / 60) == minute
                        and payload.get('elapsed_minutes') == minute
                        and payload.get('checkpoint_index') == math.floor(elapsed / 60)):
                    endpoint_audit.append({'minute': minute, 'original_index': payload['checkpoint_index'],
                        'elapsed_seconds': elapsed, 'source_sha256': snapshots[minute]['selected_sha256']})
                    payload['checkpoint_index'] = minute
            native_horizon = 180
            last = max(payloads, default=0)
            if (last and last < 180 and payloads[last].get('record_type') == 'budget_end_internal_best'
                    and isinstance(final.get('seconds'), (int, float)) and 0 < final['seconds'] <= last * 60
                    and final['seconds'] <= payloads[last].get('elapsed_seconds', -1)
                    and builder.select_formula_with_source(final)[0] == builder.select_formula_with_source(payloads[last])[0]):
                native_horizon = last
            issue = None
            try:
                payloads, _ = _normalize_checkpoint_drift(payloads, raw_snapshots=record['snapshots'], logical_key=logical_key)
                native = reconstruct_native_incumbent_trajectory(payloads, algorithm=identity['algorithm'], horizon=native_horizon)
            except EffPreparationContractError as exc:
                issue = str(exc)
                native = None
                lower, upper = 0, native_horizon
                while lower < upper:
                    midpoint = (lower + upper + 1) // 2
                    try:
                        prefix, _ = _normalize_checkpoint_drift({m: p for m, p in payloads.items() if m <= midpoint},
                            raw_snapshots=record['snapshots'], logical_key=logical_key)
                        candidate = reconstruct_native_incumbent_trajectory(prefix, algorithm=identity['algorithm'], horizon=midpoint)
                    except EffPreparationContractError:
                        upper = midpoint - 1
                    else:
                        lower, native = midpoint, candidate
            selections = {}
            if native is not None:
                for index, point in enumerate(native.points):
                    selections[index + 1] = (native.incumbent_source_minutes[index], native.objective_fields[index],
                                              native.objective_values[index], point.source)
            tool = identity['algorithm'].lower().replace('-', '')
            adapter = NATIVE_OBJECTIVE_ADAPTERS[tool]
            for minute, payload in payloads.items():
                if minute in selections or payload.get('algorithm_native_incumbent') is not True:
                    continue
                if payload.get('elapsed_minutes') != minute or not canonical_expression(payload):
                    continue
                index = payload.get('checkpoint_index')
                if not isinstance(index, int) or not 0 <= index <= minute:
                    continue
                for field, _ in adapter.objective_options:
                    value = _finite_number(payload.get(field))
                    if value is not None:
                        selections[minute] = (minute, field, value, f'audited_native_snapshot:{minute}')
                        break
            if native_horizon < 180 and native_horizon in selections:
                selected, field, value, _ = selections[native_horizon]
                for minute in range(native_horizon + 1, 181):
                    selections[minute] = (selected, field, value, f'finished_run_carry_forward:{native_horizon}')
            points = []
            replayed = {}
            cache = PerformanceReplayCache()
            for minute in range(1, 181):
                row = {'logical_key': logical_key, 'run_minute_key': f'{logical_key}::m{minute:04d}',
                    'condition': condition, 'algorithm': identity['algorithm'], 'dataset_id': identity['dataset_id'],
                    'dataset_index': f"g{int(final['task_global_index']):04d}", 'seed': int(identity['seed']),
                    'minute': minute, 'result_sha256': record['result']['sha256'],
                    'source_path': snapshots[minute].get('selected_path'),
                    'source_sha256': snapshots[minute].get('selected_sha256'),
                    'id_nmse': None, 'ood_nmse': None, 'id_quality': None, 'ood_quality': None,
                    'q': None, 'valid_output': None, 'expression_key': None, 'expression': None,
                    'sym_score': None, 'min_score': None, 'status': 'unresolved', 'reason': issue,
                    'relative_progress': None, 'cumulative_eff': None, 'evaluation_path': 'canonical_replay.v1'}
                if minute in selections:
                    selected, field, value, selection_source = selections[minute]
                    row['incumbent_source_minute'] = selected
                    row['objective_field'] = field
                    row['objective_value'] = value
                    row['selection_source'] = selection_source
                    row['native_trajectory_issue'] = issue
                    if minute > native_horizon:
                        row['selection_source'] = f'finished_run_carry_forward:{native_horizon}'
                        row['termination_evidence'] = {'result_sha256': record['result']['sha256'], 'seconds': final['seconds']}
                    row['endpoint_index_audit'] = endpoint_audit
                    if selected is None:
                        row.update(status='invalid', reason='no_budget_candidate', valid_output=False,
                            id_quality=0.0, ood_quality=0.0, q=0.0)
                    else:
                        chosen = snapshots[selected]
                        row.update(source_path=chosen['selected_path'], source_sha256=chosen['selected_sha256'])
                        if selected not in replayed:
                            payload = dict(payloads[selected])
                            for field in ('dataset_dir', 'expected_dataset_rel', 'feature_names', 'target_name'):
                                if field not in payload:
                                    payload[field] = final[field]
                            try:
                                replay = replay_payload_performance(payload, algorithm=identity['algorithm'],
                                    repo_root=ROOT, cache=cache, task_id=identity['task_id'], condition=condition,
                                    result_sha256=record['result']['sha256'])
                            except PerformanceReplayError as exc:
                                replayed[selected] = {'error': str(exc)}
                            else:
                                expression, _ = builder.select_formula_with_source({'canonical_artifact': replay['canonical_artifact']})
                                expression, mapping = builder.map_indexed_variables(expression, variables)
                                replayed[selected] = {**replay, 'expression': expression, 'variable_mapping': mapping}
                        replay = replayed[selected]
                        if replay.get('error') is not None:
                            row['reason'] = replay['error']
                        else:
                            row.update(id_quality=replay['id_quality'], ood_quality=replay['ood_quality'],
                                id_nmse=(replay['id_test'] or {}).get('nmse'), ood_nmse=(replay['ood_test'] or {}).get('nmse'),
                                valid_output=replay['valid_output'], expression=replay['expression'],
                                canonical_artifact_sha256=replay['canonical_artifact_sha256'],
                                variable_mapping=replay['variable_mapping'],
                                q=(replay['id_quality'] + replay['ood_quality']) / 2,
                                status='resolved' if replay['valid_output'] else 'invalid', reason=replay['invalid_reason'])
                            if row['status'] == 'resolved':
                                key = expression_key(identity['dataset_id'], variables, replay['expression'])
                                row['expression_key'] = key
                                expressions.setdefault(key, {'key': key, 'dataset_id': identity['dataset_id'],
                                    'dataset_index': row['dataset_index'], 'variables': variables,
                                    'expression': replay['expression'], 'source_path': row['source_path'],
                                    'source_sha256': row['source_sha256'], 'canonical_artifact_sha256': row['canonical_artifact_sha256']})
                points.append(row)
                counts[row['status']] += 1
            qualities = [point['q'] for point in points]
            if all(q is not None for q in qualities):
                score = efficiency_from_qualities(qualities)
                maximum = max(qualities)
                cumulative = 0.0
                for index, point in enumerate(points, 1):
                    relative = point['q'] / maximum if maximum else 0.0
                    cumulative += relative
                    point.update(relative_progress=relative, cumulative_eff=cumulative / index, q_star=maximum)
                if abs(points[-1]['cumulative_eff'] - score) > 1e-12:
                    raise ValueError('EFF逐分钟复算不一致')
                counts['complete_eff_runs'] += 1
            for point in points:
                target.write(canonical(point) + '\n')
            counts['runs'] += 1
    if counts['runs'] != expected_runs:
        raise ValueError(f'逐分钟运行覆盖不一致: {source_path}')
    numeric.with_suffix('.pending').replace(numeric)
    descriptors = destination / 'expressions.jsonl'
    descriptors.write_text(''.join(canonical(row) + '\n' for row in expressions.values()))
    report = {'source_path': str(source_path), 'source_sha256': sha(source_path), 'generator_version': 5,
        'generator_sha256': sha(Path(__file__)), 'counts': dict(counts),
        'numeric_path': str(numeric), 'numeric_sha256': sha(numeric), 'expressions_path': str(descriptors),
        'expressions_sha256': sha(descriptors), 'expression_count': len(expressions), 'formal_ready': False}
    write_json(destination / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=WORK / 'source_trajectory_full')
    parser.add_argument('--output', type=Path, default=WORK / 'minute_evidence_full')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    submitted = set()
    futures = {}
    completed = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(completed) < 45:
            progress = json.loads((args.source / 'progress.json').read_text())
            for key, item in progress['completed'].items():
                if key not in submitted and len(futures) < args.workers:
                    futures[pool.submit(process_shard, (Path(item['output_path']), args.output, item['tasks']))] = key
                    submitted.add(key)
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED) if futures else (set(), set())
            for future in done:
                completed[futures.pop(future)] = future.result()
                write_json(args.output / 'progress.json', {'time': time.time(), 'completed': completed, 'expected_shards': 45})
            if not futures:
                time.sleep(10)
    write_json(args.output / 'complete.json', {'completed_shards': 45,
        'run_count': sum(r['counts']['runs'] for r in completed.values()), 'formal_ready': False})


if __name__ == '__main__':
    main()
