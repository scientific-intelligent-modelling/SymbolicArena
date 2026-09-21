"""Audit fixed-suite reports without selecting models using held-out data."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from scientific_intelligent_modelling.benchmarks.normalizers import normalize_e2esr_artifact, normalize_tpsr_artifact
from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact


ROOT = Path('/home/family/workplace/scientific-intelligent-modelling')
spec = importlib.util.spec_from_file_location('probe', ROOT / 'diagnostics/three_algorithm_quality_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def replay(equation, arrays, n_features, algorithm='e2esr'):
    normalizer = normalize_e2esr_artifact if algorithm == 'e2esr' else normalize_tpsr_artifact
    artifact = normalizer(equation, expected_n_features=n_features)
    result = {}
    predictions = {}
    for split in ('train', 'id', 'ood'):
        try:
            predictions[split] = _predict_from_canonical_artifact(artifact, arrays[split + '_X'])
            result[split] = probe.metrics(arrays[split + '_y'], predictions[split])
        except Exception as exc:
            result[split] = {'finite': False, 'error': repr(exc)}
    return artifact, result, predictions


def audit_run(path, evidence):
    report = json.loads((path / 'report.json').read_text())
    arrays = np.load(path / 'inputs.npz')
    native = np.load(path / 'predictions.npz')
    history_path = path / 'native/.e2esr_native_candidates.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    eligible = [event for event in history if isinstance(event.get('training_mse'), (int, float))
                and np.isfinite(event['training_mse']) and event['training_mse'] >= 0]
    selected = report['selected_snapshot']
    n_features = report['params']['n_features']
    artifact, values, predictions = replay(selected['equation'], arrays, n_features)
    checks = {
        'worker_completed': report['worker']['status'] == 'ok' and report['execution']['returncode'] == 0,
        'selected_minimum_recorded_training_mse': selected['training_mse'] == min(e['training_mse'] for e in eligible),
        'recorded_mse_matches_native': bool(np.isclose(selected['training_mse'],
            report['metrics']['train']['native']['mse'], rtol=1e-6, atol=1e-20)),
        'current_canonical_artifact_valid': artifact['artifact_valid'],
    }
    for split in ('train', 'id', 'ood'):
        checks[split + '_current_replay_matches_native'] = bool(
            split in predictions and values[split]['finite'] and
            np.allclose(predictions[split], native[split + '_native'], rtol=1e-7, atol=1e-10))
    api = json.loads((path / 'api_recovery.json').read_text())
    checks['public_api_recovery_and_serialization_match'] = api['ok']
    audit = {'checks': checks, 'ok': all(checks.values()), 'metrics': values,
             'canonical_artifact': artifact, 'selected_equation': selected['equation'],
             'original_probe_validation': report['validation'], 'api_recovery': api,
             'evaluated_with_current_source_sha256': hashlib.sha256(
                 (ROOT / 'scientific_intelligent_modelling/benchmarks/normalizers.py').read_bytes()).hexdigest()}
    write_json(path / 'postfix_audit.json', audit)
    row = {'dataset': path.name, 'evidence': str(path.relative_to(evidence)),
           'seed': report['params']['seed'], 'features': n_features,
           'train_rows': len(arrays['train_y']), 'id_rows': len(arrays['id_y']),
           'ood_rows': len(arrays['ood_y']), 'worker_seconds': report['execution']['wall_seconds'],
           'completed_rounds': report['worker'].get('completed_rounds'),
           'refinement_type': report['worker'].get('refinement_type'),
           'original_probe_ok': report['validation']['ok'], 'postfix_audit_ok': audit['ok'],
           'train_mse': report['metrics']['train']['native']['mse']}
    for split in ('id', 'ood'):
        for key in ('r2', 'mse', 'nmse_mean_y_squared'):
            row[split + '_' + key] = report['metrics'][split]['native'].get(key)
    # Only full elapsed minutes are reconstructed; no future terminal formula is backfilled.
    records = []
    previous_sha = None
    for minute in range(1, int(report['worker']['elapsed_seconds'] // 60) + 1):
        visible = [e for e in eligible if e['source_timestamp_unix'] - e['fit_started_at_unix'] <= minute * 60]
        event = min(visible, key=lambda e: e['training_mse']) if visible else None
        record = {'condition': 'clean_as_stored', 'algorithm': 'e2esr',
                  'dataset_id': report['case'], 'seed': report['params']['seed'], 'minute': minute,
                  'attempt_id': str(path.relative_to(evidence)),
                  'logical_key': 'clean/e2esr/{}/520/{}/{}'.format(path.name, minute, path.parent.name),
                  'source_path': str(history_path.relative_to(evidence)),
                  'source_sha256': hashlib.sha256(history_path.read_bytes()).hexdigest(),
                  'selection_policy': 'e2esr_training_mse_v1',
                  'metric_contract': 'diagnostic_mse_nmse_mean_y_squared_r2_v1',
                  'judge_version': None, 'formal_ready': False,
                  'unresolved_axes': ['SYM', 'MIN', 'EFF', 'STAB'],
                  'unresolved_reason': 'No formal six-axis or paid judging run; single seed diagnostic only'}
        if event:
            current_artifact, metrics, _ = replay(event['equation'], arrays, n_features)
            record.update(equation=event['equation'], metrics=metrics,
                          candidate_sha256=event['candidate_sha256'],
                          carry_forward_from_candidate_sha256=previous_sha
                              if previous_sha == event['candidate_sha256'] else None,
                          complexity_evidence={'ast_node_count': current_artifact['ast_node_count'],
                                               'tree_depth': current_artifact['tree_depth']})
            previous_sha = event['candidate_sha256']
        else:
            record.update(equation=None, unresolved_expression=True)
        records.append(record)
    return row, records


if __name__ == '__main__':
    evidence = Path(sys.argv[1]).resolve()
    manifest = json.loads((evidence / 'initial/manifest.json').read_text())
    rows, minutes, audited = [], [], {}
    for relative in manifest['datasets']:
        name = Path(relative).name
        for stage in ('initial', 'fixed'):
            path = evidence / stage / name
            if path.exists():
                row, records = audit_run(path, evidence)
                audited[name] = row
                minutes.extend(records)
        row = audited[name]
        rows.append(row)
    tpsr = evidence / 'tpsr_korns1_smoke'
    if tpsr.exists():
        report = json.loads((tpsr / 'report.json').read_text())
        arrays = np.load(tpsr / 'inputs.npz')
        event_path = tpsr / 'emitted_candidates.jsonl'
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        for minute in range(1, int(report['worker']['elapsed_seconds'] // 60) + 1):
            visible = [e for e in events if e['probe_elapsed_seconds'] <= minute * 60]
            event = max(visible, key=lambda e: e['score']) if visible else None
            record = {'condition': 'clean_as_stored', 'algorithm': 'tpsr',
                      'dataset_id': report['case'], 'seed': 520, 'minute': minute,
                      'attempt_id': 'tpsr_korns1_smoke',
                      'logical_key': 'clean/tpsr/Korns-1/520/{}'.format(minute),
                      'source_path': str(event_path.relative_to(evidence)),
                      'source_sha256': hashlib.sha256(event_path.read_bytes()).hexdigest(),
                      'metric_contract': 'diagnostic_mse_nmse_mean_y_squared_r2_v1',
                      'selection_policy': 'native_tpsr_reward_max', 'judge_version': None,
                      'formal_ready': False, 'unresolved_axes': ['SYM', 'MIN', 'EFF', 'STAB'],
                      'unresolved_reason': 'Single-seed diagnostic; no formal six-axis or paid judging run'}
            if event:
                artifact, values, _ = replay(event['equation'], arrays, report['params']['n_features'], 'tpsr')
                record.update(equation=event['equation'], metrics=values,
                              candidate_sha256=hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest(),
                              complexity_evidence={'ast_node_count': artifact['ast_node_count'],
                                                   'tree_depth': artifact['tree_depth']})
            else:
                record.update(equation=None, unresolved_expression=True)
            minutes.append(record)
    with (evidence / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'diagnostic_only': True, 'formal_ready': False, 'paid_api_calls': 0,
               'dataset_count': len(rows), 'valid_after_fix': sum(r['postfix_audit_ok'] for r in rows),
               'id_r2_at_least_0_99': sum(r['id_r2'] is not None and r['id_r2'] >= 0.99 for r in rows),
               'ood_r2_at_least_0_99': sum(r['ood_r2'] is not None and r['ood_r2'] >= 0.99 for r in rows),
               'metrics': {'mse': 'mean((y-prediction)^2), lower is better',
                           'nmse_mean_y_squared': 'mse/mean(y^2), lower is better',
                           'r2': '1-mse/var(y), higher is better; constant targets unresolved'},
               'aggregation': 'Counts across all 10 preselected datasets, no exclusions or held-out model selection',
               'rows': rows}
    write_json(evidence / 'summary.json', summary)
    with (evidence / 'minute_evidence.jsonl').open('w') as handle:
        for record in minutes:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(evidence / 'unresolved.json', {
        'formal_ready': False, 'logical_keys': [r['logical_key'] for r in minutes],
        'axes': ['SYM', 'MIN', 'EFF', 'STAB'], 'reason': 'Short diagnostic only; not a formal six-axis release'})
    print(json.dumps(summary, indent=2))
