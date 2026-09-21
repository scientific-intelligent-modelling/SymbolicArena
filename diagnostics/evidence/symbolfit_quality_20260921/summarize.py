"""Summarize all attempts, including pre-fix failures and superseded diagnostics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FINAL = {'after_constant_column', 'final_zero_mean', 'after_shifted',
         'after_Nguyen-1', 'after_Korns-1', 'after_BPG3'}
rows = []
unresolved = []
for report_path in sorted((ROOT / 'runs').glob('*/report.json')):
    report = json.loads(report_path.read_text())
    case = report_path.parent.name
    worker = report['worker']
    arrays = np.load(report_path.parent / 'inputs.npz')
    row = {'run': case, 'selected_for_acceptance': case in FINAL,
           'status': worker.get('status'), 'validation_ok': report['validation']['ok'],
           'wall_seconds': report['execution']['wall_seconds'],
           'hard_timeout': report['execution']['hard_timeout'],
           'features': arrays['train_X'].shape[1], 'train_rows': len(arrays['train_y']),
           'id_rows': len(arrays['id_y']), 'ood_rows': len(arrays['ood_y'])}
    assert not report['execution']['hard_timeout']
    assert report['execution']['wall_seconds'] < 180
    for split in ('id', 'ood'):
        value = worker.get('metrics', {}).get(split, {}).get('native', {})
        for metric in ('r2', 'mse', 'nmse_mean_y_squared'):
            row[split + '_' + metric] = value.get(metric)
    if report['validation']['ok']:
        predictions = np.load(report_path.parent / 'predictions.npz')
        for split in ('train', 'id', 'ood'):
            actual = predictions[split + '_native']
            for kind in ('exported', 'restored', 'refit_reference'):
                np.testing.assert_allclose(actual, predictions[split + '_' + kind], rtol=1e-7, atol=1e-10)
    if report.get('data_sha256'):
        relative = report['case'].split('/sim-datasets-data/')[1]
        row['local_data_hashes_match'] = all(
            hashlib.sha256((REPO / 'sim-datasets-data' / relative / name).read_bytes()).hexdigest() == digest
            for name, digest in report['data_sha256'].items())
        assert row['local_data_hashes_match']
    else:
        row['local_data_hashes_match'] = None
    rows.append(row)
    progress = report_path.parent / 'minute_evidence.jsonl'
    if progress.exists():
        for line in progress.read_text().splitlines():
            minute = json.loads(line)
            unresolved.append({'run': case, 'logical_key': minute['logical_key'],
                               'minute': minute['minute'], 'axes': minute['unresolved_axes'],
                               'source_path': str(progress.relative_to(ROOT)),
                               'source_sha256': hashlib.sha256(progress.read_bytes()).hexdigest(),
                               'reason': minute.get('unresolved_reason', 'No formal six-axis judging run')})
assert len(rows) == 9
assert sum(row['selected_for_acceptance'] and row['validation_ok'] for row in rows) == 6
with (ROOT / 'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
summary = {'diagnostic_only': True, 'formal_ready': False, 'paid_api_calls': 0,
           'physical_runs': 9, 'selected_acceptance_scenarios': 6, 'real_datasets': 3,
           'hard_timeouts': 0, 'max_worker_seconds': max(row['wall_seconds'] for row in rows),
           'rows': rows, 'unresolved': unresolved,
           'ten_dataset_coverage': {'e2esr': 10, 'tpsr': 1, 'ragsr': 0, 'symbolfit': 3},
           'coverage_denominator': 10,
           'metric_contract': {'mse': 'mean((y-pred)^2), lower is better',
                               'nmse': 'mse/mean(y^2), lower is better',
                               'r2': '1-mse/var(y), higher is better'},
           'aggregation': 'No cross-dataset ranking; all attempts retained'}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: summary[key] for key in ('physical_runs', 'selected_acceptance_scenarios',
                                              'real_datasets', 'max_worker_seconds')}))
