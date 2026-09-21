"""Fixed sparse-column regression fixture; it is not a benchmark dataset."""

import json
from pathlib import Path

import numpy as np

root = Path('/tmp/e2esr_sparse_fixture_20260921')
root.mkdir(exist_ok=False)
rng = np.random.RandomState(520)
splits = {}
for split, count in [('train', 80), ('valid', 32), ('id_test', 64), ('ood_test', 64)]:
    varying = rng.uniform(10, 12, count) if split != 'ood_test' else rng.uniform(12, 14, count)
    x = np.column_stack([np.full(count, 50.0), varying])
    y = 3.25 * varying - 0.75
    np.savetxt(root / (split + '.csv'), np.column_stack([x, y]), delimiter=',',
               header='constant_feature,varying_feature,target', comments='')
    splits[split] = {'file': split + '.csv', 'samples': count}
(root / 'metadata.yaml').write_text(json.dumps({'dataset': {
    'name': 'diagnostic_constant_first_sparse_second', 'splits': splits,
    'features': [{'name': 'constant_feature'}, {'name': 'varying_feature'}],
    'target': {'name': 'target'}}}, indent=2))
