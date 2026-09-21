"""Verify saved E2ESR predictions through the lightweight public recovery API."""

import json
from pathlib import Path
import sys
import time

import numpy as np

from scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper import E2ESRRegressor


def forbidden_load(self):
    raise AssertionError('Recovery must not load pretrained weights')


E2ESRRegressor._load_model = forbidden_load
all_ok = True
for root_name in sys.argv[1:]:
    root = Path(root_name)
    started = time.monotonic()
    report = {'weight_loading_forbidden': True}
    try:
        job = json.loads((root / 'job.json').read_text())
        params = job['params']
        arrays = np.load(str(root / 'inputs.npz'))
        native = np.load(str(root / 'predictions.npz'))
        model = E2ESRRegressor(existing_exp_dir=str(root / 'native'),
                              n_features=params['n_features'],
                              feature_names=params['feature_names'],
                              target_name=params['target_name'])
        serialized = model.serialize()
        restored = E2ESRRegressor.deserialize(serialized)
        report['serialized_bytes'] = len(serialized.encode())
        report['equation'] = restored.get_optimal_equation()
        report['artifact'] = restored.export_canonical_symbolic_program()
        report['splits'] = {}
        for split in ('train', 'id', 'ood'):
            first = np.asarray(model.predict(arrays[split + '_X'])).reshape(-1)
            second = np.asarray(restored.predict(arrays[split + '_X'])).reshape(-1)
            expected = native[split + '_native']
            report['splits'][split] = {
                'finite': bool(np.isfinite(second).all()),
                'matches_native': bool(np.allclose(second, expected, rtol=1e-7, atol=1e-10)),
                'serialize_roundtrip_matches': bool(np.allclose(first, second, rtol=1e-7, atol=1e-10)),
                'max_abs_delta': float(np.max(np.abs(second - expected)))
                    if np.isfinite(second - expected).all() else None,
            }
        report['ok'] = all(s['finite'] and s['matches_native'] and s['serialize_roundtrip_matches']
                           for s in report['splits'].values())
    except Exception as exc:
        report.update(ok=False, error=repr(exc))
    report['elapsed_seconds'] = time.monotonic() - started
    (root / 'api_recovery.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'case': root.name, 'ok': report['ok'], 'seconds': report['elapsed_seconds']}), flush=True)
    all_ok = all_ok and report['ok']
sys.exit(0 if all_ok else 1)
