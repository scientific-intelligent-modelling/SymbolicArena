import json
from pathlib import Path
import time

import numpy as np
from scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper import E2ESRRegressor
from scientific_intelligent_modelling.srkit.subprocess_runner import (
    handle_recover_from_timeout, handle_predict, handle_get_optimal_equation,
    handle_export_canonical_symbolic_program,
)

root = Path('/tmp/e2esr_mse_policy_20260921/crk0_final_interrupted_45s')
arrays = np.load(str(root/'inputs.npz'))
started = time.monotonic()

def no_weight_loading(self):
    raise AssertionError('Recovery must not load pretrained weights')

E2ESRRegressor._load_model = no_weight_loading
result = handle_recover_from_timeout(E2ESRRegressor, {
    'tool_name': 'e2esr', 'experiment_dir': str(root/'native'),
    'params': {'n_features':2, 'feature_names':['t','A'], 'target_name':'dA_dt'},
    'data': {'X': arrays['train_X'][:8].tolist(), 'y': arrays['train_y'][:8].tolist()}})
restored = E2ESRRegressor.deserialize(result['serialized_model'])
pred = restored.predict(arrays['id_X'])
command = {'tool_name':'e2esr', 'serialized_model':result['serialized_model'],
           'data':{'X':arrays['id_X'].tolist()}}
np.testing.assert_allclose(handle_predict(E2ESRRegressor, command)['predictions'], pred)
assert handle_get_optimal_equation(E2ESRRegressor, command)['equation'] == restored.get_optimal_equation()
assert handle_export_canonical_symbolic_program(E2ESRRegressor, command)['artifact']['artifact_valid']
y = arrays['id_y']
report = {'status':'ok', 'loaded_pretrained_weights':False,
          'dispatch_predict_equation_export_without_weights':True,
          'elapsed_seconds':time.monotonic()-started,
          'equation':restored.get_optimal_equation(),
          'id_r2':float(1-np.mean((pred-y)**2)/np.var(y)),
          'serialized_bytes':len(result['serialized_model'].encode()),
          'selection_policy':restored._SELECTION_POLICY}
np.save(str(root/'api_recovered_id_predictions.npy'),pred)
(root/'api_recovery.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
