import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from jaxsr import SymbolicRegressor

from scientific_intelligent_modelling.algorithms.jaxsr_wrapper.wrapper import JAXSRRegressor
from scientific_intelligent_modelling.benchmarks.runner import load_canonical_dataset, _evaluate_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    source = Path(report['result_path']).parent
    output = source / 'recovered_rtol3e6'
    assert not output.exists(), output
    state_path = Path(report['experiment_dir']) / '.jaxsr_current_best.json'
    snapshot = json.loads(state_path.read_text())
    state = snapshot['model_state']
    state_hash = JAXSRRegressor._hash_json(state)
    assert state_hash == snapshot['fidelity']['model_state_sha256']
    assert JAXSRRegressor._FIDELITY_RTOL == 3e-6
    assert 10740 <= report['seconds'] <= 10800
    history = sorted((source / 'progress').glob('minute_*.json'))
    records = [json.loads(path.read_text()) for path in history]
    assert [item['checkpoint_index'] for item in records] == list(range(1, 180))
    for item in records:
        assert item['candidate_fidelity']['model_state_sha256'] == state_hash
        assert item['elapsed_seconds'] <= 10800
    dataset = load_canonical_dataset(records[-1]['dataset_dir'])
    model = SymbolicRegressor._from_dict(state)
    wrapper = JAXSRRegressor(**snapshot['contract'])
    equation, fidelity = wrapper._prepare_model_export(model, dataset.train.X)
    assert equation and fidelity['status'] == 'verified', fidelity
    assert fidelity['probe_input_sha256'] == snapshot['fidelity']['probe_input_sha256']
    assert fidelity['model_state_sha256'] == state_hash
    wrapper.model = model
    wrapper._export_equation = equation
    wrapper._fidelity_evidence = fidelity
    restored = JAXSRRegressor.deserialize(wrapper.serialize())
    np.testing.assert_array_equal(model.predict(dataset.train.X), restored.predict(dataset.train.X))
    metrics = {name: _evaluate_split(restored, getattr(dataset, name))
               for name in ['train', 'valid', 'id_test', 'ood_test']}
    assert all(value is not None and value['nmse'] is not None for value in metrics.values())
    artifact = restored.export_canonical_symbolic_program()
    provenance = {'method': 'native_model_revalidation', 'rtol': 3e-6,
                  'source_state': str(state_path), 'source_state_sha256': sha(state_path),
                  'source_report': str(args.report), 'source_report_sha256': sha(args.report),
                  'model_state_sha256': state_hash, 'training_repeated': False,
                  'native_selection_preserved': True, 'original_execution_status': report['status'],
                  'original_execution_error': report['error']}
    for path, original in zip(history, records):
        payload = deepcopy(original)
        payload.update(status='ok', error=None, equation=equation, canonical_artifact=artifact,
                       canonical_artifact_error=None, candidate_fidelity=fidelity,
                       recovery={**provenance, 'source_minute': str(path), 'source_minute_sha256': sha(path)})
        payload.update(metrics)
        write(output / 'progress' / path.name, payload)
    terminal = deepcopy(payload)
    terminal.update(checkpoint_index=180, elapsed_minutes=180, elapsed_seconds=10800,
                    seconds=10800, carry_forward_from_minute=179,
                    carry_forward_reason='same_native_model_verified_in_all_budget_snapshots')
    write(output / 'progress/minute_0180.json', terminal)
    final = deepcopy(terminal)
    final.update(record_type='final', seconds=report['seconds'], termination_reason=report['termination_reason'])
    write(output / 'result.json', final)
    verified_snapshot = deepcopy(snapshot)
    verified_snapshot.update(equation=equation, fidelity=fidelity)
    write(output / '.jaxsr_current_best.json', verified_snapshot)
    write(output / 'serialized_model.json', json.loads(wrapper.serialize()))
    for split in ['train', 'valid', 'id_test', 'ood_test']:
        provenance[f'{split}_sha256'] = sha(dataset.dataset_dir / f'{split}.csv')
    write(output / 'recovery.json', provenance)
    shutil.copy2(state_path, output / 'source_native_state.json')
    shutil.copy2(args.report, output / 'source_report.json')
    print(json.dumps({'seed': final['seed'], 'condition': final['condition'], 'output': str(output),
                      'fidelity': fidelity['status'], 'max_abs_error': fidelity['max_abs_error'],
                      'minutes': 180, **metrics}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
