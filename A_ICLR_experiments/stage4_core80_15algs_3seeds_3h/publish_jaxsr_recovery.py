import hashlib
import json
from pathlib import Path
import shutil
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent / '2、experiments/jaxsr/strogatz_barmag2'
QUEUE = ROOT / '.agent/work/EXP-001/controller/queue/state/stage4_core80_all_20260922.state.json'
WORK = ROOT / '.agent/work/EXP-001/oversample'


def main():
    state = json.loads(QUEUE.read_text())
    recovered = {}
    for seed, noise, host in [(520, 'noise001', 'iaaccn29'), (521, 'noise005', 'iaaccn28'), (522, 'noise005', 'iaaccn52')]:
        source = EXPERIMENTS / noise / str(seed) / 'recovered_rtol3e6'
        result = json.loads((source / 'result.json').read_text())
        assert result['status'] == 'ok' and result['candidate_fidelity']['status'] == 'verified'
        assert result['seed'] == seed and result['condition'] == noise
        assert len(list((source / 'progress').glob('minute_*.json'))) == 180
        for split in ['id_test', 'ood_test']:
            assert result[split]['r2'] is not None
        recovered[f'jaxsr_s{seed}_{noise}_g0595'] = (source, host)
    unfinished = {task_id for task_id, task in state['tasks'].items() if task['state'] != 'done'}
    assert unfinished <= set(recovered), unfinished
    controller = psutil.Process(1978557)
    assert any('run_e1_candidate200_12alg_load_queue.py' in arg for arg in controller.cmdline())
    controller.terminate()
    controller.wait(timeout=20)
    state = json.loads(QUEUE.read_text())
    backup = WORK / 'queue_before_jaxsr_recovery.json'
    assert not backup.exists()
    shutil.copy2(QUEUE, backup)
    for task_id, (source, host) in recovered.items():
        target = source.parent
        if (target / 'result.json').exists():
            shutil.copy2(target / 'result.json', source / 'previous_local_result.json')
        if (target / 'progress').exists():
            shutil.copytree(target / 'progress', source / 'previous_local_progress')
        shutil.copy2(source / 'result.json', target / 'result.json')
        shutil.copytree(source / 'progress', target / 'progress', dirs_exist_ok=True)
        task = state['tasks'][task_id]
        task.update(state='done', assigned_host=host, status_counts={'ok': 1}, error=None,
                    ended_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                    recovered_result_path=str(source / 'result.json'),
                    recovered_result_sha256=hashlib.sha256((source / 'result.json').read_bytes()).hexdigest())
        provenance = {'task_id': task_id, 'source_host': host, 'source_path': str(source),
                      'result_sha256': task['recovered_result_sha256'], 'method': 'native_model_revalidation',
                      'script': str(Path(__file__).resolve()), 'selected': True}
        (target / 'recovery_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    temporary = QUEUE.with_suffix('.pending')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(QUEUE)
    print(json.dumps({'restored_tasks': list(recovered), 'done': sum(task['state'] == 'done' for task in state['tasks'].values())}))


if __name__ == '__main__':
    main()
