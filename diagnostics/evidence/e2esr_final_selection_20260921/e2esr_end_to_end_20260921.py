import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/family/workplace/scientific-intelligent-modelling')
OUT = Path('/tmp/e2esr_mse_policy_20260921/benchmark45_final')

if '--run' in sys.argv:
    sys.path.insert(0, str(ROOT))
    from scientific_intelligent_modelling.benchmarks.runner import run_benchmark_task
    result_path = run_benchmark_task(
        tool_name='e2esr', dataset_dir=ROOT/'sim-datasets-data/llm-srbench/chem_react/CRK0',
        output_root=OUT/'runs', seed=520,
        params_override={'timeout_in_seconds': 45, 'progress_snapshot_interval_seconds': 60,
                         'max_number_bags': 1, 'n_trees_to_refine': 4, 'torch_num_threads': 1})
    result = json.loads(result_path.read_text())
    print(json.dumps({'result_path': str(result_path), 'status': result['status'],
                      'seconds': result['seconds'], 'selection_policy': result.get('selection_policy'),
                      'training_mse': result.get('training_mse'),
                      'equation': result.get('equation'), 'id_test': result['id_test'],
                      'ood_test': result['ood_test'], 'recovered_from_timeout': result.get('recovered_from_timeout')}, indent=2))
else:
    import psutil
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    children = {}
    timed_out = False
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (OUT/'run.log').open('w') as handle:
        proc = subprocess.Popen([sys.executable, '-u', __file__, '--run'], cwd=str(ROOT),
                                env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        children[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - started >= 180:
                    timed_out = True
                    break
                time.sleep(.1)
        finally:
            for child in reversed(list(children.values())):
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    report = {'wall_seconds': time.monotonic()-started, 'hard_limit_seconds':180,
              'hard_timeout':timed_out, 'returncode':proc.returncode}
    (OUT/'execution.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
