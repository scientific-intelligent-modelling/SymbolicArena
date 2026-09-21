from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/tmp/symbolfit_debug_20260921')
PYTHON = '/home/zhangziwen/anaconda3/envs/sim_symbolfit/bin/python'
CASES = [
    ('before', 'zero_mean', None), ('before', 'constant_column', None),
    ('after', 'zero_mean', None), ('after', 'constant_column', None),
    ('after', 'shifted', None),
    ('after', 'Nguyen-1', '/home/zhangziwen/sim-datasets-data/nguyen/Nguyen-1'),
    ('after', 'Korns-1', '/home/zhangziwen/sim-datasets-data/korns/Korns-1'),
    ('after', 'BPG3', '/home/zhangziwen/sim-datasets-data/llm-srbench/bio_pop_growth/BPG3'),
]


def run(case):
    version, name, dataset = case
    pkg = ROOT / version
    output = ROOT / 'runs' / (version + '_' + name)
    command = [PYTHON, '-u', str(pkg / 'diagnostics/symbolfit_quality_probe.py'),
               '--output', str(output), '--seconds', '180', '--fit-seconds', '45']
    command += ['--dataset', dataset] if dataset else ['--case', name]
    env = dict(os.environ, PYTHONPATH=str(pkg), PYTHON_JULIAPKG_PROJECT='/home/zhangziwen/pyjuliapkg_symbolfit',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', JULIA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / 'runs' / (version + '_' + name + '.log')).open('w') as log:
        completed = subprocess.run(command, cwd=pkg, env=env, stdout=log, stderr=subprocess.STDOUT)
    report_path = output / 'report.json'
    record = {'version': version, 'case': name, 'returncode': completed.returncode, 'command': command}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        record.update(execution=report['execution'], validation=report['validation'],
                      status=report['worker'].get('status'), error=report['worker'].get('traceback'))
        record['r2'] = {k: v['native']['r2'] for k, v in report['worker'].get('metrics', {}).items()}
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    (ROOT / 'runs').mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, CASES))
    (ROOT / 'execution_summary.json').write_text(json.dumps(results, indent=2))
