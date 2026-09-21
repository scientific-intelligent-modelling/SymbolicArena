"""Run a fixed diagnostic suite, retaining failures and bounding each process tree."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


ROOT = Path('/home/family/workplace/scientific-intelligent-modelling')
OUT = Path('/tmp/e2esr_multidataset_20260921')
DATASETS = [
    'nguyen/Nguyen-1', 'nguyen/Nguyen-5', 'nguyen/Nguyen-7',
    'nguyen/Nguyen-12', 'keijzer/Keijzer-5', 'korns/Korns-1',
    'vladislavleva/Vladislavleva-4',
    'srbench1.0/feynman/feynman_I_6_2b',
    'srbench1.0/feynman/feynman_I_9_18',
    'llm-srbench/bio_pop_growth/BPG3',
]


def run_one(relative):
    name = Path(relative).name
    command = [sys.executable, '-u', 'diagnostics/three_algorithm_quality_probe.py',
               '--algorithm', 'e2esr', '--python',
               '/home/family/anaconda3/envs/sim_e2esr/bin/python',
               '--dataset', str(ROOT / 'sim-datasets-data' / relative),
               '--output', str(OUT / name), '--seconds', '180', '--fit-seconds', '120']
    started = time.monotonic()
    descendants = {}
    timed_out = False
    with (OUT / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ,
                                   PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
        parent = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                for child in parent.children(recursive=True):
                    descendants[(child.pid, child.create_time())] = child
                if time.monotonic() - started >= 180:
                    timed_out = True
                    break
                time.sleep(0.05)
        finally:
            for child in reversed(list(descendants.values())):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            psutil.wait_procs(list(descendants.values()), timeout=3)
    record = {'dataset': relative, 'command': command, 'returncode': process.returncode,
              'whole_probe_hard_timeout': timed_out, 'hard_limit_seconds': 180,
              'wall_seconds_including_cleanup': time.monotonic() - started}
    (OUT / (name + '.supervisor.json')).write_text(json.dumps(record, indent=2))
    report = OUT / name / 'report.json'
    if report.exists():
        value = json.loads(report.read_text())
        record['validation'] = value['validation']
        record['r2'] = {key: split.get('native', {}).get('r2')
                        for key, split in value['metrics'].items()}
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    manifest = {'datasets': DATASETS, 'seed': 520, 'fit_seconds': 120,
                'hard_seconds': 180, 'concurrency': 2, 'diagnostic_only': True,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                      text=True).strip(),
                'data_sha256': {}}
    for relative in DATASETS:
        directory = ROOT / 'sim-datasets-data' / relative
        manifest['data_sha256'][relative] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('metadata.yaml', 'train.csv', 'valid.csv', 'id_test.csv', 'ood_test.csv')}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_one, DATASETS))
    (OUT / 'execution_summary.json').write_text(json.dumps(results, indent=2))
