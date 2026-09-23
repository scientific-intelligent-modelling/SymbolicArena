import json
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.agent/work/EXP-001/oversample'
BATCH = 'stage4_core80_replicas_20260923'
TASKS = [(520, 'noise001'), (521, 'noise005'), (522, 'noise005')]
HOSTS = ['iaaccn48', 'iaaccn50', 'iaaccn51']
REMOTE = '/data1/zhangziwen/sim-runtime/code'


def run(args):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=120).stdout


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    records = []
    for replica, host in enumerate(HOSTS, 1):
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host]
        remote_work = f'{REMOTE}/.agent/work/EXP-001/oversample'
        print(run(ssh + [f'mkdir -p {remote_work}']), flush=True)
        inputs = []
        for seed, noise in TASKS:
            name = f'jaxsr_s{seed}_{noise}_g0595'
            inputs.append(ROOT / f'.agent/work/EXP-001/controller/queue/slices/jaxsr/seed{seed}/{noise}/{name}.csv')
        inputs.extend(ROOT / f'.agent/work/EXP-001/controller/params/jaxsr__{noise}.json' for noise in ['noise001', 'noise005'])
        run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', *map(str, inputs), f'{host}:{remote_work}/'])
        for seed, noise in TASKS:
            original = f'jaxsr_s{seed}_{noise}_g0595'
            task = f'{original}_r{replica}'
            session = f'core80_extra_{task}'
            command = [
                'tmux', 'new-session', '-d', '-s', session, '/bin/bash',
                f'{REMOTE}/queue/remote/run_queue_task.sh', BATCH, task, 'jaxsr', 'jaxsr',
                str(seed), '1', 'sim_jaxsr', f'.agent/work/EXP-001/oversample/{original}.csv',
                f'.agent/work/EXP-001/oversample/jaxsr__{noise}.json', host, REMOTE,
                '/data1/zhangziwen/sim-datasets-data', 'retry', session,
            ]
            run(ssh + [shlex.join(command)])
            record = {'original_task': original, 'replica': replica, 'host': host, 'session': session,
                      'output_root': f'{REMOTE}/experiments/{BATCH}/jaxsr/seed{seed}/tasks/{task}/{host}',
                      'seed': seed, 'noise': noise, 'budget_seconds': 10800}
            records.append(record)
            (WORK / 'jaxsr_launch.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
