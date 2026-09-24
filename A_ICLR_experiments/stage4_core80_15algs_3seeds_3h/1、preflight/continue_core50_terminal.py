import argparse
from collections import Counter
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from run_core50_comparisons import prepare, rows, write_json


def main():
    sys.setrecursionlimit(20000)
    runtime = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime')
    root = runtime / 'core50_comparisons'
    lock = (root / 'terminal_continuation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plans = [row for path in (root / 'plans').glob('missing_*.jsonl') for row in rows(path)]
    if len(plans) != 1432:
        raise ValueError(f'新增化简计划数量异常: {len(plans)}')
    while True:
        states = Counter()
        for entry in plans:
            db = root / 'execution' / f"pred_simplify__{entry['condition']}" / 'state.sqlite3'
            if not db.exists():
                states['unregistered'] += 1
                continue
            with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as connection:
                record = connection.execute('SELECT state FROM tasks WHERE evaluation_key=?',
                    (entry['evaluation_key'],)).fetchone()
            states[record[0] if record else 'unregistered'] += 1
        write_json(root / 'terminal_continuation.json', {'time': time.time(), 'phase': 'simplification',
            'states': dict(states), 'expected': len(plans), 'complete': False})
        if states['frozen'] + states['exhausted'] == len(plans):
            break
        time.sleep(30)
    with (root / 'controller.lock').open('a+') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX)
    args = argparse.Namespace(root=root, runtime=runtime, prior=runtime / 'core50_new15_opus_v5',
        prepare_workers=6, workers=300, evidence_timeout_seconds=600,
        channel_settings='routify=/home/zhangziwen/.config/core50-opus/routify.json')
    write_json(root / 'terminal_continuation.json', {'time': time.time(), 'phase': 'prepare_comparisons',
        'simplification_states': dict(states), 'complete': False})
    api = subprocess.Popen(['/bin/bash', str(root / 'bin/start_core50_comparisons.sh'), 'run'])
    for preparation_attempt in range(5):
        prepare(args)
        remaining = json.loads((root / 'unresolved.json').read_text())
        if not any(row['reason'].startswith('pair_') for row in remaining):
            break
    write_json(root / 'terminal_continuation.json', {'time': time.time(), 'phase': 'comparisons', 'complete': False})
    if api.wait() != 0:
        raise RuntimeError('比较执行器退出异常')
    expected = sum(1 for path in (root / 'plans').glob('*.jsonl') for _ in rows(path))
    progress = json.loads((root / 'progress.json').read_text())
    if progress['registered'] != expected:
        subprocess.run(['/bin/bash', str(root / 'bin/start_core50_comparisons.sh'), 'run'], check=True)
    write_json(root / 'terminal_continuation.json', {'time': time.time(), 'phase': 'terminal_comparisons_finished',
        'progress': json.loads((root / 'progress.json').read_text()), 'complete': True, 'formal_ready': False})


if __name__ == '__main__':
    main()
