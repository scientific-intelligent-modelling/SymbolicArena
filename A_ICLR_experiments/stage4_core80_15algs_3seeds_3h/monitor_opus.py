import fcntl
import json
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.agent/work/EXP-001/oversample/opus'


def sample():
    progress = json.loads((WORK / 'progress.json').read_text())
    selected = json.loads((WORK / 'selected.json').read_text())
    controllers = []
    processes = {}
    for process in psutil.process_iter(['pid', 'cmdline']):
        if not any(arg.endswith('/opus_remaining_replicas.py') for arg in (process.info['cmdline'] or [])):
            continue
        try:
            controllers.append(process.pid)
            for child in [process, *process.children(recursive=True)]:
                processes[child.pid] = {'pid': child.pid, 'name': child.name(),
                                        'rss_bytes': child.memory_info().rss}
        except psutil.NoSuchProcess:
            continue
    memory = psutil.virtual_memory()
    return {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'controllers': controllers,
            'status': 'running' if controllers else 'stopped', 'selected': len(selected),
            'remaining': progress['total'] - len(selected), 'workers': progress['workers'],
            'progress_age_seconds': round(time.time() - progress['time']),
            'available_memory_bytes': memory.available, 'swap_used_bytes': psutil.swap_memory().used,
            'processes': list(processes.values()),
            'rss_bytes': sum(item['rss_bytes'] for item in processes.values())}


def main():
    lock = (WORK / 'monitor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()
    peak_rss = 0
    minimum_available = psutil.virtual_memory().available
    while True:
        current = sample()
        peak_rss = max(peak_rss, current['rss_bytes'])
        minimum_available = min(minimum_available, current['available_memory_bytes'])
        if time.monotonic() >= deadline:
            current.update(interval_seconds=60, peak_rss_bytes=peak_rss,
                           minimum_available_memory_bytes=minimum_available)
            line = json.dumps(current, ensure_ascii=False)
            with (WORK / 'monitor.jsonl').open('a') as handle:
                handle.write(line + '\n')
            temporary = WORK / 'monitor.pending'
            temporary.write_text(line + '\n')
            temporary.replace(WORK / 'monitor_latest.json')
            print(line, flush=True)
            deadline += 60
            peak_rss = 0
            minimum_available = psutil.virtual_memory().available
        time.sleep(5)


if __name__ == '__main__':
    main()
