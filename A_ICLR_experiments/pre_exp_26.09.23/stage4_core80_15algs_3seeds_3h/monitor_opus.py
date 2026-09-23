import fcntl
import json
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.agent/work/EXP-001/oversample/opus'


def sample():
    followup = ROOT / '.agent/work/EXP-001/followup'
    progress_path = followup / 'progress.json' if (followup / 'progress.json').exists() else WORK / 'progress.json'
    progress = json.loads(progress_path.read_text())
    selected_path = followup / progress['phase'] / 'selected.json' if 'phase' in progress else WORK / 'selected.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    controllers = []
    preparers = []
    processes = {}
    for process in psutil.process_iter(['pid', 'cmdline']):
        arguments = process.info['cmdline'] or []
        is_preparer = any(arg.endswith('/prepare_core80_downstream.py') for arg in arguments)
        is_controller = any(arg.endswith(('/opus_remaining_replicas.py', '/run_core80_followup.py')) for arg in arguments)
        if not (is_preparer or is_controller):
            continue
        try:
            (preparers if is_preparer else controllers).append(process.pid)
            for child in [process, *process.children(recursive=True)]:
                processes[child.pid] = {'pid': child.pid, 'name': child.name(),
                                        'rss_bytes': child.memory_info().rss}
        except psutil.NoSuchProcess:
            continue
    memory = psutil.virtual_memory()
    finished = len(selected) == progress['total'] and progress.get('preparation_complete', True)
    status = 'complete' if finished else ('running' if controllers else 'stopped')
    return {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'controllers': controllers,
            'preparers': preparers,
            'preparation_status': 'complete' if progress.get('preparation_complete', True)
                                  else ('running' if preparers else 'stopped'),
            'status': status, 'selected': len(selected),
            'by_condition': progress.get('by_condition', {}),
            'phase': progress.get('phase', 'prediction_simplify'),
            'by_task_type': progress.get('by_task_type', {}),
            'transport_version': progress.get('transport_version'),
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
