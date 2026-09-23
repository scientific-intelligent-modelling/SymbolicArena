from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.agent/work/EXP-001/followup'


def main():
    path = ROOT / '.agent/work/EXP-001/opus_postprocess/sync_results.py'
    spec = importlib.util.spec_from_file_location('core80_sync', path)
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    source_rows = sync.load_source_rows()
    queue = sync.load_queue_state()
    def collect(host):
        files = sync.result_file_list(host, source_rows, queue)
        files = [str(Path(item).parent / 'progress') + '/' for item in files
                 if not queue['tasks'][Path(item).parts[3]].get('recovered_result_path')]
        destination = sync.STAGING / host
        command = ['rsync', '-ar', '--timeout=60', '-e',
                   'ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2',
                   '--files-from=-', f'{host}:{sync.REMOTE_ROOTS[host]}/experiments/{sync.BATCH}/', str(destination)]
        result = subprocess.run(command, input='\n'.join(files)+'\n', text=True, capture_output=True, timeout=3600)
        copied = 0
        for item in files:
            task_id = Path(item).parts[3]
            task = queue['tasks'][task_id]
            identity = sync.TASK_RE.fullmatch(task_id)
            dataset = source_rows[int(identity.group('index'))]['dataset_name']
            algorithm = {'qlattice': 'QLattice', 'imcts': 'iMCTS'}.get(task['tool'], task['tool'])
            target = sync.LOCAL_EXPERIMENTS / algorithm / dataset / task['noise_tag'] / str(task['seed']) / 'progress'
            copied += sync.copy_json_tree(destination / item, target)
        return {'host': host, 'requested_runs': len(files), 'copied_snapshots': copied,
                'returncode': result.returncode, 'error_tail': result.stderr[-1200:]}
    reports = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(collect, host) for host in sync.REMOTE_ROOTS]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            sync.atomic_write(WORK / 'progress_collection.json', reports)
            print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
