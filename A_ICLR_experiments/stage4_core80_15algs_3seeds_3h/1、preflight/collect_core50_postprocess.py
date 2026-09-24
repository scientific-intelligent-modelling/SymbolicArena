import json
from pathlib import Path
import shutil
import subprocess
import time

from run_core50_comparisons import sha, write_json


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.agent/work/GOAL-CORE50'
REMOTE = '/home/zhangziwen/sim-runtime/core50-opus-runtime'
DESTINATION = ROOT / 'A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/3、metrics'


def transfer(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
               f'iaaccn22:{source}/', str(destination) + '/']
    for attempt in range(6):
        result = subprocess.run(command, timeout=14400)
        if result.returncode == 0:
            return
        if attempt < 5:
            time.sleep(30)
    raise RuntimeError(f'后处理结果同步失败: {source}')


if __name__ == '__main__':
    failures = 0
    while True:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'iaaccn22',
                   f'cat {REMOTE}/core50_minutes/pipeline_status.json']
        result = subprocess.run(command, timeout=35, capture_output=True, text=True)
        if result.returncode != 0:
            failures += 1
            write_json(WORK / 'postprocess_delivery.json', {'phase': 'poll_failed', 'failures': failures,
                'error': result.stderr[-1000:], 'time': time.time()})
            if failures >= 6:
                raise RuntimeError('连续六次无法读取后处理进度')
            time.sleep(60)
            continue
        failures = 0
        status = json.loads(result.stdout)
        write_json(WORK / 'postprocess_delivery.json', {'phase': 'waiting_remote_results',
            'remote_status': status, 'time': time.time()})
        if status.get('complete'):
            break
        time.sleep(60)
    transfer(f'{REMOTE}/core50_minutes/results', DESTINATION)
    report = json.loads((DESTINATION / 'verification.json').read_text())
    if report['run_count'] != 6750 or report['run_minute_count'] != 1215000 or report['task_minute_count'] != 405000:
        raise ValueError('交付覆盖数量不一致')
    for name, record in report['files'].items():
        if sha(DESTINATION / name) != record['sha256']:
            raise ValueError(f'交付文件哈希不一致: {name}')
    mappings = {}
    for remote, local in (('core50_comparisons', 'terminal'), ('core50_minutes', 'minutes'),
                          ('core50_new15_opus_v5', 'simplification')):
        target = DESTINATION / 'opus' / local
        transfer(f'{REMOTE}/{remote}', target)
        mappings[f'{REMOTE}/{remote}'] = str(target)
    inputs = DESTINATION / 'inputs'
    shutil.copytree(WORK / 'source_freezes_full', inputs / 'source_freezes', dirs_exist_ok=True)
    shutil.copy2(WORK / 'sixaxis_input_audit_full.json', inputs / 'sixaxis_input_audit_full.json')
    shutil.copytree(WORK / 'source_trajectory_full', inputs / 'trajectories', dirs_exist_ok=True)
    mappings[str(WORK / 'source_freezes_full')] = str(inputs / 'source_freezes')
    mappings[str(WORK / 'source_trajectory_full')] = str(inputs / 'trajectories')
    audit = json.loads((WORK / 'sixaxis_input_audit_full.json').read_text())
    for dataset in audit['dataset_sources']:
        original = ROOT / dataset['dataset_rel']
        preserved = inputs / 'datasets' / dataset['dataset_id']
        preserved.mkdir(parents=True, exist_ok=True)
        for name, expected_sha in dataset['source_checksums'].items():
            source = original / name
            if sha(source) != expected_sha:
                raise ValueError(f'数据集输入版本变化: {source}')
            shutil.copy2(source, preserved / name)
            if sha(preserved / name) != expected_sha:
                raise ValueError(f'数据集输入复制校验失败: {source}')
        mappings[str(original)] = str(preserved)
    write_json(DESTINATION / 'storage_map.json', {'prefix_mappings': mappings,
        'source_result_root': str(ROOT / 'A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments')})
    write_json(WORK / 'postprocess_delivery.json', {'phase': 'delivered', 'time': time.time(),
        'destination': str(DESTINATION), 'formal_ready': report['formal_ready'],
        'verification_sha256': sha(DESTINATION / 'verification.json')})
