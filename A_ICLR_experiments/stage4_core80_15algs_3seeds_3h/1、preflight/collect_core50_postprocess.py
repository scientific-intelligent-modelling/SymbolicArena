import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
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
    command = ['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
               f'iaaccn22:{source}/', str(destination) + '/']
    for attempt in range(6):
        result = subprocess.run(command, timeout=14400)
        if result.returncode == 0:
            return
        if attempt < 5:
            time.sleep(30)
    raise RuntimeError(f'后处理结果同步失败: {source}')


def collect_terminal():
    transfer(f'{REMOTE}/core50_terminal_metrics', DESTINATION)
    report = json.loads((DESTINATION / 'verification.json').read_text())
    if report['scope'] != 'terminal_formulas_only' or report['run_count'] != 6750:
        raise ValueError('最终公式交付范围不符')
    for name, record in report['files'].items():
        if sha(DESTINATION / name) != record['sha256']:
            raise ValueError(f'交付文件哈希不一致: {name}')
    axes = {'ID': 'id_quality', 'OOD': 'ood_quality', 'SYM': 'sym_score', 'MIN': 'min_score',
            'EFF': 'eff_score', 'STAB': 'stab_score'}
    groups, identities, tasks = defaultdict(lambda: defaultdict(list)), set(), {}
    with gzip.open(DESTINATION / 'terminal_run_metrics.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            identity = (row['condition'], row['algorithm'], row['dataset_id'], row['seed'])
            if identity in identities:
                raise ValueError('交付运行重复')
            identities.add(identity)
            for axis, field in axes.items():
                value = row[field]
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f'指标范围错误: {identity}, {axis}')
                if axis == 'STAB':
                    task = identity[:3]
                    if task in tasks:
                        if tasks[task] != value:
                            raise ValueError('同一任务STAB不一致')
                        continue
                    tasks[task] = value
                groups[identity[:2]][axis].append(value)
    if len(identities) != 6750 or len(tasks) != 2250 or len(groups) != 45:
        raise ValueError('最终交付数量错误')
    for name in ('algorithm_six_axis.csv', 'noise_supplement.csv'):
        with (DESTINATION / name).open(newline='') as handle:
            for row in csv.DictReader(handle):
                group = groups[row['condition'], row['algorithm']]
                for axis, values in group.items():
                    expected = 50 if axis == 'STAB' else 150
                    available = [value for value in values if value is not None]
                    if len(values) != expected or int(row[f'{axis}_available']) != len(available):
                        raise ValueError('汇总分母或可用数量不符')
                    partial_eff = axis == 'EFF' and row.get('EFF_aggregation') == 'mean_available_runs' and bool(available)
                    if len(available) == expected or partial_eff:
                        denominator = len(available) if partial_eff else expected
                        if not math.isclose(float(row[axis]), 100 * sum(available) / denominator, abs_tol=1e-10):
                            raise ValueError('算法汇总复算不一致')
                    elif row[axis] != '':
                        raise ValueError('缺失指标未保留空值')
    write_json(WORK / 'postprocess_delivery.json', {'phase': 'metrics_verified', 'time': time.time(),
        'destination': str(DESTINATION), 'scope': 'terminal_formulas_only', 'run_count': len(identities)})
    sources = json.loads((DESTINATION / 'opus_sources.json').read_text())
    paths, local_paths = {}, {}
    for record in sources.values():
        for path_key, hash_key in (('result_path', 'result_sha256'), ('source_plan', 'source_plan_sha256')):
            source = Path(record[path_key])
            if source.is_relative_to(ROOT):
                if str(source) in local_paths and local_paths[str(source)] != record[hash_key]:
                    raise ValueError('本地Opus计划版本冲突')
                local_paths[str(source)] = record[hash_key]
                continue
            relative = str(source.relative_to(REMOTE))
            if relative in paths and paths[relative] != record[hash_key]:
                raise ValueError('Opus证据版本冲突')
            paths[relative] = record[hash_key]
    file_list = WORK / 'terminal_opus_files.txt'
    file_list.write_text('\n'.join(sorted(paths)) + '\n')
    evidence = DESTINATION / 'opus'
    evidence.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-az', '--files-from', str(file_list), '-e',
        'ssh -o BatchMode=yes -o ConnectTimeout=10', f'iaaccn22:{REMOTE}/', str(evidence) + '/']
    subprocess.run(command, check=True, timeout=14400)
    for relative, expected in paths.items():
        if sha(evidence / relative) != expected:
            raise ValueError(f'Opus证据哈希不一致: {relative}')
    mappings = {REMOTE: str(evidence)}
    inputs = DESTINATION / 'inputs'
    for path, expected in local_paths.items():
        source = Path(path)
        if sha(source) != expected:
            raise ValueError(f'本地Opus计划哈希不一致: {path}')
        target = inputs / 'opus_plans' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        mappings[path] = str(target)
    for name in ('source_freezes_full', 'source_trajectory_full', 'minute_evidence_full', 'terminal_numeric_full'):
        shutil.copytree(WORK / name, inputs / name, dirs_exist_ok=True)
        mappings[str(WORK / name)] = str(inputs / name)
    shutil.copy2(WORK / 'sixaxis_input_audit_full.json', inputs / 'sixaxis_input_audit_full.json')
    mappings[f'{REMOTE}/core50_minutes/inputs/minutes'] = str(inputs / 'minute_evidence_full')
    mappings[f'{REMOTE}/core50_minutes/inputs/terminal_numeric'] = str(inputs / 'terminal_numeric_full')
    manifest = json.loads((DESTINATION / 'input_manifest.json').read_text())
    for original, expected in manifest['manifests'].items():
        prefix = max((prefix for prefix in mappings if original.startswith(prefix + '/')), key=len)
        preserved = Path(mappings[prefix]) / Path(original).relative_to(prefix)
        if sha(preserved) != expected:
            raise ValueError(f'输入清单哈希不一致: {original}')
        description = json.loads(preserved.read_text())
        if 'numeric_sha256' in description:
            data = preserved.parent / 'run_minutes.jsonl.gz'
            expected_data = description['numeric_sha256']
        else:
            data = preserved.with_name(preserved.name.replace('.manifest.json', '.jsonl.gz'))
            expected_data = description['output_sha256']
        if sha(data) != expected_data:
            raise ValueError(f'数值输入哈希不一致: {data}')
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
    write_json(DESTINATION / 'storage_map.json', {'prefix_mappings': mappings})
    write_json(WORK / 'postprocess_delivery.json', {'phase': 'delivered', 'time': time.time(),
        'destination': str(DESTINATION), 'scope': 'terminal_formulas_only', 'run_count': len(identities),
        'opus_evidence_files': len(paths), 'formal_ready': report['formal_ready'],
        'verification_sha256': sha(DESTINATION / 'verification.json')})
    print(json.dumps({'delivered': str(DESTINATION), 'run_count': len(identities),
                      'unresolved': report['terminal_unresolved']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal-only', action='store_true')
    args = parser.parse_args()
    if args.terminal_only:
        collect_terminal()
        raise SystemExit(0)
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
        if status.get('phase') == 'stopped_by_user':
            write_json(WORK / 'postprocess_delivery.json', {'phase': 'awaiting_terminal_metrics',
                'scope': status.get('scope'), 'remote_status': status, 'time': time.time()})
            break
        write_json(WORK / 'postprocess_delivery.json', {'phase': 'waiting_remote_results',
            'remote_status': status, 'time': time.time()})
        if status.get('complete'):
            break
        time.sleep(60)
    if status.get('phase') == 'stopped_by_user':
        raise SystemExit('逐分钟采集已停止，最终公式汇总待完成。')
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
        'source_result_root': str(ROOT / 'A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2.2 core50 experiments')})
    write_json(WORK / 'postprocess_delivery.json', {'phase': 'delivered', 'time': time.time(),
        'destination': str(DESTINATION), 'formal_ready': report['formal_ready'],
        'verification_sha256': sha(DESTINATION / 'verification.json')})
