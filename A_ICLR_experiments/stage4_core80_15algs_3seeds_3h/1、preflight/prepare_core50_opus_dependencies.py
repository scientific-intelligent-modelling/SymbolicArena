import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def read_index(path):
    return json.loads(path.read_text())


def main():
    sys.setrecursionlimit(20000)
    root = Path(__file__).resolve().parents[3]
    work = root / '.agent/work/GOAL-CORE50'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=work / 'dependency_pack')
    parser.add_argument('--remote-root', type=Path, default=Path('/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_comparisons'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_dir = args.output / 'frozen'
    frozen_dir.mkdir(exist_ok=True)
    bindings = json.loads((Path(__file__).parent / 'reuse_support/opus_bindings.json').read_text())
    by_key = {row['evaluation_key']: row for row in bindings}
    audit = json.loads((work / 'opus_cache_audit.json').read_text())
    records = []
    for cache in audit['cache_rows']:
        key = cache['historical_evaluation_key']
        binding = by_key[key]
        source_file = Path(binding['destination'])
        file_sha = digest(source_file)
        if file_sha != cache['cached_artifact_sha256']:
            raise ValueError(f'缓存文件哈希改变: {key}')
        artifact = json.loads(source_file.read_text())
        if artifact['evaluation_key'] != key or artifact['validation']['ok'] is not True:
            raise ValueError(f'缓存身份或验收记录异常: {key}')
        db = Path(binding['source_index'])
        if db.suffix == '.json':
            db = Path(read_index(db)[key]['source_db'])
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            task = dict(connection.execute('SELECT * FROM tasks WHERE evaluation_key=?', (key,)).fetchone())
            frozen = dict(connection.execute('SELECT * FROM frozen_results WHERE evaluation_key=?', (key,)).fetchone())
            attempt = dict(connection.execute('SELECT * FROM attempts WHERE attempt_id=?', (frozen['attempt_id'],)).fetchone())
        if frozen['result_sha256'] != file_sha or attempt['status'] != 'accepted' or task['state'] != 'frozen':
            raise ValueError(f'缓存状态与文件不一致: {key}')
        target = frozen_dir / f'{key}.json'
        if target.exists() and digest(target) != file_sha:
            raise ValueError(f'已有缓存副本哈希改变: {key}')
        if not target.exists():
            shutil.copy2(source_file, target)
        remote_file = args.remote_root / 'dependency_pack/frozen' / target.name
        condition = artifact['request'].get('noise_tag') or 'clean'
        plan_name = 'gt_merged_callable.jsonl' if task['task_type'] == 'gt_simplify' else f'{condition}_pred_merged_callable.jsonl'
        plan_path = work / 'opus_plan_merge_collected_v5' / plan_name
        records.append({'task': task, 'attempt': attempt,
                        'frozen': {**frozen, 'result_path': str(remote_file)},
                        'source_db': str(db), 'source_file': str(source_file),
                        'source_file_sha256': file_sha,
                        'source_plan': str(plan_path), 'source_plan_sha256': digest(plan_path),
                        'transport': artifact['metadata'].get('transport_version'),
                        'cache_audit_sha256': digest(work / 'opus_cache_audit.json')})
    manifest = args.output / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'core50.historical_dependencies.v1', 'records': records}, ensure_ascii=False) + '\n')
    print(json.dumps({'dependencies': len(records), 'manifest': str(manifest), 'sha256': digest(manifest)}))


if __name__ == '__main__':
    main()
