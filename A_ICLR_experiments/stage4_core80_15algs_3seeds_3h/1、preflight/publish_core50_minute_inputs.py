import json
from pathlib import Path
import subprocess

from run_core50_comparisons import sha, write_json


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.agent/work/GOAL-CORE50'
REMOTE = '/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_minutes'


if __name__ == '__main__':
    source = WORK / 'minute_evidence_full'
    manifests = sorted(source.glob('*/*/manifest.json'))
    if len(manifests) != 45:
        raise ValueError('逐分钟分组数量不完整')
    count = 0
    hashes = {}
    for path in manifests:
        record = json.loads(path.read_text())
        if record['generator_version'] != 5:
            raise ValueError('逐分钟生成器版本不一致')
        for name, field in (('run_minutes.jsonl.gz', 'numeric_sha256'), ('expressions.jsonl', 'expressions_sha256')):
            if sha(path.parent / name) != record[field]:
                raise ValueError(f'待同步内容哈希不一致: {path.parent / name}')
        hashes[str(path.parent.relative_to(source))] = sha(path)
        count += record['counts']['runs']
    if count != 6750:
        raise ValueError('逐分钟运行数量不完整')
    receipt = WORK / 'minute_inputs_complete.json'
    write_json(receipt, {'run_count': count, 'generator_version': 5, 'shards': hashes})
    subprocess.run(['rsync', '-a', '--include=*/', '--include=*.json', '--include=*.jsonl',
        '--include=*.jsonl.gz', '--exclude=*', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
        str(source) + '/', f'iaaccn22:{REMOTE}/inputs/minutes/'], timeout=180, check=True)
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', str(receipt),
                    f'iaaccn22:{REMOTE}/inputs.complete.json'], timeout=30, check=True)
    print({'runs': count, 'shards': len(hashes), 'receipt_sha256': sha(receipt)})
