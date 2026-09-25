import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1] / '3、metrics'
AXES = ('ID', 'OOD', 'SYM', 'MIN', 'EFF', 'STAB')
GROUPS = (
    ('Structured\nSearch', (('imcts', 'iMCTS'), ('qlattice', 'QLattice'), ('jaxsr', 'JAXSR'))),
    ('Neural Policy\nSearch', (('udsr', 'uDSR'), ('dso', 'DSO'))),
    ('Evolutionary\nSearch', (('pysr', 'PySR'), ('pyoperon', 'PyOperon'), ('gplearn', 'gplearn'), ('symbolfit', 'SymbolFit'))),
    ('LLM Assisted\nSearch', (('drsr', 'DrSR'), ('llmsr', 'LLM-SR'))),
    ('Hybrid\nSearch', (('fepysr', 'FePySR'), ('ragsr', 'RAG-SR'))),
    ('Transformer\nMethods', (('e2esr', 'E2ESR'), ('tpsr', 'TPSR'))),
)
COLORS = {1: ('#f6dddd', '#882a2a'), 2: ('#e2edf8', '#244c78')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(condition, rows, output):
    ranks, highlights, missing, partial = {}, {}, {}, {}
    for axis in AXES:
        values = sorted({Decimal(row[axis]) for row in rows.values() if row[axis]}, reverse=True)
        highlights[axis] = {}
        for rank, value in enumerate(values[:2], 1):
            selected = [algorithm for algorithm, row in rows.items() if row[axis] and Decimal(row[axis]) == value]
            highlights[axis][rank] = selected
            ranks.update({(algorithm, axis): rank for algorithm in selected})
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11.7, 7.5), facecolor='white')
    ax = fig.add_axes((0.025, 0.025, 0.95, 0.95))
    ax.set(xlim=(0, 12), ylim=(-2.05, 19.1))
    ax.axis('off')
    artists = []

    def label(x, y, value, **kwargs):
        item = ax.text(x, y, value, va='center', **kwargs)
        artists.append(item)
        return item

    label(0, 18.55, f'Six-axis profiles of 15 symbolic regression methods on Core50 ({condition})', fontsize=13, fontweight='bold')
    label(0, 17.78, 'Methods are grouped by their primary search paradigm. Scores: 0-100; higher is better.', fontsize=10)
    ax.hlines(17.2, 0, 12, colors='#222222', linewidth=1.2)
    label(0.08, 16.15, 'Paradigm', fontweight='bold')
    label(2.35, 16.15, 'Algorithm', fontweight='bold')
    centers = [4.85 + index * 1.25 for index in range(6)]
    for index, title in enumerate(('Numerical', 'Symbolic', 'Search')):
        center = (centers[2 * index] + centers[2 * index + 1]) / 2
        label(center, 16.74, title, ha='center', fontweight='bold')
        ax.hlines(16.39, centers[2 * index] - 0.5, centers[2 * index + 1] + 0.5, colors='#555555', linewidth=0.6)
    for x, axis in zip(centers, AXES):
        label(x, 15.98, axis, ha='center', fontweight='bold')
    ax.hlines(15.52, 0, 12, colors='#333333', linewidth=0.8)
    index, export = 0, []
    for paradigm, algorithms in GROUPS:
        label(0.08, 15.02 - index - (len(algorithms) - 1) / 2, paradigm, fontsize=10.5, linespacing=1.4)
        for algorithm, name in algorithms:
            row, y = rows[algorithm], 15.02 - index
            label(2.35, y, name)
            result = {'Paradigm': paradigm.replace('\n', ' '), 'Algorithm': name}
            for x, axis in zip(centers, AXES):
                score = row[axis]
                text = f'{Decimal(score):.2f}' if score else 'NA'
                if score and int(row[f'{axis}_available']) < int(row[f'{axis}_expected']):
                    text += '*'
                    partial[f'{algorithm}:{axis}'] = {'available': int(row[f'{axis}_available']), 'expected': int(row[f'{axis}_expected'])}
                result[axis] = text
                if not score:
                    missing[f'{algorithm}:{axis}'] = {'available': int(row[f'{axis}_available']), 'expected': int(row[f'{axis}_expected'])}
                rank = ranks.get((algorithm, axis))
                if rank:
                    background, foreground = COLORS[rank]
                    ax.add_patch(Rectangle((x - 0.51, y - 0.34), 1.02, 0.68, facecolor=background, edgecolor='none'))
                    label(x, y, text, ha='center', color=foreground, fontweight='bold')
                else:
                    label(x, y, text, ha='center', color='#222222' if score else '#777777')
            export.append(result)
            index += 1
        ax.hlines(15.52 - index, 0, 12, colors='#777777', linewidth=0.45)
    ax.hlines(0.52, 0, 12, colors='#222222', linewidth=1.1)
    label(0, -0.05, 'Display scale: 0-100, higher is better.', fontsize=9.5)
    for x, rank, caption in ((6.1, 1, 'Best (1st)'), (8.35, 2, 'Second-best (2nd)')):
        background, foreground = COLORS[rank]
        ax.add_patch(Rectangle((x, -0.27), 0.27, 0.42, facecolor=background, edgecolor='none'))
        label(x + 0.38, -0.05, caption, fontsize=9.5, color=foreground)
    label(0, -0.85, 'NA: incomplete metric evidence; values are not imputed. Highlights rank available, unrounded point estimates.', fontsize=9.2)
    if partial:
        counts = ', '.join(f"{name.split(':')[0]} {value['available']}/{value['expected']}" for name, value in partial.items())
        label(0, -1.48, f'* EFF mean over available runs ({counts}).', fontsize=9.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for index, box in enumerate(boxes):
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1 or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            raise ValueError(f'文字超过图像范围: {artists[index].get_text()}')
        for other in boxes[:index]:
            if box.overlaps(other):
                raise ValueError(f'文字重叠: {artists[index].get_text()}')
    stem = output / f'six_axis_{condition}'
    fig.savefig(stem.with_suffix('.png'), dpi=240, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    with stem.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Paradigm', 'Algorithm', *AXES])
        writer.writeheader()
        writer.writerows(export)
    return {'algorithms': len(export), 'highlights': highlights, 'missing': missing, 'partial': partial,
            'text_bounds_and_overlap_check': 'passed'}


def main():
    output = ROOT / 'tables'
    output.mkdir(parents=True, exist_ok=True)
    verification = json.loads((ROOT / 'verification.json').read_text())
    data, hashes = {}, {}
    algorithms = {algorithm for _, group in GROUPS for algorithm, _ in group}
    for name in ('algorithm_six_axis.csv', 'noise_supplement.csv'):
        path = ROOT / name
        hashes[name] = sha(path)
        if hashes[name] != verification['files'][name]['sha256']:
            raise ValueError(f'汇总输入版本不符: {name}')
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                key = (row['condition'], row['algorithm'])
                if key in data:
                    raise ValueError(f'重复算法: {key}')
                for axis in AXES:
                    available, expected = int(row[f'{axis}_available']), int(row[f'{axis}_expected'])
                    allowed = available == expected or (axis == 'EFF' and available > 0 and row.get('EFF_aggregation') == 'mean_available_runs')
                    if bool(row[axis]) != allowed:
                        raise ValueError(f'缺失标记与覆盖不符: {key}, {axis}')
                    if row[axis] and not 0 <= Decimal(row[axis]) <= 100:
                        raise ValueError(f'指标范围异常: {key}, {axis}')
                data[key] = row
    reports = {}
    for condition in ('clean', 'noise001', 'noise005'):
        rows = {algorithm: row for (name, algorithm), row in data.items() if name == condition}
        if set(rows) != algorithms:
            raise ValueError(f'算法覆盖不足: {condition}')
        reports[condition] = render(condition, rows, output)
    report = {'source_sha256': hashes, 'display_scale': '0-100', 'decimals': 2,
              'ranking': 'full_precision_available_values', 'conditions': reports,
              'files': {path.name: sha(path) for path in sorted(output.iterdir()) if path.suffix in ('.png', '.pdf', '.csv')}}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'conditions': reports}, ensure_ascii=False))


if __name__ == '__main__':
    main()
