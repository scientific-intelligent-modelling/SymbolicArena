import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / '.agent/work/EXP-001/opus_postprocess'
WORK = ROOT / '.agent/work/EXP-001/followup'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    sys.setrecursionlimit(100000)
    WORK.mkdir(parents=True, exist_ok=True)
    build = module(OLD / 'build_opus_simplify_plan.py', 'core80_builder')
    rows = build.source_rows()
    probes, _ = build.builder.load_dataset_probes(build.PROBES)
    old_records = []
    with build.PLAN.open() as handle:
        for line in handle:
            old_records.append(json.loads(line))
    gt = [row for row in old_records if row['task_type'] == build.builder.GT_TASK_TYPE]
    source = ROOT / 'sim-datasets-data/srbench2025/firstprinciples/first_principles_planck/formula.py'
    tree = ast.parse(source.read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    exponent = f"({constants['H']!r}*nu/({constants['K_B']!r}*T))"
    expression = f"log(2.0*{constants['H']!r}/{constants['C']!r}**2)+3.0*log(nu)-where({exponent}<50.0,log(exp({exponent})-1),{exponent})"
    native = module(source, 'planck_native')
    temperatures = np.full(6, 300.0)
    exponents = np.array([0.01, 1.0, 10.0, 49.0, 50.0, 100.0])
    frequencies = exponents * constants['K_B'] * temperatures / constants['H']
    expected = np.log(2.0*constants['H']/constants['C']**2)+3*np.log(frequencies)-np.where(exponents<50, np.log(np.expm1(exponents)), exponents)
    np.testing.assert_allclose(native.target(frequencies, temperatures), expected, rtol=1e-12, atol=1e-12)
    gt_row = {'dataset_id': 'first_principles_planck', 'target': 'target',
              'normalized_expression_input': expression, 'ordered_variables': ['nu', 'T'],
              'return_ast_dump': ast.dump(ast.parse(expression, mode='eval')),
              'return_source': expression, 'selection_reason': 'verified_log_expm1_piecewise_translation',
              'source_checksums': {'formula_py_sha256': build.sha256_file(source)}}
    gt_row['evidence_sha256'] = build.sha256_text(build.canonical_json(gt_row))
    gt.append(build.make_gt_task(gt_row, probes['first_principles_planck']))
    assert len(gt) == 80
    old_paths = {row['request']['ast_source_evidence']['result_path']:
                 row['request']['ast_source_evidence']['result_raw_sha256']
                 for row in old_records if row['task_type'] == build.builder.PRED_TASK_TYPE}
    paths = [path for path in build.LOCAL_EXPERIMENTS.glob('*/*/*/*/result.json')
             if old_paths.get(str(path.resolve())) != build.sha256_file(path)]
    # 只生成新增或更正的输入，已有化简结果继续使用原来的绑定。
    tasks, unresolved, _ = build.prediction_tasks(rows, {}, probes, result_paths=paths)
    for name, items in [('base_plan.jsonl', gt + tasks), ('unresolved.jsonl', unresolved)]:
        with (WORK / name).open('x') as handle:
            for row in items:
                handle.write(build.canonical_json(row) + '\n')
    report = {'ground_truth': len(gt), 'additional_predictions': len(tasks),
              'changed_or_missing_inputs': len(paths), 'unresolved': len(unresolved),
              'source_builder_sha256': build.sha256_file(OLD / 'build_opus_simplify_plan.py'),
              'plan_sha256': build.sha256_file(WORK / 'base_plan.jsonl')}
    (WORK / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
