import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import clean_task_builder as builder
from run_core50_comparisons import canonical, sha, write_json


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.agent/work/GOAL-CORE50'
REMOTE = Path('/home/zhangziwen/sim-runtime/core50-opus-runtime')


class FrozenParameters(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values
        self.bindings = {}

    def visit_Name(self, node):
        if node.id in self.bindings:
            return ast.Constant(self.bindings[node.id])
        return node

    def visit_Subscript(self, node):
        if not isinstance(node.value, ast.Name) or node.value.id != 'params':
            return self.generic_visit(node)
        if isinstance(node.slice, ast.Slice):
            parts = [ast.literal_eval(value) if value is not None else None
                     for value in (node.slice.lower, node.slice.upper, node.slice.step)]
            values = self.values[slice(*parts)]
            return ast.Tuple(elts=[ast.Constant(float(v)) for v in values], ctx=ast.Load())
        return ast.Constant(float(self.values[ast.literal_eval(self.visit(node.slice))]))

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name) and node.func.id == 'sum' and len(node.args) == 1
                and isinstance(node.args[0], (ast.ListComp, ast.GeneratorExp))):
            comprehension = node.args[0]
            if len(comprehension.generators) != 1:
                raise ValueError('参数求和只支持一个有界循环')
            loop = comprehension.generators[0]
            if (loop.ifs or loop.is_async or not isinstance(loop.target, ast.Name)
                    or not isinstance(loop.iter, ast.Call) or not isinstance(loop.iter.func, ast.Name)
                    or loop.iter.func.id != 'range' or loop.iter.keywords):
                raise ValueError('参数求和的循环定义不支持')
            indices = range(*(ast.literal_eval(arg) for arg in loop.iter.args))
            if len(indices) > 10000:
                raise ValueError('参数求和超出有界范围')
            result = ast.Constant(0)
            for index in indices:
                self.bindings[loop.target.id] = index
                result = ast.BinOp(left=result, op=ast.Add(), right=self.visit(copy.deepcopy(comprehension.elt)))
            self.bindings.pop(loop.target.id, None)
            return result
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ('sum', 'len') and len(node.args) == 1 and not node.keywords:
            values = ast.literal_eval(node.args[0])
            return ast.Constant(sum(values) if node.func.id == 'sum' else len(values))
        return node


def main():
    sys.setrecursionlimit(20000)
    probes, _ = builder.load_dataset_probes(ROOT / 'AAAI_experiments/stage5_metric_calculation_0831/reports/dataset_probes.jsonl')
    contract = builder._load_prompt_schema(ROOT, prompt_path=ROOT /
        'AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt')
    output = WORK / 'remaining_terminal/recovered_parameters'
    output.mkdir(parents=True, exist_ok=True)
    for seed in (520, 522):
        directory = ROOT / f'A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2.2 core50 experiments/noise005/llmsr/Nguyen-6/{seed}'
        result = json.loads((directory / 'result.json').read_text())
        experiment = directory / 'experiments' / Path(result['experiment_dir']).name
        progress = json.loads((experiment / 'progress.json').read_text())[-1]
        candidate_path = experiment / 'best_history' / f"best_sample_{progress['best_sample_order']}.json"
        candidate = json.loads(candidate_path.read_text())
        if candidate['function'] != result['equation'] or candidate['mse'] != progress['best_mse']:
            raise ValueError(f'seed{seed}: 原生最佳候选与最终公式不一致')
        witnesses = []
        for path in sorted((directory / 'progress').glob('minute_*.json')):
            snapshot = json.loads(path.read_text())
            elapsed = snapshot.get('elapsed_seconds')
            if (isinstance(elapsed, (int, float)) and elapsed <= 10800
                    and snapshot.get('source_sample_order') == candidate['sample_order']
                    and snapshot.get('internal_objective') == 'native_nmse'
                    and snapshot.get('source_internal_loss') == candidate['nmse']):
                witnesses.append({'path': str(path), 'sha256': sha(path), 'elapsed_seconds': elapsed})
        if not witnesses:
            raise ValueError(f'seed{seed}: 缺少预算内训练目标证据')
        body = builder.extract_expression_body(candidate['function'])
        tree = FrozenParameters(candidate['params']).visit(ast.parse(body, mode='eval'))
        expression = ast.unparse(ast.fix_missing_locations(tree))
        dataset = ROOT / result['expected_dataset_rel']
        for split in ('train.csv', 'id_test.csv', 'ood_test.csv'):
            frame = pd.read_csv(dataset / split)
            environment = {'np': np, 'sum': sum, 'len': len, 'range': range, 'params': np.asarray(candidate['params'])}
            environment.update({f'x{i}': frame[name].to_numpy() for i, name in enumerate(result['feature_names'])})
            original_values = eval(compile(ast.parse(body, mode='eval'), '<frozen-candidate>', 'eval'), {'__builtins__': {}, **environment})
            recovered_values = eval(compile(tree, '<recovered-candidate>', 'eval'), {'__builtins__': {}, **environment})
            np.testing.assert_allclose(original_values, recovered_values, rtol=1e-12, atol=1e-12)
        variables = result['feature_names']
        expression, variable_mapping = builder.map_indexed_variables(expression, variables)
        probe = probes[result['dataset']]
        functions, assumptions, evidence = builder._build_symbolic_request_evidence(expression=expression,
            variables=variables, probe=probe)
        source = {'result_path': str(directory / 'result.json'), 'result_raw_sha256': sha(directory / 'result.json'),
            'candidate_path': str(candidate_path), 'candidate_sha256': sha(candidate_path),
            'frozen_params': candidate['params'], 'budget_witness': witnesses[0], 'variable_mapping': variable_mapping,
            'original_equation': result['equation'], 'recovery': 'ast_constant_parameter_slice_evaluation'}
        request = {'dataset_id': result['dataset'], 'dataset_index': 'g0641', 'algorithm': 'llmsr',
            'algorithm_slug': 'llmsr', 'seed': seed, 'noise_tag': 'noise005', 'variables': variables,
            'expression': expression, 'original_expression': expression, 'allowed_functions': functions,
            'domain_assumptions': assumptions, 'probe_points': probe['points'], 'probe_source': probe['schema_version'],
            'probe_sample_sha256': probe['sample_sha256'], 'dataset_probe_evidence': probe,
            'deterministic_evidence': evidence, 'ast_source_evidence': source}
        request['evidence_hash'] = hashlib.sha256(canonical(request).encode()).hexdigest()
        task = builder._build_task_definition(logical_id=f'pred_simplify::llmsr::g0641::s{seed}::noise005',
            task_type='pred_simplify', priority=builder.PRED_PRIORITY, request=request,
            evidence_hash=request['evidence_hash'], contract=contract, condition='noise005')
        record = task.to_json_record()
        for field in ('prompt_path', 'schema_path'):
            record[field] = str(REMOTE / Path(record[field]).relative_to(ROOT))
        (output / f'recovered_params_s{seed}.jsonl').write_text(canonical(record) + '\n')
        write_json(output / f's{seed}_evidence.json', source)
        print({'seed': seed, 'evaluation_key': task.evaluation_key, 'budget_witnesses': len(witnesses)})


if __name__ == '__main__':
    main()
