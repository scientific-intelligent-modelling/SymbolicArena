import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[1]
OUTPUT = STAGE / 'experiment_config.json'
REFERENCE = ROOT / 'AAAI_experiments/stage6_multiCore60-70-80/smoke_preflight_20260918/formal_clean_params'
SELECTION = STAGE / '1、build_core30_80/core30_core80_all_sensitivity_package/core_nested_all_sensitivity/results/core80.csv'
TOOLS = ['drsr', 'dso', 'e2esr', 'fepysr', 'gplearn', 'iMCTS', 'jaxsr', 'llmsr',
         'pyoperon', 'pysr', 'QLattice', 'ragsr', 'symbolfit', 'tpsr', 'udsr']
LLM = {'model': 'deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
       'temperature': 0.6, 'max_tokens': 1024, 'base_url': 'https://api.deepinfra.com/v1/openai'}
LLM_REFERENCE_SHA = '0b85d50af7ffd2ff91448e6b915d5d772f7a461e662faf15c072035dd5ec98e3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def reference(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def build(code_commit):
    if document_hash(LLM) != LLM_REFERENCE_SHA:
        raise ValueError('LLM 参数与已核验的 Stage6 非敏感配置不一致')
    toolbox_path = ROOT / 'scientific_intelligent_modelling/config/toolbox_config.json'
    toolbox = json.loads(toolbox_path.read_text())['tool_mapping']
    if set(toolbox) != set(TOOLS):
        raise ValueError('工具注册表与15算法集合不一致')
    algorithms = {}
    required_profiles = {tool.lower() + '__clean.json' for tool in TOOLS}
    if {path.name for path in REFERENCE.glob('*.json')} != required_profiles:
        raise ValueError('Stage6 正式参数文件集合不完整或存在额外算法')
    bindings = {'seed': 'run.seed', 'train_label_noise_sigma': 'condition.sigma',
                'train_label_noise_enabled': 'condition.enabled',
                'llm_config_path': 'runtime.llm_config_path',
                'llm_model_assignment': 'training_llm.parameters.model'}
    for tool in TOOLS:
        source = REFERENCE / (tool.lower() + '__clean.json')
        params = json.loads(source.read_text())
        if params['timeout_in_seconds'] != 10800 or params['progress_snapshot_interval_seconds'] != 60:
            raise ValueError(f'{tool} 使用了非正式预算或快照间隔')
        if params['train_label_noise_sigma'] != 0 or params['train_label_noise_enabled'] is not False:
            raise ValueError(f'{tool} 的源参数并非 clean')
        moved = {key: {'reference_value': params.pop(key), 'binding': binding}
                 for key, binding in bindings.items() if key in params}
        algorithms[tool] = {'env': toolbox[tool]['env'], 'params': params,
                            'reference': reference(source), 'runtime_bindings': moved}

    with SELECTION.open(encoding='utf-8-sig', newline='') as stream:
        selected = list(csv.DictReader(stream))
    if len(selected) != 80 or any(len({row[key] for row in selected}) != 80
                                  for key in ('dataset_id', 'dataset_rel', 'basename')):
        raise ValueError('Core80 数量、身份或输出目录名称不唯一')
    datasets = [{'dataset_id': row['dataset_id'], 'dataset_rel': row['dataset_rel'],
                 'directory_name': row['basename'], 'global_index': int(row['global_index'])}
                for row in selected]
    pipeline = ROOT / 'AAAI_experiments/stage5_metric_calculation_0831'
    contract_path = pipeline / 'pipeline/claude_contract.py'
    constants = {}
    for node in ast.parse(contract_path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('CONTRACT_MODEL', 'CONTRACT_EFFORT'):
                    constants[target.id] = ast.literal_eval(node.value)

    code_paths = [ROOT / 'scientific_intelligent_modelling' / name for name in (
        'benchmarks/runner.py', 'benchmarks/normalizers.py', 'benchmarks/result_artifacts.py',
        'benchmarks/artifact_schema.py', 'srkit/regressor.py', 'srkit/subprocess_runner.py', 'srkit/llm.py')]
    code_paths += [ROOT / 'scientific_intelligent_modelling/algorithms' / (tool + '_wrapper') / 'wrapper.py'
                   for tool in TOOLS]
    conditions = [{'name': 'clean', 'enabled': False, 'sigma': 0.0},
                  {'name': 'noise001', 'enabled': True, 'sigma': 0.01},
                  {'name': 'noise005', 'enabled': True, 'sigma': 0.05}]
    config = {
        'schema_version': 'core80.experiment_config.v1', 'status': 'frozen',
        'experiment': {
            'dataset_count': 80, 'algorithm_count': 15, 'seeds': [520, 521, 522],
            'conditions': conditions, 'run_count': 10800, 'run_count_per_condition': 3600,
            'training_budget_seconds': 10800, 'snapshot_interval_seconds': 60,
            'minutes_per_run': 180, 'expected_run_minutes': 1944000,
            'condition_order': ['clean', 'noise001', 'noise005'],
            'seed_binding': 'run_benchmark_task.seed; not an algorithm-specific constant',
            'selection': 'algorithm_native_training_objective_best_within_budget',
            'select_using_test_metrics': False,
            'noise': {'target': 'training_labels_only', 'feature_noise': False,
                      'formula': 'y + sigma * std(y, ddof=0) * N(0,1)',
                      'rng': 'numpy.random.default_rng',
                      'seed_parameter': 'train_label_noise_seed',
                      'seed_rule': "int.from_bytes(sha256(f'{dataset_rel}|seed={seed}|sigma={sigma:.12g}|train_label_noise'.encode('utf-8')).digest()[:8], 'big') % (2**32)",
                      'seed_identity': 'Frozen dataset_rel including sim-datasets-data/; never the machine absolute path',
                      'evaluation_labels': 'clean', 'validation_and_test_noise': False},
        },
        'dataset_selection': {'reference': reference(SELECTION), 'datasets': datasets},
        'algorithms': algorithms,
        'training_llm': {
            'algorithms': ['llmsr', 'drsr'], 'parameters': LLM,
            'reference_public_parameters_sha256': LLM_REFERENCE_SHA,
            'reference_verification': 'Stage6 llm_config_path files checked on iaaccn22; credentials excluded',
            'credentials': 'Runtime only; never embedded in the frozen configuration',
            'runtime_config_requirement': 'Resolved config must match parameters exactly; only credentials are supplied separately',
        },
        'output': {
            'root': (STAGE / '2、experiments').relative_to(ROOT).as_posix(),
            'layout': '{algorithm}/{dataset}/{condition}/{seed}',
            'dataset_field': 'dataset_selection.datasets[].directory_name',
            'reuse_historical_results': False,
        },
        'code_reference': {'git_commit': code_commit, 'files': [reference(path) for path in code_paths],
                           'toolbox': reference(toolbox_path)},
        'postprocessing': {
            'required': True, 'model': constants['CONTRACT_MODEL'], 'effort': constants['CONTRACT_EFFORT'],
            'contract': reference(contract_path),
            'metrics_definition': reference(pipeline / 'SymbolicArena_SixAxis_Revised.md'),
            'axes': ['ID', 'OOD', 'SYM', 'MIN', 'EFF', 'STAB'],
            'minute_level_required': True, 'llm_judgments': 'independent_stateless_single_turn',
        },
        'dispatch_requirements': {
            'verify_config_and_code_hashes': True, 'freeze_input_file_hashes_before_dispatch': True,
            'bind_seed_condition_and_machine_paths_explicitly': True,
            'preserve_algorithm_thread_parameters': True,
            'paid_llm_cost_and_retry_limits_require_separate_confirmation': True,
            'experiments_started': False,
        },
    }
    for tool, item in algorithms.items():
        for condition in conditions:
            effective = dict(item['params'], train_label_noise_sigma=condition['sigma'],
                             train_label_noise_enabled=condition['enabled'])
            if effective['timeout_in_seconds'] != 10800 or 'seed' in effective:
                raise ValueError(f'{tool} 的任务参数合并错误')
    config['content_sha256'] = document_hash(config)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        frozen = json.loads(OUTPUT.read_text())
        body = dict(frozen)
        recorded_hash = body.pop('content_sha256')
        if document_hash(body) != recorded_hash:
            raise ValueError('冻结配置内容哈希不匹配')
        if build(frozen['code_reference']['git_commit']) != frozen:
            raise ValueError('参数来源、数据集名单或实现文件已改变')
        config = frozen
    else:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        config = build(commit)
        with OUTPUT.open('x', encoding='utf-8') as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'file': str(OUTPUT), 'status': 'verified' if args.check else 'frozen',
                      'algorithms': len(config['algorithms']), 'datasets': 80, 'runs': 10800,
                      'sha256': config['content_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
