# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Implementation of the LLMSR pipeline. """
from __future__ import annotations

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence
import os
import numpy as np

from drsr_420 import code_manipulation
from drsr_420 import config as config_lib
from drsr_420 import evaluator
from drsr_420 import buffer
from drsr_420 import sampler
from drsr_420 import profile
from drsr_420 import data_analyse_real

def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        **kwargs
):
    """ Launch a LLMSR experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # Profiler：直接基于 results_root（不再使用 logs 子目录）
    results_root = kwargs.get('results_root', None) or config.results_root
    llm_config = kwargs.get('llm_config', None)
    llm_client = kwargs.get('llm_client', None)
    target_variance = None
    try:
        if hasattr(inputs, 'values'):
            any_dataset = next(iter(inputs.values()))
            if isinstance(any_dataset, dict) and 'outputs' in any_dataset:
                y = np.asarray(any_dataset['outputs'])
                if y.size > 0:
                    target_variance = float(np.var(y))
    except Exception:
        target_variance = None
    # Profiler：记录样本与中间结果（包括 Top-K、历史最优与逐 iteration 进度）
    profiler = profile.Profiler(
        results_root,
        samples_per_iteration=config.samples_per_prompt,
        target_variance=target_variance,
        persist_all_samples=bool(kwargs.get('persist_all_samples', False)),
    ) if results_root else None

    seed = kwargs.get('seed', None)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,     # the data instances for the problem.
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    initial = template.get_function(function_to_evolve).body
    ini_score, error_msg ,res= evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)

    # 创建DataAnalyzer实例
    # DataAnalyzer 也写入统一结果目录（直接使用 results_root）
    # 将 LLM API 配置传递给 DataAnalyzer（通过模块级常量覆盖，最小侵入）
    if llm_config and not llm_client:
        try:
            import drsr_420.data_analyse_real as _dar
            if 'host' in llm_config: _dar.API_HOST = llm_config['host']
            if 'api_key' in llm_config: _dar.API_KEY = llm_config['api_key']
            if 'model' in llm_config: _dar.API_MODEL = llm_config['model']
            if 'max_tokens' in llm_config: _dar.MAX_TOKENS = llm_config['max_tokens']
        except Exception:
            pass
    analyzer = data_analyse_real.DataAnalyzer(timeout=600, base_dir=results_root, llm_client=llm_client, seed=seed)  # 可以自定义参数

    # 分析指定的CSV文件
    result = analyzer.analyze(
        inputs,
        # max_rows=1000,  # 可选：限制行数
        verbose=True    # 可选：显示详细信息
    )

    # 打印分析结果
    print("\n===== 分析结果 =====")
    print(result)

    # Set global max sample nums.
    prompt_ctx = kwargs.get('prompt_ctx', None)
    samplers = [sampler.Sampler(database, evaluators, 
                                config.samples_per_prompt, 
                                max_sample_nums=max_sample_nums, 
                                llm_class=class_config.llm_class,
                                config = config,
                                prompt_ctx=prompt_ctx,
                                llm_client=llm_client,
                                llm_api=None) 
                                for _ in range(config.num_samplers)]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler)
