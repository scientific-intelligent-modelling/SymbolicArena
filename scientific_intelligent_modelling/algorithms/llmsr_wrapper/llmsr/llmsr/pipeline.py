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

import numpy as np

from llmsr import code_manipulation
from llmsr import config as config_lib
from llmsr import evaluator
from llmsr import buffer
from llmsr import sampler
from llmsr import profile


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

    # 可选：从输入数据中估计目标变量的方差，用于 NMSE 计算
    target_variance = None
    try:
        # 约定输入为形如 {'data': {'inputs': X, 'outputs': y}} 的字典
        if hasattr(inputs, 'values'):
            any_dataset = next(iter(inputs.values()))
            if isinstance(any_dataset, dict) and 'outputs' in any_dataset:
                y = np.asarray(any_dataset['outputs'])
                if y.size > 0:
                    target_variance = float(np.var(y))
    except Exception:
        target_variance = None

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        # samples_per_prompt 与命令行中的 samples_per_iteration 一致
        profiler = profile.Profiler(
            log_dir,
            samples_per_iteration=config.samples_per_prompt,
            target_variance=target_variance,
            persist_all_samples=bool(kwargs.get('persist_all_samples', False)),
            wandb_run=kwargs.get('wandb_run'),
        )

    seed = kwargs.get('seed', None)
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler, seed=seed)

    # Set global max sample nums.
    samplers = [
        sampler.Sampler(
            database,
            evaluators,
            config.samples_per_prompt,
            config=config,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            llm_client=kwargs.get('llm_client')
        )
        for _ in range(config.num_samplers)
    ]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler, seed=seed)
