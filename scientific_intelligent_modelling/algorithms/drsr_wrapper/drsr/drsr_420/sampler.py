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

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

import random
from drsr_420 import evaluator
from drsr_420 import buffer
from drsr_420 import config as config_lib
import requests
import json
import http.client
import os
import traceback
from drsr_420 import prompt_config as pc
Port = '5000'

# API配置
API_HOST = "api.bltcy.ai"
API_KEY = "sk-1zejrP7CKGPUXASwGpow3vOQ1Pjl5QzeU8xCjMrOEMSbqFQd"
API_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1024
class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    # self._samples_per_prompt = 4 每一次prompt都生成四个相互独立的回答



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            prompt_ctx: pc.PromptContext | None = None,
            llm_client: object | None = None,
            llm_api: dict | None = None,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        # 覆盖模块级 API 配置（最小改动注入）
        if llm_api:
            try:
                global API_HOST, API_KEY, API_MODEL, MAX_TOKENS
                API_HOST = llm_api.get('host', API_HOST)
                API_KEY = llm_api.get('api_key', API_KEY)
                API_MODEL = llm_api.get('model', API_MODEL)
                MAX_TOKENS = llm_api.get('max_tokens', MAX_TOKENS)
            except Exception:
                pass
        self._prompt_ctx = prompt_ctx
        self._llm_client = llm_client
        # 传递上下文给 LLM，用于渲染指令与头部
        try:
            self._llm = llm_class(samples_per_prompt, prompt_ctx=self._prompt_ctx, llm_client=self._llm_client)
        except TypeError:
            # 向后兼容：旧实现不接收 prompt_ctx
            try:
                self._llm = llm_class(samples_per_prompt, prompt_ctx=self._prompt_ctx)
            except TypeError:
                self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

# 旧示例（已过时）：
# python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt
# 新推荐（动态 CSV 模式）：
# python main.py --problem_name oscillator1 --data_csv ./data/oscillator1/train.csv --background "..."
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        start_time = time.time()
        wall_limit = getattr(self.config, 'wall_time_limit_seconds', None)
        while True:
            # 基于总时长的退出：达到上限后优雅停止
            if wall_limit is not None and (time.time() - start_time) >= wall_limit:
                print(f'到达实验时长上限：{wall_limit} 秒，停止采样。')
                break

            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()    # 从岛上拿一个可参考的方程框架 - 故可以独立反思

            island_id = prompt.island_id

            best_score = self._database._best_score_per_island[island_id]
            print(f"从岛屿 {island_id} 获取prompt，最佳分数: {best_score}")

            reset_time = time.time()

            print("调用大模型处理")

            # 01 版本
            # samples, sed_rep = self._llm.draw_samples(prompt.code,self.config) # 向大模型采样出一个方程框架 - 核心
            samples = self._llm.draw_samples(prompt.code,self.config) # 向大模型采样出一个方程框架 - 核心
            
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            print("获得了samples，在95行")
            print(samples)
            # This loop can be executed in parallel on remote evaluator machines.
            score_for_sample = []
            error_for_samlple = []
            quality_for_sample = []
            residual_data = None  # 用于存储每个样本的残差数据
            best_sample = None
            if_best = False
            id = 0
            temp_best_score = []
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                score, error_msg ,residual= chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                score_for_sample.append(score)
                error_for_samlple.append(error_msg)
                id += 1
                print(best_score)
                print(score)
                print('===================从chosen_evaluator.analyse中获得残差=====================\n')
                print(residual)
                if score is not None and score > best_score:
                # if score is not None :#先为了调试，都搞一遍，上面的才是需要的
                    temp_best_score.append(score)
                    #如果score比temp_best_score中的最大值大，就更新best
                    if score >= max(temp_best_score):
                        best_id = id
                        if_best = True
                        print("我在这里变成true了")
                        residual_data=residual
                        best_sample = sample
                        best_score_for_sample = score
            # print("一共有多少个sample？",i)
                    
            print("score_for_sample: ")
            print(score_for_sample)
            print("===========error_for_samlple:============================\n ")
            print(error_for_samlple)
            print("=========================residual_data: ================\n")
            print(residual_data)
            for each_score in score_for_sample:
                if each_score == None:
                    quality_for_sample.append('None')
                elif each_score > best_score:
                    quality_for_sample.append('Good')
                else:
                    quality_for_sample.append('Bad')
            print("quality_for_sample:")
            print('================================检查一下if_best的值====================\n')
            print(if_best)
            # 调用分析函数进行分析
            try:
                #先直接进入第三次
                print("\n===== 方程和分数分析开始 =====")
                analysis_result = self.analyze_equations_with_scores(samples, quality_for_sample, error_for_samlple, prompt)
                print("总的分析结果：---------")
                print(analysis_result)
                print("===== 方程和分数分析结束 =====\n")
                
                # 添加第三次对话：残差分析
                print("\n===== 残差分析开始 =====")
                print(residual_data)
                print(if_best)
                if residual_data is not None and if_best:
                    # 只对有效样本进行残差分析
                    if_best = False
                    residual_result = self.analyze_equations_with_residual(best_sample,residual_data)
                    print(f"样本残差分析结果: {residual_result}")
                    # 创建目录存放残差分析结果
                    json_residual_file = os.path.join(self.config.results_root or ".", "residual_analyze.json")
                    
                    # 加载现有的残差分析数据（如果文件存在）
                    residual_data_list = []
                    if os.path.exists(json_residual_file):
                        try:
                            with open(json_residual_file, "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                                if isinstance(existing_data, list):
                                    residual_data_list = existing_data
                        except json.JSONDecodeError:
                            print(f"现有的残差分析JSON文件格式有误，将创建新文件")
                        except Exception as e:
                            print(f"读取现有残差分析文件时出错: {e}")
                    
                    # 创建新的残差分析记录
                    current_sample_order = self._get_global_sample_nums() - len(samples) + best_id  # 获取当前样本的顺序号
                    
                    # 计算残差统计信息
                    res_values = residual_data[:, -1]  # 第三列是残差值

                    # 创建残差分析数据结构
                    residual_record = {
                        "sample_order": current_sample_order,
                        "island_id": prompt.island_id,
                        "equation": best_sample,
                        "analysis": residual_result,
                        "best_score": best_score_for_sample,
                    }
                    
                    # 添加到残差分析数据列表
                    residual_data_list.append(residual_record)
                    
                    # 保存更新后的残差分析数据
                    try:
                        with open(json_residual_file, "w", encoding="utf-8") as f:
                            json.dump(residual_data_list, f, ensure_ascii=False, indent=2)
                        print(f"成功更新残差分析JSON文件: {json_residual_file}")
                    except Exception as e:
                        print(f"保存残差分析JSON文件时出错: {e}")
                
                print("===== 残差分析结束 =====\n")

                # 创建目录存放分析结果
                # import os
                json_experience_file = os.path.join(self.config.results_root or ".", "experiences.json")
                

                # 加载现有的经验（如果文件存在）
                experiences_data = {"None": [], "Good": [], "Bad": []}
                if os.path.exists(json_experience_file):
                    try:
                        with open(json_experience_file, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                            # 确保键存在
                            for key in ["None", "Good", "Bad"]:
                                if key in existing_data:
                                    experiences_data[key] = existing_data[key]
                    except json.JSONDecodeError:
                        print(f"现有的 JSON 文件格式有误，将创建新文件")
                    except Exception as e:
                        print(f"读取现有经验文件时出错: {e}")

                # 添加新经验
                for i, (sample_text, quality, analysis, error_msg) in enumerate(zip(samples, quality_for_sample, analysis_result, error_for_samlple)):
                    # 获取当前样本的信息
                    current_sample_order = self._get_global_sample_nums() - len(samples) + i + 1  # 计算当前样本的顺序号
                    
                    # 确定分类
                    if quality == 'Good':
                        category = "Good"
                    elif quality == 'Bad':
                        category = "Bad"
                    else:  # 'None'
                        category = "None"
                    
                    # 创建经验数据结构
                    experience = {
                        "island_id": prompt.island_id,
                        "analysis": analysis,
                        "sample_order": current_sample_order,  # 添加样本顺序号
                        "sample_time": sample_time,
                        "equation": sample_text,
                        "score": score_for_sample[i],
                    }
                    
                    # 对于 None 类型，添加错误信息
                    if category == "None" and error_msg:
                        experience["error"] = error_msg
                    
                    # 添加到相应类别
                    experiences_data[category].append(experience)

                # 保存更新后的经验数据
                try:
                    with open(json_experience_file, "w", encoding="utf-8") as f:
                        json.dump(experiences_data, f, ensure_ascii=False, indent=2)
                    print(f"成功更新经验 JSON 文件: {json_experience_file}")
                except Exception as e:
                    print(f"保存 JSON 经验文件时出错: {e}")
            except Exception as e:
                print(f"执行分析时出错: {str(e)}")

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



    # 02 版本 good/bad/none的分析
    def analyze_equations_with_scores(self, samples, quality_for_sample, error_for_sample, prompt):
        """
        让大模型分析方程及其得分，提供思考过程分析和改进建议
        
        Args:
            samples: 生成的方程样本列表
            scores: 对应的分数列表
            prompt: 原始提示，用于提供上下文
            
        Returns:
            analysis_result: 模型对方程的分析结果
        """

        analysis_results = []

        i = 0

        for sample_each in samples:

            if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None:
                new__question = self._prompt_ctx.render_analysis_question(
                    quality_for_sample[i],
                    error_for_sample[i] if quality_for_sample[i] == 'None' else None,
                )
            else:
                if quality_for_sample[i] == 'Good':
                    new__question = pc.analysis_question_good.format(
                        dependent=pc.dependent_name_in_prompt,
                        problem=pc.problem_name_in_prompt,
                    )
                elif quality_for_sample[i] == 'Bad':
                    new__question = pc.analysis_question_bad.format(
                        dependent=pc.dependent_name_in_prompt,
                        problem=pc.problem_name_in_prompt,
                    )
                elif quality_for_sample[i] == 'None':
                    new__question = pc.analysis_question_none.format(
                        dependent=pc.dependent_name_in_prompt,
                        problem=pc.problem_name_in_prompt,
                        error=error_for_sample[i],
                        budget_sentence=(
                            "Treat this failure as a negative example rather than a requirement to satisfy. "
                            "If the error is about parameter length or indexing, do not solve it by asking for more parameters. "
                            "Instead, reduce parameter usage so the equation fits the evaluator's available parameter budget.\n"
                        ),
                    )
            i += 1

            analysis_prompt = pc.analysis_conversation_template.format(
                prompt=prompt.code if hasattr(prompt, "code") else prompt,
                sample=sample_each,
                question=new__question,
            )
            # 调用远程API分析结果（仅使用注入的 llm_client）
            try:
                resp = self._llm_client.chat([{"role": "user", "content": analysis_prompt}])
                analysis_result = resp.get('content', '')
                print(f"分析结果：{analysis_result}")
                analysis_results.append(analysis_result)
            except Exception as e:
                print(f"分析请求发生错误: {str(e)}")
                analysis_results.append(f"分析请求发生错误: {str(e)}")
        return analysis_results

    def analyze_equations_with_residual(self, sample, residual) -> str:
        """
    让大模型根据输入的残差分析方程，提供对数据的思考过程分析和改进建议
    
    Args:
        sample: 生成的方程样本
        residual: 输入的数据残差
        prompt: 原始提示，用于提供上下文
            
    Returns:
        analysis_result: 模型对方程的分析结果
        """
        print("========================进入了残差分析函数========================")
        # 直接使用传入的残差数据
        # 计算残差的统计信息
        res_values = residual[:, -1]  # 第三列是残差值
        mean_res = np.mean(res_values)
        max_res = np.max(np.abs(res_values))
        std_res = np.std(res_values)
        
        try:
            import json
            import os
            import random
            residual_file = os.path.join(self.config.results_root or ".", "residual_analyze.json")
            if os.path.exists(residual_file):
                with open(residual_file ,"r", encoding="utf-8") as f:
                    experiences = json.load(f)
                
                #提取最后一条信息
                if experiences:
                    last_experience = experiences[-1]
                    last_analysis = last_experience.get("analysis", "")
        except Exception as e:
            print(f"加载残差数据时出错: {str(e)}")
            print("Error details:")
            traceback.print_exc()


        # 构建分析提示
        if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None:
            res_analyze = self._prompt_ctx.render_residual_analysis_prompt(last_analysis, residual, sample)
        else:
            res_analyze = pc.residual_analysis_prompt.format(
                last_analysis=last_analysis,
                residual=residual,
                sample=sample,
            )
        
        
        print("========这是输入的残差提示词==========\n")
        print(res_analyze)
        # 调用远程API分析结果（仅使用注入的 llm_client）
        try:
            resp = self._llm_client.chat([{"role": "user", "content": res_analyze}])
            analysis_result = resp.get('content', '')
            print(f"残差分析结果：{analysis_result}")
            return analysis_result
        except Exception as e:
            print(f"残差分析请求发生错误: {str(e)}")
            return f"分析请求发生错误: {str(e)}"


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if (line[:3] == 'def'):
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # 统一处理：直接保留函数定义后的原始缩进与内容
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True,
                 prompt_ctx: pc.PromptContext | None = None,
                 llm_client: object | None = None) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        self._prompt_ctx = prompt_ctx
        self._llm_client = llm_client
        instruction_prompt = (self._prompt_ctx.render_instruction() if self._prompt_ctx else pc.instruction_prompt)
        self._batch_inference = batch_inference
        self._instruction_prompt = instruction_prompt
        self._trim = trim

        ####################################
        # 添加会话ID存储
        # self._conversation_ids = {}  # 用于存储每个样本的对话ID
        
        


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        # 记录统一结果目录供本地路径引用
        try:
            self._base_dir = config.results_root or "."
        except Exception:
            self._base_dir = "."

        # 统一走本地批量请求（已使用注入的 llm_client）
        return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:

###############################################
                    # 现在_do_request返回两组响应，只取第一组(方程实现)
                    print("运行了_draw_samples_local的_batch_inference分支")

                    first_responses = self._do_request(prompt)
                    # 01 版本
                    # first_responses, second_responses = self._do_request(prompt)
                    print("成功运行first_responses = self._do_request(prompt)")

                    # all_samples = first_responses


                    # 原代码
                    # response = self._do_request(prompt)
                    # for res in response:

##################################################### 
                    print(first_responses)                   
                    for res in first_responses:
                        all_samples.append(res)

                    # print("-------jjjjjjjjjjjjjjjj")
                else:
                    for _ in range(self._samples_per_prompt):
                        # response = self._do_request(prompt)
                        # all_samples.append(response)
                        # print("________")
                        first_responses, second_responses = self._do_request(prompt)
                        all_samples.append(first_responses)

                # print(all_samples)


                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                # print("处理过的all_samples")
                # print(all_samples)
                

                # 01版本
                # return all_samples, second_responses
            
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        for _ in range(self._samples_per_prompt):
            try:
                resp = self._llm_client.chat([{"role": "user", "content": prompt}])
                response = resp.get('content', '')
                if self._trim:
                    response = _extract_body(response, config)
                all_samples.append(response)
            except Exception:
                all_samples.append("")
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1

        # print('ttttttttttttttttttttttttttttttttttttttttt')
        # print(content)
        # print('ttttttttttttttttttttttttttttttttttttttttt')
        # 添加 idea_lib

        # 尝试加载经验数据
        try:
        # if True:


            # 获取当前的 sample_order
            # current_sample_order = self._get_global_sample_nums()
            # print(f"当前 sample_order: {current_sample_order}")
            
            # 计算经验文件中所有类别经验的总数
            current_sample_order = 0

            experience_file = os.path.join(getattr(self, "_base_dir", "."), "experiences.json")

            if os.path.exists(experience_file):
                with open(experience_file, "r", encoding="utf-8") as f:
                    experiences = json.load(f)
                
                # 统计所有类别的经验总数
                for category in ["None", "Good", "Bad"]:
                    if category in experiences:
                        current_sample_order += len(experiences[category])
                
                # print(f"经验库中共有 {current_sample_order} 条经验")


                
                # 准备存储筛选后的各类经验。
                # 规则：
                # - None：始终参与注入，最多 3 条；
                # - Good / Bad：各自独立以 0.5 概率参与注入，最多 2 条；
                # - 经验筛选范围与原逻辑保持一致。
                filtered_experiences = {"None": [], "Good": [], "Bad": []}
                optional_category_probability = 0.5
                category_max_samples = {"None": 3, "Good": 2, "Bad": 2}
                
                # 根据当前 sample_order 选择合适的经验
                for category in ["None", "Good", "Bad"]:
                    if category not in experiences or not experiences[category]:
                        continue

                    # None 类经验始终注入；Good / Bad 先按概率决定是否注入。
                    if category != "None" and random.random() >= optional_category_probability:
                        continue

                    # 筛选符合条件的经验
                    if current_sample_order <= 50:
                        # sample_order < 50 时，不限制经验的 sample_order
                        filtered_category = experiences[category]
                    else:
                        # sample_order > 50 时，只选择 sample_order 在当前值的 0.7~1 倍范围内的经验
                        min_order = current_sample_order * 0.7
                        max_order = current_sample_order
                        filtered_category = [
                            exp for exp in experiences[category] 
                            if "sample_order" in exp and min_order <= exp["sample_order"] <= max_order
                        ]
                    
                    # 按类别上限随机抽样
                    if filtered_category:
                        max_selected = category_max_samples[category]
                        selected = random.sample(filtered_category, min(max_selected, len(filtered_category)))
                        filtered_experiences[category] = selected
                
                # 合并所有类别的经验
                all_selected_experiences = []
                for category, exps in filtered_experiences.items():
                    for exp in exps:
                        experience_entry = {
                            "type": category,
                            "analysis": exp.get("analysis", ""),
                            "sample_order": exp.get("sample_order", "unknown"),
                        }
                        
                        # 对于 None 类别，添加错误信息（如果有）
                        if category == "None" and "error" in exp:
                            error_msg = exp["error"]
                            # 移除特定错误信息（如果需要）
                            if error_msg == "Execution Error: too many values to unpack (expected 5)":
                                error_msg = ""
                            
                            if error_msg:
                                experience_entry["error"] = error_msg
                        
                        all_selected_experiences.append(experience_entry)
                
                # 如果有经验可用，构建经验提示
                if all_selected_experiences:
                    experience_prompt = pc.ideas_block_title
                    
                    # 为每个经验分配编号
                    for i, exp in enumerate(all_selected_experiences, 1):
                        experience_prompt += pc.idea_item_prefix.format(index=i)
                        # experience_prompt += f"(sample_order: {exp['sample_order']})\n"
                        print("=================================sample_order: ==================================\n", exp['sample_order'])
                        
                        # 限制经验分析文本最多100个字符
                        analysis_text = exp["analysis"] if exp.get("analysis") else ""
                        if len(analysis_text) > 500:
                            analysis_text = analysis_text[:500] + "..."
                        experience_prompt += analysis_text
                        
                        experience_prompt += "\n---\n\n"
                    
                    # 将经验添加到原始内容中
                    content_with_lib = experience_prompt + "\n\n" + content
                    # print(f"已添加 {len(all_selected_experiences)} 条历史经验到提示中")

                    content = content_with_lib

            #有p的几率进入以下代码：
            p = 0.5  # 设置执行概率为50%，你可以根据需要调整这个值
            
            if random.random() < p and os.path.exists(experience_file):
                print("use residual_analyze: True")

                residual_file = os.path.join(getattr(self, "_base_dir", "."), "residual_analyze.json")
                if os.path.exists(residual_file):
                    with open(residual_file ,"r", encoding="utf-8") as f:
                        experiences = json.load(f)
                    
                    #提取最后一条信息
                    if experiences:
                        last_experience = experiences[-1]
                        last_analysis = last_experience.get("analysis", "")
                        last_sample_order = last_experience.get("sample_order", "unknown")
                        last_equation = last_experience.get("equation", "")
                        if last_equation is not None:
                        
                            # 构建提示
                            experience_prompt = (
                                self._prompt_ctx.render_residual_block_title()
                                if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None
                                else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
                            )
                            # experience_prompt += f"经验{last_sample_order}：\n"
                            # experience_prompt += f"(sample_order: {last_sample_order})\n"
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis
                            print("=================================sample_order: ==================================\n", last_sample_order)
                            # 将经验添加到原始内容中
                            content_with_residual = experience_prompt + "\n\n" + content

                            content = content_with_residual
                        
                            # print(f"残差分析库中共有 {len(experiences)} 条经验")
                        
                        else:
                            # 构建提示
                            experience_prompt = (
                                self._prompt_ctx.render_residual_block_title()
                                if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None
                                else pc.residual_block_title.format(problem=pc.problem_name_in_prompt)
                            )
                            # experience_prompt += f"经验{last_sample_order}：\n"
                            # experience_prompt += f"(sample_order: {last_sample_order})\n"
                            if isinstance(last_analysis, list):
                                last_analysis = last_analysis[0] if last_analysis else ""
                            # print(f"===========================last_analysis:=====================================\n {last_analysis}")
                            if len(last_analysis) > 2000:
                                last_analysis = last_analysis[:2000] + "..."
                            experience_prompt += last_analysis

                            # 将经验添加到原始内容中
                            content_with_residual = experience_prompt + "\n\n" + content

                            content = content_with_residual
                        
                            # print(f"残差分析库中共有 {len(experiences)} 条经验")


        except Exception as e:
            print(f"加载经验数据时出错: {str(e)}")
            print("Error details:")
            traceback.print_exc()  # 输出详细的错误堆栈信息
        
        # 重复提示以进行批量推理
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1


        if hasattr(self, "_prompt_ctx") and self._prompt_ctx is not None:
            head = self._prompt_ctx.render_head()
        else:
            head = pc.head_template.format(
                dependent=pc.dependent_name_in_prompt,
                problem=pc.problem_name_in_prompt,
                independent=pc.independent_name_in_prompt,
            )
        content = head +'\n'+ content
        print("========================最终输入给大模型的content========================\n")
        print(content)

        # 优先使用传入的 llm_client
        client = getattr(self, "_llm_client", None)
        if client is not None:
            try:
                responses = []
                for _ in range(repeat_prompt):
                    resp = client.chat([{"role": "user", "content": content}])
                    responses.append(resp.get('content', ''))
                return responses if self._batch_inference else responses[0]
            except Exception as e:
                print(f"API请求发生错误: {str(e)}")
                return [""] * repeat_prompt if self._batch_inference else ""
        
        # fallback 到 http.client
        try:
            responses = []
            for _ in range(repeat_prompt):
                resp = self._llm_client.chat([{"role": "user", "content": content}])
                responses.append(resp.get('content', ''))
            return responses if self._batch_inference else responses[0]
        except Exception as e:
            print(f"API请求发生错误: {str(e)}")
            return [""] * repeat_prompt if self._batch_inference else ""
