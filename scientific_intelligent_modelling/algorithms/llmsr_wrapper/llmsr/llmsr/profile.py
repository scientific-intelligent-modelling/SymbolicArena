# 实验采样/评估的简单记录器（去除 TensorBoard 依赖，仅保留 JSON 与控制台输出）

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import logging
import json
import llm
from llmsr import code_manipulation


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
            samples_per_iteration: int | None = None,
            target_variance: Optional[float] = None,
            persist_all_samples: bool = False,
            wandb_run=None,
    ):
        """
        参数说明：
            log_dir     : 日志目录（用于保存 samples/*.json）。
            pkl_dir     : 预留参数（未使用）。
            max_log_nums: 最多记录条数上限。
            samples_per_iteration: 每个 iteration 的样本数（用于进度记录）。
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        # 存储“历史最佳样本”的目录（每次出现新全局最优时保存一份）
        self._best_history_dir = os.path.join(log_dir, 'best_history')
        os.makedirs(self._best_history_dir, exist_ok=True)
        # 每个 iteration 样本数，用于把 sample_order 映射到 iteration
        self._samples_per_iteration = samples_per_iteration or 1
        self._persist_all_samples = bool(persist_all_samples)
        # 目标变量的方差，用于计算和排序 NMSE（越小越好）
        self._target_variance: Optional[float] = target_variance
        # 进度记录文件路径：保存每个 iteration 的最佳信息
        self._progress_json_path = os.path.join(log_dir, 'progress.json')
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        # 当前全局最优的 MSE / NMSE（用于 progress.json）
        self._cur_best_program_mse: Optional[float] = None
        self._cur_best_program_nmse: Optional[float] = None
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        # 维护 Top-K 方程（默认前 10 个）
        self._top_k = 10

        # 去除 TensorBoard 依赖，不再创建 SummaryWriter
        self._writer = None

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        # iteration -> 该 iteration 结束时的最佳统计
        self._iteration_progress: Dict[int, Dict[str, Any]] = {}
        self._wandb_run = wandb_run

    def _write_tensorboard(self):
        # 已移除 TensorBoard 功能：保持空实现以兼容调用点
        return

    def _compute_iteration_index(self, sample_order: int | None) -> int | None:
        """根据样本序号计算 iteration 编号（从 1 开始）。

        若样本序号缺失，则返回 None。
        """
        if sample_order is None:
            return None
        if self._samples_per_iteration <= 0:
            return sample_order
        return int((int(sample_order) - 1) // self._samples_per_iteration) + 1

    def _build_content(self, programs: code_manipulation.Function) -> Dict:
        """按照约定字段顺序构建 JSON 内容。

        约定顺序：
        1. iteration
        2. sample_order
        3. nmse
        4. mse
        5. function
        6. params
        """
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        iteration_idx = self._compute_iteration_index(sample_order)
        function_str = str(programs)
        score = programs.score
        # 原始 score 为 -MSE，这里转换为正的 MSE，并根据目标方差计算 NMSE
        if score is None:
            mse = None
            nmse = None
        else:
            mse = -float(score)
            if self._target_variance is not None and self._target_variance > 0:
                nmse = mse / float(self._target_variance)
            else:
                nmse = None
        params = programs.params
        # 先写 nmse 再写 mse，便于直接扫 nmse 排序
        content = {
            'iteration': iteration_idx,
            'sample_order': sample_order,
            'nmse': nmse,
            'mse': mse,
            'function': function_str,
            # 可选：参数，如存在则写入
            'params': params,
        }
        return content

    def _write_json(self, programs: code_manipulation.Function):
        """写出单个样本对应的 JSON 文件。"""
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        content = self._build_content(programs)
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _write_topk_json(self):
        """根据当前采样结果，维护前 K 个最优方程文件。

        命名规则：
        - top01_*.json, top02_*.json, ..., top10_*.json
        内容字段顺序同 _build_content。
        """
        # 先清理旧的 top 文件，确保 samples 目录中只保留当前这一轮的 Top-K
        try:
            for fname in os.listdir(self._json_dir):
                if fname.startswith('top') and fname.endswith('.json'):
                    try:
                        os.remove(os.path.join(self._json_dir, fname))
                    except Exception:
                        # 删除失败不影响后续写入新的 top 文件
                        continue
        except Exception:
            # 目录不存在或其他错误时直接跳过清理
            pass

        # 根据 NMSE（若可用）或 MSE 升序排序，忽略 score 为空的样本
        scored_items = []
        for order, func in self._all_sampled_functions.items():
            score = getattr(func, 'score', None)
            if score is None:
                continue
            mse = -float(score)
            if self._target_variance is not None and self._target_variance > 0:
                nmse = mse / float(self._target_variance)
            else:
                nmse = None
            scored_items.append((order, func, mse, nmse))

        if not scored_items:
            return

        def _sort_key(item):
            _, _, mse_val, nmse_val = item
            key_val = nmse_val if nmse_val is not None else mse_val
            return key_val if key_val is not None else float('inf')

        scored_items.sort(key=_sort_key)
        top_items = scored_items[: self._top_k]

        for idx, (order, func, _, _) in enumerate(top_items, start=1):
            prefix = f'top{idx:02d}_'
            content = self._build_content(func)
            filename = f'{prefix}samples_{order}.json'
            path = os.path.join(self._json_dir, filename)
            with open(path, 'w') as json_file:
                json.dump(content, json_file)

    def _save_best_history_sample(self, programs: code_manipulation.Function, sample_orders: int):
        """在出现新全局最优时，保存一份完整样本到 best_history 目录。"""
        if self._best_history_dir is None:
            return
        content = self._build_content(programs)
        filename = f'best_sample_{sample_orders}.json'
        path = os.path.join(self._best_history_dir, filename)
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _update_iteration_progress(self, sample_orders: int):
        """根据当前全局最优情况，更新每个 iteration 的进度记录并写入单一 JSON 文件。

        JSON 中每一项包含：
        - iteration: 当前迭代编号（从 1 开始）
        - best_mse: 截止该迭代为止的全局最小 MSE
        - best_nmse: 截止该迭代为止的全局最小 NMSE（若可计算）
        - best_sample_order: 产生当前全局最优得分的样本索引
        """
        if self._cur_best_program_sample_order is None:
            # 还没有任何成功样本，不记录
            return

        iteration_idx = self._compute_iteration_index(sample_orders)
        if iteration_idx is None:
            return

        prev_record = self._iteration_progress.get(iteration_idx)
        # 先写 best_nmse 再写 best_mse，保持与样本 JSON 一致的关注顺序
        record = {
            'iteration': iteration_idx,
            'best_nmse': self._cur_best_program_nmse,
            'best_mse': self._cur_best_program_mse,
            'best_sample_order': self._cur_best_program_sample_order,
        }
        # 追加大模型统计信息：总 tokens 与总耗时（秒，保留两位小数）
        try:
            tokens = llm.get_global_tokens()
        except Exception:
            tokens = {}
        try:
            total_time = llm.get_global_time()
        except Exception:
            total_time = None

        record['llm_tokens'] = tokens
        if total_time is not None:
            record['llm_time_seconds'] = round(float(total_time), 2)
        self._iteration_progress[iteration_idx] = record

        # 将 per-iteration 数据推送到 WandB（若启用），确保仅在数据发生变化时推送
        if self._wandb_run and record != prev_record:
            try:
                self._wandb_run.log(record, step=iteration_idx)
            except Exception:
                pass

        # 将所有 iteration 按编号排序后写入统一 JSON 文件
        history = [self._iteration_progress[k] for k in sorted(self._iteration_progress.keys())]
        try:
            with open(self._progress_json_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            # 不影响主流程
            pass

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            if self._persist_all_samples:
                self._write_json(programs)
            # 无论是否保留全量样本，都刷新 Top-K 方程文件。
            self._write_topk_json()
            # 更新按 iteration 统计的进度信息
            self._update_iteration_progress(sample_orders)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        # 计算当前样本的 MSE / NMSE（score = -MSE）
        mse = None
        nmse = None
        if score is not None:
            mse = -float(score)
            if self._target_variance is not None and self._target_variance > 0:
                nmse = mse / float(self._target_variance)
            else:
                nmse = None

        # log attributes of the function（打印 MSE 与 NMSE）
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'MSE         : {mse}')
        print(f'NMSE        : {nmse}')
        print(f'Sample time : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve（优先以 NMSE 最小为优，若不可用则回退到 MSE）
        if nmse is not None:
            if (self._cur_best_program_nmse is None) or (nmse < self._cur_best_program_nmse):
                self._cur_best_program_nmse = nmse
                self._cur_best_program_mse = mse
                self._cur_best_program_sample_order = sample_orders
                self._cur_best_program_str = function_str
                # 新的全局最优，额外保存一份到 best_history 目录
                self._save_best_history_sample(function, sample_orders)
        elif mse is not None:
            if (self._cur_best_program_mse is None) or (mse < self._cur_best_program_mse):
                self._cur_best_program_mse = mse
                self._cur_best_program_nmse = None
                self._cur_best_program_sample_order = sample_orders
                self._cur_best_program_str = function_str
                self._save_best_history_sample(function, sample_orders)

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
