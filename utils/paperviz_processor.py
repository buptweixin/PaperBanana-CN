# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processing pipeline of PaperVizAgent
"""

import asyncio
import traceback
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional

import numpy as np
from tqdm.asyncio import tqdm

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent

from .config import ExpConfig
from .eval_toolkits import get_score_for_image_referenced


class PaperVizProcessor:
    """Main class for multimodal document processor"""

    def __init__(
        self,
        exp_config: ExpConfig,
        vanilla_agent: VanillaAgent,
        planner_agent: PlannerAgent,
        visualizer_agent: VisualizerAgent,
        stylist_agent: StylistAgent,
        critic_agent: CriticAgent,
        retriever_agent: RetrieverAgent,
        polish_agent: PolishAgent,
    ):
        self.exp_config = exp_config
        self.vanilla_agent = vanilla_agent
        self.planner_agent = planner_agent
        self.visualizer_agent = visualizer_agent
        self.stylist_agent = stylist_agent
        self.critic_agent = critic_agent
        self.retriever_agent = retriever_agent
        self.polish_agent = polish_agent

    async def _notify_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
        *,
        candidate_id: Any,
        stage: str,
        status: str,
        data: Optional[Dict[str, Any]] = None,
        message: str = "",
        completed: int | None = None,
        total: int | None = None,
    ):
        """向外部 UI 发出进度事件。"""
        if progress_callback is None:
            return

        event = {
            "candidate_id": candidate_id,
            "stage": stage,
            "status": status,
            "message": message,
            "data": data,
            "completed": completed,
            "total": total,
        }
        result = progress_callback(event)
        if asyncio.iscoroutine(result):
            await result

    async def _run_critic_iterations(
        self,
        data: Dict[str, Any],
        task_name: str,
        max_rounds: int = 3,
        source: str = "stylist",
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-round critic iteration (up to max_rounds).
        Returns the data with critic suggestions and updated eval_image_field.
        
        Args:
            data: Input data dictionary
            task_name: Name of the task (e.g., "diagram", "plot")
            max_rounds: Maximum number of critic iterations
            source: Source of the input for round 0 critique ("stylist" or "planner")
        """
        # Determine initial fallback image key based on source
        if source == "planner":
            current_best_image_key = f"target_{task_name}_desc0_base64_jpg"
        else: # default to stylist
            current_best_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            
        for round_idx in range(max_rounds):
            data["current_critic_round"] = round_idx
            await self._notify_progress(
                progress_callback,
                candidate_id=data.get("candidate_id", "N/A"),
                stage=f"critic_{round_idx}",
                status="running",
                data=data,
                message=f"正在执行第 {round_idx + 1} 轮评审",
            )
            data = await self.critic_agent.process(data, source=source)
            
            critic_suggestions_key = f"target_{task_name}_critic_suggestions{round_idx}"
            critic_suggestions = data.get(critic_suggestions_key, "")
            
            if critic_suggestions.strip() == "No changes needed.":
                print(f"[Critic Round {round_idx}] No changes needed. Stopping iteration.")
                await self._notify_progress(
                    progress_callback,
                    candidate_id=data.get("candidate_id", "N/A"),
                    stage=f"critic_{round_idx}",
                    status="success",
                    data=data,
                    message=f"第 {round_idx + 1} 轮评审认为无需继续修改",
                )
                break
            
            data = await self.visualizer_agent.process(data)
            
            # Check if visualization validation succeeded
            new_image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
            if new_image_key in data and data[new_image_key]:
                current_best_image_key = new_image_key
                print(f"[Critic Round {round_idx}] Completed iteration. Visualization SUCCESS.")
                await self._notify_progress(
                    progress_callback,
                    candidate_id=data.get("candidate_id", "N/A"),
                    stage=f"critic_{round_idx}",
                    status="success",
                    data=data,
                    message=f"第 {round_idx + 1} 轮评审完成并生成新图像",
                )
            else:
                print(f"[Critic Round {round_idx}] Visualization FAILED (No valid image). Rolling back to previous best: {current_best_image_key}")
                await self._notify_progress(
                    progress_callback,
                    candidate_id=data.get("candidate_id", "N/A"),
                    stage=f"critic_{round_idx}",
                    status="failed",
                    data=data,
                    message=f"第 {round_idx + 1} 轮评审后可视化失败，保留上一版本",
                )
                break
        
        data["eval_image_field"] = current_best_image_key
        return data

    async def process_single_query(
        self,
        data: Dict[str, Any],
        do_eval=True,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single query
        """
        candidate_id = data.get('candidate_id', 'N/A')
        exp_mode = self.exp_config.exp_mode
        task_name = self.exp_config.task_name.lower()
        retrieval_setting = self.exp_config.retrieval_setting
        print(f"\n[DEBUG] ── process_single_query 开始 ── candidate={candidate_id}")
        print(
            f"[DEBUG]   exp_mode={exp_mode}, task={task_name}, retrieval={retrieval_setting}, "
            f"text_provider={self.exp_config.text_provider}, image_provider={self.exp_config.image_provider}"
        )
        await self._notify_progress(
            progress_callback,
            candidate_id=candidate_id,
            stage="queued",
            status="running",
            data=data,
            message="候选方案已进入处理队列",
        )

        if exp_mode == "vanilla":
            print(f"[DEBUG] [{candidate_id}] 流水线: vanilla_agent")
            data = await self.vanilla_agent.process(data)
            data["eval_image_field"] = f"vanilla_{task_name}_base64_jpg"
            await self._notify_progress(
                progress_callback,
                candidate_id=candidate_id,
                stage="vanilla",
                status="success",
                data=data,
                message="基础流水线执行完成",
            )

        elif exp_mode == "dev_planner":
            print(f"[DEBUG] [{candidate_id}] 流水线: retriever → planner → visualizer")
            data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
            print(f"[DEBUG] [{candidate_id}] ✓ retriever 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="retriever", status="success", data=data, message="检索完成")
            data = await self.planner_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ planner 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="planner", status="success", data=data, message="规划描述已生成")
            data = await self.visualizer_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ visualizer 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="visualizer", status="success", data=data, message="首版图像已生成")
            data["eval_image_field"] = f"target_{task_name}_desc0_base64_jpg"

        elif exp_mode == "dev_planner_stylist":
            print(f"[DEBUG] [{candidate_id}] 流水线: retriever → planner → stylist → visualizer")
            data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
            print(f"[DEBUG] [{candidate_id}] ✓ retriever 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="retriever", status="success", data=data, message="检索完成")
            data = await self.planner_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ planner 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="planner", status="success", data=data, message="规划描述已生成")
            data = await self.stylist_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ stylist 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="stylist", status="success", data=data, message="风格优化完成")
            data = await self.visualizer_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ visualizer 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="visualizer", status="success", data=data, message="首版图像已生成")
            data["eval_image_field"] = f"target_{task_name}_stylist_desc0_base64_jpg"

        elif exp_mode in ["dev_planner_critic", "demo_planner_critic"]:
            max_rounds = data.get("max_critic_rounds", 3)
            print(f"[DEBUG] [{candidate_id}] 流水线: retriever → planner → visualizer → critic×{max_rounds}")
            data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
            print(f"[DEBUG] [{candidate_id}] ✓ retriever 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="retriever", status="success", data=data, message="检索完成")
            data = await self.planner_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ planner 完成, desc0 长度={len(data.get(f'target_{task_name}_desc0', ''))}")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="planner", status="success", data=data, message="规划描述已生成")
            data = await self.visualizer_agent.process(data)
            has_img = f"target_{task_name}_desc0_base64_jpg" in data and bool(data.get(f"target_{task_name}_desc0_base64_jpg"))
            print(f"[DEBUG] [{candidate_id}] ✓ visualizer 完成, 图像生成={'成功' if has_img else '失败'}")
            await self._notify_progress(
                progress_callback,
                candidate_id=candidate_id,
                stage="visualizer",
                status="success" if has_img else "failed",
                data=data,
                message="首版图像已生成" if has_img else "首版图像生成失败",
            )
            data = await self._run_critic_iterations(
                data,
                task_name,
                max_rounds=max_rounds,
                source="planner",
                progress_callback=progress_callback,
            )
            print(f"[DEBUG] [{candidate_id}] ✓ critic 迭代完成, eval_image_field={data.get('eval_image_field')}")
            if "demo" in exp_mode: do_eval = False

        elif exp_mode in ["dev_full", "demo_full"]:
            max_rounds = data.get("max_critic_rounds", self.exp_config.max_critic_rounds)
            print(f"[DEBUG] [{candidate_id}] 流水线: retriever → planner → stylist → visualizer → critic×{max_rounds}")
            data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
            print(f"[DEBUG] [{candidate_id}] ✓ retriever 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="retriever", status="success", data=data, message="检索完成")
            data = await self.planner_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ planner 完成, desc0 长度={len(data.get(f'target_{task_name}_desc0', ''))}")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="planner", status="success", data=data, message="规划描述已生成")
            data = await self.stylist_agent.process(data)
            print(f"[DEBUG] [{candidate_id}] ✓ stylist 完成")
            await self._notify_progress(progress_callback, candidate_id=candidate_id, stage="stylist", status="success", data=data, message="风格优化完成")
            data = await self.visualizer_agent.process(data)
            has_img = f"target_{task_name}_stylist_desc0_base64_jpg" in data and bool(data.get(f"target_{task_name}_stylist_desc0_base64_jpg"))
            print(f"[DEBUG] [{candidate_id}] ✓ visualizer 完成, 图像生成={'成功' if has_img else '失败'}")
            await self._notify_progress(
                progress_callback,
                candidate_id=candidate_id,
                stage="visualizer",
                status="success" if has_img else "failed",
                data=data,
                message="首版图像已生成" if has_img else "首版图像生成失败",
            )
            data = await self._run_critic_iterations(
                data,
                task_name,
                max_rounds=max_rounds,
                source="stylist",
                progress_callback=progress_callback,
            )
            print(f"[DEBUG] [{candidate_id}] ✓ critic 迭代完成, eval_image_field={data.get('eval_image_field')}")
            if "demo" in exp_mode: do_eval = False

        elif exp_mode == "dev_polish":
            print(f"[DEBUG] [{candidate_id}] 流水线: polish_agent")
            data = await self.polish_agent.process(data)
            data["eval_image_field"] = f"polished_{task_name}_base64_jpg"

        elif exp_mode == "dev_retriever":
            print(f"[DEBUG] [{candidate_id}] 流水线: retriever_agent")
            data = await self.retriever_agent.process(data)
            do_eval = False

        else:
            raise ValueError(f"Unknown experiment name: {exp_mode}")

        print(f"[DEBUG] [{candidate_id}] ── process_single_query 完成 ──")
        await self._notify_progress(
            progress_callback,
            candidate_id=candidate_id,
            stage="completed",
            status="success",
            data=data,
            message="候选方案处理完成",
        )

        if do_eval:
            data_with_eval = await self.evaluation_function(data, exp_config=self.exp_config)
            return data_with_eval
        else:
            return data

    async def process_queries_batch(
        self,
        data_list: List[Dict[str, Any]],
        max_concurrent: int = 50,
        do_eval: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Batch process queries with concurrency support
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(doc):
            async with semaphore:
                try:
                    result = await self.process_single_query(
                        doc,
                        do_eval=do_eval,
                        progress_callback=progress_callback,
                    )
                    result["processing_status"] = "success"
                    return result
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    failed_result = dict(doc)
                    failed_result["processing_status"] = "failed"
                    failed_result["processing_error"] = str(e)
                    failed_result["processing_error_type"] = type(e).__name__
                    failed_result["processing_traceback"] = traceback.format_exc()
                    print(
                        f"[ERROR] [PaperVizProcessor] candidate={doc.get('candidate_id', 'N/A')} "
                        f"处理失败: {type(e).__name__}: {e}"
                    )
                    await self._notify_progress(
                        progress_callback,
                        candidate_id=doc.get("candidate_id", "N/A"),
                        stage="failed",
                        status="failed",
                        data=failed_result,
                        message=f"{type(e).__name__}: {e}",
                    )
                    return failed_result

        # Create all tasks
        tasks = []
        for data in data_list:
            task = asyncio.create_task(process_with_semaphore(data))
            tasks.append(task)
        
        all_result_list = []
        eval_dims = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]

        with tqdm(total=len(tasks), desc="Processing concurrently") as pbar:
            # Iterate through completed tasks returned by as_completed
            for future in asyncio.as_completed(tasks):
                result_data = await future
                all_result_list.append(result_data)
                await self._notify_progress(
                    progress_callback,
                    candidate_id=result_data.get("candidate_id", "N/A"),
                    stage="batch_progress",
                    status=result_data.get("processing_status", "success"),
                    data=result_data,
                    completed=len(all_result_list),
                    total=len(tasks),
                    message="批量任务进度更新",
                )
                postfix_dict = {}

                for dim in eval_dims:
                    winner_key = f"{dim}_outcome"

                    if winner_key in result_data:
                        winners = [d.get(winner_key) for d in all_result_list]
                        total = len(winners)

                        if total > 0:
                            h_cnt = winners.count("Human")
                            m_cnt = winners.count("Model")
                            t_cnt = winners.count("Tie") + winners.count("Both are good") + winners.count("Both are bad")

                            h_rate = (h_cnt / total) * 100
                            m_rate = (m_cnt / total) * 100
                            t_rate = (t_cnt / total) * 100

                            display_key = dim[:5].capitalize()
                            postfix_dict[display_key] = f"{m_rate:.0f}/{t_rate:.0f}/{h_rate:.0f}"

                pbar.set_postfix(postfix_dict)
                pbar.update(1)
                yield result_data

    async def evaluation_function(
        self, data: Dict[str, Any], exp_config: ExpConfig
    ) -> Dict[str, Any]:
        """
        Evaluation function - uses referenced setting (GT shown first)
        """
        data = await get_score_for_image_referenced(
            data, task_name=exp_config.task_name, work_dir=exp_config.work_dir
        )
        return data
