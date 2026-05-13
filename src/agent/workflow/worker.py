from __future__ import annotations
import asyncio
from typing import Optional

from .task_queue import task_queue, AsyncTask
from ..orchestrator import Orchestrator
from ..persistence import save_agent_task, save_thinking_trace, cleanup_old_records
from ..factory import create_orchestrator
import time


class AgentWorker:
    """Worker：从队列中取出任务，执行 Agent 编排"""

    def __init__(self, orchestrator_factory=None):
        self._running = False
        self._task = None
        self._orchestrator_factory = orchestrator_factory or (
            lambda **kw: create_orchestrator(**kw)
        )

    async def start(self):
        self._running = True
        self._last_cleanup_time = time.time()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self):
        while self._running:
            try:
                async_task = await task_queue.dequeue()
                if async_task is None:
                    # 定期清理过期数据（每小时）
                    now = time.time()
                    if now - self._last_cleanup_time > 3600:
                        self._last_cleanup_time = now
                        try:
                            counts = cleanup_old_records()
                            if any(counts.values()):
                                print(f"[Worker] Cleaned up old records: {counts}")
                        except Exception as e:
                            print(f"[Worker] Cleanup error: {e}")
                    continue

                await self._process_task(async_task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Worker] Error: {e}")

    async def _process_task(self, async_task: AsyncTask) -> None:
        """执行单个异步任务"""
        try:
            task_queue.update_task_progress(
                async_task.id, 5.0,
                message="正在初始化 Agent 系统...",
                step="initializing",
            )

            orchestrator = self._orchestrator_factory(
                async_task_id=async_task.id,
                task_queue=task_queue,
            )

            task_queue.update_task_progress(
                async_task.id, 10.0,
                message="正在分析需求并拆解任务...",
                step="decomposing",
            )

            result = await orchestrator.run(
                user_request=async_task.prompt,
                context={"async_task_id": async_task.id, "user_id": async_task.user_id},
            )

            task_queue.update_task_progress(
                async_task.id, 90.0,
                message="完成，正在整理结果...",
                step="finalizing",
            )

            if result.status.value in ("completed", "partial"):
                task_queue.complete_task(async_task.id, result.to_dict())
                # 持久化到数据库
                trace_steps = result.thinking_trace.get("steps", []) if result.thinking_trace else []
                save_agent_task(
                    task_id=async_task.id,
                    user_id=async_task.user_id,
                    prompt=async_task.prompt,
                    status=result.status,
                    result={"agent_result": result.to_dict()},
                    progress=100.0,
                    total_steps=len(trace_steps) or 1,
                    completed_steps=len(trace_steps),
                    total_tokens=result.usage.tokens_used if result.usage else 0,
                    thinking_trace=result.thinking_trace,
                )
            else:
                task_queue.fail_task(async_task.id, result.error or "Unknown error")
                save_agent_task(
                    task_id=async_task.id,
                    user_id=async_task.user_id,
                    prompt=async_task.prompt,
                    status="failed",
                    error=result.error,
                    progress=0.0,
                )

        except Exception as e:
            task_queue.fail_task(async_task.id, str(e))


# 全局 Worker 实例
worker = AgentWorker()
