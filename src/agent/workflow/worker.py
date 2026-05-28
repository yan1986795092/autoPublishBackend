from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional

from .task_queue import task_queue, AsyncTask
from ..orchestrator import Orchestrator
from ..persistence import save_agent_task, save_thinking_trace, cleanup_old_records
from ..factory import create_orchestrator
from ..cache import agent_cache
from src.mcp.client import get_mcp_client, shutdown_mcp_client

logger = logging.getLogger(__name__)


class AgentWorker:
    """Worker：从队列中取出任务，执行 Agent 编排"""

    def __init__(self, orchestrator_factory=None, max_retries: Optional[int] = None):
        self._running = False
        self._task = None
        from src.config import get_settings
        settings = get_settings()
        self._max_retries = max_retries if max_retries is not None else settings.worker_max_retries
        self._orchestrator_factory = orchestrator_factory or (
            lambda **kw: create_orchestrator(**kw)
        )
        self._mcp_client = None
        self._mcp_tools = []

    async def start(self):
        self._running = True
        self._last_cleanup_time = time.monotonic()

        # 启动 MCP WebSearch Client
        try:
            self._mcp_client = await get_mcp_client()
            if self._mcp_client.is_started:
                self._mcp_tools = self._mcp_client.get_tools()
                logger.info("MCP WebSearch started with %d tools", len(self._mcp_tools))
            else:
                logger.warning("MCP WebSearch not available (network/config issue)")
        except Exception as e:
            logger.error("Failed to start MCP WebSearch: %s", e)

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

        # 关闭 MCP Client
        try:
            await shutdown_mcp_client()
        except Exception as e:
            logger.warning("MCP shutdown error: %s", e)

    async def _run_loop(self):
        while self._running:
            try:
                async_task = await task_queue.dequeue()
                if async_task is None:
                    # 定期清理过期数据（每小时）
                    now = time.monotonic()
                    if now - self._last_cleanup_time > 3600:
                        self._last_cleanup_time = time.monotonic()
                        try:
                            counts = cleanup_old_records()
                            if any(counts.values()):
                                logger.info("Cleaned up old records: %s", counts)
                        except Exception as e:
                            logger.warning("Cleanup error: %s", e)
                    continue

                await self._process_task(async_task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Worker loop error: %s", e)

    async def _process_task(self, async_task: AsyncTask) -> None:
        """执行单个异步任务，带重试逻辑"""
        last_error = ""

        for attempt in range(self._max_retries):
            try:
                if attempt > 0:
                    wait = min(2 ** attempt, 30)
                    logger.info("Retry %d/%d for task %s, waiting %ds",
                                attempt + 1, self._max_retries, async_task.id, wait)
                    await asyncio.sleep(wait)

                task_queue.update_task_progress(
                    async_task.id, 5.0,
                    message="正在初始化 Agent 系统..." + (f" (第{attempt + 1}次尝试)" if attempt > 0 else ""),
                    step="initializing",
                )

                orchestrator = self._orchestrator_factory(
                    async_task_id=async_task.id,
                    task_queue=task_queue,
                    mcp_tools=self._mcp_tools,
                )

                task_queue.update_task_progress(
                    async_task.id, 10.0,
                    message="正在分析需求并拆解任务...",
                    step="decomposing",
                )

                context: dict = {"async_task_id": async_task.id, "user_id": async_task.user_id}
                if async_task.plan:
                    context["precomputed_plan"] = async_task.plan
                if async_task.skill_context:
                    context.update(async_task.skill_context)
                result = await orchestrator.run(
                    user_request=async_task.prompt,
                    context=context,
                )

                task_queue.update_task_progress(
                    async_task.id, 90.0,
                    message="完成，正在整理结果...",
                    step="finalizing",
                )

                if result.status.value == "completed":
                    task_queue.complete_task(async_task.id, result.to_dict())
                    trace_steps = result.thinking_trace.get("steps", []) if result.thinking_trace else []
                    save_agent_task(
                        task_id=async_task.id,
                        user_id=async_task.user_id,
                        prompt=async_task.prompt,
                        status="completed",
                        result={"agent_result": result.to_dict()},
                        progress=100.0,
                        total_steps=len(trace_steps) or 1,
                        completed_steps=len(trace_steps),
                        total_tokens=result.usage.tokens_used if result.usage else 0,
                        thinking_trace=result.thinking_trace,
                    )
                    return  # 完全成功，退出重试循环

                # partial 状态：有输出但质量不足，记录警告但不退出重试
                if result.status.value == "partial":
                    last_error = result.error or "partial_result"
                    logger.warning("Task %s attempt %d produced partial result: %s",
                                   async_task.id, attempt + 1, last_error)
                    # partial 结果可以继续（当达到最大重试次数时退化为 partial 完成）
                    if attempt >= self._max_retries - 1:
                        task_queue.complete_task(async_task.id, result.to_dict())
                        save_agent_task(
                            task_id=async_task.id,
                            user_id=async_task.user_id,
                            prompt=async_task.prompt,
                            status="partial",
                            result={"agent_result": result.to_dict()},
                            progress=50.0,
                            total_steps=1,
                            completed_steps=0,
                            total_tokens=0,
                            thinking_trace=result.thinking_trace,
                        )
                        return
                    continue

                last_error = result.error or "Unknown error"
                logger.warning("Task %s attempt %d failed: %s",
                               async_task.id, attempt + 1, last_error)

            except Exception as e:
                last_error = str(e)
                logger.error("Task %s attempt %d error: %s",
                             async_task.id, attempt + 1, e)
                # 网络/超时类错误可重试
                if isinstance(e, (asyncio.TimeoutError, ConnectionError, OSError)):
                    continue
                # 其它错误仅在最后一次尝试时继续
                if attempt < self._max_retries - 1:
                    continue
                break

        # 所有重试耗尽，进入死信队列
        logger.warning("Task %s exhausted all %d retries. Sending to DLQ.",
                       async_task.id, self._max_retries)
        task_queue.send_to_dlq(async_task.id, reason=last_error)
        save_agent_task(
            task_id=async_task.id,
            user_id=async_task.user_id,
            prompt=async_task.prompt,
            status="failed",
            error=last_error,
            progress=0.0,
        )


# 全局 Worker 实例
worker = AgentWorker()
