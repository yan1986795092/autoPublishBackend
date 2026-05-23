"""BaseAgent — 角色化对话智能体基类

包装现有 HANDLER_MAP 中的 handler 函数，注入角色身份信息。
Handler 函数保持不变，BaseAgent 只是在其上包装一层 profile 上下文。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_openai import ChatOpenAI

from .profile import AgentProfile
from ..handlers import HANDLER_MAP, handle_fallback

logger = logging.getLogger(__name__)


class BaseAgent:
    """角色化对话智能体

    每个 BaseAgent 实例 = 一个 AgentProfile + 一个 LLM 实例 + 一个 handler 函数。
    本质是对 handler 函数的带身份包装。
    """

    def __init__(
        self,
        profile: AgentProfile,
        llm: Optional[ChatOpenAI] = None,
    ):
        self.profile = profile
        self._llm = llm
        self._handler_fn: Optional[Callable] = None

    # ── 属性快捷访问 ──

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def role(self) -> str:
        return self.profile.role

    # ── 核心执行 ──

    async def execute(
        self,
        task_input: dict,
        mcp_tools: Optional[list] = None,
    ) -> dict:
        """执行任务

        将 profile 注入 task_input，然后委托给对应的 handler 函数。
        返回值与 handler 函数完全一致。
        """
        # 注入 profile 到 task_input（handler 可选使用）
        enriched = {**task_input, "_agent_profile": self.profile}

        # 获取 handler
        handler = self._resolve_handler()

        # 调用 handler
        if handler is not None:
            return await handler(enriched, self._llm, mcp_tools)

        # Fallback
        fallback_input = {
            "agent_type": self.profile.name,
            "input": task_input,
        }
        return await handle_fallback(fallback_input, self._llm, None)

    async def execute_with_memory(
        self,
        task_input: dict,
        mcp_tools: Optional[list] = None,
        memory_context: str = "",
    ) -> dict:
        """执行任务并注入记忆上下文"""
        if memory_context:
            task_input = {**task_input, "_memory_context": memory_context}
        return await self.execute(task_input, mcp_tools)

    # ── 内部 ──

    def _resolve_handler(self) -> Optional[Callable]:
        if self._handler_fn is not None:
            return self._handler_fn

        # 按 handler_key 或 name 查找
        key = self.profile.handler_key or self.profile.name
        self._handler_fn = HANDLER_MAP.get(key)
        if self._handler_fn is None:
            logger.warning("BaseAgent[%s]: no handler found for key=%s", self.name, key)
        return self._handler_fn
