"""BaseAgent — 角色化对话智能体基类

包装现有 HANDLER_MAP 中的 handler 函数，注入角色身份信息。
Handler 函数保持不变，BaseAgent 只是在其上包装一层 profile 上下文。
新增 ToolRegistry 感知能力，支持为 LLM 绑定可见工具。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import structlog
from langchain_openai import ChatOpenAI

from .profile import AgentProfile
from ..handlers import HANDLER_MAP, handle_fallback

logger = structlog.get_logger(__name__)


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
        自动采集执行快照。
        """
        enriched = {**task_input, "_agent_profile": self.profile}

        # ── 创建 SnapshotCollector ──
        collector = None
        try:
            from src.agent.snapshot.collector import SnapshotCollector, set_collector
            collector = SnapshotCollector(
                agent_name=self.name,
                profile=self.profile.to_dict(),
                task_id=task_input.get("_task_id"),
            )
            set_collector(collector)

            # 注册 PromptBuilder hook
            from src.prompts import builder as _prompt_builder
            _prompt_builder.set_on_build_complete(
                lambda composed, messages: _record_builder_messages(collector, messages)
            )
        except ImportError:
            pass

        # 获取 handler（原有逻辑不变）
        handler = self._resolve_handler()

        try:
            if handler is not None:
                result = await handler(enriched, self._llm, mcp_tools)
            else:
                fallback_input = {
                    "agent_type": self.profile.name,
                    "input": task_input,
                }
                result = await handle_fallback(fallback_input, self._llm, None)

            # ── 采集快照并持久化 ──
            if collector is not None:
                snapshot = collector.finalize(result)
                try:
                    from src.agent.snapshot.storage import SnapshotStorage
                    storage = SnapshotStorage()
                    storage.save(snapshot)
                    result["_snapshot_id"] = snapshot.snapshot_id
                except Exception as e:
                    logger.warning("Failed to persist snapshot: %s", e)

            return result
        finally:
            # 清理 context
            if collector is not None:
                try:
                    from src.agent.snapshot.collector import set_collector
                    set_collector(None)
                    from src.prompts import builder as _prompt_builder
                    _prompt_builder.set_on_build_complete(None)
                except ImportError:
                    pass

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

    # ── 工具绑定 ──

    def bind_tools(self, tool_registry: Any) -> None:
        """从 ToolRegistry 为 Agent 的 LLM 绑定可见的工具

        调用后 self._llm 被替换为绑定了工具的 LLM。
        生成工具描述并缓存到 self._tool_descriptions。
        """
        if self._llm is None or tool_registry is None:
            return

        from ..tools import bind_agent_tools, tool_prompt_section

        bound = bind_agent_tools(self._llm, self.profile.name, tool_registry)
        if bound is not self._llm:
            self._llm = bound
            self._tool_descriptions = tool_prompt_section(
                self.profile.name, tool_registry,
            )
            logger.info(
                "Agent tools bound",
                agent_name=self.name,
                tool_preview=self._tool_descriptions[:100] if self._tool_descriptions else "none",
            )

    # ── 内部 ──

    def _resolve_handler(self) -> Optional[Callable]:
        if self._handler_fn is not None:
            return self._handler_fn

        # 按 handler_key 或 name 查找
        key = self.profile.handler_key or self.profile.name
        self._handler_fn = HANDLER_MAP.get(key)
        if self._handler_fn is None:
            logger.warning("No handler found for agent", agent_name=self.name, handler_key=key)
        return self._handler_fn


def _record_builder_messages(collector, messages):
    """将 PromptBuilder 构建的消息注册到 collector（模块级函数，避免 lambda 序列化问题）。"""
    from src.agent.snapshot.models import PromptMessageSnapshot
    for m in messages:
        collector.record_prompt_message(PromptMessageSnapshot(
            role=m["role"],
            source="build_result",
            resolved_content=m["content"],
        ))
