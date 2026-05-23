"""记忆工具 — 将 MemoryManager 方法暴露为 AgentTool

依赖：memory_manager 模块级单例（自动注入）
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..base import AgentTool, AgentToolMeta

logger = logging.getLogger(__name__)


def create_memory_tools() -> list[AgentTool]:
    """创建所有记忆相关工具"""

    # ── memory_search：语义记忆检索 ──
    async def _memory_search(query: str, namespace: str = "general", top_k: int = 3) -> str:
        """从语义记忆中检索相关内容

        Args:
            query: 搜索关键词
            namespace: 命名空间（general/content/feedback/knowledge/domain）
            top_k: 返回结果数
        """
        try:
            from src.memory import memory_manager
            content = memory_manager.semantic.retrieve_formatted(
                query=query, namespace=namespace, top_k=top_k,
            )
            return content or "未找到相关记忆。"
        except Exception as e:
            logger.warning("memory_search 失败: %s", e)
            return f"记忆检索失败: {e}"

    # ── memory_store：存入语义记忆 ──
    async def _memory_store(
        text: str,
        namespace: str = "general",
        agent_type: str = "",
    ) -> str:
        """将内容存入语义记忆

        Args:
            text: 要存储的文本内容
            namespace: 命名空间
            agent_type: 来源 Agent 类型
        """
        try:
            from src.memory import memory_manager
            memory_manager.semantic.store(
                text=text,
                namespace=namespace,
                metadata={"agent_type": agent_type, "source": "tool_call"},
            )
            return f"已存入 {namespace} 记忆。"
        except Exception as e:
            logger.warning("memory_store 失败: %s", e)
            return f"记忆存储失败: {e}"

    # ── memory_context：构建 Agent 记忆上下文 ──
    async def _memory_context(
        agent_type: str,
        task_summary: str = "",
        user_id: str = "",
    ) -> str:
        """为 Agent 构建增强记忆上下文

        Args:
            agent_type: Agent 类型
            task_summary: 任务摘要（用于检索）
            user_id: 用户 ID
        """
        try:
            from src.memory import memory_manager
            task_input = {"topic": task_summary} if task_summary else {}
            context = await memory_manager.build_agent_context(
                agent_type=agent_type,
                task_input=task_input,
                user_id=user_id,
            )
            return context or "无可用记忆上下文。"
        except Exception as e:
            logger.warning("memory_context 失败: %s", e)
            return f"记忆上下文构建失败: {e}"

    return [
        AgentTool(
            name="memory_search",
            description="从语义记忆中检索与 query 相关的内容",
            execute_func=_memory_search,
            meta=AgentToolMeta(
                category="memory",
                agent_types=["WritingAgent", "SEOAgent", "ReviewAgent", "ManagerAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="memory_store",
            description="将文本内容存入语义记忆以供后续检索",
            execute_func=_memory_store,
            meta=AgentToolMeta(
                category="memory",
                agent_types=["WritingAgent", "SEOAgent", "ReviewAgent", "ManagerAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="memory_context",
            description="为指定 Agent 类型构建增强记忆上下文（聚合语义/情景/工作记忆）",
            execute_func=_memory_context,
            meta=AgentToolMeta(
                category="memory",
                agent_types=["ManagerAgent"],
                builtin=True,
            ),
        ),
    ]
