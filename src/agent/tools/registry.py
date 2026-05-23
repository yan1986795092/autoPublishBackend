"""ToolRegistry — 工具注册中心

集中管理所有 Agent 可用工具的注册、查询和绑定。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .base import AgentTool, AgentToolMeta

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册中心

    职责：
    1. 注册系统内置工具和外部工具
    2. 按 Agent 类型 + 分类查询可用工具
    3. 生成 LLM function calling 所需的工具描述
    """

    def __init__(self):
        self._tools: dict[str, AgentTool] = {}  # name -> AgentTool

    # ── 注册 ──

    def register(self, tool: AgentTool) -> None:
        """注册一个工具"""
        if tool.name in self._tools:
            logger.warning("ToolRegistry: 覆盖已注册的工具: %s", tool.name)
        self._tools[tool.name] = tool

    def register_many(self, tools: list[AgentTool]) -> None:
        """批量注册工具"""
        for tool in tools:
            self.register(tool)

    def register_from_langchain(
        self,
        lc_tool: Any,
        meta: Optional[AgentToolMeta] = None,
    ) -> AgentTool:
        """从 LangChain BaseTool 注册"""
        tool = AgentTool.from_langchain(lc_tool, meta)
        self.register(tool)
        return tool

    # ── 查询 ──

    def get(self, name: str) -> Optional[AgentTool]:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[AgentTool]:
        """列出所有已注册工具"""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    # ── 按 Agent 类型过滤 ──

    def get_for_agent(self, agent_type: str) -> list[AgentTool]:
        """获取指定 Agent 类型可用的工具列表

        过滤规则：
        - agent_types 为空 = 所有 Agent 可用
        - agent_types 包含 agent_type = 该 Agent 可用
        - not visible = 系统内部工具，不暴露给 LLM
        """
        result = []
        for tool in self._tools.values():
            if not tool.meta.visible:
                continue
            if not tool.meta.agent_types:
                result.append(tool)
            elif agent_type in tool.meta.agent_types:
                result.append(tool)
        return result

    def get_by_category(self, category: str) -> list[AgentTool]:
        """按分类获取工具"""
        return [
            t for t in self._tools.values()
            if t.meta.category == category and t.meta.visible
        ]

    # ── LLM Function Calling ──

    def to_openai_functions(self, agent_type: str) -> list[dict]:
        """为指定 Agent 类型生成 OpenAI function calling 格式的工具列表"""
        tools = self.get_for_agent(agent_type)
        result = []
        for tool in tools:
            lc = tool.to_langchain_tool()
            # LangChain BaseTool 自带 to_openai_function 方法
            if hasattr(lc, "to_openai_function"):
                func = lc.to_openai_function()
                if func:
                    result.append(func)
            elif hasattr(lc, "to_openai_functions"):
                funcs = lc.to_openai_functions()
                result.extend(funcs)
        return result

    def bind_to_llm(self, agent_type: str, llm: Any) -> Any:
        """将 Agent 可用工具绑定到 LLM 实例

        返回绑定了 tools 的 LLM（通过 bind_tools）。
        """
        tools = self.get_for_agent(agent_type)
        if not tools:
            return llm

        lc_tools = [t.to_langchain_tool() for t in tools]
        return llm.bind_tools(lc_tools)

    # ── 工具描述（供 Prompt 注入）──

    def describe_for_agent(self, agent_type: str) -> str:
        """生成 Agent 可读的工具列表描述（供 system prompt 使用）"""
        tools = self.get_for_agent(agent_type)
        if not tools:
            return ""

        lines = ["可用工具："]
        for t in tools:
            lines.append(f"  - {t.name}: {t.description}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({self.count} tools)"
