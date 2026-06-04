"""AgentTool — 统一工具基类

包装 LangChain BaseTool，添加 Agent 系统所需的元数据。
所有工具（内置 / MCP / 遗留）统一适配到此接口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.tools import BaseTool as LangChainBaseTool


@dataclass
class AgentToolMeta:
    """工具元数据"""
    category: str = "general"            # memory / communication / reflection / hitl / search / code
    agent_types: list[str] = field(default_factory=list)  # 允许使用的 Agent 类型列表，空=全部
    builtin: bool = False                # 是否为内置工具
    visible: bool = True                 # 是否对 LLM 可见（False=仅供系统内部使用）


class AgentTool:
    """Agent 工具包装

    统一适配三种工具来源：
    1. 内置工具（直接调用系统模块）
    2. MCP 工具（从 MCP Server 加载的 LangChain BaseTool）
    3. 遗留工具（tools/langchain/ 中的 BaseTool 子类）

    对 LLM 表现为 LangChain BaseTool（支持 bind_tools + function calling）。
    对 Agent 系统表现为 AgentTool（包含元数据和分类）。
    """

    def __init__(
        self,
        name: str,
        description: str,
        execute_func: Any,           # async callable(**kwargs) -> str
        meta: Optional[AgentToolMeta] = None,
        args_schema: Optional[Any] = None,  # Pydantic model
    ):
        self.name = name
        self.description = description
        self._execute_func = execute_func
        self.meta = meta or AgentToolMeta()
        self._args_schema = args_schema

    # ── 执行 ──

    async def execute(self, **kwargs: Any) -> str:
        """执行工具（所有工具统一的调用接口）"""
        return await self._execute_func(**kwargs)

    # ── 转为 LangChain BaseTool（用于 bind_tools）──

    def to_langchain_tool(self) -> LangChainBaseTool:
        """将 AgentTool 转为 LangChain BaseTool

        返回的 BaseTool 可以通过 llm.bind_tools() 绑定到 ChatOpenAI，
        使 LLM 可以通过 function calling 调用此工具。
        """
        from langchain_core.tools import tool

        name = self.name
        description = self.description

        lc_tool = tool(
            name, description=description, args_schema=self._args_schema,
        )(self._execute_func)
        return lc_tool

    # ── 从 LangChain BaseTool 创建 ──

    @classmethod
    def from_langchain(
        cls,
        lc_tool: LangChainBaseTool,
        meta: Optional[AgentToolMeta] = None,
    ) -> AgentTool:
        """从 LangChain BaseTool 创建 AgentTool"""
        async def _execute(**kwargs: Any) -> str:
            if hasattr(lc_tool, "_arun"):
                # LangChain StructuredTool 要求 config 参数
                try:
                    from langchain_core.runnables import RunnableConfig
                    return await lc_tool._arun(**kwargs, config=RunnableConfig())
                except TypeError:
                    # 如果 config 参数不被接受，回退到不带 config 调用
                    return await lc_tool._arun(**kwargs)
            return lc_tool.run(kwargs)

        return cls(
            name=lc_tool.name,
            description=lc_tool.description or "",
            execute_func=_execute,
            meta=meta or AgentToolMeta(category="external", visible=True),
            args_schema=getattr(lc_tool, "args_schema", None),
        )

    def __repr__(self) -> str:
        return f"AgentTool({self.name}, cat={self.meta.category})"
