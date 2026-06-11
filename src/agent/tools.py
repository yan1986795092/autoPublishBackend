"""Agent Tool 抽象层 — AgentTool / AgentToolMeta

MCP Client 通过 AgentTool.from_langchain() 将 LangChain BaseTool 适配为内部工具体系，
使 MCP 工具可以无缝注册到 ToolRegistry 并按 category/agent_type 过滤。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from langchain_core.tools import BaseTool


@dataclass
class AgentToolMeta:
    """MCP / 内置工具的元信息"""

    category: str = "search"
    agent_types: list[str] = field(default_factory=list)
    builtin: bool = False
    visible: bool = True
    description: str = ""
    tags: list[str] = field(default_factory=list)


class AgentTool:
    """LangChain BaseTool 的内部包装器

    Attributes:
        name: 工具名（与底层 BaseTool.name 一致）
        description: 工具描述
        meta: 工具元信息（category / agent_types / builtin / visible）
        _tool: 底层 LangChain BaseTool
    """

    # 注册表：name -> AgentTool 实例
    _registry: ClassVar[dict[str, "AgentTool"]] = {}

    def __init__(
        self,
        tool: BaseTool,
        meta: AgentToolMeta | None = None,
        func: Callable[..., Any] | None = None,
    ):
        self._tool = tool
        self.name: str = tool.name
        self.description: str = tool.description
        self.meta: AgentToolMeta = meta or AgentToolMeta()
        self._func: Callable[..., Any] | None = func
        AgentTool._registry[self.name] = self

    # ── LangChain BaseTool 代理 ──────────────────────

    @property
    def args_schema(self):
        return self._tool.args_schema

    def invoke(self, input: Any, **kwargs) -> Any:
        return self._tool.invoke(input, **kwargs)

    async def ainvoke(self, input: Any, **kwargs) -> Any:
        return await self._tool.ainvoke(input, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        return self._tool.run(*args, **kwargs)

    async def arun(self, *args, **kwargs) -> Any:
        return await self._tool.arun(*args, **kwargs)

    # ── 工厂方法 ──────────────────────────────────────

    @classmethod
    def from_langchain(cls, tool: BaseTool, meta: AgentToolMeta | None = None) -> "AgentTool":
        """从 LangChain BaseTool 创建 AgentTool（幂等，同名工具不重复注册）"""
        if tool.name in cls._registry:
            return cls._registry[tool.name]
        return cls(tool=tool, meta=meta)

    # ── 工具注册表 ────────────────────────────────────

    @classmethod
    def get_registry(cls) -> dict[str, "AgentTool"]:
        return dict(cls._registry)

    @classmethod
    def get_by_category(cls, category: str) -> list["AgentTool"]:
        return [t for t in cls._registry.values() if t.meta.category == category]

    @classmethod
    def get_by_agent_type(cls, agent_type: str) -> list["AgentTool"]:
        result = []
        for t in cls._registry.values():
            if not t.meta.agent_types or agent_type in t.meta.agent_types:
                result.append(t)
        return result

    @classmethod
    def clear_registry(cls) -> None:
        cls._registry.clear()

    def __repr__(self) -> str:
        return f"AgentTool(name={self.name!r}, category={self.meta.category!r}, builtin={self.meta.builtin})"
