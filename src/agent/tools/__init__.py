"""Agent 工具体系 — 统一的工具定义、注册、绑定

架构：
  AgentTool (LangChain BaseTool 包装)
    ├── 元数据：category, agent_types, builtin, visible
    ├── 内置工具：memory / communication / reflection / hitl
    │   └── tools/builtins/ 中的工厂函数
    ├── MCP 工具：通过 register_from_langchain() 注册
    └── 外部工具：直接 register() 注册

  ToolRegistry
    ├── 注册 / 查询 / 按 Agent 过滤
    ├── to_openai_functions() → LLM function calling
    └── bind_to_llm() → llm.bind_tools()

  binding.py
    ├── bind_agent_tools() — 为 LLM 绑定可见工具
    └── tool_prompt_section() — 生成工具列表供 System Prompt 注入
"""
from .base import AgentTool, AgentToolMeta
from .registry import ToolRegistry
from .binding import bind_agent_tools, tool_prompt_section
from .builtins import (
    create_memory_tools,
    create_communication_tools,
    create_reflection_tools,
    create_hitl_tools,
)

__all__ = [
    "AgentTool", "AgentToolMeta", "ToolRegistry",
    "bind_agent_tools", "tool_prompt_section",
    "create_memory_tools", "create_communication_tools",
    "create_reflection_tools", "create_hitl_tools",
]
