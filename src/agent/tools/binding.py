"""Tool Binding — 工具绑定与描述

提供两个核心函数：
  - bind_agent_tools(): 为 LLM 绑定 Agent 可见的工具（bind_tools）
  - tool_prompt_section(): 生成 Agent 可读的工具列表描述
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def bind_agent_tools(
    llm: Any,
    agent_type: str,
    tool_registry: Any,
) -> Any:
    """为指定 Agent 类型的 LLM 绑定可用工具

    从 ToolRegistry 获取 agent_type 可见的工具，通过 llm.bind_tools() 绑定。
    如果 Agent 没有可用工具或 LLM 为 None，原样返回 LLM。

    Args:
        llm: ChatOpenAI 实例
        agent_type: Agent 类型名
        tool_registry: ToolRegistry 实例

    Returns:
        绑定了工具的 LLM（或原 LLM）
    """
    if llm is None or tool_registry is None:
        return llm

    tools = tool_registry.get_for_agent(agent_type)
    if not tools:
        return llm

    lc_tools = [t.to_langchain_tool() for t in tools]
    try:
        return llm.bind_tools(lc_tools)
    except Exception as e:
        logger.warning("bind_agent_tools(%s) 失败: %s", agent_type, e)
        return llm


def tool_prompt_section(
    agent_type: str,
    tool_registry: Any,
    include_category: bool = True,
) -> str:
    """生成 Agent 可读的工具列表描述（供 system prompt 注入）

    Args:
        agent_type: Agent 类型名
        tool_registry: ToolRegistry 实例
        include_category: 是否按分类分组展示

    Returns:
        格式化的工具描述文本，空字符串表示无可用工具
    """
    if tool_registry is None:
        return ""

    tools = tool_registry.get_for_agent(agent_type)
    if not tools:
        return ""

    if not include_category:
        lines = ["可用工具："]
        for t in tools:
            lines.append(f"  - {t.name}: {t.description}")
        return "\n".join(lines)

    # 按分类分组
    by_category: dict[str, list] = {}
    for t in tools:
        cat = t.meta.category or "general"
        by_category.setdefault(cat, []).append(t)

    lines = ["可用工具："]
    cat_labels = {
        "memory": "📝 记忆",
        "communication": "💬 通信",
        "reflection": "🔍 反思",
        "hitl": "👤 人机协同",
        "general": "🔧 通用",
        "search": "🌐 搜索",
        "code": "💻 代码",
    }
    for cat, cat_tools in by_category.items():
        label = cat_labels.get(cat, cat)
        lines.append(f"\n[{label}]")
        for t in cat_tools:
            lines.append(f"  - {t.name}: {t.description}")

    return "\n".join(lines)
