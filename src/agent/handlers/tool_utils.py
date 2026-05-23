"""Tool-aware LLM 调用工具

为 handler 提供透明的工具绑定+执行支持：
1. bind_tools() → 将 ToolRegistry 中的工具绑定到 LLM
2. execute_tool_calls() → 执行 LLM 返回的 tool_calls
3. call_llm() → 一站式工具感知 LLM 调用

用法：
    from .tool_utils import call_llm

    # 在 handler 中替换 llm.ainvoke() / llm.astream()
    text = await call_llm(
        llm, messages,
        tool_registry=task_input.get("_tool_registry"),
        agent_type="WritingAgent",
        stream=True,
        on_chunk=task_input.get("_on_llm_chunk"),
    )
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 最大工具调用轮次，防止无限循环
MAX_TOOL_ROUNDS = 5


def bind_tools(
    llm: ChatOpenAI,
    tool_registry: Any,
    agent_type: str,
) -> ChatOpenAI:
    """将 Agent 可见的工具绑定到 LLM

    如果 tool_registry 为空或该 Agent 无可见工具，返回原始 LLM。
    """
    if tool_registry is None or not agent_type:
        return llm

    tools = tool_registry.get_for_agent(agent_type)
    if not tools:
        return llm

    lc_tools = [t.to_langchain_tool() for t in tools]
    try:
        bound = llm.bind_tools(lc_tools)
        logger.debug("工具绑定 %s: %d 个工具", agent_type, len(tools))
        return bound
    except Exception as e:
        logger.warning("bind_tools 失败 (%s): %s", agent_type, e)
        return llm


async def execute_tool_calls(
    tool_calls: list[dict],
    tool_registry: Any,
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """执行 tool_calls 并追加 ToolMessage 到消息列表

    Args:
        tool_calls: LLM 返回的 tool_calls 列表
        tool_registry: ToolRegistry 实例
        messages: 当前消息列表（会被追加 ToolMessage）

    Returns:
        追加了 ToolMessage 后的消息列表
    """
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "") or tc.get("tool_call_id", "")

        if tool_registry is None:
            result_text = f"错误: 工具系统未启用"
        else:
            tool = tool_registry.get(tool_name)
            if tool is None:
                logger.warning("工具未找到: %s", tool_name)
                result_text = f"错误: 工具 {tool_name} 不可用"
            else:
                try:
                    result_text = await tool.execute(**tool_args)
                except Exception as e:
                    logger.warning("工具执行失败 %s: %s", tool_name, e)
                    result_text = f"错误: {e}"

        messages.append(ToolMessage(
            content=str(result_text)[:4000],
            tool_call_id=tool_id,
        ))

    return messages


async def call_llm(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    *,
    tool_registry: Any = None,
    agent_type: str = "",
    stream: bool = False,
    on_chunk: Optional[Callable] = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> str:
    """工具感知的 LLM 调用

    当 tool_registry 和 agent_type 可用时：
      Phase 1: 非流式调用 LLM，处理 tool_calls（最多 max_tool_rounds 轮）
      Phase 2: 流式输出最终文本（仅 stream=True 时）

    无工具可用时退化为原始行为（直接流式/非流式调用）。

    Args:
        llm: ChatOpenAI 实例
        messages: LangChain 消息列表
        tool_registry: ToolRegistry 实例（可选）
        agent_type: Agent 类型（用于过滤工具）
        stream: 是否流式输出
        on_chunk: 流式回调 async on_chunk(chunk: str) -> None
        max_tool_rounds: 最大工具调用轮次

    Returns:
        LLM 返回的文本内容
    """
    has_tools = _check_tools(tool_registry, agent_type)

    # ── 无工具：直接走原始路径 ──
    if not has_tools:
        if stream:
            return await _stream_only(llm, messages, on_chunk)
        result = await llm.ainvoke(messages)
        return result.content or ""

    # ── 有工具：绑定工具并执行工具循环 ──
    tool_llm = bind_tools(llm, tool_registry, agent_type)
    current_messages = list(messages)

    # Phase 1: 工具调用循环（非流式）
    for _round in range(max_tool_rounds):
        response = await tool_llm.ainvoke(current_messages)
        current_messages.append(response)

        tool_calls = _get_tool_calls(response)
        if not tool_calls:
            # 纯文本回复 — 结束工具循环
            final_text = response.content or ""
            # 如果请求了流式，用流式重新生成文本（给前端实时感）
            if stream and on_chunk:
                return await _stream_final(llm, current_messages, on_chunk)
            return final_text

        # 执行工具调用
        current_messages = await execute_tool_calls(
            tool_calls, tool_registry, current_messages,
        )

    # 超轮次兜底
    logger.warning("超出最大工具调用轮次 %d", max_tool_rounds)
    return current_messages[-1].content or ""


# ── 内部辅助 ──


def _check_tools(tool_registry: Any, agent_type: str) -> bool:
    """检查是否有可用工具"""
    if tool_registry is None or not agent_type:
        return False
    try:
        return len(tool_registry.get_for_agent(agent_type)) > 0
    except Exception:
        return False


def _get_tool_calls(response: AIMessage) -> list[dict]:
    """从 AIMessage 中提取 tool_calls"""
    raw = getattr(response, "tool_calls", [])
    if not raw:
        return []
    # LangChain tool_calls 格式: [{"name": ..., "args": ..., "id": ...}, ...]
    return [
        {
            "name": tc.get("name", tc.get("function", {}).get("name", "")),
            "args": tc.get("args", tc.get("function", {}).get("arguments", {})),
            "id": tc.get("id", tc.get("tool_call_id", "")),
        }
        for tc in raw
    ]


async def _stream_only(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    on_chunk: Optional[Callable],
) -> str:
    """纯流式调用（无工具）"""
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_chunk:
                await on_chunk(chunk.content)
    return "".join(chunks)


async def _stream_final(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    on_chunk: Optional[Callable],
) -> str:
    """工具循环结束后，流式输出最终文本"""
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_chunk:
                await on_chunk(chunk.content)
    return "".join(chunks)
