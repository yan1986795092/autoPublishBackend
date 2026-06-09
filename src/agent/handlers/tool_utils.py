"""简化的工具感知 LLM 调用工具

删除了 ToolRegistry 绑定逻辑，保留基础 LLM 调用 + 流式支持。
"""

from __future__ import annotations
import time as _time
from typing import Any, Callable, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.logging_shared.config import get_logger

logger = get_logger("agent.tool_utils")


async def call_llm(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    *,
    agent_type: str = "",
    stream: bool = False,
    on_chunk: Optional[Callable] = None,
) -> str:
    """调用 LLM，支持流式输出

    Args:
        llm: ChatOpenAI 实例
        messages: LangChain 消息列表
        agent_type: Agent 类型（仅用于日志）
        stream: 是否流式输出
        on_chunk: 流式回调 async on_chunk(chunk: str) -> None

    Returns:
        LLM 返回的文本内容
    """
    if stream:
        return await _stream_only(llm, messages, on_chunk)

    result = await llm.ainvoke(messages)
    return result.content or ""


async def _stream_only(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    on_chunk: Optional[Callable],
) -> str:
    """纯流式调用"""
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_chunk:
                await on_chunk(chunk.content)
    return "".join(chunks)
