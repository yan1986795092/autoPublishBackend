"""简化的工具感知 LLM 调用工具

保留基础 LLM 调用 + 流式支持。
从 LLM 响应中提取实际 token 用量（不再用 estimate_tokens 估算）。
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.logging_shared.config import get_logger

logger = get_logger("agent.tool_utils")


@dataclass
class LLMResult:
    """LLM 调用结果，含实际 token 用量"""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


async def call_llm(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    *,
    agent_type: str = "",
    stream: bool = False,
    on_chunk: Optional[Callable] = None,
) -> LLMResult:
    """调用 LLM，返回内容 + 实际 token 用量

    Args:
        llm: ChatOpenAI 实例
        messages: LangChain 消息列表
        agent_type: Agent 类型（仅用于日志）
        stream: 是否流式输出
        on_chunk: 流式回调 async on_chunk(chunk: str) -> None

    Returns:
        LLMResult: 包含 content / prompt_tokens / completion_tokens
    """
    if stream:
        start = time.perf_counter()
        result = await _stream_only(llm, messages, on_chunk)
        duration_ms = (time.perf_counter() - start) * 1000
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        logger.info(
            f"LLM {model_name}",
            _type="llm",
            model=model_name,
            duration_ms=round(duration_ms, 2),
            agent_type=agent_type,
        )
        return result

    start = time.perf_counter()
    result = await llm.ainvoke(messages)
    duration_ms = (time.perf_counter() - start) * 1000
    content = result.content or ""

    # 从 response_metadata 提取实际 token 用量
    prompt_tokens = 0
    completion_tokens = 0
    try:
        meta = result.response_metadata or {}
        usage = meta.get("token_usage", {}) or meta
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
    except (ValueError, TypeError, AttributeError):
        pass

    if prompt_tokens == 0 and completion_tokens == 0:
        logger.debug("未能从 LLM 响应中提取 token 用量", agent_type=agent_type)

    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    logger.info(
        f"LLM {model_name}",
        _type="llm",
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        duration_ms=round(duration_ms, 2),
        agent_type=agent_type,
    )

    return LLMResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _stream_only(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    on_chunk: Optional[Callable],
) -> LLMResult:
    """纯流式调用（流式模式下无法获取 token 用量，返回 0）"""
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_chunk:
                await on_chunk(chunk.content)
    return LLMResult(content="".join(chunks))
