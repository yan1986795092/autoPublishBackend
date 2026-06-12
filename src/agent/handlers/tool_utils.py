"""简化的工具感知 LLM 调用工具

保留基础 LLM 调用 + 流式支持。
从 LLM 响应中提取实际 token 用量（不再用 estimate_tokens 估算）。
自动上报调用指标到 StatsCollector + Prometheus。

流式模式下（stream=True），LLM Gateway 的 SSE 响应会在最终 chunk 中
注入 usage 信息（格式：{"type": "usage", "prompt_tokens": X, "completion_tokens": Y}），
供后端从流中提取。若上游未返回 usage，使用字符数估算作为降级。
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.logging_shared.config import get_logger
from src.observability.metrics import stats_collector

logger = get_logger("agent.tool_utils")

# 中文/混合文本的 token 估算因子：平均 2-3 字符/token
_CHARS_PER_TOKEN_ESTIMATE = 2.5


@dataclass
class LLMResult:
    """LLM 调用结果，含实际 token 用量"""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_tokens(text: str | list, factor: float = _CHARS_PER_TOKEN_ESTIMATE) -> int:
    """估算 token 数量（降级方案：按字符数估算）

    Args:
        text: 要估算的文本（str）或消息列表
        factor: 每 token 平均字符数（中文约 1.5-2，英文约 4）

    Returns:
        估算的 token 数，最小为 1
    """
    if isinstance(text, list):
        # messages 列表：累加所有 content 长度
        total = 0
        for msg in text:
            if isinstance(msg, dict):
                total += len(msg.get("content", "") or "")
            elif hasattr(msg, "content"):
                total += len(getattr(msg, "content", "") or "")
        text_len = total
    elif isinstance(text, str):
        text_len = len(text)
    else:
        return 0
    return max(1, int(text_len / factor))


def _extract_token_usage_from_stream_chunks(
    chunks: list[Any] | list[str],
) -> tuple[int, int]:
    """从流式 chunks 中尝试提取 token 用量。

    支持两种格式：
    1. LangChain AIMessageChunk 对象（携带 response_metadata）
    2. 纯文本 chunks（无法提取，返回 0）
    """
    if not chunks:
        return 0, 0

    # 检查最后一个 chunk 是否携带了 usage 元数据
    last = chunks[-1]
    if hasattr(last, "response_metadata"):
        meta = last.response_metadata or {}
        usage = meta.get("token_usage", {}) or meta.get("usage", {})
        if isinstance(usage, dict):
            pt = int(usage.get("prompt_tokens", 0))
            ct = int(usage.get("completion_tokens", 0))
            if pt > 0 or ct > 0:
                return pt, ct

    # 检查是否有 AIMessageChunk 中途携带了 usage
    for chunk in chunks:
        if hasattr(chunk, "response_metadata"):
            meta = chunk.response_metadata or {}
            usage = meta.get("token_usage", {}) or meta.get("usage", {})
            if isinstance(usage, dict):
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                if pt > 0 or ct > 0:
                    return pt, ct

    return 0, 0


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
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            duration_ms=round(duration_ms, 2),
            agent_type=agent_type,
        )

        provider = getattr(llm, "model_kwargs", {}).get("provider", "") or "unknown"
        stats_collector.record_llm_call(
            model=model_name, provider=provider, status="success",
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            duration_s=duration_ms / 1000,
        )
        return result

    start = time.perf_counter()
    try:
        result = await llm.ainvoke(messages)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        provider = getattr(llm, "model_kwargs", {}).get("provider", "") or "unknown"
        logger.warning(f"LLM {model_name} 调用出错: {exc}", _type="llm", model=model_name, duration_ms=round(duration_ms, 2))
        stats_collector.record_llm_call(
            model=model_name, provider=provider, status="error",
            prompt_tokens=0, completion_tokens=0, duration_s=duration_ms / 1000,
        )
        raise
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

    # 降级：response_metadata 中没有 token_usage 时估算
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _estimate_tokens(messages)
        completion_tokens = _estimate_tokens(content)
        logger.debug("使用估算 token 用量（响应无 token_usage）", agent_type=agent_type)

    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    provider = getattr(llm, "model_kwargs", {}).get("provider", "") or "unknown"
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

    # 报告到 StatsCollector + Prometheus
    stats_collector.record_llm_call(
        model=model_name, provider=provider, status="success",
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        duration_s=duration_ms / 1000,
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
    """流式调用 LLM，尝试从流中提取 token 用量，失败时估算。

    策略（按优先级）：
    1. 从最后一个 chunk 的 response_metadata 中提取实际 token 用量
    2. 降级：按字符数估算 prompt_tokens 和 completion_tokens
    """
    raw_chunks: list[Any] = []
    full_content_chunks: list[str] = []

    async for chunk in llm.astream(messages):
        raw_chunks.append(chunk)
        if chunk.content:
            full_content_chunks.append(chunk.content)
            if on_chunk:
                await on_chunk(chunk.content)

    full_content = "".join(full_content_chunks)

    # 策略 1：从 chunk 元数据中提取
    prompt_tokens, completion_tokens = _extract_token_usage_from_stream_chunks(raw_chunks)

    # 策略 2：降级估算
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _estimate_tokens(messages)
        completion_tokens = _estimate_tokens(full_content)

    return LLMResult(
        content=full_content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
