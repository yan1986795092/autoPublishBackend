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

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.logging_shared.config import get_logger
from src.logging_shared.genai_records import PromptRecord, CompletionRecord
from src.logging_shared.genai_dispatcher import (
    dispatch_prompt,
    dispatch_completion,
    serialize_messages,
)
from src.logging_shared.error_codes import ErrorCode
from src.logging_shared.trace_context_var import (
    TraceContextVar,
    get_current_trace,
    new_span_id,
    new_trace_id,
)
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


def _log_llm_messages(messages: list[BaseMessage], agent_type: str, model_name: str) -> None:
    """格式化打印 LLM messages，带角色标签"""
    parts = [f"\n{'='*60}", f"  LLM 输入 [{agent_type}] [{model_name}]", f"{'='*60}"]
    for msg in messages:
        role = "SYSTEM" if isinstance(msg, SystemMessage) else \
               "USER" if isinstance(msg, HumanMessage) else \
               "ASSISTANT" if isinstance(msg, AIMessage) else \
               "TOOL" if isinstance(msg, ToolMessage) else \
               type(msg).__name__.upper()
        content = msg.content or ""
        parts.append(f"\n──[{role}]──\n{content}")
    parts.append(f"\n{'='*60}")
    try:
        logger.info("📋 LLM Prompt", _type="system", event_name="llm_prompt",
                     message="".join(parts), agent_type=agent_type, model=model_name)
    except UnicodeEncodeError:
        pass


async def call_llm(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    *,
    agent_type: str = "",
    stream: bool = False,
    on_chunk: Optional[Callable] = None,
    on_reasoning: Optional[Callable] = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    tool_definitions: list[dict] | None = None,
    raw: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """调用 LLM，返回内容 + 实际 token 用量。

    事故追溯增强（Phase 0）：
    - 调用前后分别 dispatch PromptRecord / CompletionRecord 到 agent-events-* 索引
      (event_type=llm_prompt / llm_completion)，与 console 输出独立。
    - 新增参数：trace_id / span_id / parent_span_id / tool_definitions / raw。
      全部默认 None 或 False，向后兼容现有 12 个 handler 调用方。
    - raw=True 表示跳过 PII 脱敏（业务侧确认无敏感信息时可显式 opt-in）。

    创作偏好增强：
    - temperature: 覆盖 LLM 实例默认 temperature，控制输出随机性（0=确定性，1=发散）。
      由前端"创造力"参数映射而来（creativity 0→0.2，0.5→0.6，1→0.9）。
    - max_tokens: 覆盖 LLM 实例默认 max_tokens，控制最大输出长度。
      由前端"篇幅"参数映射而来（short→800，medium→2000，long→4000）。

    Args:
        llm: ChatOpenAI 实例
        messages: LangChain 消息列表
        agent_type: Agent 类型（仅用于日志）
        stream: 是否流式输出
        on_chunk: 流式回调 async on_chunk(chunk: str) -> None
        trace_id: W3C trace context 主 trace id；缺省时自动生成
        span_id: 本次 LLM 调用的 span id；缺省时自动生成
        parent_span_id: 上游 Agent 的 span id（如果有）
        tool_definitions: 调用 LLM 时可用的工具 schema（写入 PromptRecord）
        raw: True = 不脱敏
        temperature: 可选，per-call 覆盖 LLM 的 temperature（用于创作偏好）
        max_tokens: 可选，per-call 覆盖 LLM 的 max_tokens（用于篇幅控制）

    Returns:
        LLMResult: 包含 content / prompt_tokens / completion_tokens
    """
    model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    # per-call 参数优先于 LLM 实例属性
    effective_temperature = temperature if temperature is not None else getattr(llm, "temperature", None)
    effective_max_tokens = max_tokens

    # 构建 per-call LLM config（用于 ainvoke / astream）
    call_config: dict[str, Any] = {}
    if effective_temperature is not None:
        call_config["temperature"] = effective_temperature
    if effective_max_tokens is not None:
        call_config["max_tokens"] = effective_max_tokens

    # 从 contextvars 提取 trace 元数据 — 业务侧 push_trace() 后即使 handler 不传参也能落库
    _cur = get_current_trace()
    if _cur is not None and not trace_id:
        effective_trace_id = _cur.trace_id
    else:
        effective_trace_id = trace_id or new_trace_id()
    if _cur is not None and not span_id:
        effective_span_id = _cur.span_id
    else:
        effective_span_id = span_id or new_span_id()
    if _cur is not None and parent_span_id is None:
        effective_parent_span_id = _cur.parent_span_id or _cur.span_id
    else:
        effective_parent_span_id = parent_span_id
    if _cur is not None and not raw:
        effective_raw = _cur.raw
    else:
        effective_raw = raw

    # ── Phase 0 关键改动：先 dispatch PromptRecord（默认 PII 脱敏） ──
    messages_dict = [serialize_messages(m) for m in messages]
    prompt_record = PromptRecord(
        trace_id=effective_trace_id,
        span_id=effective_span_id,
        parent_span_id=effective_parent_span_id,
        agent_name=agent_type,
        model=model_name,
        messages=messages_dict,
        tool_definitions=tool_definitions,
        temperature=effective_temperature,
        raw=effective_raw,
    )
    try:
        await dispatch_prompt(prompt_record)
    except Exception:
        # Phase 0 兜底：dispatch 抛错不影响 LLM 调用本身
        pass
    _log_llm_messages(messages, agent_type, model_name)
    if tool_definitions:
        _tool_names = [t.get("function", {}).get("name", t.get("name", "?")) for t in tool_definitions if isinstance(t, dict)]
        logger.info(
            "LLM 工具绑定: agent=%s tools=%s",
            agent_type, _tool_names,
        )

    if stream:
        start = time.perf_counter()
        result = await _stream_only(llm, messages, on_chunk, on_reasoning, call_config)
        duration_ms = (time.perf_counter() - start) * 1000
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))

        # Phase 0 关键改动：dispatch CompletionRecord
        try:
            completion_record = CompletionRecord(
                trace_id=effective_trace_id,
                span_id=prompt_record.span_id,
                parent_span_id=effective_parent_span_id,
                agent_name=agent_type,
                model=model_name,
                finish_reason="stop",
                content=result.content or "",
                tool_calls=None,  # 流式暂不取
                completion_tokens=result.completion_tokens,
                raw=effective_raw,
            )
            await dispatch_completion(completion_record)
        except Exception:
            pass

        logger.info(
            "LLM 流式完成: agent=%s model=%s text_len=%s",
            agent_type, model_name, len(result.content or ""),
        )
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
        if call_config:
            result = await llm.ainvoke(messages, config=call_config)
        else:
            result = await llm.ainvoke(messages)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        provider = getattr(llm, "model_kwargs", {}).get("provider", "") or "unknown"
        # P2.3: 将异常映射为标准错误码
        _exc_name = type(exc).__name__
        if "RateLimit" in _exc_name or "rate_limit" in str(exc).lower():
            _error_code = ErrorCode.LLM_RATE_LIMIT
        elif "Timeout" in _exc_name or "timeout" in str(exc).lower():
            _error_code = ErrorCode.LLM_TIMEOUT
        elif "Auth" in _exc_name or "Unauthorized" in _exc_name or "401" in str(exc):
            _error_code = ErrorCode.LLM_AUTH_ERROR
        elif "Quota" in _exc_name or "quota" in str(exc).lower():
            _error_code = ErrorCode.LLM_QUOTA_EXCEEDED
        else:
            _error_code = ErrorCode.LLM_INVALID_RESPONSE
        logger.warning(
            f"LLM {model_name} 调用出错: {exc}",
            _type="llm", model=model_name,
            duration_ms=round(duration_ms, 2),
            error_code=_error_code.value,
        )

        # Phase 0 关键改动：失败也 dispatch 一条 CompletionRecord，finish_reason=error
        try:
            error_record = CompletionRecord(
                trace_id=effective_trace_id,
                span_id=prompt_record.span_id,
                parent_span_id=effective_parent_span_id,
                agent_name=agent_type,
                model=model_name,
                finish_reason="error",
                content=str(exc)[:2000],
                tool_calls=None,
                completion_tokens=0,
                raw=effective_raw,
            )
            await dispatch_completion(error_record)
        except Exception:
            pass

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

    # Phase 0 关键改动：成功路径 dispatch CompletionRecord（含 tool_calls / finish_reason）
    try:
        tool_calls_serialized: list[dict] | None = None
        if hasattr(result, "tool_calls") and result.tool_calls:
            tool_calls_serialized = [
                {
                    "name": (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")),
                    "arguments": (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})),
                }
                for tc in result.tool_calls
            ]
            _called = [tc.get("name", tc.get("function", {}).get("name", "?")) if isinstance(tc, dict) else getattr(tc, "name", "?") for tc in result.tool_calls]
            logger.info(
                "LLM 返回 tool_calls: agent=%s tools=%s n=%s",
                agent_type, _called, len(_called),
            )
        else:
            logger.info(
                "LLM 调用完成（无 tool_calls）: agent=%s text_len=%s",
                agent_type, len(content),
            )
        completion_record = CompletionRecord(
            trace_id=effective_trace_id,
            span_id=prompt_record.span_id,
            parent_span_id=effective_parent_span_id,
            agent_name=agent_type,
            model=model_name,
            finish_reason=str(result.response_metadata.get("finish_reason", "stop") or "stop"),
            content=content,
            tool_calls=tool_calls_serialized,
            completion_tokens=completion_tokens,
            raw=effective_raw,
        )
        await dispatch_completion(completion_record)
    except Exception:
        pass

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
    on_reasoning: Optional[Callable] = None,
    call_config: dict | None = None,
) -> LLMResult:
    """流式调用 LLM，尝试从流中提取 token 用量，失败时估算。

    策略（按优先级）：
    1. 从最后一个 chunk 的 response_metadata 中提取实际 token 用量
    2. 降级：按字符数估算 prompt_tokens 和 completion_tokens

    Args:
        on_reasoning: 可选的 reasoning 回调，用于接收 LLM 的思考过程
        call_config: 可选的 per-call config dict，透传 temperature / max_tokens
    """
    raw_chunks: list[Any] = []
    full_content_chunks: list[str] = []

    # 流式调用：temperature / max_tokens 通过 astream(config={...}) 注入
    if call_config:
        stream_input = (messages, {"configurable": call_config})
    else:
        stream_input = (messages,)

    async for chunk in llm.astream(*stream_input):
        raw_chunks.append(chunk)

        # 检查 chunk 是否包含 reasoning 内容
        # LangChain 的 ChatOpenAI 流可能通过 additional_kwargs 或其他方式返回 reasoning
        reasoning_content = None
        if hasattr(chunk, "additional_kwargs"):
            additional = chunk.additional_kwargs or {}
            reasoning_content = additional.get("thinking") or additional.get("reasoning")

        # 处理 reasoning 回调
        if reasoning_content and on_reasoning:
            await on_reasoning(reasoning_content)

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
