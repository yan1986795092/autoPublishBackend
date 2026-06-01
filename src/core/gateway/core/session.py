"""GatewaySession — LangChain-compatible wrapper around the full Gateway pipeline.

Handlers and tool_utils.call_llm() depend on exactly three methods:
  - ainvoke(messages)      → AIMessage with .content (str) and .tool_calls (list)
  - astream(messages)      → AsyncIterator of AIMessageChunk with .content (str)
  - bind_tools(tools)      → returns NEW GatewaySession with tools bound

GatewaySession implements all three, routing each call through:
  1. EnhancedRouter.select()      → provider/model selection
  2. RateLimiter.check()          → RPM/TPM enforcement
  3. CircuitBreaker.is_available() → failure isolation (via LiteLLMClient)
  4. LiteLLMClient.completion()   → actual LLM call with OTel spans
  5. LatencyTracker.record()      → P95 feedback loop
  6. CostTracker.record()         → usage persistence
"""

from __future__ import annotations

import json
import logging
import math
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool

from .router import EnhancedRouter, ModelCandidate
from ..config import get_gateway_settings

logger = logging.getLogger(__name__)

# Rough token estimation: ~3 characters per token on average
_TOKEN_CHARS_PER_TOKEN = 3


# ── Conversion helpers ────────────────────────────────────────────────────

def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count from message content for rate limiting."""
    total = 0
    for m in messages:
        content = str(m.get("content", ""))
        total += math.ceil(len(content) / _TOKEN_CHARS_PER_TOKEN)
    return max(total, 1)


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain BaseMessage list to dict list for LiteLLMClient."""
    result = []
    for m in messages:
        role = getattr(m, "type", "user")
        content = getattr(m, "content", "")
        msg_dict = {"role": role, "content": content}

        # Preserve tool_call_id for ToolMessage
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            msg_dict["tool_call_id"] = tool_call_id

        # Preserve tool_calls for AIMessage (needed for multi-turn tool loops)
        tc = getattr(m, "tool_calls", None)
        if tc:
            msg_dict["tool_calls"] = tc

        result.append(msg_dict)
    return result


def _convert_tools_to_openai(tools: list[BaseTool]) -> list[dict]:
    """Convert LangChain BaseTool list to OpenAI function-calling format.

    LangChain BaseTool:
      - .name: str
      - .description: str
      - .args_schema: Pydantic model (has .schema() method)

    OpenAI format:
      [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    """
    openai_tools = []
    for tool in tools:
        try:
            if hasattr(tool, "args_schema") and tool.args_schema is not None:
                parameters = tool.args_schema.schema()
                # Remove Pydantic-specific keys that OpenAI doesn't expect
                parameters.pop("title", None)
                parameters.pop("description", None)
            else:
                parameters = {"type": "object", "properties": {}}
        except Exception:
            parameters = {"type": "object", "properties": {}}

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        })
    return openai_tools


def _parse_tool_calls(response: Any) -> list[dict]:
    """Parse OpenAI-format tool_calls from litellm response to LangChain format.

    Litellm returns tool_calls like OpenAI:
      response.choices[0].message.tool_calls = [
        {"id": "call_xxx", "function": {"name": "...", "arguments": "{...}"}, "type": "function"}
      ]

    LangChain AIMessage.tool_calls format (consumed by _get_tool_calls in tool_utils.py):
      [{"name": "...", "args": {...}, "id": "..."}]
    """
    if not hasattr(response, "choices") or not response.choices:
        return []

    msg = response.choices[0].message
    raw_calls = getattr(msg, "tool_calls", None)
    if not raw_calls:
        return []

    result = []
    for tc in raw_calls:
        # Handle both dict and object forms from litellm
        if isinstance(tc, dict):
            func = tc.get("function", {})
            name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
            args_str = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
            tc_id = tc.get("id", "")
        else:
            func = getattr(tc, "function", None)
            name = getattr(func, "name", "") if func else ""
            args_str = getattr(func, "arguments", "{}") if func else "{}"
            tc_id = getattr(tc, "id", "")

        # Parse JSON arguments string
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": str(args_str)}

        result.append({
            "name": name,
            "args": args,
            "id": tc_id,
        })

    return result


# ── GatewaySession ────────────────────────────────────────────────────────

class GatewaySession:
    """LangChain-compatible wrapper around the full Gateway pipeline.

    Drop-in replacement for ChatOpenAI in the Agent system. All handlers
    and tool_utils.call_llm() use this exactly like ChatOpenAI — they call
    ainvoke(), astream(), and bind_tools() without knowing the difference.

    Each call goes through the full Gateway pipeline:
      router → rate limiter → circuit breaker → litellm → cost tracker
    """

    def __init__(
        self,
        agent_type: str,
        candidates: list[ModelCandidate],
        container: Any,  # GatewayContainer (lazy import to avoid circular deps)
        hint: str = "balanced",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        source_app: str = "agent-service",
    ):
        self.agent_type = agent_type
        self.candidates = list(candidates)  # shallow copy
        self._container = container
        self.hint = hint
        self.temperature = temperature
        self._max_tokens = max_tokens
        self.source_app = source_app
        self._bound_tools: Optional[list[dict]] = None  # OpenAI-format tools

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def max_tokens(self) -> int:
        """Exposed for Token Budget checks (nodes.py:735 uses getattr)."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self._max_tokens = value

    # ── bind_tools — MUST return a new instance ────────────────────────

    def bind_tools(self, tools: list[BaseTool]) -> "GatewaySession":
        """Return a NEW GatewaySession with tools bound in OpenAI format.

        IMPORTANT: Returns a clone, not self. This is required because
        call_llm() uses the original (unbound) llm for _stream_final()
        after the tool_llm (bound) completes the tool loop.

        See: tool_utils.py call_llm() — tool_llm = bind_tools(llm, ...)
             then later _stream_final(llm, ...) uses the ORIGINAL llm.
        """
        clone = GatewaySession(
            agent_type=self.agent_type,
            candidates=self.candidates,
            container=self._container,
            hint=self.hint,
            temperature=self.temperature,
            max_tokens=self._max_tokens,
            source_app=self.source_app,
        )
        clone._bound_tools = _convert_tools_to_openai(tools)
        return clone

    # ── generate_sync / generate — orchestrator backward compat ──────────

    async def generate_sync(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Synchronous (non-streaming) completion with plain dict messages.

        Backward-compatible with LLMGateway.generate_sync() used by
        Orchestrator._decompose() and Orchestrator._fallback_llm_call().

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            temperature: Override session temperature if provided
            max_tokens: Override session max_tokens if provided

        Returns:
            Response text string
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage as LcAIMessage

        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(LcAIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Temporarily override temperature/max_tokens if provided
        orig_temp = self.temperature
        orig_max = self._max_tokens
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self._max_tokens = max_tokens

        try:
            result = await self.ainvoke(lc_messages)
            return result.content or ""
        finally:
            self.temperature = orig_temp
            self._max_tokens = orig_max

    async def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
    ):
        """Streaming completion with plain dict messages.

        Backward-compatible with LLMGateway.generate().
        Not currently used by orchestrator but kept for compatibility.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            temperature: Override session temperature
            max_tokens: Override session max_tokens
            stream: If True, yields text chunks; if False, yields full response
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage as LcAIMessage

        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(LcAIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        orig_temp = self.temperature
        orig_max = self._max_tokens
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self._max_tokens = max_tokens

        try:
            if stream:
                async for chunk in self.astream(lc_messages):
                    yield chunk.content
            else:
                result = await self.ainvoke(lc_messages)
                yield result.content or ""
        finally:
            self.temperature = orig_temp
            self._max_tokens = orig_max

    # ── ainvoke — non-streaming completion ─────────────────────────────

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """Non-streaming completion through the full Gateway pipeline.

        Args:
            messages: LangChain BaseMessage list (SystemMessage, HumanMessage, etc.)
            **kwargs: Additional args (ignored; kept for LangChain compatibility)

        Returns:
            AIMessage with .content (str) and .tool_calls (list[dict])
        """
        messages_dict = _messages_to_dicts(messages)
        token_estimate = _estimate_tokens(messages_dict)
        start_time = time.time()

        # 1. Route — select best provider/model
        candidate = await self._select_candidate()

        # 2. Rate limit
        await self._check_rate_limit(candidate, token_estimate)

        # 3. Call LiteLLMClient
        response = await self._call_completion(candidate, messages_dict, stream=False)

        # 4. Parse response
        content = ""
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", "") or ""

        tool_calls = _parse_tool_calls(response)

        # 5. Record cost + latency feedback
        latency_ms = (time.time() - start_time) * 1000
        await self._record_usage(candidate, response, token_estimate, latency_ms)

        # 6. Build AIMessage (compatible with _get_tool_calls in tool_utils.py)
        return AIMessage(
            content=content,
            tool_calls=tool_calls if tool_calls else [],
        )

    # ── astream — streaming completion ─────────────────────────────────

    async def astream(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Streaming completion through the full Gateway pipeline.

        Args:
            messages: LangChain BaseMessage list
            **kwargs: Additional args (ignored)

        Yields:
            AIMessageChunk with .content (str) for each text delta
        """
        messages_dict = _messages_to_dicts(messages)
        token_estimate = _estimate_tokens(messages_dict)
        start_time = time.time()

        # 1. Route
        candidate = await self._select_candidate()

        # 2. Rate limit
        await self._check_rate_limit(candidate, token_estimate)

        # 3. Call LiteLLMClient (streaming)
        response = await self._call_completion(candidate, messages_dict, stream=True)

        full_content = ""
        try:
            async for chunk in response:
                delta = ""
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = getattr(chunk.choices[0].delta, "content", "") or ""
                    if delta is None:
                        delta = ""
                elif isinstance(chunk, str):
                    delta = chunk

                if delta:
                    full_content += delta
                    yield AIMessageChunk(content=delta)
        finally:
            # Always record cost, even if streaming is interrupted
            latency_ms = (time.time() - start_time) * 1000
            completion_tokens = max(math.ceil(len(full_content) / _TOKEN_CHARS_PER_TOKEN), 1)
            await self._record_usage_estimated(
                candidate,
                prompt_tokens=token_estimate,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

    # ── Internal pipeline steps ────────────────────────────────────────

    async def _select_candidate(self) -> ModelCandidate:
        """Select the best available model via EnhancedRouter.

        Creates a fresh EnhancedRouter per call (lightweight — the real state
        lives in the shared circuit_breaker and latency_tracker).
        """
        cb = self._container.circuit_breaker
        router = EnhancedRouter(
            candidates=self.candidates,
            circuit_breaker=cb,
            weights=self._container.settings.router_default_weights,
            hints=self._container.settings.router_hints,
            latency_tracker=self._container.latency_tracker,
        )
        candidate = await router.select(self.hint)
        if candidate is None:
            raise RuntimeError(
                f"GatewaySession[{self.agent_type}]: all providers unavailable. "
                "Check circuit breaker status and API key configuration."
            )
        return candidate

    async def _check_rate_limit(self, candidate: ModelCandidate, token_estimate: int) -> None:
        """Check rate limits for the selected provider."""
        result = await self._container.rate_limiter.check(
            provider=candidate.provider,
            source_app=self.source_app,
            token_count=token_estimate,
        )
        if not result.allowed:
            raise RuntimeError(
                f"GatewaySession[{self.agent_type}]: rate limit exceeded "
                f"for {candidate.provider}. Wait {result.wait_seconds:.1f}s."
            )

    async def _call_completion(
        self,
        candidate: ModelCandidate,
        messages_dict: list[dict],
        stream: bool = False,
    ) -> Any:
        """Call LiteLLMClient.completion() with the selected candidate."""
        extra_kwargs = {}
        if self._bound_tools:
            extra_kwargs["tools"] = self._bound_tools

        return await self._container.litellm_client.completion(
            model=candidate.model,
            messages=messages_dict,
            api_key=candidate.api_key,
            base_url=candidate.base_url,
            temperature=self.temperature,
            max_tokens=self._max_tokens,
            stream=stream,
            timeout=self._container.settings.global_request_timeout,
            source_app=self.source_app,
            gateway_hint=self.hint,
            provider=candidate.provider,
            **extra_kwargs,
        )

    async def _record_usage(
        self,
        candidate: ModelCandidate,
        response: Any,
        token_estimate: int,
        latency_ms: float,
    ) -> None:
        """Extract actual usage from litellm response and record to CostTracker."""
        from .cost_tracker import CostRecord

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", token_estimate) if usage else token_estimate
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        await self._container.latency_tracker.record(candidate.model, latency_ms)
        await self._container.circuit_breaker.record_success(candidate.model)

        await self._container.cost_tracker.record(CostRecord(
            timestamp=datetime.now(),
            trace_id="",
            source_app=self.source_app,
            provider=candidate.provider,
            model=candidate.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            estimated_cost=self._container.cost_tracker.calculate_cost(
                candidate.model, prompt_tokens, completion_tokens,
            ),
        ))

    async def _record_usage_estimated(
        self,
        candidate: ModelCandidate,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record estimated usage for streaming (no usage object in stream)."""
        from .cost_tracker import CostRecord

        await self._container.latency_tracker.record(candidate.model, latency_ms)
        await self._container.circuit_breaker.record_success(candidate.model)

        await self._container.cost_tracker.record(CostRecord(
            timestamp=datetime.now(),
            trace_id="",
            source_app=self.source_app,
            provider=candidate.provider,
            model=candidate.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            estimated_cost=self._container.cost_tracker.calculate_cost(
                candidate.model, prompt_tokens, completion_tokens,
            ),
        ))
