import time
import asyncio
import math
import structlog
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from opentelemetry import trace

from ..config import get_gateway_settings
from ..core.router import EnhancedRouter, ModelCandidate, LatencyTracker
from ..core.circuit_breaker import CircuitBreaker
from ..core.rate_limiter import MultiDimensionalRateLimiter
from ..core.security import validate_request_body, mask_message
from ..core.cost_tracker import CostRecord, CostTracker
from ..litellm_client import GatewayCallbacks, LiteLLMClient

logger = structlog.get_logger()
router = APIRouter(prefix="/v1", tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False
    gateway_hint: Optional[str] = Field(None, description="simple | complex | balanced")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# Rough token estimation: ~3 chars per token on average.
_TOKEN_CHARS_PER_TOKEN = 3


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = str(m.get("content", ""))
        total += math.ceil(len(content) / _TOKEN_CHARS_PER_TOKEN)
    return max(total, 1)


def _get_trace_id(http_request: Request) -> str:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    header_id = http_request.headers.get("X-Trace-ID", "")
    if header_id:
        return header_id
    return uuid.uuid4().hex[:16]


def _build_router(
    settings,
    cb: CircuitBreaker,
    latency_tracker: LatencyTracker,
) -> EnhancedRouter:
    candidates = []
    if settings.openai_api_key:
        candidates.append(ModelCandidate(
            provider="openai",
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            input_price=settings.pricing.get(settings.openai_model, {}).get("input", 0.002),
            output_price=settings.pricing.get(settings.openai_model, {}).get("output", 0.002),
            static_latency_ms=1000,
        ))
    if settings.zhipu_api_key:
        candidates.append(ModelCandidate(
            provider="zhipuai",
            model=settings.zhipu_model,
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
            input_price=settings.pricing.get(settings.zhipu_model, {}).get("input", 0.0005),
            output_price=settings.pricing.get(settings.zhipu_model, {}).get("output", 0.0005),
            static_latency_ms=800,
        ))
    if settings.qwen_api_key:
        candidates.append(ModelCandidate(
            provider="qwen",
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            input_price=settings.pricing.get(settings.qwen_model, {}).get("input", 0.004),
            output_price=settings.pricing.get(settings.qwen_model, {}).get("output", 0.012),
            static_latency_ms=1200,
        ))
    return EnhancedRouter(
        candidates=candidates,
        circuit_breaker=cb,
        weights=settings.router_default_weights,
        hints=settings.router_hints,
        latency_tracker=latency_tracker,
    )


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
):
    settings = get_gateway_settings()
    span = trace.get_current_span()
    source_app = http_request.headers.get("X-Source-App", "default")
    trace_id = _get_trace_id(http_request)

    span.set_attribute("source_app", source_app)
    span.set_attribute("gateway_hint", request.gateway_hint or "balanced")
    span.set_attribute("model", request.model)
    span.set_attribute("trace_id", trace_id)

    logger.info(
        "chat_completion_request",
        source_app=source_app,
        model=request.model,
        gateway_hint=request.gateway_hint,
        message_count=len(request.messages),
        masked_messages=[mask_message(m.content) for m in request.messages[:3]],
    )

    error = validate_request_body(
        request.model_dump(),
        settings.allowed_models,
        settings.max_content_length,
    )
    if error:
        raise HTTPException(status_code=400, detail=error)

    enh_router = _build_router(
        settings,
        http_request.state.circuit_breaker,
        http_request.state.latency_tracker,
    )
    rate_limiter: MultiDimensionalRateLimiter = http_request.state.rate_limiter
    cost_tracker: CostTracker = http_request.state.cost_tracker
    litellm_client: LiteLLMClient = http_request.state.litellm_client

    token_estimate = _estimate_tokens([m.model_dump() for m in request.messages])

    # Respect the requested model if allowed and available, otherwise fall back to router.
    candidate: Optional[ModelCandidate] = None
    if request.model in settings.allowed_models:
        candidate = next(
            (c for c in enh_router.candidates
             if c.model == request.model and enh_router.cb.is_available(c.model)),
            None,
        )

    if candidate is None:
        candidate = await enh_router.select(request.gateway_hint)

    if not candidate:
        raise HTTPException(
            status_code=503,
            detail="All providers are currently unavailable. Please retry later.",
            headers={"Retry-After": "30"},
        )
    span.set_attribute("selected_provider", candidate.provider)
    span.set_attribute("selected_model", candidate.model)

    rate_result = await rate_limiter.check(candidate.provider, source_app, token_estimate)
    if not rate_result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for provider '{candidate.provider}'. Retry after {rate_result.wait_seconds:.1f}s.",
            headers={"Retry-After": str(int(rate_result.wait_seconds) + 1)},
        )
    span.set_attribute("rate_limiter_wait_ms", rate_result.wait_seconds * 1000)

    start_time = time.time()
    try:
        response = await asyncio.wait_for(
            litellm_client.completion(
                model=candidate.model,
                messages=[m.model_dump() for m in request.messages],
                api_key=candidate.api_key,
                base_url=candidate.base_url,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 2000,
                stream=request.stream or False,
                timeout=settings.global_request_timeout,
            ),
            timeout=settings.global_request_timeout,
        )

        if request.stream:
            async def stream_wrapper():
                full_content = ""
                try:
                    async for chunk in response:
                        content = getattr(chunk.choices[0].delta, "content", "") if hasattr(chunk, "choices") else str(chunk)
                        full_content += content
                        yield f"data: {chunk.model_dump_json()}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                finally:
                    latency_ms = (time.time() - start_time) * 1000
                    completion_chars = len(full_content)
                    completion_tokens = max(math.ceil(completion_chars / _TOKEN_CHARS_PER_TOKEN), 1)
                    await enh_router.record_success(candidate.model, latency_ms)
                    await cost_tracker.record(CostRecord(
                        timestamp=datetime.now(),
                        trace_id=trace_id,
                        source_app=source_app,
                        provider=candidate.provider,
                        model=candidate.model,
                        prompt_tokens=token_estimate,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        estimated_cost=cost_tracker.calculate_cost(candidate.model, token_estimate, completion_tokens),
                    ))
                    logger.info(
                        "chat_completion_stream_done",
                        provider=candidate.provider,
                        model=candidate.model,
                        latency_ms=latency_ms,
                    )

            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
        else:
            latency_ms = (time.time() - start_time) * 1000
            await enh_router.record_success(candidate.model, latency_ms)

            msg_content = response.choices[0].message.content
            completion_chars = len(msg_content)
            completion_tokens = max(math.ceil(completion_chars / _TOKEN_CHARS_PER_TOKEN), 1)

            await cost_tracker.record(CostRecord(
                timestamp=datetime.now(),
                trace_id=trace_id,
                source_app=source_app,
                provider=candidate.provider,
                model=candidate.model,
                prompt_tokens=token_estimate,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                estimated_cost=cost_tracker.calculate_cost(candidate.model, token_estimate, completion_tokens),
            ))

            usage = response.usage if hasattr(response, "usage") and response.usage else None
            actual_prompt = usage.prompt_tokens if usage else token_estimate
            actual_completion = usage.completion_tokens if usage else completion_tokens

            logger.info(
                "chat_completion_done",
                provider=candidate.provider,
                model=candidate.model,
                latency_ms=latency_ms,
                total_tokens=actual_prompt + actual_completion,
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{trace_id[:8]}",
                created=int(time.time()),
                model=candidate.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": msg_content},
                    finish_reason="stop",
                )],
                usage=Usage(
                    prompt_tokens=actual_prompt,
                    completion_tokens=actual_completion,
                    total_tokens=actual_prompt + actual_completion,
                ),
            )

    except asyncio.TimeoutError:
        await enh_router.record_failure(candidate.model)
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except Exception as e:
        await enh_router.record_failure(candidate.model)
        logger.error(
            "chat_completion_error",
            error=str(e),
            provider=candidate.provider,
            model=candidate.model,
        )
        raise HTTPException(status_code=500, detail=str(e))
