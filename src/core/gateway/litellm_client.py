import json
import time
from typing import Any, Optional, AsyncGenerator

import structlog
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

import litellm

# Lazy imports — observability may not be fully configured at import time
# (e.g., elasticsearch not installed in dev environment)
_tracer = None
_get_tracer_fn = None
_record_gateway_request = None
_record_gateway_error = None

logger = structlog.get_logger(__name__)


def _get_tracer():
    global _tracer, _get_tracer_fn
    if _get_tracer_fn is None:
        from src.observability import get_tracer as _fn
        _get_tracer_fn = _fn
    if _tracer is None:
        _tracer = _get_tracer_fn("gateway.litellm")
    return _tracer


def _do_record_gateway_request(*args, **kwargs):
    global _record_gateway_request
    if _record_gateway_request is None:
        from src.observability.metrics import record_gateway_request as _fn
        _record_gateway_request = _fn
    return _record_gateway_request(*args, **kwargs)


def _do_record_gateway_error(*args, **kwargs):
    global _record_gateway_error
    if _record_gateway_error is None:
        from src.observability.metrics import record_gateway_error as _fn
        _record_gateway_error = _fn
    return _record_gateway_error(*args, **kwargs)


class GatewayCallbacks:
    """LiteLLM success/failure callbacks — also emit OTel spans and Prometheus metrics."""

    def __init__(
        self,
        on_success=None,
        on_failure=None,
        latency_tracker=None,
        circuit_breaker=None,
        rate_limiter=None,
        source_app: str = "default",
    ):
        self.on_success = on_success
        self.on_failure = on_failure
        self.latency_tracker = latency_tracker
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.source_app = source_app

    async def async_success_handler(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
        model: str,
        provider: str,
        trace_id: str = "",
    ):
        latency_ms = (end_time - start_time) * 1000
        if self.latency_tracker:
            await self.latency_tracker.record(model, latency_ms)
        if self.circuit_breaker:
            await self.circuit_breaker.record_success(model)
        if self.on_success:
            await self.on_success(kwargs, response_obj, latency_ms)

    async def async_failure_handler(
        self,
        kwargs: dict,
        exception: Exception,
        start_time: float,
        end_time: float,
        model: str,
        provider: str,
        trace_id: str = "",
    ):
        if self.circuit_breaker:
            await self.circuit_breaker.record_failure(model)
        if self.on_failure:
            await self.on_failure(model, exception)


class LiteLLMClient:
    """LiteLLM wrapper with full OTel tracing + Prometheus metrics.

    Creates spans for:
      - model.request (parent)
      - model.router
      - model.circuit_breaker
      - model.rate_limit
    """

    def __init__(
        self,
        success_callback=None,
        failure_callback=None,
        circuit_breaker=None,
        rate_limiter=None,
    ):
        litellm.drop_params = True
        litellm.set_verbose = False
        self.success_callback = success_callback
        self.failure_callback = failure_callback
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        if success_callback:
            litellm.success_callback.append(success_callback.async_success_handler)
        if failure_callback:
            litellm.failure_callback.append(failure_callback.async_failure_handler)

    async def completion(
        self,
        model: str,
        messages: list[dict],
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        timeout: int = 60,
        source_app: str = "default",
        gateway_hint: str = "balanced",
        routing_decision: dict | None = None,
        selected_model: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        tracer = _get_tracer()
        start_time = time.perf_counter()

        # Extract actual model from routing decision if provided
        actual_model = selected_model or model
        actual_provider = provider
        if routing_decision:
            actual_provider = routing_decision.get("provider", provider)
            actual_model = routing_decision.get("model", actual_model)

        # ── [插桩] model.request span ──────────────────────────────────────
        with tracer.start_as_current_span(
            "model.request",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": actual_model,
                "gen_ai.request.temperature": temperature,
                "gen_ai.request.max_tokens": max_tokens,
                "gen_ai.request.stream": stream,
                "custom.provider": actual_provider,
                "custom.source_app": source_app,
                "custom.gateway_hint": gateway_hint,
                "llm.model_requested": model,
            },
        ) as request_span:
            # ── [插桩] model.circuit_breaker span ───────────────────────
            cb_state = "closed"
            cb_allowed = True
            if self.circuit_breaker:
                with tracer.start_as_current_span(
                    "model.circuit_breaker",
                    kind=trace.SpanKind.INTERNAL,
                    attributes={
                        "circuit_breaker.model": actual_model,
                        "circuit_breaker.provider": actual_provider,
                    },
                ) as cb_span:
                    cb_allowed = self.circuit_breaker.is_available(actual_model)
                    cb_state = self.circuit_breaker.get_state(actual_model)
                    cb_span.set_attribute("circuit_breaker.state", cb_state)
                    cb_span.set_attribute("circuit_breaker.allowed", cb_allowed)
                    cb_span.set_status(Status(StatusCode.OK))

                    if not cb_allowed:
                        msg = f"Circuit breaker OPEN for {actual_model}"
                        request_span.set_status(Status(StatusCode.ERROR, msg))
                        raise RuntimeError(msg)

            # ── [插桩] model.rate_limit span ────────────────────────────
            rl_result = None
            if self.rate_limiter:
                with tracer.start_as_current_span(
                    "model.rate_limit",
                    kind=trace.SpanKind.INTERNAL,
                    attributes={
                        "rate_limit.provider": actual_provider,
                        "rate_limit.source_app": source_app,
                    },
                ) as rl_span:
                    # Estimate token count for rate limiting
                    est_tokens = sum(
                        len(json.dumps(m.get("content", ""), ensure_ascii=False)) // 4
                        for m in messages
                    )
                    rl_result = await self.rate_limiter.check(
                        provider=actual_provider,
                        source_app=source_app,
                        token_count=est_tokens,
                    )
                    rl_span.set_attribute("rate_limit.allowed", rl_result.allowed)
                    rl_span.set_attribute("rate_limit.remaining_rpm", rl_result.remaining_rpm)
                    rl_span.set_attribute("rate_limit.remaining_tpm", rl_result.remaining_tpm)
                    rl_span.set_attribute("rate_limit.wait_seconds", rl_result.wait_seconds)
                    rl_span.set_status(Status(StatusCode.OK))

                    if not rl_result.allowed:
                        msg = f"Rate limit exceeded (wait {rl_result.wait_seconds}s)"
                        request_span.set_status(Status(StatusCode.ERROR, msg))
                        raise RuntimeError(msg)

            # ── Actual LLM call ───────────────────────────────────────────
            litellm_params = {
                "model": actual_model,
                "messages": messages,
                "api_key": api_key,
                "api_base": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "timeout": timeout,
                **kwargs,
            }

            request_span.add_event("llm.call_start", {"model": actual_model})
            try:
                response = await litellm.acompletion(**litellm_params)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Extract usage info
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

                # Record to span
                request_span.set_attribute("gen_ai.response.model", getattr(response, "model", actual_model))
                request_span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                request_span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                request_span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
                request_span.set_attribute("gen_ai.response.latency_ms", elapsed_ms)
                request_span.set_attribute("llm.circuit_breaker_state", cb_state)
                request_span.set_status(Status(StatusCode.OK))
                request_span.add_event("llm.call_end")

                # Prometheus metrics
                _do_record_gateway_request(
                    source_app=source_app,
                    provider=actual_provider,
                    model=actual_model,
                    status="success",
                    status_code=200,
                    duration_s=elapsed_ms / 1000,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=0.0,
                )

                # Callbacks
                if self.success_callback:
                    await self.success_callback.async_success_handler(
                        litellm_params, response, start_time, time.perf_counter(),
                        actual_model, actual_provider,
                    )

                return response

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                request_span.record_exception(e)
                request_span.set_status(Status(StatusCode.ERROR, str(e)))
                request_span.set_attribute("llm.error_type", type(e).__name__)

                _do_record_gateway_error(
                    provider=actual_provider,
                    model=actual_model,
                    error_type=type(e).__name__,
                )
                _do_record_gateway_request(
                    source_app=source_app,
                    provider=actual_provider,
                    model=actual_model,
                    status="error",
                    status_code=500,
                    duration_s=elapsed_ms / 1000,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                )

                if self.failure_callback:
                    await self.failure_callback.async_failure_handler(
                        litellm_params, e, start_time, time.perf_counter(),
                        actual_model, actual_provider,
                    )

                raise

    def completion_sync(
        self,
        model: str,
        messages: list[dict],
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        source_app: str = "default",
        provider: str = "openai",
        **kwargs,
    ):
        tracer = _get_tracer()
        start_time = time.perf_counter()

        with tracer.start_as_current_span(
            "model.request.sync",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": model,
                "custom.provider": provider,
                "custom.source_app": source_app,
            },
        ) as request_span:
            litellm_params = {
                "model": model,
                "messages": messages,
                "api_key": api_key,
                "api_base": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                **kwargs,
            }
            try:
                response = litellm.completion(**litellm_params)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                request_span.set_attribute("gen_ai.response.latency_ms", elapsed_ms)
                request_span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                request_span.record_exception(e)
                request_span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
