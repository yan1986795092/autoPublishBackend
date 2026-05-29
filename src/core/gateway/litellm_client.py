import litellm
from typing import Any, Optional, AsyncGenerator
from opentelemetry import trace


class GatewayCallbacks:
    def __init__(
        self,
        on_success=None,
        on_failure=None,
        latency_tracker=None,
    ):
        self.on_success = on_success
        self.on_failure = on_failure
        self.latency_tracker = latency_tracker

    async def async_success_handler(self, kwargs: dict, response_obj: Any, start_time: float, end_time: float):
        latency_ms = (end_time - start_time) * 1000
        model = kwargs.get("model", "")
        if self.latency_tracker:
            await self.latency_tracker.record(model, latency_ms)
        if self.on_success:
            await self.on_success(kwargs, response_obj, latency_ms)

    async def async_failure_handler(self, kwargs: dict, exception: Exception, start_time: float, end_time: float):
        model = kwargs.get("model", "")
        if self.on_failure:
            await self.on_failure(model, exception)


class LiteLLMClient:
    def __init__(
        self,
        success_callback=None,
        failure_callback=None,
    ):
        litellm.drop_params = True
        litellm.set_verbose = False
        self.success_callback = success_callback
        self.failure_callback = failure_callback
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
        **kwargs,
    ):
        span = trace.get_current_span()
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.api_base", base_url)
        span.set_attribute("llm.stream", stream)

        litellm_params = {
            "model": model,
            "messages": messages,
            "api_key": api_key,
            "api_base": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "timeout": timeout,
            **kwargs,
        }
        span.add_event("llm.call_start")
        try:
            response = await litellm.acompletion(**litellm_params)
            span.add_event("llm.call_end")
            span.set_attribute("llm.status", "success")
            return response
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("llm.status", "error")
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
        **kwargs,
    ):
        span = trace.get_current_span()
        span.set_attribute("llm.model", model)
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
            span.set_attribute("llm.status", "success")
            return response
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("llm.status", "error")
            raise
