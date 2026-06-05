"""LLM 网关客户端 — 统一经过 LLM Gateway（8001）

保持原有接口不变，内部实现从 LangChain 改为 httpx 直调 Gateway。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)

_GATEWAY_URL = "http://localhost:8001"


class LLMGateway:
    """统一 LLM 调用客户端 — 内部使用 httpx 调用 LLM Gateway。"""

    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        model: str = "",
        base_url: str = "",
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        gateway_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.gateway_url = gateway_url or _GATEWAY_URL
        from src.config import get_settings
        _settings = get_settings()
        self.default_temperature = (
            default_temperature if default_temperature is not None
            else _settings.default_temperature
        )
        self.default_max_tokens = (
            default_max_tokens if default_max_tokens is not None
            else _settings.default_max_tokens
        )
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.gateway_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Source-App": "autopublish-api",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        """将各种格式的消息标准化为 [{role, content}, ...]"""
        result = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            result.append({"role": role, "content": content})
        return result

    async def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """流式生成 — 通过 LLM Gateway SSE 流式响应。"""
        t = temperature if temperature is not None else self.default_temperature
        mt = max_tokens if max_tokens is not None else self.default_max_tokens
        client = self._get_client()

        trace_id = uuid.uuid4().hex[:16]
        payload = {
            "model": self.model or "auto",
            "messages": self._normalize_messages(messages),
            "temperature": t,
            "max_tokens": mt,
            "stream": True,
        }

        logger.info(
            "llm_request_start",
            **{
                "trace_id": trace_id,
                "event": "llm_request_start",
                "model": self.model or "auto",
                "provider": self.provider,
                "temperature": t,
                "max_tokens": mt,
                "stream": True,
                "message_count": len(messages),
            },
        )

        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers={"X-Trace-Id": trace_id},
            ) as response:
                response.raise_for_status()
                full_content = ""
                start = time.perf_counter()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                full_content += delta
                                yield delta
                        except json.JSONDecodeError:
                            continue

                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "llm_request_end",
                    **{
                        "trace_id": trace_id,
                        "event": "llm_request_end",
                        "model": self.model or "auto",
                        "provider": self.provider,
                        "llm_latency_ms": elapsed_ms,
                        "response_chars": len(full_content),
                        "status_code": 200,
                    },
                )

        except httpx.HTTPStatusError as e:
            logger.error(
                "llm_request_error",
                **{
                    "trace_id": trace_id,
                    "event": "llm_request_error",
                    "status_code": e.response.status_code,
                    "error_type": "HTTPStatusError",
                    "error_message": str(e),
                },
            )
            raise
        except Exception as e:
            logger.error(
                "llm_request_error",
                **{
                    "trace_id": trace_id,
                    "event": "llm_request_error",
                    "status_code": 500,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise

    async def generate_sync(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """同步生成 — 通过 LLM Gateway 获取完整响应。"""
        t = temperature if temperature is not None else self.default_temperature
        mt = max_tokens if max_tokens is not None else self.default_max_tokens
        client = self._get_client()

        trace_id = uuid.uuid4().hex[:16]
        payload = {
            "model": self.model or "auto",
            "messages": self._normalize_messages(messages),
            "temperature": t,
            "max_tokens": mt,
            "stream": False,
        }

        start = time.perf_counter()

        logger.info(
            "llm_request_start",
            **{
                "trace_id": trace_id,
                "event": "llm_request_start",
                "model": self.model or "auto",
                "provider": self.provider,
                "temperature": t,
                "max_tokens": mt,
                "stream": False,
                "message_count": len(messages),
            },
        )

        try:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            usage = data.get("usage", {})

            logger.info(
                "llm_request_end",
                **{
                    "trace_id": trace_id,
                    "event": "llm_request_end",
                    "model": self.model or "auto",
                    "provider": self.provider,
                    "llm_latency_ms": elapsed_ms,
                    "llm_prompt_tokens": usage.get("prompt_tokens", 0),
                    "llm_completion_tokens": usage.get("completion_tokens", 0),
                    "llm_total_tokens": usage.get("total_tokens", 0),
                    "status_code": 200,
                },
            )

            return content

        except httpx.HTTPStatusError as e:
            logger.error(
                "llm_request_error",
                **{
                    "trace_id": trace_id,
                    "event": "llm_request_error",
                    "status_code": e.response.status_code,
                    "error_type": "HTTPStatusError",
                    "error_message": str(e),
                },
            )
            raise
        except Exception as e:
            logger.error(
                "llm_request_error",
                **{
                    "trace_id": trace_id,
                    "event": "llm_request_error",
                    "status_code": 500,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise
