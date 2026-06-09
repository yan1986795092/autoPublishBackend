"""ModelRouter — 按 Agent 角色创建 LangChain ChatOpenAI 实例，指向 LLM Gateway"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.logging_shared.config import get_logger

logger = get_logger("core.model_router")

_GATEWAY_OPENAI_BASE_URL = "http://localhost:9010/v1"


class ModelRouter:
    """按角色创建并缓存 ChatOpenAI 实例，指向 LLM Gateway。"""

    def __init__(self, gateway_url: str | None = None):
        self._settings = get_settings()
        self._cache: dict[str, Any] = {}
        self._gateway_url = gateway_url or _GATEWAY_OPENAI_BASE_URL

    def _get_llm_config(self) -> tuple[str, str, str]:
        """Return (provider, model, base_url) from settings.

        base_url 统一指向 LLM Gateway。
        """
        s = self._settings
        for key, provider in [("openai", "openai"), ("zhipu", "zhipu"), ("qwen", "qwen")]:
            api_key = getattr(s, f"{key}_api_key", None)
            if api_key:
                model = getattr(s, f"{key}_model", "") or self._default_model(provider)
                return provider, model, self._gateway_url

        raise RuntimeError(
            "No LLM provider configured. Set OPENAI_API_KEY, ZHIPUAI_API_KEY, or QWEN_API_KEY."
        )

    def _default_model(self, provider: str) -> str:
        defaults = {"openai": "gpt-4o-mini", "zhipu": "glm-4-flash", "qwen": "qwen-plus"}
        return defaults.get(provider, "gpt-4o-mini")

    def get_chat_model(self, agent_type: str) -> ChatOpenAI:
        """返回 ChatOpenAI 实例（base_url 指向 LLM Gateway 的 OpenAI 兼容 `/v1` 根路径）。"""
        if agent_type in self._cache:
            return self._cache[agent_type]

        provider, model, base_url = self._get_llm_config()
        api_key = getattr(self._settings, f"{provider}_api_key", "")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            streaming=True,
            timeout=120.0,
        )

        self._cache[agent_type] = llm
        logger.info("ModelRouter 指向 LLM Gateway",
                    agent_type=agent_type, provider=provider, model=model, gateway=base_url)
        return llm

    def get_gateway(self, agent_type: str) -> ChatOpenAI:
        """Alias — 向后兼容。"""
        return self.get_chat_model(agent_type)

    def get_all_models(self) -> dict[str, ChatOpenAI]:
        agent_types = [
            "ManagerAgent", "WritingAgent", "DiagramAgent", "ImageAgent",
            "ReviewAgent", "SEOAgent", "HotTopicAgent", "CodeAgent",
            "WebSearch", "PlatformAdapter", "TranslationAgent",
        ]
        return {at: self.get_chat_model(at) for at in agent_types}

    def invalidate_cache(self) -> None:
        self._cache.clear()
