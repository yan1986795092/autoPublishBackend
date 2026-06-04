"""ModelRouter — 按 Agent 角色创建 LangChain ChatOpenAI 实例。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

# Provider → base_url 映射
_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


class ModelRouter:
    """按角色创建并缓存 ChatOpenAI 实例。"""

    def __init__(self):
        self._settings = get_settings()
        self._cache: dict[str, Any] = {}

    def _get_llm_config(self) -> tuple[str, str, str]:
        """Return (provider, model, base_url) from settings."""
        s = self._settings
        for key, provider in [("openai", "openai"), ("zhipu", "zhipu"), ("qwen", "qwen")]:
            api_key = getattr(s, f"{key}_api_key", None)
            if api_key:
                model = getattr(s, f"{key}_model", "") or self._default_model(provider)
                base_url = getattr(s, f"{key}_base_url", "") or _PROVIDER_BASE_URLS.get(provider, "")
                return provider, model, base_url

        raise RuntimeError(
            "No LLM provider configured. Set OPENAI_API_KEY, ZHIPUAI_API_KEY, or QWEN_API_KEY."
        )

    def _default_model(self, provider: str) -> str:
        defaults = {"openai": "gpt-4o-mini", "zhipu": "glm-4-flash", "qwen": "qwen-plus"}
        return defaults.get(provider, "gpt-4o-mini")

    def get_chat_model(self, agent_type: str) -> ChatOpenAI:
        """返回 ChatOpenAI 实例（替代 ChatOpenAI）。"""
        if agent_type in self._cache:
            return self._cache[agent_type]

        provider, model, base_url = self._get_llm_config()
        api_key = getattr(self._settings, f"{provider}_api_key", "")

        # streaming=True 保证 astream() 正常输出。
        # 非流式 ainvoke() 虽然会多一点点开销，但主要提速靠取消 WritingAgent 重试。
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url or None,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            streaming=True,
        )

        self._cache[agent_type] = llm
        logger.info(
            "ModelRouter: created ChatOpenAI for %s (provider=%s, model=%s)",
            agent_type, provider, model,
        )
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
