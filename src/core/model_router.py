"""ModelRouter: 按 Agent 角色分配不同的 LLM 配置。

每个 Agent 类型可以独立配置 provider/model/api_key/base_url，
未配置的角色会回退到 default_llm_provider 的配置。
"""
from __future__ import annotations
from typing import Optional

from src.config import get_settings
from src.core.llm_gateway import LLMGateway
from langchain_openai import ChatOpenAI


# Agent 类型 -> 配置角色名映射
ROLE_MAP: dict[str, str] = {
    "ManagerAgent": "manager",
    "WritingAgent": "writing",
    "DiagramAgent": "writing",       # 图表生成复用生文模型
    "ImageAgent": "image",
    "ReviewAgent": "subagent",
    "SEOAgent": "subagent",
    "HotTopicAgent": "subagent",
    "CodeAgent": "subagent",
    "WebSearch": "subagent",
    "PlatformAdapter": "subagent",
}


class ModelRouter:
    """按角色创建并缓存 LLMGateway 和 ChatOpenAI 实例。"""

    def __init__(self):
        self._settings = get_settings()
        self._cache: dict[str, LLMGateway] = {}
        self._chat_cache: dict[str, ChatOpenAI] = {}

    def get_gateway(self, agent_type: str) -> LLMGateway:
        role = ROLE_MAP.get(agent_type, "__default__")
        if role in self._cache:
            return self._cache[role]

        if role == "__default__":
            gateway = self._build_default()
        else:
            cfg = self._settings.get_role_config(role)
            gateway = LLMGateway(
                api_key=cfg["api_key"],
                provider=cfg["provider"],
                model=cfg["model"],
                base_url=cfg["base_url"],
            )
        self._cache[role] = gateway
        return gateway

    def get_chat_model(self, agent_type: str) -> ChatOpenAI:
        """返回 LangChain ChatOpenAI 实例（供 LangGraph 节点直接使用）"""
        role = ROLE_MAP.get(agent_type, "__default__")
        if role in self._chat_cache:
            return self._chat_cache[role]

        if role == "__default__":
            cfg = self._build_default_config()
        else:
            cfg = self._settings.get_role_config(role)

        model = ChatOpenAI(
            api_key=cfg["api_key"],
            model=cfg["model"] or "gpt-3.5-turbo",
            base_url=cfg["base_url"],
            temperature=0.7,
            max_tokens=2000,
            streaming=True,
        )
        self._chat_cache[role] = model
        return model

    def get_all_models(self) -> dict[str, ChatOpenAI]:
        """返回所有角色的 ChatOpenAI 实例字典（供 LangGraph 调用）"""
        import logging
        logger = logging.getLogger(__name__)
        models = {}
        for agent_type in ROLE_MAP:
            try:
                models[agent_type] = self.get_chat_model(agent_type)
            except Exception as e:
                logger.warning(f"Failed to create model for {agent_type}: {e}")
        return models

    def _build_default_config(self) -> dict:
        """构建默认配置 dict（不创建 LLMGateway）"""
        settings = self._settings
        provider = settings.default_llm_provider
        api_key = getattr(settings, f"{provider}_api_key", "")
        if not api_key:
            for p in ["openai", "zhipu", "qwen"]:
                api_key = getattr(settings, f"{p}_api_key", "")
                if api_key:
                    provider = p
                    break
        return {
            "api_key": api_key,
            "provider": provider,
            "model": getattr(settings, f"{provider}_model", ""),
            "base_url": getattr(settings, f"{provider}_base_url", ""),
        }

    def _build_default(self) -> LLMGateway:
        settings = self._settings
        provider = settings.default_llm_provider
        api_key = getattr(settings, f"{provider}_api_key", "")
        if not api_key:
            for p in ["openai", "zhipu", "qwen"]:
                api_key = getattr(settings, f"{p}_api_key", "")
                if api_key:
                    provider = p
                    break
        return LLMGateway(
            api_key=api_key,
            provider=provider,
            model=getattr(settings, f"{provider}_model", ""),
            base_url=getattr(settings, f"{provider}_base_url", ""),
        )

    def invalidate_cache(self) -> None:
        self._cache.clear()
