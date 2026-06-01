"""ModelRouter: 按 Agent 角色创建 GatewaySession 实例。

每个 GatewaySession 通过完整的 Gateway 管线调用 LLM：
  EnhancedRouter → CircuitBreaker → RateLimiter → LiteLLMClient → CostTracker
"""

from __future__ import annotations
import logging
from typing import Any, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)

# Agent 类型 → routing hint 映射（决定 EnhancedRouter 的评分权重）
HINT_MAP: dict[str, str] = {
    "ManagerAgent": "complex",
    "WritingAgent": "complex",
    "DiagramAgent": "simple",
    "ImageAgent": "simple",
    "ReviewAgent": "simple",
    "SEOAgent": "simple",
    "HotTopicAgent": "simple",
    "CodeAgent": "balanced",
    "WebSearch": "simple",
    "PlatformAdapter": "simple",
    "TranslationAgent": "balanced",
}

# Agent 类型 → 配置角色名映射
ROLE_MAP: dict[str, str] = {
    "ManagerAgent": "manager",
    "WritingAgent": "writing",
    "DiagramAgent": "writing",
    "ImageAgent": "image",
    "ReviewAgent": "subagent",
    "SEOAgent": "subagent",
    "HotTopicAgent": "subagent",
    "CodeAgent": "subagent",
    "WebSearch": "subagent",
    "PlatformAdapter": "subagent",
    "TranslationAgent": "subagent",
}


class ModelRouter:
    """按角色创建并缓存 GatewaySession 实例。

    每个 GatewaySession 暴露 LangChain 兼容接口（ainvoke/astream/bind_tools），
    但内部通过完整的 Gateway 管线调用 LLM。
    """

    def __init__(self):
        self._settings = get_settings()
        # Lazy init — GatewayContainer may not be started yet
        self._container: Any = None
        self._cache: dict[str, Any] = {}  # GatewaySession cache

    def _get_container(self) -> Any:
        """Lazy-init the GatewayContainer singleton."""
        if self._container is None:
            from src.core.gateway.core.container import GatewayContainer
            self._container = GatewayContainer.get_instance_sync()
        return self._container

    def get_chat_model(self, agent_type: str):
        """返回 GatewaySession 实例（替代 ChatOpenAI）。

        GatewaySession 提供 ainvoke/astream/bind_tools 三个方法，
        与 ChatOpenAI 接口兼容，所有 handler 无需修改。

        Args:
            agent_type: Agent 类型，决定 routing hint 和温度等参数
        """
        cache_key = agent_type
        if cache_key in self._cache:
            return self._cache[cache_key]

        container = self._get_container()
        candidates = container.build_candidates_from_settings(self._settings)

        # 如果该 agent_type 有专属 role 配置，prepend 为优先候选
        role = ROLE_MAP.get(agent_type)
        if role:
            try:
                role_cfg = self._settings.get_role_config(role)
                if role_cfg.get("api_key"):
                    from src.core.gateway.core.router import ModelCandidate

                    static_latency = {"openai": 1000, "zhipu": 800, "qwen": 1200}
                    role_candidate = ModelCandidate(
                        provider=role_cfg["provider"],
                        model=role_cfg["model"],
                        api_key=role_cfg["api_key"],
                        base_url=role_cfg["base_url"],
                        input_price=container.settings.pricing.get(
                            role_cfg["model"], {}
                        ).get("input", 0.002),
                        output_price=container.settings.pricing.get(
                            role_cfg["model"], {}
                        ).get("output", 0.002),
                        static_latency_ms=static_latency.get(role_cfg["provider"], 1000),
                    )
                    # Prepend: role-specific candidate has priority
                    candidates.insert(0, role_candidate)
            except Exception as e:
                logger.warning("ModelRouter: failed to build role candidate for %s: %s", role, e)

        hint = HINT_MAP.get(agent_type, "balanced")

        from src.core.gateway.core.session import GatewaySession
        session = GatewaySession(
            agent_type=agent_type,
            candidates=candidates,
            container=container,
            hint=hint,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
        )
        self._cache[cache_key] = session
        logger.info(
            "ModelRouter: created GatewaySession for %s (hint=%s, %d candidates)",
            agent_type, hint, len(candidates),
        )
        return session

    def get_gateway(self, agent_type: str):
        """Alias for get_chat_model — 向后兼容 Orchestrator 的 llm_gateway 参数。

        Orchestrator 使用 get_gateway("ManagerAgent") 获取 LLM 实例用于
        _decompose 和 _fallback_llm_call。GatewaySession 的 ainvoke 完全兼容。
        """
        return self.get_chat_model(agent_type)

    def get_all_models(self) -> dict[str, Any]:
        """返回所有已知 agent_type 的 GatewaySession 字典。"""
        models = {}
        for agent_type in HINT_MAP:
            try:
                models[agent_type] = self.get_chat_model(agent_type)
            except Exception as e:
                logger.warning("ModelRouter: failed to create session for %s: %s", agent_type, e)
        return models

    def invalidate_cache(self) -> None:
        """清空 GatewaySession 缓存，下次调用时重建。"""
        self._cache.clear()
