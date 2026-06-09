"""ModelRouter — 按 Agent 角色创建 LangChain ChatOpenAI 实例

所有请求统一走 LLM Gateway（http://localhost:9010/v1），
Gateway 根据 model 名称路由到对应 provider。
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.logging_shared.config import get_logger

logger = get_logger("core.model_router")

_GATEWAY_URL = "http://localhost:9010/v1"


class ModelRouter:
    """按角色创建并缓存 ChatOpenAI 实例，统一指向 LLM Gateway。

    路由逻辑：
    - ManagerAgent → manager 角色 → gateway 转发到 DeepSeek
    - WritingAgent → writing 角色 → gateway 转发到 智谱
    - ReviewAgent/SEO/... → subagent 角色 → gateway 转发到 智谱
    """

    def __init__(self, gateway_url: str | None = None):
        self._settings = get_settings()
        self._cache: dict[str, Any] = {}
        self._gateway_url = gateway_url or _GATEWAY_URL

    def _agent_type_to_role(self, agent_type: str) -> str:
        role_map = {
            "ManagerAgent": "manager",
            "WritingAgent": "writing",
        }
        return role_map.get(agent_type, "subagent")

    def _get_model_for_role(self, agent_type: str) -> str:
        """获取指定 Agent 类型使用的模型名（发给 Gateway，由其路由到对应 provider）"""
        role = self._agent_type_to_role(agent_type)
        role_config = self._settings.get_role_config(role)
        return role_config["model"] or "glm-4-flash"

    def get_chat_model(self, agent_type: str) -> ChatOpenAI:
        """返回 ChatOpenAI 实例，base_url 统一指向 LLM Gateway"""
        if agent_type in self._cache:
            return self._cache[agent_type]

        model = self._get_model_for_role(agent_type)

        llm = ChatOpenAI(
            model=model,
            api_key="gateway",  # Gateway 不校验 key，用自己配置的 provider key
            base_url=self._gateway_url,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            streaming=True,
            timeout=120.0,
        )

        self._cache[agent_type] = llm
        logger.info("ModelRouter -> Gateway",
                    agent_type=agent_type,
                    role=self._agent_type_to_role(agent_type),
                    model=model,
                    gateway=self._gateway_url)
        return llm

    def get_gateway(self, agent_type: str) -> ChatOpenAI:
        return self.get_chat_model(agent_type)

    def invalidate_cache(self) -> None:
        self._cache.clear()
