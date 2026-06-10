"""ModelRouter — 按 Agent 角色创建 LangChain ChatOpenAI 实例

所有请求统一走 LLM Gateway，
Gateway 根据 model 名称路由到对应 provider。

重大改动(2026-06-10):
  1. Gateway URL 改为从配置读取，支持环境变量 LLM_GATEWAY_URL
  2. 发送 model="auto" + X-Role header，让 Gateway 的 AutoRouter 做智能路由
  3. 新增 default_model 配置项，兜底用
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.logging_shared.config import get_logger

logger = get_logger("core.model_router")

_DEFAULT_GATEWAY_URL = "http://localhost:9010/v1"


class ModelRouter:
    """按角色创建并缓存 ChatOpenAI 实例，统一指向 LLM Gateway。

    路由逻辑：
    - 由 Gateway 的 AutoRouter 自动选择最优 provider
    - 每个 Agent 角色通过 default_headers 携带 task_type 提示
    """

    # Agent 类型 → 路由提示（供 Gateway 的 AutoRouter 选择策略参考）
    _ROLE_HINTS = {
        "ManagerAgent": "planning",    # 需要强推理能力
        "WritingAgent": "creative",    # 需要内容生成质量
        "SEOAgent": "creative",
        "ReviewAgent": "analysis",     # 需要准确性
        "DiagramAgent": "creative",
        "CodeAgent": "code",
        "WebSearch": "simple",         # 搜索简单回答即可
    }

    def __init__(self, gateway_url: str | None = None):
        self._settings = get_settings()
        self._cache: dict[str, Any] = {}
        self._gateway_url = (
            gateway_url
            or os.getenv("LLM_GATEWAY_URL")
            or self._settings.llm_gateway_url
            or _DEFAULT_GATEWAY_URL
        )

    def _get_route_hint(self, agent_type: str) -> str:
        return self._ROLE_HINTS.get(agent_type, "balanced")

    def get_chat_model(self, agent_type: str) -> ChatOpenAI:
        """返回 ChatOpenAI 实例，base_url 统一指向 LLM Gateway

        每次请求都发 model="auto"，让 Gateway 的 AutoRouter
        根据请求内容 + route_hint 动态选择最优 provider。
        """
        if agent_type in self._cache:
            return self._cache[agent_type]

        route_hint = self._get_route_hint(agent_type)

        llm = ChatOpenAI(
            model="auto",             # ← 关键变更：让 Gateway 做路由决策
            api_key="gateway",        # Gateway 不校验 key，用自己配置的 provider key
            base_url=self._gateway_url,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            streaming=True,
            timeout=120.0,
            default_headers={
                "X-Route-Hint": route_hint,
                "X-Agent-Type": agent_type,
            },
        )

        self._cache[agent_type] = llm
        logger.info("ModelRouter -> Gateway (auto-route)",
                    agent_type=agent_type,
                    hint=route_hint,
                    gateway=self._gateway_url)
        return llm

    def get_gateway(self, agent_type: str) -> ChatOpenAI:
        return self.get_chat_model(agent_type)

    def invalidate_cache(self) -> None:
        self._cache.clear()
