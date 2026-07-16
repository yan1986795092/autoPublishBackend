"""ModelRouter — 统一 LLM 创建入口，统一指向 LLM Gateway

两种模式：
  1. get_chat_model(agent_type)  — 自动路由模式
     发送 model="auto" + X-Route-Hint，交给 Gateway 的 AutoRouter 按成本选 cheapest
     适用于：WritingAgent, SEOAgent, ReviewAgent, WebSearch, AIEditorService  # TEMP: DiagramAgent disabled

     2. get_chat_model_for_model(model, agent_type) — 指定模型模式
     发送具体模型名（如 "deepseek-chat"），Gateway 精确匹配到对应 adapter
     适用于：ManagerAgent、CodeAgent（高推理规划，固定走 DeepSeek）

重大改动(2026-06-10):
  1. 拆分为两种模式，支持策略化路由
  2. 移除旧的 _ROLE_HINTS 系统
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
    """统一 LLM 创建入口，所有 ChatOpenAI 实例均指向 LLM Gateway。

    路由策略：
    - 高推理 Agent（ManagerAgent、CodeAgent）→ 指定模型 → Gateway 精确匹配
    - 低成本 Agent（其余所有）→ model="auto" + hint → Gateway AutoRouter 按成本选 cheapest
    """

    def __init__(self, gateway_url: str | None = None):
        self._settings = get_settings()
        self._cache: dict[str, ChatOpenAI] = {}
        self._gateway_url = (
            gateway_url
            or os.getenv("LLM_GATEWAY_URL")
            or self._settings.llm_gateway_url
            or _DEFAULT_GATEWAY_URL
        )

    # ──────────────────────────────────────────
    # 模式 1：自动路由（低成本 Agent）
    # ──────────────────────────────────────────

    @staticmethod
    def _role_to_hint(agent_type: str) -> str:
        """Agent 类型 → Gateway 路由提示（供 AutoRouter 选择策略参考）"""
        mapping = {
            "WritingAgent": "creative",
            "SEOAgent": "creative",
            "ReviewAgent": "analysis",
            # "DiagramAgent": "creative",  # TEMP: DiagramAgent disabled
            "WebSearch": "simple",
            "AIEditorService": "simple",
            "ContentService": "simple",
        }
        return mapping.get(agent_type, "balanced")

    def get_chat_model(self, agent_type: str) -> ChatOpenAI:
        """自动路由模式 — 发送 model='auto' + X-Route-Hint

        Gateway 的 AutoRouter 根据 hint + 消息内容分析，
        按成本优化权重（w_cost=0.7）选 cheapest provider。
        """
        if agent_type in self._cache:
            return self._cache[agent_type]

        hint = self._role_to_hint(agent_type)

        llm = ChatOpenAI(
            model="auto",
            api_key="gateway",       # Gateway 不校验 key，用自己配置的 provider key
            base_url=self._gateway_url,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            streaming=True,
            timeout=120.0,
            default_headers={
                "X-Route-Hint": hint,
                "X-Agent-Type": agent_type,
            },
        )

        self._cache[agent_type] = llm
        logger.info("ModelRouter -> Gateway (auto-route)",
                    agent_type=agent_type,
                    hint=hint,
                    gateway=self._gateway_url)
        return llm

    # ──────────────────────────────────────────
    # 模式 2：指定模型（高推理 Agent）
    # ──────────────────────────────────────────

    def get_chat_model_for_model(
        self,
        model: str,
        agent_type: str = "unknown",
    ) -> ChatOpenAI:
        """指定模型模式 — 发送具体模型名，Gateway 精确匹配到对应 adapter

        当前适用于 ManagerAgent / CodeAgent → 固定走 DeepSeek。
        不传 X-Route-Hint，避免 Gateway 误入 AutoRouter 路径。

        缓存实例（以 model:agent_type 为键），避免 repeated create_orchestrator 调用
        重复创建相同的 LLM 连接。
        """
        cache_key = f"{model}:{agent_type}"
        if cache_key in self._cache:
            logger.debug("ModelRouter cache hit (exact-model)",
                         agent_type=agent_type, model=model)
            return self._cache[cache_key]

        llm = ChatOpenAI(
            model=model,
            api_key="gateway",
            base_url=self._gateway_url,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            streaming=True,
            timeout=120.0,
            default_headers={
                "X-Agent-Type": agent_type,
            },
        )

        self._cache[cache_key] = llm
        logger.info("ModelRouter -> Gateway (exact-model)",
                    agent_type=agent_type,
                    model=model,
                    gateway=self._gateway_url)
        return llm

    def invalidate_cache(self) -> None:
        self._cache.clear()
