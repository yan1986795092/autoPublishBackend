"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。

模型分配由 registry.py 中的 AgentCard 驱动：
新增 Agent 只需在 registry 注册卡片，无需修改此文件。
"""
from __future__ import annotations
import logging
from typing import Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator
from .registry import all as get_all_cards

logger = logging.getLogger(__name__)

# 模块级单例 — ModelRouter 内部有 ChatOpenAI 实例缓存
_shared_router: ModelRouter | None = None


def _get_router() -> ModelRouter:
    global _shared_router
    if _shared_router is None:
        _shared_router = ModelRouter()
    return _shared_router


def create_orchestrator(
    router: Optional[ModelRouter] = None,
    mcp_tools: Optional[list] = None,
    **kwargs,
) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    模型分配策略来自 registry 中每个 AgentCard 的 model 字段。

    Args:
        router: ModelRouter 实例，不传则使用共享单例
        mcp_tools: 可选的 MCP 工具列表
        **kwargs: 传递给 Orchestrator 的额外参数

    Returns:
        已就绪的 Orchestrator 实例
    """
    router = router or _get_router()

    # 从 registry 动态分配模型：遍历所有注册的 AgentCard
    chat_models = {}

    for card in get_all_cards():
        try:
            chat_models[card.name] = router.get_chat_model_for_model(
                model=card.model,
                agent_type=card.name,
            )
        except Exception:
            logger.warning("无法为 %s 创建 LLM（模型: %s）", card.name, card.model)

    # ManagerAgent 单独处理（规划 Agent，固定使用 deepseek-chat）
    for agent_type in ["ManagerAgent"]:
        if agent_type not in chat_models:
            try:
                chat_models[agent_type] = router.get_chat_model_for_model(
                    model="deepseek-chat",
                    agent_type=agent_type,
                )
            except Exception:
                logger.warning("无法为 %s 创建 LLM", agent_type)

    orchestrator = Orchestrator(
        chat_models=chat_models,
        mcp_tools=mcp_tools,
        **kwargs,
    )
    return orchestrator
