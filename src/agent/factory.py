"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。"""
from __future__ import annotations
import logging
from typing import Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def create_orchestrator(
    router: Optional[ModelRouter] = None,
    mcp_tools: Optional[list] = None,
    **kwargs,
) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    Args:
        router: ModelRouter 实例，不传则自动创建
        mcp_tools: 可选的 MCP 工具列表
        **kwargs: 传递给 Orchestrator 的额外参数

    Returns:
        已就绪的 Orchestrator 实例
    """
    router = router or ModelRouter()

    # 按 Agent 类型获取 LLM
    chat_models = {}

    # 高推理 Agent → 指定模型（DeepSeek → Gateway 精确匹配）
    for agent_type in ["ManagerAgent", "CodeAgent"]:
        try:
            chat_models[agent_type] = router.get_chat_model_for_model(
                model="deepseek-chat",
                agent_type=agent_type,
            )
        except Exception:
            logger.warning("Failed to create LLM for %s", agent_type)

    # 低成本 Agent → 指定智谱模型（绕过 AutoRouter，Gateway 精确匹配到智谱 adapter）
    for agent_type in [
        "WritingAgent", "SEOAgent", "ReviewAgent",
        "DiagramAgent", "WebSearch",
    ]:
        try:
            chat_models[agent_type] = router.get_chat_model_for_model(
                model="glm-4-flash",
                agent_type=agent_type,
            )
        except Exception:
            pass

    orchestrator = Orchestrator(
        chat_models=chat_models,
        mcp_tools=mcp_tools,
        **kwargs,
    )
    return orchestrator
