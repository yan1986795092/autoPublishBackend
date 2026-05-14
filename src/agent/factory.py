"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。"""
from __future__ import annotations
from typing import Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator


def create_orchestrator(
    router: Optional[ModelRouter] = None,
    mcp_tools: Optional[list] = None,
    **kwargs,
) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    LangGraph 版本：不再手动注册子 Agent，而是将 chat_models 字典传入，
    Graph 节点函数在 state._llm_models 中按 agent_type 获取对应的 ChatOpenAI。

    Args:
        router: ModelRouter 实例，不传则自动创建
        mcp_tools: 可选的 MCP 工具列表（由 langchain-mcp-adapters 加载），
                   会绑定到各 Agent 的 LLM 实例上，使其可以调用搜索等工具
        **kwargs: 传递给 Orchestrator 的额外参数（如 async_task_id, task_queue）

    Returns:
        已就绪的 Orchestrator 实例
    """
    router = router or ModelRouter()

    # 按 Agent 类型获取已绑定 MCP 工具的 LLM
    chat_models = {}
    for agent_type in [
        "ManagerAgent", "WritingAgent", "SEOAgent", "ReviewAgent",
        "DiagramAgent", "CodeAgent", "WebSearch", "HotTopicAgent",
        "PlatformAdapter", "TranslationAgent",
    ]:
        try:
            # WebSearch/WebResearch 类型始终绑定 MCP 工具
            needs_tools = agent_type in ("WebSearch", "ManagerAgent")
            chat_models[agent_type] = router.get_chat_model(
                agent_type,
                tools=mcp_tools if needs_tools and mcp_tools else None,
            )
        except Exception:
            pass

    orchestrator = Orchestrator(
        llm_gateway=router.get_gateway("ManagerAgent"),
        chat_models=chat_models,
        mcp_tools=mcp_tools,
        **kwargs,
    )
    return orchestrator
