"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。"""
from __future__ import annotations
from typing import Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator


def create_orchestrator(router: Optional[ModelRouter] = None, **kwargs) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    LangGraph 版本：不再手动注册子 Agent，而是将 chat_models 字典传入，
    Graph 节点函数在 state._llm_models 中按 agent_type 获取对应的 ChatOpenAI。

    Args:
        router: ModelRouter 实例，不传则自动创建
        **kwargs: 传递给 Orchestrator 的额外参数（如 async_task_id, task_queue）

    Returns:
        已就绪的 Orchestrator 实例
    """
    router = router or ModelRouter()
    orchestrator = Orchestrator(
        llm_gateway=router.get_gateway("ManagerAgent"),
        chat_models=router.get_all_models(),
        **kwargs,
    )
    return orchestrator
