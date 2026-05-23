"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。"""
from __future__ import annotations
from typing import Any, Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator


def create_orchestrator(
    router: Optional[ModelRouter] = None,
    mcp_tools: Optional[list] = None,
    enable_agent_registry: bool = True,
    enable_message_bus: bool = False,
    enable_reflection: bool = False,
    enable_hitl: bool = False,
    **kwargs,
) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    支持 Phase 1-5 所有组件（条件启用，默认关闭确保向后兼容）。

    Args:
        router: ModelRouter 实例，不传则自动创建
        mcp_tools: 可选的 MCP 工具列表
        enable_agent_registry: 是否启用 AgentRegistry（Phase 1）
        enable_message_bus: 是否启用 P2P 通信（Phase 2）
        enable_reflection: 是否启用反思回路（Phase 3）
        enable_hitl: 是否启用 HITL（Phase 4）
        **kwargs: 传递给 Orchestrator 的额外参数

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
            # WebSearch/ManagerAgent 始终绑定 MCP 工具
            needs_tools = agent_type in ("WebSearch", "ManagerAgent")
            chat_models[agent_type] = router.get_chat_model(
                agent_type,
                tools=mcp_tools if needs_tools and mcp_tools else None,
            )
        except Exception:
            pass

    # ── Phase 1: AgentRegistry ──
    agent_registry = _build_agent_registry(chat_models) if enable_agent_registry else None

    # ── Phase 2: MessageBus ──
    message_bus = _build_message_bus(agent_registry) if enable_message_bus and agent_registry else None

    # ── Phase 3: ReflectionEngine ──
    reflection_engine = _build_reflection_engine(agent_registry) if enable_reflection else None

    # ── Phase 4: HITLManager ──
    hitl_manager = _build_hitl_manager(kwargs.get("task_queue")) if enable_hitl else None

    orchestrator = Orchestrator(
        llm_gateway=router.get_gateway("ManagerAgent"),
        chat_models=chat_models,
        mcp_tools=mcp_tools,
        agent_registry=agent_registry,
        message_bus=message_bus,
        reflection_engine=reflection_engine,
        hitl_manager=hitl_manager,
        **kwargs,
    )
    return orchestrator


def _build_agent_registry(chat_models: dict) -> Any:
    """构建 AgentRegistry"""
    from .agents import AgentRegistry
    return AgentRegistry(llm_models=chat_models)


def _build_message_bus(registry: Any) -> Any:
    """构建 MessageBus"""
    from .communication import MessageBus
    return MessageBus(registry=registry)


def _build_reflection_engine(registry: Any) -> Any:
    """构建 ReflectionEngine"""
    from .reflection import ReflectionEngine, ReflectionConfig
    return ReflectionEngine(config=ReflectionConfig(), registry=registry)


def _build_hitl_manager(task_queue: Any) -> Any:
    """构建 HITLManager"""
    from .hitl import HITLManager
    return HITLManager(task_queue=task_queue)
