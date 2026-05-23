"""共享工厂：创建带 ModelRouter + LangGraph 的 Orchestrator 实例。"""
from __future__ import annotations
import logging
from typing import Any, Optional

from src.core.model_router import ModelRouter
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def create_orchestrator(
    router: Optional[ModelRouter] = None,
    mcp_tools: Optional[list] = None,
    enable_agent_registry: bool = True,
    enable_message_bus: bool = False,
    enable_reflection: bool = False,
    enable_hitl: bool = False,
    enable_tools: bool = True,
    **kwargs,
) -> Orchestrator:
    """创建 Orchestrator 实例，用 ModelRouter 为各 Agent 分配独立 LLM 实例。

    支持 Phase 1-5 所有组件 + Tool System（条件启用，默认关闭确保向后兼容）。

    Args:
        router: ModelRouter 实例，不传则自动创建
        mcp_tools: 可选的 MCP 工具列表
        enable_agent_registry: 是否启用 AgentRegistry（Phase 1）
        enable_message_bus: 是否启用 P2P 通信（Phase 2）
        enable_reflection: 是否启用反思回路（Phase 3）
        enable_hitl: 是否启用 HITL（Phase 4）
        enable_tools: 是否启用 ToolRegistry（Tool System）
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

    # ── Tool System: ToolRegistry ──
    tool_registry = _build_tool_registry(
        agent_registry=agent_registry,
        mcp_tools=mcp_tools,
        message_bus=message_bus,
        reflection_engine=reflection_engine,
        hitl_manager=hitl_manager,
    ) if enable_tools else None

    # 如果 ToolRegistry 就绪且 AgentRegistry 就绪，为各 Agent LLM 绑定工具
    if tool_registry and agent_registry:
        for agent_type in agent_registry.list_types():
            agent = agent_registry.get(agent_type)
            if agent and agent.profile.tools_enabled:
                agent.bind_tools(tool_registry)

    orchestrator = Orchestrator(
        llm_gateway=router.get_gateway("ManagerAgent"),
        chat_models=chat_models,
        mcp_tools=mcp_tools,
        agent_registry=agent_registry,
        message_bus=message_bus,
        reflection_engine=reflection_engine,
        hitl_manager=hitl_manager,
        tool_registry=tool_registry,
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


def _build_tool_registry(
    agent_registry: Any = None,
    mcp_tools: Optional[list] = None,
    message_bus: Any = None,
    reflection_engine: Any = None,
    hitl_manager: Any = None,
) -> Any:
    """构建 ToolRegistry 并注册所有工具

    注册顺序：
    1. 内置工具（memory / communication / reflection / hitl）
    2. MCP 工具（如果提供了 mcp_tools）
    """
    from .tools import ToolRegistry, create_memory_tools

    registry = ToolRegistry()

    # 1. 记忆工具（始终可用，依赖 memory_manager 单例）
    for tool in create_memory_tools():
        registry.register(tool)

    # 2. 通信工具（依赖 MessageBus 实例）
    if message_bus is not None:
        from .tools import create_communication_tools
        for tool in create_communication_tools(message_bus):
            registry.register(tool)

    # 3. 反思工具（依赖 ReflectionEngine 实例）
    if reflection_engine is not None:
        from .tools import create_reflection_tools
        for tool in create_reflection_tools(reflection_engine):
            registry.register(tool)

    # 4. HITL 工具（依赖 HITLManager 实例）
    if hitl_manager is not None:
        from .tools import create_hitl_tools
        for tool in create_hitl_tools(hitl_manager):
            registry.register(tool)

    # 5. MCP 工具
    if mcp_tools:
        for lc_tool in mcp_tools:
            from .tools import AgentToolMeta
            registry.register_from_langchain(
                lc_tool,
                meta=AgentToolMeta(category="search", builtin=False),
            )

    logger.info(
        "ToolRegistry: 已注册 %d 个工具（记忆=%s, 通信=%s, 反思=%s, HITL=%s, MCP=%d）",
        registry.count,
        "✓", "✓" if message_bus else "✗",
        "✓" if reflection_engine else "✗",
        "✓" if hitl_manager else "✗",
        len(mcp_tools) if mcp_tools else 0,
    )
    return registry
